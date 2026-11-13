"""Download a Hugging Face checkpoint and split it into per-layer shards.

Pure standard library.  A safetensors file is a tiny JSON header followed
by raw tensor bytes, so sharding is just: parse each header, regroup the
tensor entries by layer, then stream the byte ranges into new files with
rebased offsets.  No torch, no safetensors package, no numpy - which is
what makes ``openllm pull`` work on a phone.
"""

import json
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

from . import hub

MANIFEST = "manifest.json"
FORMAT = "openllm-layer-shard-v1"

TEXT_SUFFIXES = (".json", ".txt", ".model", ".jinja")
SKIP_TEXT = ("model.safetensors.index.json", MANIFEST)

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


# --------------------------------------------------------------------------- naming


def layer_index(name):
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def bucket(name):
    index = layer_index(name)
    if index is not None:
        return "layer:%d" % index
    if "lm_head" in name:
        return "head"
    if "embed" in name or name.endswith("wte.weight") or "shared" in name:
        return "embed"
    return "other"


# --------------------------------------------------------------------------- download


def download(repo, dest, token=None, progress=None):
    """Fetch only the safetensors, config and tokenizer files."""
    say = progress or (lambda message: None)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = hub.list_files(repo, token_value=token)
    say("found %d files in %s" % (len(files), repo))
    hub.download(repo, files, dest, token_value=token, progress=say)
    return dest


# --------------------------------------------------------------------------- safetensors


def read_header(path):
    """Return (header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise IOError("%s is too small to be safetensors" % path)
        size = struct.unpack("<Q", prefix)[0]
        if size <= 0 or size > 200 * 1024 * 1024:
            raise IOError("implausible safetensors header in %s" % path)
        raw = handle.read(size)
    header = json.loads(raw.decode("utf-8"))
    return header, 8 + size


def pack_header(header):
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    return struct.pack("<Q", len(body)) + body


def scan(source):
    """Map every tensor name to its location, without reading tensor data."""
    source = Path(source)
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("no .safetensors files in %s" % source)

    info = {}
    metadata = None
    for path in files:
        header, data_start = read_header(path)
        meta = header.pop("__metadata__", None)
        if meta and metadata is None:
            metadata = meta
        for name, entry in header.items():
            begin, end = entry.get("data_offsets", (0, 0))
            if end < begin:
                raise IOError("bad data offsets for %s in %s" % (name, path))
            info[name] = {
                "path": path,
                "offset": data_start + begin,
                "length": end - begin,
                "dtype": entry.get("dtype", "F32"),
                "shape": entry.get("shape", []),
            }
    return info, metadata, files


def write_shard(names, info, out_path, metadata=None):
    header = {}
    if metadata:
        header["__metadata__"] = metadata

    cursor = 0
    for name in names:
        item = info[name]
        header[name] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "data_offsets": [cursor, cursor + item["length"]],
        }
        cursor += item["length"]

    with open(out_path, "wb") as dst:
        dst.write(pack_header(header))
        for name in names:
            item = info[name]
            with open(item["path"], "rb") as src:
                src.seek(item["offset"])
                remaining = item["length"]
                while remaining > 0:
                    chunk = src.read(min(1 << 20, remaining))
                    if not chunk:
                        raise IOError("unexpected end of %s" % item["path"])
                    dst.write(chunk)
                    remaining -= len(chunk)

    return out_path.stat().st_size


# --------------------------------------------------------------------------- split


def split(repo, source, dest, keep_source=False, progress=None):
    say = progress or (lambda message: None)
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    info, metadata, weight_files = scan(source)
    say("indexed %d tensors across %d file(s)" % (len(info), len(weight_files)))

    groups = {}
    for name in info:
        groups.setdefault(bucket(name), []).append(name)

    layer_ids = sorted(int(g.split(":")[1]) for g in groups if g.startswith("layer:"))
    if not layer_ids:
        raise RuntimeError(
            "no transformer layers found; this repo is probably not a supported decoder"
        )
    if layer_ids != list(range(len(layer_ids))):
        raise RuntimeError("non-contiguous layer numbering: %s" % layer_ids)

    manifest = {
        "format": FORMAT,
        "repo": repo,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": len(layer_ids),
        "tensors": len(info),
        "layer_bytes": [],
        "dtype": None,
        "files": {},
    }

    order = ["embed"] + ["layer:%d" % i for i in layer_ids] + ["head", "other"]
    total = 0
    for group in order:
        names = groups.get(group)
        if not names:
            continue
        names.sort()

        if group.startswith("layer:"):
            index = int(group.split(":")[1])
            out = dest / ("layer_%02d.safetensors" % index)
            manifest["files"].setdefault("layer_files", []).append(out.name)
        else:
            out = dest / ("%s.safetensors" % group)
            manifest["files"][group] = out.name

        say("writing %s (%d tensors)" % (out.name, len(names)))
        size = write_shard(names, info, out, metadata)
        total += size
        if manifest["dtype"] is None:
            manifest["dtype"] = info[names[0]]["dtype"]
        if group.startswith("layer:"):
            manifest["layer_bytes"].append(size)

    manifest["total_bytes"] = total

    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES and path.name not in SKIP_TEXT:
            shutil.copy2(path, dest / path.name)

    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    say("wrote %d layer shards, %.2f GB total" % (len(layer_ids), total / 1e9))

    if not keep_source:
        shutil.rmtree(source, ignore_errors=True)

    return manifest
