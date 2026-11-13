"""Download a Hugging Face checkpoint and split it into per-layer shards.

This is the real thing: ``split()`` reads the safetensors index, groups
every tensor by its transformer layer (``layers.N.`` / ``h.N.``), and
writes one file per layer plus ``embed``, ``head`` and ``other`` groups,
with a ``manifest.json`` describing the result.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "manifest.json"
FORMAT = "openllm-layer-shard-v1"
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"

HUB_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    "special_tokens*",
]
IGNORE_PATTERNS = [
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "consolidated*",
    "*.pth",
]

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


def download(repo, dest, token=None, revision=None, progress=None):
    """Fetch only the files we need (safetensors + config + tokenizer)."""
    from huggingface_hub import snapshot_download

    say = progress or (lambda msg: None)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    say("downloading %s from huggingface.co" % repo)
    return snapshot_download(
        repo_id=repo,
        revision=revision,
        token=token,
        local_dir=str(dest),
        allow_patterns=HUB_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )


# --------------------------------------------------------------------------- index


def weight_files(source):
    source = Path(source)
    index = source / INDEX_FILE
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        files = sorted({source / name for name in data.get("weight_map", {}).values()})
        if files:
            return files
    single = source / SINGLE_FILE
    if single.exists():
        return [single]
    found = sorted(source.glob("*.safetensors"))
    if not found:
        raise FileNotFoundError("no .safetensors weights inside %s" % source)
    return found


def key_map(files):
    """Map every tensor name to the archive that holds it, without loading data."""
    from safetensors import safe_open

    mapping = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                mapping[key] = path
    return mapping


def _write_group(mapping, names, out_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    by_file = {}
    for name in names:
        by_file.setdefault(mapping[name], []).append(name)

    tensors = {}
    for path, keys in by_file.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key).contiguous()

    save_file(tensors, str(out_path))
    size = out_path.stat().st_size
    del tensors
    return size


def _dtype_of(mapping, names):
    from safetensors import safe_open

    key = sorted(names)[0]
    try:
        with safe_open(str(mapping[key]), framework="pt", device="cpu") as handle:
            return str(handle.get_slice(key).get_dtype())
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- split


def split(repo, source, dest, keep_source=False, progress=None):
    """Split ``source`` (a HF snapshot) into layer shards under ``dest``."""
    say = progress or (lambda msg: None)
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = weight_files(source)
    say("indexing %d weight archive(s)" % len(files))
    mapping = key_map(files)

    groups = {}
    for name in mapping:
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
        "tensors": len(mapping),
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
        if group.startswith("layer:"):
            index = int(group.split(":")[1])
            out = dest / ("layer_%02d.safetensors" % index)
            manifest["files"].setdefault("layer_files", []).append(out.name)
        else:
            out = dest / ("%s.safetensors" % group)
            manifest["files"][group] = out.name

        say("writing %s (%d tensors)" % (out.name, len(names)))
        size = _write_group(mapping, names, out)
        total += size
        if manifest["dtype"] is None:
            manifest["dtype"] = _dtype_of(mapping, names)
        if group.startswith("layer:"):
            manifest["layer_bytes"].append(size)

    manifest["total_bytes"] = total

    for pattern in ("*.json", "*.txt", "*.model", "*.jinja"):
        for path in source.glob(pattern):
            if path.name in (INDEX_FILE, MANIFEST):
                continue
            shutil.copy2(path, dest / path.name)

    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    say("wrote %d layer shards, %.2f GB total" % (len(layer_ids), total / 1e9))

    if not keep_source:
        say("removing the original download in %s" % source)
        shutil.rmtree(source, ignore_errors=True)

    return manifest
