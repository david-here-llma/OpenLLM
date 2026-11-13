"""Self-contained checks, so the program can be verified without a shell.

    openllm selftest              # everything
    openllm selftest qwen2.5-0.5b # one installed model

Exit code is non-zero if any check fails, so it can gate a setup script.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import catalog, splitter

PASS = "\033[38;5;42mPASS\033[0m"
WARN = "\033[38;5;214mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class Report:
    def __init__(self):
        self.failed = 0
        self.warned = 0

    def ok(self, message):
        print("  %s  %s" % (PASS, message), flush=True)

    def warn(self, message):
        self.warned += 1
        print("  %s  %s" % (WARN, message), flush=True)

    def fail(self, message):
        self.failed += 1
        print("  %s  %s" % (FAIL, message), flush=True)


# --------------------------------------------------------------------------- checks


def check_python(report):
    version = sys.version.split()[0]
    if sys.version_info < (3, 8):
        report.fail("python %s is too old (need 3.8 or newer)" % version)
    else:
        report.ok("python %s" % version)


def check_roundtrip(report):
    """Write a tiny safetensors file, split it, and compare every byte."""
    work = Path(tempfile.mkdtemp(prefix="openllm-selftest-"))
    try:
        source = work / "src"
        source.mkdir()
        tensors = [
            ("model.embed_tokens.weight", "F32", [2], b"\x00\x01\x02\x03" * 2),
            ("model.layers.0.self_attn.q_proj.weight", "BF16", [2], b"\x10\x11\x12\x13"),
            ("model.layers.1.mlp.down_proj.weight", "BF16", [4], b"\x20\x21\x22\x23" * 2),
            ("model.layers.0.input_layernorm.weight", "F32", [2], b"\x30\x31\x32\x33" * 2),
            ("lm_head.weight", "F32", [1], b"\x40\x41\x42\x43"),
            ("model.norm.weight", "F32", [1], b"\x50\x51\x52\x53"),
        ]

        header = {}
        cursor = 0
        for name, dtype, shape, raw in tensors:
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [cursor, cursor + len(raw)]}
            cursor += len(raw)

        path = source / "model.safetensors"
        with open(path, "wb") as handle:
            handle.write(splitter.pack_header(header))
            for _, _, _, raw in tensors:
                handle.write(raw)

        info, _, _ = splitter.scan(source)
        if len(info) != len(tensors):
            report.fail("round-trip: indexed %d of %d tensors" % (len(info), len(tensors)))
            return

        names = [item[0] for item in tensors]
        out = work / "shard.safetensors"
        splitter.write_shard(names, info, out)
        new_header, data_start = splitter.read_header(out)
        expected = {name: raw for name, _, _, raw in tensors}

        problems = []
        for name in names:
            entry = new_header.get(name)
            if entry is None:
                problems.append("missing " + name)
                continue
            begin, end = entry["data_offsets"]
            with open(out, "rb") as handle:
                handle.seek(data_start + begin)
                got = handle.read(end - begin)
            if got != expected[name]:
                problems.append("bytes differ for " + name)

        if problems:
            report.fail("round-trip: " + "; ".join(problems[:3]))
        else:
            report.ok("safetensors round-trip (%d tensors, %d bytes)"
                      % (len(names), cursor))
    except Exception as exc:
        report.fail("round-trip crashed: %s: %s" % (type(exc).__name__, exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check_buckets(report):
    cases = {
        "model.layers.7.mlp.gate_proj.weight": "layer:7",
        "transformer.h.3.attn.c_attn.weight": "layer:3",
        "model.embed_tokens.weight": "embed",
        "lm_head.weight": "head",
        "model.norm.weight": "other",
    }
    bad = [name for name, want in cases.items() if splitter.bucket(name) != want]
    if bad:
        report.fail("layer grouping wrong for: " + ", ".join(bad[:3]))
    else:
        report.ok("layer grouping (%d naming cases)" % len(cases))


def check_lookup(report):
    try:
        upper = catalog.lookup("Qwen2.5-0.5B")
        lower = catalog.lookup("qwen2.5-0.5b")
        rejected = False
        try:
            catalog.lookup("definitely-not-a-model")
        except catalog.UnknownModel:
            rejected = True
    except Exception as exc:
        report.fail("catalog lookup: %s: %s" % (type(exc).__name__, exc))
        return

    if upper.key != "qwen2.5-0.5b" or lower.key != "qwen2.5-0.5b":
        report.fail("catalog lookup returned the wrong entry")
    elif not rejected:
        report.fail("catalog lookup accepts nonsense without an error")
    else:
        report.ok("catalog lookup (case-insensitive)")


def check_hub(report):
    try:
        from . import hub
        files = hub.list_files("HuggingFaceTB/SmolLM2-135M-Instruct")
    except Exception as exc:
        report.warn("hub unreachable: %s" % (str(exc) or type(exc).__name__))
        return

    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        report.fail("hub returned no weight archives for SmolLM2-135M")
    else:
        report.ok("hub reachable (%d files, %d weight archive(s))"
                  % (len(files), len(weights)))


def check_runtime(report):
    missing = []
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            __import__(name)
        except Exception as exc:
            missing.append("%s (%s)" % (name, type(exc).__name__))
    if missing:
        report.warn("chat runtime incomplete: " + ", ".join(missing))
    else:
        report.ok("chat runtime (torch, transformers, accelerate, safetensors)")


def check_models(report, only=None):
    from . import cli

    root = cli.models_dir()
    found = 0
    for path in sorted(root.iterdir()):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            continue
        if only and path.name != only and path.name != catalog.slug_for(only):
            continue
        found += 1

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.fail("%s: unreadable manifest (%s)" % (path.name, exc))
            continue

        layer_files = manifest.get("files", {}).get("layer_files", [])
        declared = int(manifest.get("layers", 0))
        if len(layer_files) != declared:
            report.fail("%s: manifest says %d layers but lists %d shards"
                        % (path.name, declared, len(layer_files)))
            continue

        problems = []
        tensors = 0
        for name in layer_files:
            shard = path / name
            if not shard.is_file():
                problems.append(name + " missing")
                continue
            try:
                header, data_start = splitter.read_header(shard)
                tensors += len(header)
                size = shard.stat().st_size
                for key, entry in header.items():
                    begin, end = entry.get("data_offsets", (0, 0))
                    if data_start + end > size:
                        problems.append("%s: %s overruns the file" % (name, key))
                        break
            except Exception as exc:
                problems.append("%s: %s" % (name, exc))

        for group in ("embed", "head", "other"):
            name = manifest.get("files", {}).get(group)
            if not name:
                continue
            shard = path / name
            if not shard.is_file():
                problems.append(name + " missing")
                continue
            try:
                header, _ = splitter.read_header(shard)
                tensors += len(header)
            except Exception as exc:
                problems.append("%s: %s" % (name, exc))

        config_path = path / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                depth = config.get("num_hidden_layers")
                if depth is not None and int(depth) != declared:
                    problems.append("config has %s layers, shards have %d"
                                    % (depth, declared))
            except Exception as exc:
                problems.append("config.json: %s" % exc)
        else:
            problems.append("config.json missing")

        if problems:
            report.fail("%s: %s" % (path.name, "; ".join(problems[:3])))
        else:
            report.ok("%s (%d layers, %d tensors)" % (path.name, declared, tensors))

    if not found:
        report.warn("no models installed (try: openllm pull qwen2.5-0.5b)")


# --------------------------------------------------------------------------- entry


def run(argv=None):
    argv = [item for item in (argv or []) if not item.startswith("-")]
    only = argv[0] if argv else None

    report = Report()
    print()
    print("  OpenLLM selftest")
    print("  " + "-" * 62)
    check_python(report)
    check_roundtrip(report)
    check_buckets(report)
    check_lookup(report)
    check_hub(report)
    check_runtime(report)
    check_models(report, only)
    print("  " + "-" * 62)
    if report.failed:
        print("  %d failed, %d warning(s)" % (report.failed, report.warned))
    elif report.warned:
        print("  all checks passed, %d warning(s)" % report.warned)
    else:
        print("  all checks passed")
    print()
    return 1 if report.failed else 0
