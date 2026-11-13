"""Command line entry point: catalog, picker, download, chat."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import catalog, splitter
from . import __version__


def home():
    return Path(os.environ.get("OPENLLM_HOME") or (Path.home() / ".openllm"))


def models_dir():
    path = home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(entry):
    return models_dir() / catalog.slug_for(entry.repo)


def installed(entry):
    return (model_dir(entry) / splitter.MANIFEST).exists()


def log(message):
    print("  " + message, flush=True)


def disk_free():
    try:
        return shutil.disk_usage(str(models_dir())).free
    except OSError:
        return 0


def human_gb(value):
    return "%.2f GB" % (value / 1e9)


# --------------------------------------------------------------------------- list


def cmd_list(args):
    free = disk_free()
    print()
    print("  %-3s %-18s %-34s %9s  %s" % ("", "key", "huggingface repo", "size", "state"))
    print("  " + "-" * 76)
    for entry in catalog.CATALOG:
        state = "installed" if installed(entry) else ("gated" if entry.gated else "-")
        if entry.approx_gb * 1e9 > free and not installed(entry):
            state += ", low disk"
        print("  %-3s %-18s %-34s %9.2f  %s" % (
            "*" if installed(entry) else " ", entry.key, entry.repo,
            entry.approx_gb, state))
    print()
    print("  models live in %s" % models_dir())
    print("  free disk: %s   max size: %.0fB params" % (human_gb(free), catalog.MAX_PARAMS_B))
    print()
    return 0


# --------------------------------------------------------------------------- picker


def pick(only_installed=False):
    entries = [e for e in catalog.CATALOG if installed(e)] if only_installed else list(catalog.CATALOG)
    if not entries:
        log("no models installed yet - run: openllm list")
        return None

    free = disk_free()
    print()
    print("  Select a model")
    print("  " + "-" * 76)
    for index, entry in enumerate(entries, 1):
        state = "installed" if installed(entry) else "%5.2f GB" % entry.approx_gb
        print("  %2d. %-30s %-12s %s" % (index, entry.label, entry.key, state))
    print()

    while True:
        try:
            raw = input("  number (enter to cancel) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        if not raw.isdigit() or not (1 <= int(raw) <= len(entries)):
            log("pick a number between 1 and %d" % len(entries))
            continue
        entry = entries[int(raw) - 1]
        if not installed(entry) and entry.approx_gb * 1e9 > free:
            log("not enough disk: need about %.2f GB" % entry.approx_gb)
            continue
        return entry


# --------------------------------------------------------------------------- pull


def cmd_pull(args):
    entry = catalog.lookup(args.model)
    if not entry.fits_limit:
        log("%s is above the %.0fB ceiling for this build" % (entry.key, catalog.MAX_PARAMS_B))
        return 1

    dest = model_dir(entry)
    if installed(entry) and not args.force:
        log("%s is already installed in %s" % (entry.label, dest))
        log("use --force to re-download and re-split it")
        return 0

    free = disk_free()
    need = entry.approx_gb * 1e9 * 1.15
    if need > free and not args.force:
        log("need about %s free, only %s available" % (human_gb(need), human_gb(free)))
        log("pass --force to try anyway")
        return 1

    staging = home() / "downloads" / catalog.slug_for(entry.repo)
    try:
        splitter.download(entry.repo, staging, progress=log)
    except Exception as exc:
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            log("this repo is gated; accept the licence on the HF page, then set HF_TOKEN")
        else:
            log("download failed: %s: %s" % (name, exc))
        return 1

    try:
        manifest = splitter.split(entry.repo, staging, dest,
                                  keep_source=args.keep_source, progress=log)
    except Exception as exc:
        log("split failed: %s: %s" % (type(exc).__name__, exc))
        return 1

    log("done: %d layers, %s" % (manifest["layers"], human_gb(manifest["total_bytes"])))
    log("chat with it:  openllm chat %s" % entry.key)
    return 0


# --------------------------------------------------------------------------- chat


def runtime_ok():
    missing = []
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            __import__(name)
        except Exception as exc:
            missing.append("%s (%s)" % (name, type(exc).__name__))
    return missing


def cmd_chat(args):
    entry = None
    if args.model:
        try:
            entry = catalog.lookup(args.model)
        except KeyError as exc:
            log(str(exc))
            return 1
    else:
        entry = pick(only_installed=True)
        if entry is None:
            return 1

    if not installed(entry):
        log("%s is not installed yet - run: openllm pull %s" % (entry.label, entry.key))
        return 1

    missing = runtime_ok()
    if missing or args.mock:
        if missing:
            log("real runtime unavailable: " + ", ".join(missing))
        log("falling back to the simulated backend (shards are ignored)")
        from . import mock
        return mock.chat(entry.label, args.max_new_tokens)

    from .runner import LayerStreamer
    from .ui import Renderer

    target = model_dir(entry)
    log("building skeleton from %d shards" % len(list(target.glob("layer_*.safetensors"))))
    streamer = LayerStreamer(
        target,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        streamer.build()
    except Exception as exc:
        log("could not build the model: %s: %s" % (type(exc).__name__, exc))
        return 1

    log("ready - layer %d, device %s, dtype %s" % (
        streamer.layer_count, args.device, streamer.dtype))

    messages = []
    while True:
        try:
            raw = input("\n  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("/exit", "/quit", "exit", "quit"):
            break
        if low == "/clear":
            messages = []
            log("history cleared")
            continue

        messages.append({"role": "user", "content": raw})
        error = None
        view = Renderer(streamer.layer_count, entry.label, args.max_new_tokens)
        with view:
            view.set_phase("prefill")
            for event in streamer.stream(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            ):
                if event.type == "layer_start":
                    view.on_layer(event.index)
                elif event.type == "token":
                    view.on_token(event.text)
                elif event.type == "error":
                    error = event.text
            view.set_phase("done")

        if error:
            messages.pop()
            log("generation failed: " + error)
            continue

        view.flush_text()
        messages.append({"role": "assistant", "content": view.text.strip()})

    streamer.close()
    return 0


# --------------------------------------------------------------------------- rm / doctor


def cmd_rm(args):
    entry = catalog.lookup(args.model)
    target = model_dir(entry)
    if not target.exists():
        log("%s is not installed" % entry.key)
        return 1
    shutil.rmtree(target)
    log("removed %s" % target)
    return 0


def cmd_doctor(args):
    print()
    log("openllm %s" % __version__)
    log("python %s" % sys.version.split()[0])
    log("home %s" % home())
    log("free disk %s" % human_gb(disk_free()))
    for name in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub"):
        try:
            module = __import__(name)
            log("%-16s %s" % (name, getattr(module, "__version__", "?")))
        except Exception as exc:
            log("%-16s MISSING (%s)" % (name, type(exc).__name__))
    log("HF_TOKEN %s" % ("set" if os.environ.get("HF_TOKEN") else "not set (needed for gated repos)"))
    try:
        import torch
        log("device %s" % ("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:
        pass
    found = [p.name for p in models_dir().iterdir() if (p / splitter.MANIFEST).exists()]
    log("installed %s" % (", ".join(found) if found else "none"))
    print()
    return 0


# --------------------------------------------------------------------------- main


def interactive():
    from . import mock

    entries = [e for e in catalog.CATALOG if installed(e)]
    if not entries:
        log("no models installed - showing the catalog")
        cmd_list(None)
        entry = pick()
        if entry is None:
            log("run 'openllm pull <key>' to download one")
            return 1
        if not installed(entry):
            log("then: openllm pull %s" % entry.key)
            return 1
    else:
        entry = pick(only_installed=True)
    if entry is None:
        return 1

    ns = argparse.Namespace(
        model=entry.key, device="cpu", dtype="auto",
        max_new_tokens=256, temperature=0.7, mock=False,
    )
    return cmd_chat(ns)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openllm",
        description="Layer-streamed inference for Hugging Face models.",
    )
    parser.add_argument("--version", action="version", version="openllm " + __version__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="show the model catalog").set_defaults(func=cmd_list)

    pull = sub.add_parser("pull", help="download and split a model")
    pull.add_argument("model", help="catalog key or owner/repo")
    pull.add_argument("--keep-source", action="store_true", help="keep the raw download")
    pull.add_argument("--force", action="store_true", help="ignore disk checks and reinstall")
    pull.set_defaults(func=cmd_pull)

    chat = sub.add_parser("chat", help="chat with an installed model")
    chat.add_argument("model", nargs="?", help="catalog key (omit to pick)")
    chat.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    chat.add_argument("--dtype", default="auto", help="auto, float32, float16, bfloat16")
    chat.add_argument("--max-new-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--mock", action="store_true", help="force the simulated backend")
    chat.set_defaults(func=cmd_chat)

    rm = sub.add_parser("rm", help="delete an installed model")
    rm.add_argument("model")
    rm.set_defaults(func=cmd_rm)

    sub.add_parser("doctor", help="check the runtime").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        return interactive()
    return args.func(args)
