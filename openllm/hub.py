"""Minimal Hugging Face Hub client built on the standard library only.

No ``huggingface_hub``, no ``torch``: just urllib.  That is what lets
``openllm pull`` work in plain Termux, where those packages are not
available.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models/{repo}"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"

WANTED = re.compile(
    r"(\.safetensors$|\.json$|\.txt$|\.model$|\.jinja$|^tokenizer|^special_tokens)",
    re.IGNORECASE,
)
SKIP = re.compile(
    r"(\.index\.json$|\.bin$|\.h5$|\.msgpack$|\.gguf$|\.onnx|\.pth$|onnx/)",
    re.IGNORECASE,
)


class HubError(RuntimeError):
    pass


def token(explicit=None):
    return (explicit or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _request(url, token_value=None):
    headers = {"User-Agent": "openllm/0.2"}
    if token_value:
        headers["Authorization"] = "Bearer " + token_value
    return urllib.request.Request(url, headers=headers)


def _explain(exc, repo, path=None):
    if exc.code in (401, 403):
        return ("%s is gated or private. Accept the licence at "
                "https://huggingface.co/%s then: export HF_TOKEN=..." % (repo, repo))
    if exc.code == 404:
        return "%s not found on huggingface.co%s" % (
            repo, "" if not path else " (%s)" % path)
    return "huggingface.co returned HTTP %s for %s" % (exc.code, repo)


def list_files(repo, token_value=None):
    """Return the file names we need: weights, config and tokenizer."""
    request = _request(API.format(repo=repo), token(token_value))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise HubError(_explain(exc, repo)) from exc
    except urllib.error.URLError as exc:
        raise HubError("cannot reach huggingface.co: %s" % exc.reason) from exc

    names = [item.get("rfilename", "") for item in payload.get("siblings", [])]
    wanted = []
    for name in names:
        if not name or "/" in name:
            continue
        if WANTED.search(name) and not SKIP.search(name):
            wanted.append(name)

    wanted.sort(key=lambda item: (not item.endswith(".safetensors"), item))
    if not any(name.endswith(".safetensors") for name in wanted):
        raise HubError("no .safetensors weights in %s (repo may be GGUF-only)" % repo)
    return wanted


def download(repo, files, dest, token_value=None, progress=None):
    say = progress or (lambda message: None)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in files:
        target = dest / name
        url = RESOLVE.format(repo=repo, path=name)
        try:
            with urllib.request.urlopen(_request(url, token(token_value)), timeout=120) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with open(target, "wb") as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
        except urllib.error.HTTPError as exc:
            raise HubError(_explain(exc, repo, name)) from exc
        except urllib.error.URLError as exc:
            raise HubError("download failed: %s" % exc.reason) from exc

        if total and done != total:
            raise HubError("short download for %s (%d of %d bytes)" % (name, done, total))
        say("%-48s %7.1f MB" % (name, done / 1e6))

    return dest
