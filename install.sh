#!/usr/bin/env bash
# OpenLLM installer - Termux and desktop Linux.
set -eu

REPO_OWNER="david-here-llma"
REPO_NAME="OpenLLM"
BRANCH="main"
DEST="${OPENLLM_HOME:-$HOME/.openllm}"

step() { printf '\033[38;5;45m==>\033[0m %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }
warn() { printf '\033[38;5;214m[!]\033[0m %s\n' "$1"; }

step "Checking dependencies"
if command -v pkg >/dev/null 2>&1; then
  missing=""
  command -v git >/dev/null 2>&1 || missing="$missing git"
  command -v python3 >/dev/null 2>&1 || missing="$missing python"
  command -v curl >/dev/null 2>&1 || missing="$missing curl"
  if [ -n "$missing" ]; then
    note "installing:$missing"
    pkg install -y $missing
  else
    note "git, python3 and curl are already present"
  fi
else
  command -v git >/dev/null 2>&1 || { warn "git is required but missing"; exit 1; }
  command -v python3 >/dev/null 2>&1 || { warn "python3 is required but missing"; exit 1; }
  note "git and python3 found"
fi

step "Fetching OpenLLM"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --depth=1 origin "$BRANCH"
  git -C "$DEST" reset --hard "origin/$BRANCH" >/dev/null
  note "updated $DEST"
else
  rm -rf "$DEST"
  git clone --depth=1 --branch "$BRANCH" "https://github.com/$REPO_OWNER/$REPO_NAME.git" "$DEST"
fi

step "Installing the openllm command"
if [ -n "${PREFIX:-}" ] && [ -d "${PREFIX}/bin" ]; then
  BIN="${PREFIX}/bin"
elif [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
  BIN="/usr/local/bin"
else
  BIN="$HOME/.local/bin"
  mkdir -p "$BIN"
fi

printf '#!/usr/bin/env bash\nexport PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}"\nexec python3 -m openllm "$@"\n' "$DEST" > "$BIN/openllm"
chmod +x "$BIN/openllm"
note "wrote $BIN/openllm"

if [ "${OPENLLM_SKIP_DEPS:-0}" = "1" ]; then
  warn "skipping Python dependencies (OPENLLM_SKIP_DEPS=1)"
else
  step "Installing Python dependencies"
  python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
  if python3 -m pip install -r "$DEST/requirements.txt"; then
    note "torch, transformers, accelerate, safetensors, huggingface_hub ready"
  else
    warn "dependency install failed"
    note "plain Termux has no PyTorch wheel; use proot-distro (Ubuntu) or a desktop"
    note "the CLI still runs: 'openllm list' works and 'openllm chat --mock' works"
  fi
fi

step "Ready"
note "openllm list      # catalog, smallest to 70B"
note "openllm doctor    # what is installed"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) warn "add this to your shell profile: export PATH=\"$BIN:\$PATH\"" ;;
esac
