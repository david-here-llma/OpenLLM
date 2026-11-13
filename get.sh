#!/usr/bin/env bash
# OpenLLM one-shot installer.
#
#   curl -fsSL https://raw.githubusercontent.com/david-here-llma/OpenLLM/main/get.sh | bash
#
# It does everything, in one go:
#   * Termux  -> installs proot-distro + an Ubuntu sandbox (once), then
#                python + torch + transformers inside it,
#   * Linux   -> uses the system packages and a venv,
#   * installs the `openllm` command,
#   * launches OpenLLM at the end.
#
# Safe to re-run: every step checks before it acts.
set -eu

REPO_URL="https://github.com/david-here-llma/OpenLLM.git"
DISTRO="ubuntu"
VENV="$HOME/ollm"
SRC="$HOME/OpenLLM"
AUTOLAUNCH="${OPENLLM_NO_LAUNCH:-0}"

say()  { printf '\033[38;5;45m==>\033[0m %s\n' "$1"; }
sub()  { printf '    %s\n' "$1"; }
warn() { printf '\033[38;5;214m[!]\033[0m %s\n' "$1"; }

is_termux() {
  if [ -n "${PREFIX:-}" ] && [ -x "${PREFIX}/bin/pkg" ]; then return 0; fi
  if [ -n "${TERMUX_VERSION:-}" ]; then return 0; fi
  return 1
}

# --------------------------------------------------------------------- termux

install_termux() {
  say "Setting up Termux"
  pkg install -y git curl proot-distro >/dev/null 2>&1 || {
    warn "pkg install failed; run: pkg update && pkg install git curl proot-distro"
    exit 1
  }
  sub "git, curl and proot-distro ready"

  ROOTFS="${PREFIX}/var/lib/proot-distro/installed-rootfs/${DISTRO}"
  if [ ! -d "$ROOTFS" ]; then
    say "Installing the Ubuntu sandbox (once, about 200 MB)"
    proot-distro install "$DISTRO" || {
      warn "proot-distro install failed; check your connection and re-run"
      exit 1
    }
  else
    sub "ubuntu sandbox already installed"
  fi

  say "Installing python, torch and transformers inside ubuntu"
  sub "this is the slow step: a few hundred MB, give it 5-15 minutes"
  proot-distro login "$DISTRO" -- bash -s <<'INNER' || {
set -eu
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update -qq || apt-get update
apt-get install -y -qq python3 python3-venv python3-pip git curl ca-certificates >/dev/null
if [ ! -d "$HOME/OpenLLM/.git" ]; then
  git clone --depth=1 https://github.com/david-here-llma/OpenLLM.git "$HOME/OpenLLM"
else
  git -C "$HOME/OpenLLM" fetch --depth=1 origin main || true
  git -C "$HOME/OpenLLM" reset --hard origin/main || true
fi
if [ ! -x "$HOME/ollm/bin/python" ]; then
  python3 -m venv "$HOME/ollm"
fi
. "$HOME/ollm/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || pip install -q torch
pip install -q transformers accelerate safetensors huggingface_hub
python - <<'CHECK'
import torch, transformers, accelerate, safetensors
print("    torch %s | transformers %s | accelerate %s" % (
    torch.__version__, transformers.__version__, accelerate.__version__))
CHECK
INNER
    warn "the ubuntu setup step failed"
    warn "re-run this same command; finished steps are skipped"
    exit 1
  }

  say "Installing the 'openllm' command"
  cat > "${PREFIX}/bin/openllm" <<'LAUNCH'
#!/usr/bin/env bash
set -eu
quoted=""
for arg in "$@"; do
  quoted="$quoted $(printf '%q' "$arg")"
done
exec proot-distro login ubuntu -- bash -lc 'cd "$HOME/OpenLLM" && . "$HOME/ollm/bin/activate" && exec python3 -m openllm '"$quoted"
LAUNCH
  chmod +x "${PREFIX}/bin/openllm"
  sub "${PREFIX}/bin/openllm"
  LAUNCHER="${PREFIX}/bin/openllm"
}

# --------------------------------------------------------------------- linux

install_linux() {
  say "Setting up Linux"
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    [ "$(id -u)" -eq 0 ] || SUDO="sudo"
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl ca-certificates >/dev/null
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip git curl >/dev/null
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm python python-pip git curl >/dev/null
  else
    warn "unknown package manager; install python3, python3-venv and git yourself"
  fi

  if [ ! -d "$SRC/.git" ]; then
    git clone --depth=1 "$REPO_URL" "$SRC"
  else
    git -C "$SRC" fetch --depth=1 origin main || true
    git -C "$SRC" reset --hard origin/main || true
  fi

  say "Installing python packages"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || "$VENV/bin/pip" install -q torch
  "$VENV/bin/pip" install -q transformers accelerate safetensors huggingface_hub

  say "Installing the 'openllm' command"
  if [ -w /usr/local/bin ]; then
    BIN="/usr/local/bin"
  else
    BIN="$HOME/.local/bin"
    mkdir -p "$BIN"
  fi
  cat > "$BIN/openllm" <<LAUNCH
#!/usr/bin/env bash
set -eu
cd "$SRC"
exec "$VENV/bin/python" -m openllm "\$@"
LAUNCH
  chmod +x "$BIN/openllm"
  sub "$BIN/openllm"
  LAUNCHER="$BIN/openllm"

  case ":$PATH:" in
    *":$BIN:"*) ;;
    *) warn "add to your profile: export PATH=\"$BIN:\$PATH\"" ;;
  esac
}

# --------------------------------------------------------------------- main

if is_termux; then
  install_termux
else
  install_linux
fi

say "Ready"
sub "next time just run:  openllm"
sub "try:  openllm list    then    openllm pull qwen2.5-0.5b"

export PATH="${BIN:-}:$PATH" 2>/dev/null || true

if [ "$AUTOLAUNCH" != "1" ] && [ -t 0 ] && [ -t 1 ]; then
  say "Starting OpenLLM"
  exec "$LAUNCHER"
fi
