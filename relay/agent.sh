#!/usr/bin/env bash
# OpenLLM relay agent.
#
# Polls the 'relay' branch of this repo. When relay/seq changes it runs
# relay/task.sh and pushes the output to relay/result.txt.
#
#   export RELAY_TOKEN=github_pat_...
#   bash ~/OpenLLM-relay/relay/agent.sh
#
# Ctrl-C to stop. Nothing is opened to the network: no sshd, no ports.
set -u

OWNER="${RELAY_USER:-david-here-llma}"
REPO="${RELAY_REPO:-OpenLLM}"
BRANCH="${RELAY_BRANCH:-relay}"
INTERVAL="${RELAY_INTERVAL:-10}"
TOKEN="${RELAY_TOKEN:-}"
DIR="${RELAY_DIR:-$HOME/OpenLLM-relay}"

if [ -z "$TOKEN" ]; then
  echo "set RELAY_TOKEN to a GitHub token with contents:write on $OWNER/$REPO" >&2
  exit 1
fi
command -v git >/dev/null 2>&1 || { echo "git is required: pkg install git" >&2; exit 1; }

URL="https://x-access-token:${TOKEN}@github.com/${OWNER}/${REPO}.git"

if [ ! -d "$DIR/.git" ]; then
  echo "cloning $OWNER/$REPO#$BRANCH into $DIR"
  git clone --depth=1 --branch "$BRANCH" "$URL" "$DIR" >/dev/null 2>&1 || {
    echo "clone failed - check the token and repo name" >&2
    exit 1
  }
fi

cd "$DIR" || exit 1
git remote set-url origin "$URL"
chmod 600 .git/config 2>/dev/null || true

git config user.name "openllm-relay"
git config user.email "relay@localhost"

last="$(cat .relay-last 2>/dev/null || echo 0)"
echo "relay ready: $OWNER/$REPO#$BRANCH every ${INTERVAL}s, last task $last"
echo "leave this running, Ctrl-C to stop"

while true; do
  if git fetch --depth=1 origin "$BRANCH" -q 2>/dev/null; then
    git reset --hard "origin/$BRANCH" -q 2>/dev/null || true
    seq="$(cat relay/seq 2>/dev/null || echo 0)"

    if [ "$seq" != "$last" ] && [ -f relay/task.sh ]; then
      echo "--- task $seq starting ---"
      {
        echo "# seq $seq"
        echo "# started $(date -u +%FT%TZ)"
        echo "# host $(uname -s) $(uname -m) pwd=$PWD"
        echo
        bash relay/task.sh 2>&1
        code=$?
        echo
        echo "# exit $code"
      } > relay/result.txt

      last="$seq"
      echo "$last" > .relay-last
      cp relay/result.txt "$HOME/.openllm-relay-result.txt" 2>/dev/null || true

      git add -f relay/result.txt
      git commit -q -m "result $seq" 2>/dev/null || true

      pushed=0
      for _try in 1 2 3 4; do
        if git push -q origin "$BRANCH" 2>/dev/null; then
          pushed=1
          break
        fi
        git pull --rebase -q origin "$BRANCH" >/dev/null 2>&1 || true
        sleep 2
      done

      if [ "$pushed" = "1" ]; then
        echo "--- task $seq done, output pushed ---"
      else
        echo "--- push failed; output kept at ~/.openllm-relay-result.txt ---"
      fi
    fi
  fi
  sleep "$INTERVAL"
done
