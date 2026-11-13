# Relay channel (store-and-forward shell)

This branch is a command channel, not a server. Nothing listens on a port and
no SSH daemon is involved.

How it works:

1. `relay/seq` is a counter. When it changes, there is new work.
2. `relay/task.sh` is the script to run.
3. `relay/agent.sh` on your device polls this branch every 10 seconds, runs
   `task.sh` when `seq` changes, and pushes the output to `relay/result.txt`.
4. The assistant reads `result.txt` and writes the next `task.sh` + `seq`.

## Security

- **Anything printed to your terminal ends up in this repo.** If the repo is
  public, that output is public. Make the repo private before using this.
- Use a **fine-grained personal access token** limited to this repo with only
  `Contents: read and write`, with a short expiry.
- The token is stored in `.git/config`; `agent.sh` sets that file to `0600`.
- Revoke the token when you are finished.
- Stop the agent with Ctrl-C. It is a `sleep` loop, nothing else.

## Not interactive

One task per assistant turn. A task script may itself run many commands, so a
single turn can gather a whole battery of output.
