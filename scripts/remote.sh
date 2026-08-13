#!/usr/bin/env bash
# Sync the repo to the server via rsync, then execute a command there.
#
# Usage:
#   scripts/remote.sh <command...>   # sync code, then run the command on the server
#   scripts/remote.sh                # sync only
#
# Environment overrides:
#   SERVER_HOST   ssh host (default: autodl)
#   SERVER_DIR    remote repo directory (default: ~/autodl-tmp/llm)

set -euo pipefail

HOST="${SERVER_HOST:-autodl}"
REMOTE_DIR="${SERVER_DIR:-~/autodl-tmp/llm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Syncing $REPO_DIR -> $HOST:$REMOTE_DIR"
rsync -az --delete -e ssh \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  "$REPO_DIR/" "$HOST:$REMOTE_DIR/"

if (($# > 0)); then
  echo "==> Running on $HOST: $*"
  ssh "$HOST" "git config --global --add safe.directory $REMOTE_DIR; cd $REMOTE_DIR && $*"
fi
