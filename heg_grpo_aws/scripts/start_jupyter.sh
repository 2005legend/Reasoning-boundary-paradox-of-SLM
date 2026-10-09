#!/usr/bin/env bash
# JupyterLab in a tmux session, bound to localhost only (reach it through an SSH tunnel).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"

if tmux has-session -t jupyter 2>/dev/null; then
  echo "JupyterLab already running."
else
  tmux new-session -d -s jupyter \
    "source '$VENV/bin/activate' && cd '$REPO' && jupyter lab --no-browser --ip 127.0.0.1 --port 8888"
  sleep 5
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
jupyter server list
cat <<EOF

Open the http://127.0.0.1:8888/?token=... URL above in your laptop's browser. It works while the
'ssh heg' window stays open: that connection carries the tunnel (LocalForward 8888, README C1).
EOF
