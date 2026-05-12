#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Creating Python venv..."
  rm -rf .venv
  if ! python3 -m venv .venv; then
    echo
    echo "Failed to create venv."
    echo "On Ubuntu/Debian, run:"
    echo "  sudo apt install python3-venv python3-pip"
    exit 1
  fi
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python foxy_stream.py "$@"
