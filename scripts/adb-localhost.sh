#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8766}"

adb devices
adb reverse "tcp:${PORT}" "tcp:${PORT}"

echo
echo "Open this in Quest Browser:"
echo "  http://localhost:${PORT}"
