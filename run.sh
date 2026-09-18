#!/usr/bin/env bash
# Start ADS-B Camera. Extra arguments are passed through, e.g.
#   ./run.sh --source http://192.168.1.50:8080/stream.mjpeg --port 8091
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then VPY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe
else
  echo "No .venv found - run ./install.sh first." >&2
  exit 1
fi
[ -f config.yaml ] || cp config.example.yaml config.yaml
exec "$VPY" -m adsb_camera -c config.yaml "$@"
