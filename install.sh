#!/usr/bin/env bash
# Set up ADS-B Camera: pick a Python, create .venv, install requirements,
# create config.yaml and the data folder.
#
#   ./install.sh              interactive
#   ./install.sh -y           recreate .venv without asking
#   ./install.sh --python /usr/bin/python3.12
set -euo pipefail
cd "$(dirname "$0")"

MIN_MAJOR=3; MIN_MINOR=9
FORCE=0; FORCED_PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes|--force) FORCE=1 ;;
    --python) FORCED_PY="${2:-}"; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# Prints "X.Y.Z" if $1 is a working Python >= MIN, otherwise nothing.
py_version() {
  "$1" -c "import sys
if sys.version_info[:2] >= ($MIN_MAJOR, $MIN_MINOR): print('.'.join(map(str, sys.version_info[:3])))" 2>/dev/null || true
}

# ── choose the interpreter ────────────────────────────────────────────────
PY=""
if [ -n "$FORCED_PY" ]; then
  [ -n "$(py_version "$FORCED_PY")" ] || die "$FORCED_PY is not Python >= $MIN_MAJOR.$MIN_MINOR"
  PY="$FORCED_PY"
else
  V_PYTHON=""; V_PYTHON3=""
  command -v python  >/dev/null 2>&1 && V_PYTHON="$(py_version python)"
  command -v python3 >/dev/null 2>&1 && V_PYTHON3="$(py_version python3)"

  if [ -n "$V_PYTHON" ] && [ -n "$V_PYTHON3" ] && [ "$V_PYTHON" != "$V_PYTHON3" ]; then
    echo "Two different Python versions were found:"
    echo "  1) python   $V_PYTHON   ($(command -v python))"
    echo "  2) python3  $V_PYTHON3   ($(command -v python3))"
    if [ -t 0 ]; then
      read -r -p "Which one should create the .venv? [1/2] (default 2): " CHOICE
    else
      CHOICE=2; echo "Non-interactive shell: using python3."
    fi
    case "${CHOICE:-2}" in
      1) PY=python ;;
      2) PY=python3 ;;
      *) die "Please answer 1 or 2." ;;
    esac
  elif [ -n "$V_PYTHON3" ]; then PY=python3
  elif [ -n "$V_PYTHON" ];  then PY=python
  else
    die "No Python >= $MIN_MAJOR.$MIN_MINOR found as 'python' or 'python3'.
    Debian/Ubuntu/Raspberry Pi OS:  sudo apt install python3 python3-venv
    macOS:                          brew install python"
  fi
fi
say "Using $PY $(py_version "$PY") ($(command -v "$PY" || echo "$PY"))"

# ── virtual environment ───────────────────────────────────────────────────
if [ -d .venv ]; then
  if [ "$FORCE" = 1 ]; then
    REPLY=y
  elif [ -t 0 ]; then
    read -r -p ".venv already exists. Recreate it? [y/N]: " REPLY
  else
    REPLY=n
  fi
  case "$REPLY" in
    y|Y|yes) say "Removing old .venv"; rm -rf .venv ;;
    *) say "Keeping existing .venv (dependencies will be updated)" ;;
  esac
fi

if [ ! -d .venv ]; then
  say "Creating .venv"
  "$PY" -m venv .venv || die "Could not create the venv. On Debian/Ubuntu: sudo apt install python3-venv"
fi

if [ -x .venv/bin/python ]; then VPY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe   # Git Bash on Windows
else die ".venv looks broken - rerun with -y to recreate it."; fi

say "Upgrading pip"
"$VPY" -m pip install --upgrade pip wheel >/dev/null
say "Installing requirements (numpy, scipy, opencv, fastapi, paho-mqtt ...)"
"$VPY" -m pip install -r requirements.txt

# ── project files ─────────────────────────────────────────────────────────
mkdir -p data
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  say "Created config.yaml - set video.source, mqtt.host and site.lat/lon"
else
  say "config.yaml already exists - left unchanged"
fi
[ -f .env ] || cp .env.example .env 2>/dev/null || true
chmod +x run.sh install.sh 2>/dev/null || true

say "Checking the install"
"$VPY" -c "import cv2, numpy, scipy, fastapi, uvicorn, paho.mqtt, yaml; print('  OpenCV', cv2.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"

echo
say "Done. Next:"
echo "    1. edit config.yaml (camera stream, MQTT broker, camera location)"
echo "    2. ./run.sh"
echo "    3. open http://localhost:8090"
