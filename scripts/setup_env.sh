#!/usr/bin/env bash
# setup_env.sh — create a virtual environment and install all dependencies
set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "✓ Environment ready. Activate with:"
echo "    source $VENV_DIR/bin/activate"
