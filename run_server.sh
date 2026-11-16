#!/usr/bin/env bash
# Launch the ASL Recognition & Generation API and Web Client
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/uvicorn" ]; then
    echo "================================================================="
    echo "  🤟 ASL Multimodal Recognition & Generation System"
    echo "  Web UI   : http://localhost:8000 (or http://localhost:8000/app)"
    echo "  API Docs : http://localhost:8000/docs"
    echo "  Health   : http://localhost:8000/health"
    echo "================================================================="
    exec "$SCRIPT_DIR/.venv/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Error: Virtual environment (.venv) not found."
    echo "Please set up the environment with:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

