#!/usr/bin/env bash
# QuantaMED Clinical Dashboard Launcher
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "===================================================================="
echo "  Starting QuantaMED Clinical Screening Dashboard..."
echo "  Access URL: http://localhost:8501"
echo "===================================================================="

streamlit run frontend/app/main.py --server.port 8501
