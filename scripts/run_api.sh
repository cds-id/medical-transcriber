#!/bin/bash

# Run the Medical STT API server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Activate venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found. Run: python3 -m venv venv"
    exit 1
fi

cd "$PROJECT_DIR"

HOST="${1:-0.0.0.0}"
PORT="${2:-8000}"

echo -e "${GREEN}Starting Medical STT API server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo ""
echo "Endpoints:"
echo "  - POST /api/transcribe/upload    - Upload audio file"
echo "  - WS   /api/transcribe/stream    - Stream audio via WebSocket"
echo "  - GET  /api/models               - List available models"
echo ""

python -m uvicorn src.medical_stt.api:app --host "$HOST" --port "$PORT"
