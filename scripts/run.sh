#!/bin/bash

# Medical STT Runner
# Run the medical speech-to-text application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Activate venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found. Run: python3 -m venv venv"
    exit 1
fi

cd "$PROJECT_DIR"

# Parse arguments
MODE="${1:-demo}"

case $MODE in
    demo)
        echo -e "${GREEN}Running demo transcription...${NC}"
        python3 -m src.medical_stt.demo
        ;;
    mic)
        echo -e "${GREEN}Starting microphone streaming...${NC}"
        python3 -m src.medical_stt.mic_stream
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        pytest tests/ -v
        ;;
    *)
        echo "Usage: $0 {demo|mic|test}"
        echo ""
        echo "  demo  - Run demo with test audio"
        echo "  mic   - Stream from microphone"
        echo "  test  - Run test suite"
        exit 1
        ;;
esac
