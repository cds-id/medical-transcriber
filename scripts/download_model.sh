#!/bin/bash

# Medical STT Model Downloader
# Downloads HuggingFace models for medical speech-to-text

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# HuggingFace token (can be set via environment or .env file)
HF_TOKEN="${HF_TOKEN:-}"

# Load from .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
fi

# Available models
MODELS=(
    "openai/whisper-large-v3"
)

print_header() {
    echo -e "${GREEN}"
    echo "========================================"
    echo "  Medical STT Model Downloader"
    echo "========================================"
    echo -e "${NC}"
}

print_models() {
    echo -e "${YELLOW}Available models:${NC}"
    echo ""
    echo "  1) openai/whisper-large-v3      (~3GB)    - OpenAI Whisper Large V3"
    echo ""
}

activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}"
        echo "Please run: python3 -m venv venv && pip install -r requirements.txt"
        exit 1
    fi
}

install_dependencies() {
    echo -e "${GREEN}Checking dependencies...${NC}"
    pip install transformers torch --quiet
}

download_model() {
    local model_id="$1"
    echo ""
    echo -e "${GREEN}Downloading: ${YELLOW}$model_id${NC}"
    if [ -n "$HF_TOKEN" ]; then
        echo -e "${GREEN}Using HuggingFace token${NC}"
    fi
    echo "This may take a while depending on your connection..."
    echo ""

    python3 << EOF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import os

model_id = "$model_id"
hf_token = os.environ.get("HF_TOKEN") or "$HF_TOKEN" or None

# Prepare token kwargs
token_kwargs = {"token": hf_token} if hf_token else {}

print(f"Downloading processor for {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, **token_kwargs)

print(f"Downloading model for {model_id}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    **token_kwargs,
)

print(f"✓ Successfully downloaded {model_id}")
EOF
}

main() {
    print_header
    activate_venv
    install_dependencies

    print_models
    echo -e "${GREEN}Downloading openai/whisper-large-v3...${NC}"
    download_model "${MODELS[0]}"

    echo ""
    echo -e "${GREEN}========================================"
    echo "  Download complete!"
    echo "========================================"
    echo -e "${NC}"
    echo "Models are cached in: ~/.cache/huggingface/"
    echo ""
    echo "To run the app:"
    echo "  ./scripts/run.sh"
    echo ""
}

main "$@"
