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

# Available models
MODELS=(
    "ayoubkirouane/whisper-small-medical"
    "MohamedRashad/whisper-large-v2-medical"
    "primeLine/whisper-large-v3-german-medical"
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
    echo "  1) whisper-small-medical        (~500MB)  - Fast, lightweight"
    echo "  2) whisper-large-v2-medical     (~3GB)    - English medical"
    echo "  3) whisper-large-v3-german      (~3GB)    - German medical"
    echo "  4) all                          (~6.5GB)  - Download all models"
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
    echo "This may take a while depending on your connection..."
    echo ""

    python3 << EOF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "$model_id"
print(f"Downloading processor for {model_id}...")
processor = AutoProcessor.from_pretrained(model_id)

print(f"Downloading model for {model_id}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

print(f"✓ Successfully downloaded {model_id}")
EOF
}

main() {
    print_header
    activate_venv
    install_dependencies

    if [ -n "$1" ]; then
        choice="$1"
    else
        print_models
        read -p "Select model to download (1-4): " choice
    fi

    case $choice in
        1)
            download_model "${MODELS[0]}"
            ;;
        2)
            download_model "${MODELS[1]}"
            ;;
        3)
            download_model "${MODELS[2]}"
            ;;
        4|all)
            for model in "${MODELS[@]}"; do
                download_model "$model"
            done
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            exit 1
            ;;
    esac

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
