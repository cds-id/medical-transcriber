#!/bin/bash

# Download sample audio files for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SAMPLES_DIR="$PROJECT_DIR/samples"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$SAMPLES_DIR"

echo -e "${GREEN}Downloading sample audio files...${NC}"
echo ""

# LibriSpeech sample (English speech)
SAMPLE_URL="https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
SAMPLE_FILE="$SAMPLES_DIR/sample_english.wav"

if [ ! -f "$SAMPLE_FILE" ]; then
    echo "Downloading English sample..."
    curl -L -o "$SAMPLE_FILE" "$SAMPLE_URL" 2>/dev/null || wget -O "$SAMPLE_FILE" "$SAMPLE_URL" 2>/dev/null
    echo -e "${GREEN}✓ Downloaded: sample_english.wav${NC}"
else
    echo -e "${YELLOW}Already exists: sample_english.wav${NC}"
fi

echo ""
echo -e "${GREEN}Sample audio files are in: $SAMPLES_DIR${NC}"
echo ""
echo "To test transcription:"
echo "  source venv/bin/activate"
echo "  python -m src.medical_stt.demo small samples/sample_english.wav"
echo ""
echo -e "${YELLOW}For medical-specific audio, please provide your own recordings.${NC}"
echo ""
