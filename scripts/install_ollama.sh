#!/bin/bash

# Install Ollama and download models for Medical STT post-processing

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Ollama Installation Script ===${NC}"
echo ""

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is already installed${NC}"
    ollama --version
else
    echo -e "${YELLOW}Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

echo ""

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    ollama serve &
    sleep 3
    echo -e "${GREEN}✓ Ollama service started${NC}"
else
    echo -e "${GREEN}✓ Ollama service is already running${NC}"
fi

echo ""

# Download default model
MODEL="${1:-llama3.2:3b}"

echo -e "${YELLOW}Downloading model: ${MODEL}${NC}"
echo "This may take a few minutes..."
echo ""

ollama pull "$MODEL"

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Available models:"
ollama list
echo ""
echo -e "${GREEN}Usage:${NC}"
echo "  Default model (llama3.2:3b):"
echo "    curl -X POST 'http://localhost:8000/api/transcribe/upload?postprocess=true' -F 'file=@audio.wav'"
echo ""
echo "  Custom model:"
echo "    curl -X POST 'http://localhost:8000/api/transcribe/upload?postprocess=true&llm_model=${MODEL}' -F 'file=@audio.wav'"
echo ""
echo -e "${YELLOW}Optional: Download additional models${NC}"
echo "  ollama pull llama3.1:8b      # Better quality (~5GB)"
echo "  ollama pull llama3.2:1b      # Faster, smaller (~1.3GB)"
echo ""
