#!/bin/bash
# ===========================================
# Vera Model Downloader
# ===========================================
# Downloads required GGUF models from HuggingFace
# Requires HF_TOKEN environment variable for private repos

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN environment variable not set${NC}"
    echo "Please set your HuggingFace token: export HF_TOKEN=hf_your_token"
    exit 1
fi

# Base directory (relative to deployment folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$BASE_DIR/models"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Vera Model Downloader${NC}"
echo -e "${GREEN}============================================${NC}"
echo "Models directory: $MODELS_DIR"
echo ""

# Function to download a model
download_model() {
    local dir="$1"
    local filename="$2"
    local url="$3"
    local full_path="$MODELS_DIR/$dir/$filename"
    
    # Create directory if it doesn't exist
    mkdir -p "$MODELS_DIR/$dir"
    
    if [ -f "$full_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $dir/$filename already exists"
        return 0
    fi
    
    echo -e "${GREEN}[DOWNLOAD]${NC} Downloading $dir/$filename..."
    echo "  URL: $url"
    
    curl -L \
        -H "Authorization: Bearer $HF_TOKEN" \
        -o "$full_path" \
        --progress-bar \
        "$url"
    
    if [ -f "$full_path" ]; then
        local size=$(du -h "$full_path" | cut -f1)
        echo -e "${GREEN}[SUCCESS]${NC} Downloaded $dir/$filename ($size)"
    else
        echo -e "${RED}[ERROR]${NC} Failed to download $dir/$filename"
        return 1
    fi
}

# ===========================================
# Standard Mode Models
# ===========================================
echo ""
echo -e "${GREEN}--- Standard Mode Models ---${NC}"

# Decomposer (Standard)
download_model "decomposer" \
    "${DECOMPOSER_MODEL:-LFM2.5-1.2B-Instruct.F16.gguf}" \
    "${DECOMPOSER_DOWNLOAD_URL:-https://huggingface.co/werstal/vera-decomposer-lfm2.5-1.2b-gguf/resolve/main/LFM2.5-1.2B-Instruct.F16.gguf}"

# ===========================================
# CoT/Reasoning Mode Models
# ===========================================
echo ""
echo -e "${GREEN}--- CoT/Reasoning Mode Models ---${NC}"

# Decomposer (CoT)
download_model "decomposer-cot" \
    "${DECOMPOSER_COT_MODEL:-LFM2.5-1.2B-Instruct.F16.gguf}" \
    "${DECOMPOSER_COT_DOWNLOAD_URL:-https://huggingface.co/werstal/vera-decomposer-cot-lfm2.5-1.2b-gguf/resolve/main/LFM2.5-1.2B-Instruct.F16.gguf}"

# Verifier (CoT)
download_model "verifier-cot" \
    "${VERIFIER_COT_MODEL:-LFM2.5-1.2B-Instruct.F16.gguf}" \
    "${VERIFIER_COT_DOWNLOAD_URL:-https://huggingface.co/werstal/vera-verifier-cot-lfm2.5-1.2b-gguf/resolve/main/LFM2.5-1.2B-Instruct.F16.gguf}"

# ===========================================
# Summary
# ===========================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Models downloaded to: $MODELS_DIR"
ls -lh "$MODELS_DIR"/*/ 2>/dev/null || echo "No models found"
echo ""
