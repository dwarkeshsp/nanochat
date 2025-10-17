#!/bin/bash
set -e

echo "=== NanoChat Weights Download Script ==="
echo ""

# Setup directories
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20"

mkdir -p "$TOKENIZER_DIR"
mkdir -p "$CHECKPOINT_DIR"

# HuggingFace model repository
BASE_URL="https://huggingface.co/sdobson/nanochat/resolve/main"

# Download tokenizer files
echo "=== Downloading tokenizer files ==="
cd "$TOKENIZER_DIR"
[ -f "tokenizer.pkl" ] || wget "$BASE_URL/tokenizer.pkl"
[ -f "token_bytes.pt" ] || wget "$BASE_URL/token_bytes.pt"

# Download model checkpoint files (~2GB total)
echo ""
echo "=== Downloading model checkpoint files ==="
cd "$CHECKPOINT_DIR"
[ -f "meta_000650.json" ] || wget "$BASE_URL/meta_000650.json"
[ -f "model_000650.pt" ] || wget "$BASE_URL/model_000650.pt"

# Setup Python environment
echo ""
echo "=== Setting up Python environment ==="
cd "$(dirname "$0")"
uv sync

echo ""
echo "=== Download Complete! ==="
