#!/bin/bash
set -e  # Exit on error

echo "=== NanoChat Setup Script ==="
echo ""

# Define directories
CACHE_DIR="$HOME/.cache/nanochat"
TOKENIZER_DIR="$CACHE_DIR/tokenizer"
CHECKPOINT_DIR="$CACHE_DIR/chatsft_checkpoints/d20"

# Create directories if they don't exist
echo "Creating necessary directories..."
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Base URL for HuggingFace model files
BASE_URL="https://huggingface.co/sdobson/nanochat/resolve/main"

# Download tokenizer files
echo ""
echo "=== Downloading tokenizer files ==="
cd "$TOKENIZER_DIR"

if [ -f "tokenizer.pkl" ]; then
    echo "tokenizer.pkl already exists, skipping..."
else
    echo "Downloading tokenizer.pkl..."
    wget "$BASE_URL/tokenizer.pkl"
fi

if [ -f "token_bytes.pt" ]; then
    echo "token_bytes.pt already exists, skipping..."
else
    echo "Downloading token_bytes.pt..."
    wget "$BASE_URL/token_bytes.pt"
fi

# Download model checkpoint files
echo ""
echo "=== Downloading model checkpoint files ==="
cd "$CHECKPOINT_DIR"

if [ -f "meta_000650.json" ]; then
    echo "meta_000650.json already exists, skipping..."
else
    echo "Downloading meta_000650.json..."
    wget "$BASE_URL/meta_000650.json"
fi

if [ -f "model_000650.pt" ]; then
    echo "model_000650.pt already exists, skipping..."
else
    echo "Downloading model_000650.pt (this is ~2GB, may take a while)..."
    wget "$BASE_URL/model_000650.pt"
fi

# Clone repository if not already present
echo ""
echo "=== Checking nanochat repository ==="
REPO_DIR="$HOME/nanochat"

if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists at $REPO_DIR"
    cd "$REPO_DIR"
else
    echo "Cloning nanochat repository..."
    cd "$HOME"
    git clone https://github.com/dwarkesh_sp/nanochat
    cd nanochat
fi

# Setup Python environment and dependencies
echo ""
echo "=== Setting up Python environment ==="
echo "Running uv sync..."
uv sync

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start the chat web interface, run:"
echo "  cd $REPO_DIR"
echo "  uv run python -m scripts.chat_web"
echo ""
echo "Or run this script with the 'run' argument to start immediately:"
echo "  $0 run"

# If 'run' argument is passed, start the chat web interface
if [ "$1" = "run" ]; then
    echo ""
    echo "=== Starting chat web interface ==="
    cd "$REPO_DIR"
    uv run python -m scripts.chat_web
fi

