#!/bin/bash
# Install LiveWorld and all local dependencies
set -e

echo "Installing LiveWorld..."
pip install -e .

echo "Installing SAM3..."
pip install -e misc/sam3

echo "Adding STream3R to Python path..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/misc/STream3R" > "$SITE_PACKAGES/stream3r.pth"

echo "Done. All dependencies installed."
