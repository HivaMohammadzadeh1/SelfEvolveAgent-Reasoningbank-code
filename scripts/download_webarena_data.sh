#!/bin/bash
# Script to download and setup WebArena benchmark data
# Based on: https://github.com/web-arena-x/webarena

set -e

echo "=========================================="
echo "WebArena Data Download and Setup"
echo "=========================================="
echo ""

# Create data directory
mkdir -p data
cd data

# Clone WebArena repository
if [ -d "webarena_repo" ]; then
    echo "WebArena repo already exists, pulling latest..."
    cd webarena_repo
    git pull
    cd ..
else
    echo "Cloning WebArena repository..."
    git clone https://github.com/web-arena-x/webarena.git webarena_repo
fi

echo ""
echo "Installing WebArena dependencies..."
cd webarena_repo

# Install WebArena package
pip install -e .

echo ""
echo "Generating test data config files..."
# This generates the 812 config files (*.json) in config_files/
python scripts/generate_test_data.py

echo ""
echo "Copying config files to data directory..."
cd ..
mkdir -p webarena/config_files
cp -r webarena_repo/config_files/* webarena/config_files/

# Count files
num_configs=$(ls webarena/config_files/*.json 2>/dev/null | wc -l)
echo ""
echo "✓ Copied $num_configs config files"

echo ""
echo "=========================================="
echo "WebArena Data Setup Complete!"
echo "=========================================="
echo ""
echo "Config files location: data/webarena/config_files/"
echo ""
echo "Dataset breakdown:"
echo "  - Total: 812 tasks"
echo "  - Shopping: 187 tasks"
echo "  - Admin: 182 tasks"
echo "  - GitLab: 180 tasks"
echo "  - Reddit: 106 tasks"
echo "  - Map: 128 tasks (excluded by default)"
echo "  - Multi: 29 tasks"
echo ""
echo "Note: These are config files only."
echo "To run evaluations with real websites, you need to:"
echo "  1. Set up WebArena environment (Docker containers)"
echo "  2. Configure URLs in .env"
echo ""
echo "For now, the agent will use mock browser interactions."
echo "Real browser integration requires WebArena Docker setup."
echo ""
echo "Ready to use! Run:"
echo "  python run_eval.py --mode no_memory --subset admin --seed 42"
echo ""
