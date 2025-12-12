#!/bin/bash
# Setup script for Mind2Web test splits
# This script helps download and prepare the Mind2Web test data

set -e

echo "=============================================="
echo "Mind2Web Test Data Setup"
echo "=============================================="
echo ""

# Create data directory
mkdir -p data/mind2web
cd data/mind2web

echo "The Mind2Web test splits are password-protected to prevent data contamination."
echo ""
echo "Please follow these steps:"
echo ""
echo "1. Visit: https://huggingface.co/datasets/osunlp/Mind2Web/tree/main"
echo ""
echo "2. Download the following ZIP files:"
echo "   - test_task.zip (Cross-Task, 252 tasks)"
echo "   - test_website.zip (Cross-Website, 177 tasks)"
echo "   - test_domain.zip (Cross-Domain, 912 tasks)"
echo ""
echo "3. Place the ZIP files in: $(pwd)"
echo ""
echo "4. Extract them with password: mind2web"
echo ""
echo "Example commands after downloading:"
echo "  unzip -P mind2web test_task.zip"
echo "  unzip -P mind2web test_website.zip"
echo "  unzip -P mind2web test_domain.zip"
echo ""
echo "Expected structure after extraction:"
echo "  data/mind2web/test_task/test_task.json"
echo "  data/mind2web/test_website/test_website.json"
echo "  data/mind2web/test_domain/test_domain.json"
echo ""
echo "=============================================="
echo ""

# Check if files already exist
if [ -f "test_task/test_task.json" ] && [ -f "test_website/test_website.json" ] && [ -f "test_domain/test_domain.json" ]; then
    echo "✓ All test splits are already set up!"
    echo ""
    echo "You can now run Mind2Web evaluation:"
    echo "  python run_mind2web.py --mode reasoningbank --split test_task"
else
    echo "⚠ Test data not found. Please download and extract the files as described above."
    echo ""

    # Check for ZIP files
    if ls *.zip 1> /dev/null 2>&1; then
        echo "Found ZIP files in directory. Attempting to extract with password 'mind2web'..."
        echo ""

        for zipfile in test_task.zip test_website.zip test_domain.zip; do
            if [ -f "$zipfile" ]; then
                echo "Extracting $zipfile..."
                unzip -P mind2web -o "$zipfile" || echo "Failed to extract $zipfile (check password)"
            fi
        done

        echo ""
        echo "Extraction complete. Verifying..."

        if [ -f "test_task/test_task.json" ] && [ -f "test_website/test_website.json" ] && [ -f "test_domain/test_domain.json" ]; then
            echo "✓ All test splits successfully set up!"
        else
            echo "⚠ Some test splits are still missing. Please check the extraction."
        fi
    fi
fi

cd ../..
