#!/bin/bash
# Setup script for BrowserGym environment
# Based on BrowserGym paper: https://openreview.net/pdf?id=5298fKGmv3

set -e

echo "================================"
echo "BrowserGym Setup for WebArena"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core BrowserGym packages
echo "Installing BrowserGym ecosystem..."
pip install browsergym>=0.5.0
pip install browsergym-webarena>=0.5.0
pip install gymnasium>=0.29.0

# Install Playwright for browser automation
echo "Installing Playwright..."
pip install playwright>=1.40.0

# Install Playwright browsers
echo "Installing Playwright browsers (this may take a few minutes)..."
playwright install chromium

# Install remaining requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying BrowserGym installation..."
python -c "import browsergym; print(f'✓ BrowserGym version: {browsergym.__version__}')" || echo "✗ BrowserGym not found"
python -c "import gymnasium; print(f'✓ Gymnasium version: {gymnasium.__version__}')" || echo "✗ Gymnasium not found"
python -c "import playwright; print('✓ Playwright installed')" || echo "✗ Playwright not found"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Note: To use BrowserGym with WebArena, you need to:"
echo "1. Set up WebArena Docker containers (see WEBARENA_SETUP.md)"
echo "2. Configure environment variables for WebArena URLs"
echo "3. Set use_real_browser: true in config.yaml"
echo ""
echo "For mock testing without Docker, set use_real_browser: false"
echo ""

