#!/bin/bash
#
# Vision Pro - Installation Script
#
# 시스템을 자동으로 설치하고 설정합니다.
#

set -e  # Exit on error

echo "======================================"
echo "Vision Pro Installation Script"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}➜${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found!"
    echo "Please install Python 3.8+ first:"
    echo "  - Ubuntu: sudo apt install python3 python3-pip"
    echo "  - macOS: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Create necessary directories
print_info "Creating directories..."
mkdir -p logs static/screenshots static/recordings
print_success "Directories created"

# Create .env file if not exists
print_info "Checking .env file..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from .env.example"
        print_info "Please edit .env and add your API keys!"
    else
        print_info ".env.example not found, skipping..."
    fi
else
    print_info ".env file already exists, skipping..."
fi

# Check webcam
print_info "Checking webcam..."
if [ -e /dev/video0 ]; then
    print_success "Webcam found at /dev/video0"
else
    print_info "No webcam found, you may need to configure it"
fi

# Download models (optional)
echo ""
print_info "Models will be downloaded automatically on first run"
print_info "YOLOv8n: ~6 MB"
print_info "Depth Anything V3: ~100 MB"

# Done
echo ""
echo "======================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "       source venv/bin/activate"
echo ""
echo "  2. (Optional) Edit .env file with your API keys:"
echo "       nano .env"
echo ""
echo "  3. Run the server:"
echo "       python app.py"
echo ""
echo "  4. Open your browser:"
echo "       http://localhost:8080/monitor"
echo ""
echo "======================================"
