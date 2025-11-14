#!/bin/bash
# Setup script for split CPU/iGPU inference development
# Installs Intel oneAPI, dependencies, and builds the scheduler

set -e  # Exit on error

echo "======================================================================"
echo "SPLIT CPU/iGPU INFERENCE - DEVELOPMENT ENVIRONMENT SETUP"
echo "======================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS${NC}"

# Check if running on Intel hardware
if [ "$OS" == "linux" ]; then
    CPU_INFO=$(lscpu | grep "Model name" || echo "Unknown")
    echo "CPU: $CPU_INFO"
    
    if lscpu | grep -q "Intel"; then
        echo -e "${GREEN}✓ Intel CPU detected${NC}"
    else
        echo -e "${YELLOW}⚠️  Non-Intel CPU detected - iGPU features may not work${NC}"
    fi
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install system dependencies
echo ""
echo "======================================================================" 
echo "Step 1: Installing system dependencies"
echo "======================================================================"

if [ "$OS" == "linux" ]; then
    # Detect package manager
    if command_exists apt-get; then
        PKG_MGR="apt-get"
        echo -e "${GREEN}Using apt-get${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            pkg-config \
            libzmq3-dev \
            wget \
            curl \
            git \
            python3 \
            python3-pip \
            python3-venv
    elif command_exists yum; then
        PKG_MGR="yum"
        echo -e "${GREEN}Using yum${NC}"
        sudo yum install -y \
            gcc gcc-c++ \
            cmake \
            pkgconfig \
            zeromq-devel \
            wget \
            curl \
            git \
            python3 \
            python3-pip
    else
        echo -e "${RED}Unsupported package manager${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Step 2: Install Intel oneAPI
echo ""
echo "======================================================================"
echo "Step 2: Installing Intel oneAPI Base Toolkit"
echo "======================================================================"

ONEAPI_INSTALLER_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.1.0.596_offline.sh"

if [ -d "/opt/intel/oneapi" ]; then
    echo -e "${YELLOW}oneAPI already installed at /opt/intel/oneapi${NC}"
    read -p "Reinstall? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        SKIP_ONEAPI=1
    fi
fi

if [ -z "$SKIP_ONEAPI" ]; then
    echo "Downloading oneAPI installer..."
    wget -O /tmp/oneapi_installer.sh "$ONEAPI_INSTALLER_URL"
    
    echo "Installing oneAPI (this may take 10-15 minutes)..."
    chmod +x /tmp/oneapi_installer.sh
    sudo /tmp/oneapi_installer.sh -a --silent --eula accept
    
    rm /tmp/oneapi_installer.sh
    echo -e "${GREEN}✓ oneAPI installed${NC}"
else
    echo -e "${YELLOW}Skipping oneAPI installation${NC}"
fi

# Source oneAPI environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
    echo -e "${GREEN}✓ oneAPI environment loaded${NC}"
else
    echo -e "${RED}✗ oneAPI installation not found${NC}"
    echo "Please install manually from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
fi

# Step 3: Install Python dependencies
echo ""
echo "======================================================================"
echo "Step 3: Installing Python dependencies"
echo "======================================================================"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip
pip install \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    numpy \
    psutil \
    einops \
    pyzmq \
    PyYAML \
    onnx \
    onnxruntime

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Step 4: Build C++ scheduler
echo ""
echo "======================================================================"
echo "Step 4: Building C++ scheduler"
echo "======================================================================"

cd split_inference/cpp

# Clean previous build
rm -rf build
mkdir build
cd build

echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SYCL=OFF \
    -DENABLE_ONEDNN=OFF \
    -DENABLE_VTUNE=OFF

echo "Building..."
make -j$(nproc)

echo -e "${GREEN}✓ Scheduler built successfully${NC}"

cd ../../..

# Step 5: Verify installation
echo ""
echo "======================================================================"
echo "Step 5: Verifying installation"
echo "======================================================================"

echo "Checking scheduler executable..."
if [ -f "split_inference/cpp/build/scheduler" ]; then
    echo -e "${GREEN}✓ Scheduler executable found${NC}"
else
    echo -e "${RED}✗ Scheduler executable not found${NC}"
fi

echo "Checking Python environment..."
python3 -c "import torch; import transformers; import zmq; print('✓ Python imports OK')"

echo "Checking oneAPI..."
if command_exists icpx; then
    echo -e "${GREEN}✓ Intel DPC++ compiler available${NC}"
    icpx --version | head -n 1
else
    echo -e "${YELLOW}⚠️  Intel DPC++ compiler not found (needed for SYCL)${NC}"
fi

# Step 6: Create helper scripts
echo ""
echo "======================================================================"
echo "Step 6: Creating helper scripts"
echo "======================================================================"

# Create run_scheduler.sh
cat > run_scheduler.sh << 'EOF'
#!/bin/bash
# Start the C++ scheduler

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

./split_inference/cpp/build/scheduler tcp://*:5555
EOF

chmod +x run_scheduler.sh

# Create run_orchestrator.sh
cat > run_orchestrator.sh << 'EOF'
#!/bin/bash
# Start the Python orchestrator

source venv/bin/activate
python3 split_inference/python/orchestrator.py "$@"
EOF

chmod +x run_orchestrator.sh

# Create enable_oneapi.sh
cat > enable_oneapi.sh << 'EOF'
#!/bin/bash
# Source oneAPI environment variables

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
    echo "✓ oneAPI environment enabled"
else
    echo "✗ oneAPI not found at /opt/intel/oneapi"
fi
EOF

chmod +x enable_oneapi.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

# Summary
echo ""
echo "======================================================================"
echo "SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Rebuild scheduler with SYCL support (optional):"
echo "     cd split_inference/cpp/build"
echo "     cmake .. -DENABLE_SYCL=ON -DENABLE_ONEDNN=ON"
echo "     make -j\$(nproc)"
echo ""
echo "  2. Start the scheduler:"
echo "     ./run_scheduler.sh"
echo ""
echo "  3. In another terminal, run the orchestrator:"
echo "     ./run_orchestrator.sh"
echo ""
echo "  4. Run ONNX analysis:"
echo "     source venv/bin/activate"
echo "     python3 export_to_onnx.py"
echo ""
echo "Environment setup:"
echo "  • Python venv: source venv/bin/activate"
echo "  • oneAPI: source enable_oneapi.sh"
echo ""
echo -e "${GREEN}Happy hacking!${NC}"
