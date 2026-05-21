#!/bin/bash

# SloughGPT Enterprise AI Framework - Installation Script
# Automated setup and dependency resolution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}🚀 SloughGPT Enterprise AI Framework${NC}"
echo -e "${BLUE}====================================${NC}"
echo

# Check Python version
check_python() {
    echo -e "${YELLOW}🔍 Checking Python version...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
            PYTHON_CMD="python3"
        else
            echo -e "${RED}❌ Python $PYTHON_VERSION found (requires >= $REQUIRED_VERSION)${NC}"
            echo -e "${YELLOW}Please install Python 3.9 or higher${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Python 3 not found${NC}"
        echo -e "${YELLOW}Please install Python 3.9 or higher${NC}"
        exit 1
    fi
}

# Check system requirements
check_system() {
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="Windows"
    else
        OS="Unknown"
    fi
    
    echo -e "${GREEN}✅ Operating System: $OS${NC}"
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(($(free -g | awk '/^Mem:/{print $2}')))
        echo -e "${GREEN}✅ Available Memory: ${MEMORY_GB}GB${NC}"
    elif command -v sysctl &> /dev/null; then
        MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
        echo -e "${GREEN}✅ Available Memory: ${MEMORY_GB}GB${NC}"
    fi
    
    # Check disk space
    DISK_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    echo -e "${GREEN}✅ Available Disk: ${DISK_GB}GB${NC}"
    echo
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    
    # Upgrade pip
    echo -e "${BLUE}   Upgrading pip...${NC}"
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel
    
    # Fix NumPy compatibility first
    echo -e "${BLUE}   Installing NumPy compatible version...${NC}"
    $PYTHON_CMD -m pip install "numpy<2.0"
    
    # Install core dependencies without torch first
    echo -e "${BLUE}   Installing core dependencies...${NC}"
    $PYTHON_CMD -m pip install fastapi uvicorn sqlalchemy alembic redis
    $PYTHON_CMD -m pip install pydantic "python-jose[cryptography]" "passlib[bcrypt]"
    $PYTHON_CMD -m pip install python-multipart psutil prometheus-client
    $PYTHON_CMD -m pip install pytest pytest-asyncio pytest-cov
    $PYTHON_CMD -m pip install aiosmtplib aiofiles aiohttp websockets
    
    # Install PyTorch with compatibility
    echo -e "${BLUE}   Installing PyTorch (CPU version)...${NC}"
    $PYTHON_CMD -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    
    # Install additional dependencies
    echo -e "${BLUE}   Installing additional dependencies...${NC}"
    $PYTHON_CMD -m pip install scikit-learn pandas matplotlib seaborn
    $PYTHON_CMD -m pip install jupyter ipykernel notebook plotly dash streamlit
    
    echo -e "${GREEN}✅ Python dependencies installed successfully${NC}"
    echo
}

# Create virtual environment (optional)
create_venv() {
    if [[ "$1" == "--venv" ]]; then
        echo -e "${YELLOW}🐍 Creating virtual environment...${NC}"
        
        VENV_DIR="sloughgpt-venv"
        
        # Create virtual environment
        $PYTHON_CMD -m venv $VENV_DIR
        
        # Activate it
        source $VENV_DIR/bin/activate
        
        echo -e "${GREEN}✅ Virtual environment created: $VENV_DIR${NC}"
        echo -e "${YELLOW}To activate: source $VENV_DIR/bin/activate${NC}"
        echo
        
        PYTHON_CMD="$VENV_DIR/bin/python"
    fi
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}🧪 Verifying installation...${NC}"
    
    # Test critical imports
    $PYTHON_CMD -c "
import sys
modules_to_test = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'redis',
    'pydantic', 'torch', 'numpy', 'psutil'
]
failed = []
for module in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        failed.append(module)
        print(f'❌ {module}')
        
if failed:
    print(f'\\n⚠️  Missing modules: {failed}')
    sys.exit(1)
else:
    print('\\n🎉 All critical dependencies verified!')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Installation verification passed${NC}"
    else
        echo -e "${RED}❌ Installation verification failed${NC}"
        exit 1
    fi
}

# Setup configuration
setup_config() {
    echo -e "${YELLOW}⚙️  Setting up configuration...${NC}"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        echo -e "${BLUE}   Creating .env file...${NC}"
        cat > .env << EOF
# SloughGPT Configuration
DATABASE_URL=sqlite:///sloughgpt.db
JWT_SECRET_KEY=$(openssl rand -hex 32)
BCRYPT_ROUNDS=12

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
# Web UI (local): cd apps/web && npm run dev — http://localhost:3000

# Logging
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true

# Cost Management
DEFAULT_MONTHLY_BUDGET=1000
COST_ALERT_THRESHOLD=0.8

# Security
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=100
EOF
        echo -e "${GREEN}✅ Created .env file with secure defaults${NC}"
    else
        echo -e "${BLUE}   .env file already exists${NC}"
    fi
    
    # Create data directories
    mkdir -p data logs models checkpoints
    echo -e "${GREEN}✅ Created data directories${NC}"
    echo
}

# Run tests
run_tests() {
    echo -e "${YELLOW}🧪 Running basic tests...${NC}"
    
    # Test SloughGPT import
    echo -e "${BLUE}   Testing SloughGPT import...${NC}"
    $PYTHON_CMD -c "
try:
    import domains
    print('✅ domains package import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    import sys
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Basic tests passed${NC}"
    else
        echo -e "${RED}❌ Basic tests failed${NC}"
        exit 1
    fi
}

# Show next steps
show_next_steps() {
    echo -e "${BLUE}🎯 Installation Complete! Next Steps:${NC}"
    echo
    echo -e "${GREEN}1. Start API (from cloned repo root):${NC}"
    echo -e "   $PYTHON_CMD apps/api/server/main.py"
    echo
    echo -e "${GREEN}2. Web UI (separate terminal; Node per .nvmrc):${NC}"
    echo -e "   cd apps/web && npm install && npm run dev"
    echo
    echo -e "${GREEN}3. CLI:${NC}"
    echo -e "   $PYTHON_CMD cli.py --help   # or: sloughgpt --help (after $PYTHON_CMD -m pip install -e .)"
    echo
    echo -e "${GREEN}4. View Documentation:${NC}"
    echo -e "   📖 README.md - Complete overview"
    echo -e "   🌐 docs/API.md - API documentation"
    echo -e "   🚀 QUICKSTART.md - Quick start guide"
    echo -e "   🔧 docs/INSTALL.md - Installation details"
    echo -e "   📋 docs/TODO.md - Roadmap / backlog"
    echo
    echo -e "${YELLOW}🌟 SloughGPT Enterprise AI Framework - Ready to Use!${NC}"
}

# Main installation flow
main() {
    echo -e "${BLUE}🚀 Starting SloughGPT Installation${NC}"
    echo -e "${BLUE}==================================${NC}"
    echo
    
    # Parse command line arguments
    CREATE_VENV=false
    if [[ "$1" == "--venv" ]]; then
        CREATE_VENV=true
    fi
    
    # Run installation steps
    check_python
    check_system
    create_venv $([[ "$CREATE_VENV" == true ]] && echo "--venv")
    install_python_deps
    verify_installation
    setup_config
    run_tests
    show_next_steps
}

# Help function
show_help() {
    echo -e "${BLUE}SloughGPT Installation Script${NC}"
    echo -e "${BLUE}Usage: $0 [--venv]${NC}"
    echo
    echo "Options:"
    echo "  --venv    Create and use a virtual environment"
    echo
    echo "Examples:"
    echo "  $0              Install with system Python"
    echo "  $0 --venv       Install in virtual environment"
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"