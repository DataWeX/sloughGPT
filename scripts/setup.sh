#!/bin/bash
# SloughGPT Setup Script
# Quick start for local development and Docker deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PYTHON_CMD="python3"
VENV_DIR=".venv"
CONDA_ENV="sloughgpt"
MODE="all"
GPU_SUPPORT=false
CUDA_VERSION="cpu"
USE_CONDA=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_SUPPORT=true
            CUDA_VERSION="cu118"
            shift
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --conda)
            USE_CONDA=true
            shift
            ;;
        --conda-env)
            CONDA_ENV="$2"
            USE_CONDA=true
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --docker-only)
            MODE="docker"
            shift
            ;;
        --local-only)
            MODE="local"
            shift
            ;;
        --help|-h)
            echo "SloughGPT Setup Script"
            echo ""
            echo "Usage: ./scripts/setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu              Enable GPU support (requires CUDA)"
            echo "  --venv DIR         Virtual environment directory (default: .venv)"
            echo "  --conda            Use conda (recommended for macOS Intel)"
            echo "  --conda-env NAME   Conda environment name (default: sloughgpt)"
            echo "  --python CMD       Python command (default: python3)"
            echo "  --docker-only      Setup Docker only"
            echo "  --local-only       Setup local development only"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/setup.sh                    # Full setup (local + Docker)"
            echo "  ./scripts/setup.sh --conda            # Use conda (recommended for macOS)"
            echo "  ./scripts/setup.sh --gpu              # Setup with GPU support"
            echo "  ./scripts/setup.sh --docker-only      # Docker only"
            echo "  ./scripts/setup.sh --venv myenv       # Custom venv directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       SloughGPT Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if command -v git &> /dev/null; then
        print_status "Git found"
    else
        print_error "Git not found. Please install Git."
        exit 1
    fi
    
    if command -v docker &> /dev/null; then
        print_status "Docker found"
        DOCKER_VERSION=$(docker --version)
        print_info "Docker version: $DOCKER_VERSION"
    else
        print_warning "Docker not found. Docker setup will be skipped."
        MODE="local"
    fi
    
    # macOS fix: Disable library injection for PyTorch
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_INSERT_LIBRARIES=""
        print_info "Applied macOS PyTorch compatibility fix"
    fi
    
    if command -v $PYTHON_CMD &> /dev/null; then
        PYTHON_VERSION=$($PYTHON_CMD --version)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    echo ""
}

# Create .env file if it doesn't exist
setup_env() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_status "Created .env from .env.example"
        else
            cat > .env << 'EOF'
# SloughGPT Environment Configuration

# Application
SLOUGHGPT_ENV=development
SLOUGHGPT_HOST=0.0.0.0
SLOUGHGPT_PORT=8000
SLOUGHGPT_LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///./sloughgpt.db

# Model Settings
MODEL_CACHE_DIR=./models
DEFAULT_MODEL=gpt2
QUANTIZATION_TYPE=int8_dynamic

# API Keys (optional)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Security
SECRET_KEY=change_me_in_production
JWT_SECRET=change_me_in_production
EOF
            print_status "Created .env file"
        fi
    else
        print_info ".env already exists, skipping"
    fi
    
    echo ""
}

# Create necessary directories
setup_directories() {
    print_info "Creating directories..."
    
    DIRS=("models" "datasets" "data" "checkpoints" "experiments" "logs" "cache")
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created $dir/"
        else
            print_info "$dir/ already exists"
        fi
    done
    
    echo ""
}

# Install Python dependencies
install_python_deps() {
    # Check if conda should be used
    if [ "$USE_CONDA" = true ]; then
        install_conda_deps
        return
    fi
    
    print_info "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_status "Created virtual environment at $VENV_DIR/"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    print_info "Installing core dependencies..."
    python -m pip install fastapi uvicorn pydantic pydantic-settings httpx aiofiles PyJWT python-multipart
    
    print_info "Installing ML dependencies..."
    if [ "$GPU_SUPPORT" = true ]; then
        print_info "Installing PyTorch with CUDA $CUDA_VERSION..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_VERSION
    else
        python -m pip install torch torchvision torchaudio
    fi
    
    python -m pip install transformers accelerate bitsandbytes scipy
    
    print_info "Installing data processing..."
    python -m pip install networkx numpy pandas pillow
    
    print_info "Installing development tools..."
    python -m pip install pytest pytest-asyncio pytest-cov ruff black mypy
    
    print_status "Python dependencies installed"
    echo ""
}

# Install Python dependencies using conda
install_conda_deps() {
    print_info "Setting up with conda..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found! Please install Miniconda or Anaconda."
        print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        print_info "Or run: brew install --cask miniconda"
        echo ""
        print_info "Falling back to venv..."
        USE_CONDA=false
        install_python_deps
        return
    fi
    
    print_info "Conda found: $(conda --version)"
    
    # Create conda environment
    print_info "Creating conda environment: $CONDA_ENV"
    if conda env list | grep -q "^$CONDA_ENV "; then
        print_info "Conda environment '$CONDA_ENV' already exists"
    else
        conda create -n "$CONDA_ENV" python=3.11 -y
        print_status "Created conda environment: $CONDA_ENV"
    fi
    
    # Activate conda environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Install PyTorch via conda (best for macOS)
    print_info "Installing PyTorch via conda..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use default channel (works on both Intel and Apple Silicon)
        conda install pytorch torchvision torchaudio -c pytorch -y
    else
        # Linux
        if [ "$GPU_SUPPORT" = true ] && command -v nvidia-smi &> /dev/null; then
            print_info "Installing PyTorch with CUDA..."
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        else
            conda install pytorch torchvision torchaudio -c pytorch -y
        fi
    fi
    
    print_info "Installing other dependencies via pip..."
    python -m pip install -r requirements.txt
    
    print_status "Conda environment setup complete!"
    echo ""
    echo "To activate this environment in the future, run:"
    echo "  conda activate $CONDA_ENV"
    echo ""
}

# Setup Docker
setup_docker() {
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not available, skipping Docker setup"
        return
    fi
    
    print_info "Setting up Docker..."
    
    # Create .dockerignore if it doesn't exist
    if [ ! -f .dockerignore ]; then
        cat > .dockerignore << 'EOF'
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.env
.venv
venv
.env.local
.coverage
.pytest_cache
.mypy_cache
.ruff_cache
*.log
.DS_Store
Thumbs.db
EOF
        print_status "Created .dockerignore"
    fi
    
    # Build Docker images
    print_info "Building Docker images..."
    
    if [ "$GPU_SUPPORT" = true ]; then
        print_info "Building GPU-enabled images..."
        docker build -t sloughgpt:gpu -f infra/docker/Dockerfile.gpu . 2>/dev/null || \
            docker build -t sloughgpt:latest -f infra/docker/Dockerfile . || true
    else
        docker build -t sloughgpt:latest -f infra/docker/Dockerfile .
    fi
    
    print_status "Docker image built successfully"
    echo ""
}

# Create convenient scripts
create_scripts() {
    print_info "Creating convenience scripts..."
    
    # Start script (do not overwrite repo-shipped scripts/deploy/start.sh)
    mkdir -p scripts/deploy
    if [ ! -f scripts/deploy/start.sh ]; then
    cat > scripts/deploy/start.sh << 'EOF'
#!/bin/bash
# Start SloughGPT API server
set -e
ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT/apps/api/server"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x scripts/deploy/start.sh
    print_status "Created scripts/deploy/start.sh"
    else
    print_status "Keeping existing scripts/deploy/start.sh"
    fi
    
    # Development script
    if [ ! -f dev.sh ]; then
    cat > dev.sh << 'EOF'
#!/bin/bash
# Start SloughGPT in development mode with debug logging
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/.venv/bin/activate"
export SLOUGHGPT_ENV=development
export SLOUGHGPT_LOG_LEVEL=DEBUG
cd "$ROOT/apps/api/server"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
EOF
    chmod +x dev.sh
    print_status "Created dev.sh"
    else
    print_status "Keeping existing dev.sh"
    fi
    
    # Docker start script
    if [ ! -f docker-start.sh ]; then
    cat > docker-start.sh << 'EOF'
#!/bin/bash
# Start SloughGPT with Docker Compose
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if docker compose version &>/dev/null; then
  docker compose -f infra/docker/docker-compose.yml up -d api
else
  docker-compose -f infra/docker/docker-compose.yml up -d api
fi
echo "API running at http://localhost:8000"
echo "Docs at http://localhost:8000/docs"
EOF
    chmod +x docker-start.sh
    print_status "Created docker-start.sh"
    else
    print_status "Keeping existing docker-start.sh"
    fi
    
    # Test script
    if [ ! -f test.sh ]; then
    cat > test.sh << 'EOF'
#!/bin/bash
# Run SloughGPT tests

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

echo "Running unit tests..."
python3 -m pytest tests/test_*.py -v

echo ""
echo "Running with coverage..."
python3 -m pytest tests/ --cov=domains --cov-report=html --cov-report=term
EOF
    chmod +x test.sh
    print_status "Created test.sh"
    else
    print_status "Keeping existing test.sh"
    fi
    
    # Benchmark script
    if [ ! -f benchmark.sh ]; then
    cat > benchmark.sh << 'EOF'
#!/bin/bash
# Run performance benchmarks

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

echo "Running benchmarks..."
python3 -c "
from domains.inference.engine import create_engine
from domains.ml_infrastructure.benchmarking import benchmark_model

print('Loading model...')
engine = create_engine('gpt2', device='cpu')

print('Running benchmark...')
result = benchmark_model(engine.model, engine.tokenizer, device='cpu')
print(f'Model: {result.model_name}')
print(f'Parameters: {result.num_parameters:,}')
print(f'Memory: {result.memory_mb:.2f} MB')
print(f'Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec')
"
EOF
    chmod +x benchmark.sh
    print_status "Created benchmark.sh"
    else
    print_status "Keeping existing benchmark.sh"
    fi
    
    echo ""
}

# Print next steps
print_next_steps() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}       Setup Complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    if [ "$USE_CONDA" = true ]; then
        echo "Conda environment: ${GREEN}$CONDA_ENV${NC}"
        echo ""
        echo "Next steps:"
        echo ""
        echo "1. Activate conda environment:"
        echo -e "   ${GREEN}conda activate $CONDA_ENV${NC}"
        echo ""
        echo "2. Or use conda run (no activation needed):"
        echo -e "   ${GREEN}conda run -n $CONDA_ENV python3 cli.py train --epochs 3${NC}"
        echo -e "   ${GREEN}./run.sh python3 -c \"import torch; print(torch.__version__)\"${NC}"
        echo ""
    else
        echo "Virtual environment: ${GREEN}$VENV_DIR${NC}"
        echo ""
        echo "Next steps:"
        echo ""
        echo "1. Activate virtual environment:"
        echo -e "   ${GREEN}source $VENV_DIR/bin/activate${NC}"
        echo ""
        echo "2. Or use ./run.sh (no activation needed):"
        echo -e "   ${GREEN}./run.sh python3 cli.py train --epochs 3${NC}"
        echo ""
    fi
    
    echo "3. Start the API server:"
    echo -e "   ${GREEN}./scripts/deploy/start.sh${NC}"
    echo "   or"
    echo -e "   ${GREEN}./run.sh python3 -m uvicorn main:app --app-dir apps/api/server --reload${NC}"
    echo ""
    echo "4. Access the API:"
    echo "   - API: http://localhost:8000"
    echo "   - Docs: http://localhost:8000/docs"
    echo "   - ReDoc: http://localhost:8000/redoc"
    echo ""
    
    if command -v docker &> /dev/null; then
        echo "5. Docker deployment:"
        echo -e "   ${GREEN}./docker-start.sh${NC}"
        echo "   or"
        echo -e "   ${GREEN}docker compose -f infra/docker/docker-compose.yml up -d api${NC}"
        echo ""
    fi
    
    echo "6. Run tests:"
    echo -e "   ${GREEN}./run.sh python3 -m pytest tests/ -x${NC}"
    echo ""
    
    echo "7. Verify PyTorch:"
    echo -e "   ${GREEN}./run.sh python3 -c \"import torch; print(f'PyTorch {torch.__version__}')\"${NC}"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_env
    setup_directories
    
    if [ "$MODE" != "docker" ]; then
        install_python_deps
    fi
    
    if [ "$MODE" != "local" ]; then
        setup_docker
    fi
    
    create_scripts
    print_next_steps
}

main
