#!/bin/bash

# SloughGPT Startup Script
# Usage: ./start.sh [mode] [port]

set -e

ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
COMPOSE="docker compose -f $ROOT/infra/docker/docker-compose.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Banner
echo -e "${CYAN}"
echo "  _____ _ _       _____     _____"
echo " |_   _(_) |     / ____|   / ____|"
echo "   | |  _| | ___| (___  ___| (___   ___ __ _ _ __  _ __   ___ _ __"
echo "   | | |/ | |/ __|\___ \/ __|\___ \ / __/ _\` | '_ \| '_ \ / _ \ '__|"
echo "   | |   <| | (__ ____) | (___)__) | (_| (_| | |_) | |_) |  __/ |"
echo "   |_|  \_\_\_|\___|_____/ \____/|___\___\__,_| .__/| .__/ \___|_|"
echo "                                              | |   | |"
echo "                                              |_|   |_|"
echo -e "${NC}"
echo "Enterprise AI Framework - Version 1.0.0"
echo ""

# Check Python
log_info "Checking Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_success "Python $PYTHON_VERSION"

# Check dependencies
log_info "Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    log_warn "Installing dependencies..."
    pip3 install fastapi uvicorn pydantic torch transformers --quiet
}
log_success "Dependencies OK"

# Check environment
if [ -f .env ]; then
    log_success ".env file found"
else
    if [ -f .env.example ]; then
        log_warn ".env not found. Creating..."
        cp .env.example .env
        log_info "Please edit .env and add your API keys"
    fi
fi

# Parse arguments
MODE=${1:-development}
PORT=${2:-8000}

case $MODE in
    development|dev)
        log_info "Starting DEVELOPMENT mode on port $PORT..."
        export SLOUGHGPT_ENV=development
        cd "$ROOT/apps/api/server"
        python3 main.py
        ;;
    production|prod)
        log_info "Starting PRODUCTION mode on port $PORT..."
        export SLOUGHGPT_ENV=production
        uvicorn main:app --app-dir "$ROOT/apps/api/server" --host 0.0.0.0 --port "$PORT" --workers 4
        ;;
    docker)
        log_info "Starting with Docker..."
        eval "$COMPOSE up -d api"
        eval "$COMPOSE logs -f api"
        ;;
    kubernetes|k8s)
        log_info "Checking Kubernetes..."
        kubectl get pods -n sloughgpt 2>/dev/null || log_error "kubectl not configured"
        ;;
    docker-gpu|gpu)
        log_info "Starting with Docker GPU..."
        docker compose -f "$ROOT/infra/docker/docker-compose.yml" --profile gpu up -d
        ;;
    all)
        log_info "Starting ALL services..."
        eval "$COMPOSE up -d"
        ( cd "$ROOT/apps/api/server" && python3 main.py ) &
        npm run dev --prefix "$ROOT/apps/web" &
        log_success "All services started"
        echo "  API: http://localhost:8000"
        echo "  UI:  http://localhost:3000"
        ;;
    help|-h|--help)
        echo "Usage: $0 [MODE] [PORT]"
        echo ""
        echo "Modes:"
        echo "  development, dev    Start in development mode (default)"
        echo "  production, prod    Start in production mode"
        echo "  docker              Start with Docker Compose"
        echo "  docker-gpu, gpu    Start with Docker GPU"
        echo "  kubernetes, k8s    Check Kubernetes status"
        echo "  all                Start all services"
        echo ""
        echo "Examples:"
        echo "  $0                    # Dev mode, port 8000"
        echo "  $0 production 8080     # Prod API (uvicorn) on port 8080"
        echo "  $0 docker             # Docker mode"
        ;;
    *)
        log_error "Unknown mode: $MODE"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
