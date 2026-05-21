#!/bin/bash
# SloughGPT Deployment Scripts
# Production deployment automation

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
COMPOSE_FILE="$REPO_ROOT/infra/docker/docker-compose.yml"
DOCKERFILE="$REPO_ROOT/infra/docker/Dockerfile"
DOCKERFILE_GPU="$REPO_ROOT/infra/docker/Dockerfile.gpu"

compose() {
    if docker compose version &>/dev/null; then
        docker compose -f "$COMPOSE_FILE" "$@"
    else
        docker-compose -f "$COMPOSE_FILE" "$@"
    fi
}

echo "🚀 SloughGPT Production Deployment"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Install Docker Compose v2 or docker-compose."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+."
        exit 1
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        GPU_AVAILABLE=true
    else
        print_warning "No CUDA GPU detected. Will use CPU inference."
        GPU_AVAILABLE=false
    fi
    
    print_status "Prerequisites check completed."
}

# Build Docker image
build_image() {
    print_status "Building SloughGPT Docker image..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        docker build -t sloughgpt:latest-gpu -f "$DOCKERFILE_GPU" "$SCRIPT_DIR"
        docker tag sloughgpt:latest-gpu sloughgpt:latest
    else
        docker build -t sloughgpt:latest -f "$DOCKERFILE" "$REPO_ROOT"
    fi
    
    print_status "Docker image built successfully."
}

# Setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p models data checkpoints logs
    mkdir -p data/train data/val
    mkdir -p checkpoints/best checkpoints/latest
    
    # Set permissions
    chmod -R 755 models data checkpoints logs
    
    print_status "Directory structure created."
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env.production << EOF
# SloughGPT Production Environment
SLUGHGPT_ENV=production
SLUGHGPT_LOG_LEVEL=info
SLUGHGPT_HOST=0.0.0.0
SLUGHGPT_PORT=8000

# Model Configuration
SLUGHGPT_MODEL_PATH=/app/models
SLUGHGPT_DATA_PATH=/app/data
SLUGHGPT_CHECKPOINT_PATH=/app/checkpoints

# Performance Settings
SLUGHGPT_WORKERS=4
SLUGHGPT_MAX_CONNECTIONS=100
SLUGHGPT_TIMEOUT=300

# GPU Settings
EOF

    if [ "$GPU_AVAILABLE" = true ]; then
        cat >> .env.production << EOF
CUDA_VISIBLE_DEVICES=0
SLUGHGPT_DEVICE=cuda
EOF
    else
        cat >> .env.production << EOF
CUDA_VISIBLE_DEVICES=""
SLUGHGPT_DEVICE=cpu
EOF
    fi
    
    print_status "Environment file created: .env.production"
}

# Start services
start_services() {
    print_status "Starting SloughGPT services..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Starting with GPU support..."
        compose --profile gpu up -d
    else
        print_status "Starting CPU-only version..."
        compose up -d
    fi
    
    print_status "Services started successfully."
}

# Wait for service to be ready
wait_for_service() {
    print_status "Waiting for SloughGPT service to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_status "SloughGPT service is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Service not ready yet..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "Service failed to start within timeout."
    return 1
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check API health
    echo "Checking API health..."
    curl -s http://localhost:8000/health | python3 -m json.tool || {
        print_error "API health check failed"
        return 1
    }
    
    # Check model info
    echo "Checking model information..."
    curl -s http://localhost:8000/model/info | python3 -m json.tool || {
        print_warning "Model info check failed (may still be loading)"
    }
    
    # Check web interface
    echo "Checking web interface..."
    curl -s -I http://localhost:8000/ | head -1 || {
        print_error "Web interface check failed"
        return 1
    }
    
    print_status "Health checks completed."
}

# Show service status
show_status() {
    print_status "Service Status:"
    compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "  🌐 Web Interface: http://localhost:8000"
    echo "  📚 API Documentation: http://localhost:8000/docs"
    echo "  🔗 API Endpoint: http://localhost:8000"
    echo "  💓 Health Check: http://localhost:8000/health"
}

# Stop services
stop_services() {
    print_status "Stopping SloughGPT services..."
    compose down
    print_status "Services stopped."
}

# Cleanup deployment
cleanup() {
    print_status "Cleaning up deployment..."
    
    # Stop and remove containers
    compose down -v
    
    # Remove images
    docker rmi sloughgpt:latest 2>/dev/null || true
    
    # Clean up volumes (optional)
    read -p "Remove all data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        rm -rf models data checkpoints logs
        print_status "Volumes removed."
    fi
    
    print_status "Cleanup completed."
}

# Show logs
show_logs() {
    print_status "Showing service logs..."
    compose logs -f
}

# Scale services
scale_services() {
    local replicas=${1:-2}
    print_status "Scaling SloughGPT services to $replicas replicas..."
    compose up -d --scale "api=$replicas"
    print_status "Services scaled to $replicas replicas."
}

# Main deployment function
deploy() {
    print_status "Starting SloughGPT deployment..."
    
    check_prerequisites
    setup_directories
    create_env_file
    build_image
    start_services
    
    if wait_for_service; then
        run_health_checks
        show_status
        
        print_status "🎉 SloughGPT deployment completed successfully!"
        echo ""
        print_status "To stop services:    $0 stop"
        print_status "To view logs:       $0 logs"
        print_status "To scale services:    $0 scale <number>"
        print_status "To cleanup:           $0 cleanup"
    else
        print_error "Deployment failed. Check logs for details."
        show_logs
    fi
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 5
        deploy
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "scale")
        scale_services $2
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        run_health_checks
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|scale|cleanup|health}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy SloughGPT services (default)"
        echo "  stop    - Stop all services"
        echo "  restart  - Stop and restart services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  scale   - Scale services to N replicas"
        echo "  cleanup - Remove all containers and volumes"
        echo "  health  - Run health checks"
        exit 1
        ;;
esac