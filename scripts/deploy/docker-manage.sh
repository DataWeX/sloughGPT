#!/bin/bash

# SloughGPT Docker Management Script
# Comprehensive Docker deployment and management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (run this script from the repository root)
PROJECT_NAME="sloughgpt"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker/docker-compose.yml}"
ENV_FILE=".env"

compose() {
    if docker compose version &> /dev/null; then
        docker compose -f "$COMPOSE_FILE" "$@"
    else
        docker-compose -f "$COMPOSE_FILE" "$@"
    fi
}

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example "$ENV_FILE"
            log_success "Created .env from .env.example"
            log_warning "Please review and update .env with your configuration"
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p docker/nginx/conf.d docker/ssl docker/postgres docker/grafana/provisioning
    
    log_success "Environment setup completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    docker build -t ${PROJECT_NAME}:latest -f infra/docker/Dockerfile .
    docker build -t ${PROJECT_NAME}:dev -f infra/docker/Dockerfile.dev .
    docker build -t ${PROJECT_NAME}:test -f infra/docker/Dockerfile.test .
    docker build -t ${PROJECT_NAME}:sdk -f infra/docker/Dockerfile.sdk .
    
    log_success "Docker images built successfully"
}

# Start production services
start_production() {
    log_info "Starting production services..."
    
    check_dependencies
    setup_environment
    
    compose up -d
    
    log_success "Production services started"
    show_status
}

# Start development services
start_development() {
    log_info "Starting development services..."
    
    check_dependencies
    setup_environment
    
    compose --profile dev up -d
    
    log_success "Development services started"
    show_status
}

# Start services with GPU support
start_gpu() {
    log_info "Starting GPU-enabled services..."
    
    check_dependencies
    setup_environment
    
    # Check for NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime not available. Please install nvidia-docker2."
        exit 1
    fi
    
    compose --profile gpu up -d
    
    log_success "GPU-enabled services started"
    show_status
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    
    compose down
    
    log_success "Services stopped"
}

# Stop all services including volumes
clean_services() {
    log_info "Stopping and removing all services and volumes..."
    
    compose down -v --remove-orphans
    
    log_success "All services and volumes removed"
}

# Show service status
show_status() {
    log_info "Service status:"
    
    compose ps
    
    echo ""
    log_info "Service URLs:"
    
    # Get the port mappings
    API_PORT=$(compose port api 8000 2>/dev/null | cut -d: -f2)
    REDIS_PORT=$(compose port redis 6379 2>/dev/null | cut -d: -f2)
    POSTGRES_PORT=$(compose port postgres 5432 2>/dev/null | cut -d: -f2)
    
    if [ ! -z "$API_PORT" ]; then
        echo "  API:          http://localhost:$API_PORT"
        echo "  Health Check: http://localhost:$API_PORT/health"
    fi
    
    if [ ! -z "$POSTGRES_PORT" ]; then
        echo "  PostgreSQL:   localhost:$POSTGRES_PORT"
    fi
    
    if [ ! -z "$REDIS_PORT" ]; then
        echo "  Redis:        localhost:$REDIS_PORT"
    fi
}

# Show logs
show_logs() {
    local service=${1:-}
    
    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        compose logs -f
    else
        log_info "Showing logs for $service..."
        compose logs -f "$service"
    fi
}

# Run tests (core pytest suite in disposable image; see infra/docker/Dockerfile.test)
run_tests() {
    log_info "Running tests in Docker..."
    
    check_dependencies
    
    docker build -t ${PROJECT_NAME}:test -f infra/docker/Dockerfile.test .
    docker run --rm ${PROJECT_NAME}:test
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "Tests failed with exit code: $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Backup data
backup_data() {
    log_info "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup PostgreSQL
    if compose ps postgres 2>/dev/null | grep -q "Up"; then
        compose exec -T postgres pg_dump -U sloughgpt sloughgpt > "$BACKUP_DIR/postgres_backup.sql"
        log_success "PostgreSQL data backed up"
    fi
    
    # Backup application data
    if docker volume ls | grep -q "sloughgpt_sloughgpt_data"; then
        docker run --rm -v sloughgpt_sloughgpt_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/data_backup.tar.gz -C /data .
        log_success "Application data backed up"
    fi
    
    log_success "Backup created in $BACKUP_DIR"
}

# Restore data
restore_data() {
    local backup_dir=${1:-}
    
    if [ -z "$backup_dir" ] || [ ! -d "$backup_dir" ]; then
        log_error "Please provide a valid backup directory"
        exit 1
    fi
    
    log_warning "This will replace all current data. Are you sure? (y/N)"
    read -r response
    
    if [[ ! $response =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Restoring data from $backup_dir..."
    
    # Restore PostgreSQL
    if [ -f "$backup_dir/postgres_backup.sql" ]; then
        compose exec -T postgres psql -U sloughgpt -c "DROP DATABASE IF EXISTS sloughgpt;"
        compose exec -T postgres psql -U sloughgpt -c "CREATE DATABASE sloughgpt;"
        compose exec -T postgres psql -U sloughgpt sloughgpt < "$backup_dir/postgres_backup.sql"
        log_success "PostgreSQL data restored"
    fi
    
    # Restore application data
    if [ -f "$backup_dir/data_backup.tar.gz" ]; then
        docker run --rm -v sloughgpt_sloughgpt_data:/data -v "$(pwd)/$backup_dir":/backup alpine tar xzf /backup/data_backup.tar.gz -C /data
        log_success "Application data restored"
    fi
    
    log_success "Data restoration completed"
}

# Update services
update_services() {
    log_info "Updating services..."
    
    # Pull latest code
    git pull origin main
    
    # Rebuild images
    build_images
    
    # Restart services
    stop_services
    start_production
    
    log_success "Services updated successfully"
}

# Scale services
scale_services() {
    local service=${1:-api}
    local replicas=${2:-2}
    
    log_info "Scaling $service to $replicas replicas..."
    
    compose up -d --scale "$service=$replicas"
    
    log_success "$service scaled to $replicas replicas"
}

# Show help
show_help() {
    echo "SloughGPT Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start          Start production services"
    echo "  dev            Start development services"
    echo "  gpu            Start GPU-enabled services"
    echo "  stop           Stop all services"
    echo "  clean          Stop and remove all services and volumes"
    echo "  restart        Restart services"
    echo "  status         Show service status"
    echo "  logs [service] Show logs for all services or specific service"
    echo "  test           Run tests in Docker"
    echo "  build          Build Docker images"
    echo "  backup         Create data backup"
    echo "  restore [dir]  Restore data from backup directory"
    echo "  update         Update services with latest code"
    echo "  scale [svc] [n] Scale service to n replicas"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start production services"
    echo "  $0 dev                      # Start development services"
    echo "  $0 logs api                 # Show logs for API service"
    echo "  $0 scale api 3             # Scale api to 3 replicas"
    echo "  $0 restore backups/20231201_120000  # Restore from backup"
}

# Main script logic
case "${1:-}" in
    "start")
        start_production
        ;;
    "dev")
        start_development
        ;;
    "gpu")
        start_gpu
        ;;
    "stop")
        stop_services
        ;;
    "clean")
        clean_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_production
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "${2:-}"
        ;;
    "test")
        run_tests
        ;;
    "build")
        build_images
        ;;
    "backup")
        backup_data
        ;;
    "restore")
        restore_data "${2:-}"
        ;;
    "update")
        update_services
        ;;
    "scale")
        scale_services "${2:-api}" "${3:-2}"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        log_error "No command specified. Use 'help' for usage information."
        exit 1
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac