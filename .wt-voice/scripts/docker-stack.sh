#!/bin/bash
# SloughGPT Docker Stack Startup Script

set -e

echo "🚀 Starting SloughGPT Stack..."

# Check for .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env from example..."
    cp infra/docker/.env.example .env
    echo "⚠️  Please edit .env with your configuration before restarting"
fi

# Parse arguments
PROFILE="${1:-full}"

case "$PROFILE" in
    full)
        echo "📦 Starting full stack (API + Web + Monitoring)..."
        docker compose -f infra/docker/docker-compose.yml --profile full up -d
        ;;
    api)
        echo "📦 Starting API only..."
        docker compose -f infra/docker/docker-compose.yml up -d api
        ;;
    gpu)
        echo "📦 Starting GPU stack..."
        docker compose -f infra/docker/docker-compose.yml --profile gpu up -d
        ;;
    dev)
        echo "📦 Starting development stack with hot reload..."
        docker compose -f infra/docker/docker-compose.yml --profile dev up -d
        ;;
    stop)
        echo "🛑 Stopping all containers..."
        docker compose -f infra/docker/docker-compose.yml down
        ;;
    logs)
        echo "📜 Following API logs..."
        docker logs -f sloughgpt-api
        ;;
    status)
        echo "📊 Container status:"
        docker compose -f infra/docker/docker-compose.yml ps
        ;;
    *)
        echo "Usage: $0 [full|api|gpu|dev|stop|logs|status]"
        echo ""
        echo "Profiles:"
        echo "  full  - Start everything (default)"
        echo "  api   - API server only"
        echo "  gpu   - GPU-enabled API"
        echo "  dev   - Development with hot reload"
        echo "  stop  - Stop all containers"
        echo "  logs  - Follow API logs"
        echo "  status - Show container status"
        exit 1
        ;;
esac

echo "✅ Done!"
