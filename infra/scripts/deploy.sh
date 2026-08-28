#!/usr/bin/env bash
set -euo pipefail

# SloughGPT Cloud Deployment Script
# Usage: ./deploy.sh [api|web|all]
# Requirements: docker, docker compose, curl

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/infra/docker/docker-compose.prod.yml"
ENV_FILE="$REPO_ROOT/infra/docker/.env"

REGISTRY="${REGISTRY:-ghcr.io/iamtowbee}"
TAG="${TAG:-latest}"
TARGET="${1:-all}"

log() { echo -e "\033[1;34m[deploy]\033[0m $*"; }
err() { echo -e "\033[1;31m[deploy]\033[0m $*" >&2; exit 1; }

check_deps() {
    command -v docker >/dev/null 2>&1 || err "docker not found"
    command -v docker compose >/dev/null 2>&1 || err "docker compose not found"
    command -v curl >/dev/null 2>&1 || err "curl not found"
}

build() {
    log "Building API image..."
    docker build -t "$REGISTRY/sloughgpt-api:$TAG" \
        -f "$REPO_ROOT/infra/docker/Dockerfile" "$REPO_ROOT"

    log "Building Web image..."
    docker build -t "$REGISTRY/sloughgpt-web:$TAG" \
        -f "$REPO_ROOT/apps/web/Dockerfile" "$REPO_ROOT"
}

push() {
    log "Pushing images to $REGISTRY..."
    docker push "$REGISTRY/sloughgpt-api:$TAG"
    docker push "$REGISTRY/sloughgpt-web:$TAG"
}

pull() {
    log "Pulling images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
}

deploy() {
    log "Deploying $TARGET..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "$TARGET"
}

health() {
    log "Waiting for health checks..."
    sleep 5
    curl -fsS http://localhost:8000/health || err "API health check failed"
    curl -fsS http://localhost:3000/ || err "Web health check failed"
    log "All services healthy."
}

case "${TARGET}" in
    api|web|all)
        check_deps
        if [[ "${PULL:-1}" == "1" ]]; then
            pull
        else
            build
            push
            pull
        fi
        deploy
        health
        log "Deployment complete: https://your-domain.com"
        ;;
    *)
        echo "Usage: $0 [api|web|all]"
        echo "  all  - Deploy both API and Web (default)"
        echo "  api  - Deploy API only"
        echo "  web  - Deploy Web only"
        exit 1
        ;;
esac
