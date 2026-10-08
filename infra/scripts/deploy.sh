#!/usr/bin/env bash
set -euo pipefail

# SloughGPT Cloud Deployment Script
# Usage: ./deploy.sh [api|web|all]
# Requirements: docker, docker compose
#
# Port note (card 257d310b): docker-compose.prod.yml publishes ONLY nginx
# (:80/:443) — nothing listens on host :8000/:3000. Health is therefore read
# from container health status (docker inspect), not host curls.

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
    # "all" is not a compose service name — omit the selector to bring up
    # the whole project (compose v2 errors on `up -d all`).
    if [[ "$TARGET" == "all" ]]; then
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "$TARGET"
    fi
}

wait_healthy() {
    local name=$1 st="absent" i
    # prod publishes only nginx — container-internal healthchecks are the
    # source of truth (web's own wget localhost:3000 runs INSIDE the web
    # container, where :3000 is the real listener).
    for i in $(seq 1 90); do
        st=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$name" 2>/dev/null || echo absent)
        case "$st" in
            healthy|running)
                log "$name: $st"
                return 0
                ;;
        esac
        sleep 2
    done
    err "$name not healthy after 180s (last status: $st)"
}

health() {
    log "Waiting for health checks (container health — prod has no host :8000/:3000)..."
    case "$TARGET" in
        api) wait_healthy sloughgpt-api ;;
        web) wait_healthy sloughgpt-web ;;
        all)
            wait_healthy sloughgpt-api
            wait_healthy sloughgpt-web
            wait_healthy sloughgpt-nginx
            ;;
    esac
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
