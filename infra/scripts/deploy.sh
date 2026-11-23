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

# ONE registry scheme — the one ci_cd.yml publishes to
# (ghcr.io/datawex/sloughgpt{,-web,-cli}), sha-tagged from CI. This used to
# default to the stale fork owner ghcr.io/iamtowbee, whose tags no compose
# file ever pulled. Override with REGISTRY=.../TAG=... for an experiment.
REGISTRY="${REGISTRY:-ghcr.io/datawex}"
TAG="${TAG:-latest}"
TARGET="${1:-all}"

# infra/docker/.env is gitignored, so a fresh clone has none and compose v2
# hard-errors on `--env-file <missing>`. Fall back to the tracked
# .env.production; if neither exists, omit --env-file so compose's own
# `${VAR:?}` guards report exactly which key to set.
if [[ ! -f "$ENV_FILE" && -f "$REPO_ROOT/infra/docker/.env.production" ]]; then
    ENV_FILE="$REPO_ROOT/infra/docker/.env.production"
fi
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [[ -f "$ENV_FILE" ]]; then
    COMPOSE_ARGS+=(--env-file "$ENV_FILE")
fi

log() { echo -e "\033[1;34m[deploy]\033[0m $*"; }
err() { echo -e "\033[1;31m[deploy]\033[0m $*" >&2; exit 1; }

check_deps() {
    command -v docker >/dev/null 2>&1 || err "docker not found"
    command -v docker compose >/dev/null 2>&1 || err "docker compose not found"
}

build() {
    # Names match compose's `image:` (and therefore what `pull` fetches):
    # the push half and the pull half finally speak the same tag.
    log "Building API image..."
    docker build -t "$REGISTRY/sloughgpt:$TAG" \
        -f "$REPO_ROOT/infra/docker/Dockerfile" "$REPO_ROOT"

    log "Building Web image..."
    docker build -t "$REGISTRY/sloughgpt-web:$TAG" \
        -f "$REPO_ROOT/apps/web/Dockerfile" "$REPO_ROOT"
}

push() {
    log "Pushing images to $REGISTRY..."
    docker push "$REGISTRY/sloughgpt:$TAG"
    docker push "$REGISTRY/sloughgpt-web:$TAG"
}

pull() {
    log "Pulling published images (api/web only)..."
    # Only the services CI publishes (ghcr.io/datawex/sloughgpt{,-web}).
    # gateway/gpu/dev have no CI build job — pulling them would fail on a
    # registry tag that does not exist (card 20261121_150816).
    docker compose "${COMPOSE_ARGS[@]}" pull api web
}

deploy() {
    log "Deploying $TARGET..."
    # "all" is not a compose service name — omit the selector to bring up
    # the whole project (compose v2 errors on `up -d all`).
    if [[ "$TARGET" == "all" ]]; then
        docker compose "${COMPOSE_ARGS[@]}" up -d
    else
        docker compose "${COMPOSE_ARGS[@]}" up -d "$TARGET"
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
