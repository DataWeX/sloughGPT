#!/bin/bash
# SloughGPT Production Deploy
# One-command deployment: nginx + postgres + redis + api + web + prometheus + grafana + backup
#
# Usage:
#   ./deploy.sh                    # Deploy full stack
#   ./deploy.sh --build            # Rebuild images then deploy
#   ./deploy.sh --down             # Tear down stack
#   ./deploy.sh --status           # Show service status
#   ./deploy.sh --logs [service]   # Tail logs
#   ./deploy.sh --backup           # Manual backup
#   ./deploy.sh --restore <file>   # Restore from backup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
ENV_FILE="${SCRIPT_DIR}/.env.production"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${BLUE}[deploy]${NC} $1"; }
ok()    { echo -e "${GREEN}[  ok ]${NC} $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $1"; }
err()   { echo -e "${RED}[error]${NC} $1"; }

# ─── Preflight ─────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in docker; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if ! docker compose version &>/dev/null 2>&1; then
        missing+=("docker-compose")
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing: ${missing[*]}"
        exit 1
    fi
}

check_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        err ".env.production not found at $ENV_FILE"
        err "Copy infra/docker/.env.production to .env.production and set secrets"
        exit 1
    fi
    # Check critical secrets
    source "$ENV_FILE"
    local warnings=()
    [[ "${POSTGRES_PASSWORD:-}" == "CHANGE_ME" || -z "${POSTGRES_PASSWORD:-}" ]] && warnings+=("POSTGRES_PASSWORD")
    [[ "${REDIS_PASSWORD:-}" == "CHANGE_ME" || -z "${REDIS_PASSWORD:-}" ]] && warnings+=("REDIS_PASSWORD")
    [[ "${GRAFANA_ADMIN_PASSWORD:-}" == "CHANGE_ME" || -z "${GRAFANA_ADMIN_PASSWORD:-}" ]] && warnings+=("GRAFANA_ADMIN_PASSWORD")
    [[ "${SLO_JWT_SECRET:-}" == "CHANGE_ME" || -z "${SLO_JWT_SECRET:-}" ]] && warnings+=("SLO_JWT_SECRET")
    if [[ ${#warnings[@]} -gt 0 ]]; then
        warn "Default secrets detected: ${warnings[*]}"
        warn "Run: openssl rand -base64 32 to generate secrets"
        echo -n "Continue anyway? [y/N] "
        read -r answer
        [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    fi
}

compose() {
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
}

# ─── Self-signed TLS (for local/dev) ──────────────────────────
generate_self_signed_cert() {
    local ssl_dir="${SCRIPT_DIR}/docker/nginx/ssl"
    if [[ -f "${ssl_dir}/cert.pem" && -f "${ssl_dir}/key.pem" ]]; then
        ok "TLS certs already exist"
        return
    fi
    log "Generating self-signed TLS certificate..."
    mkdir -p "$ssl_dir"
    openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout "${ssl_dir}/key.pem" \
        -out "${ssl_dir}/cert.pem" \
        -subj "/CN=localhost/O=SloughGPT/C=US" \
        2>/dev/null
    ok "Self-signed cert generated at ${ssl_dir}/"
}

# ─── Build ─────────────────────────────────────────────────────
build() {
    log "Building Docker images..."
    compose build --parallel
    ok "Images built"
}

# ─── Deploy ────────────────────────────────────────────────────
deploy() {
    log "Deploying SloughGPT production stack..."
    generate_self_signed_cert
    compose up -d --remove-orphans
    log "Waiting for services to become healthy..."

    local timeout=120
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        local unhealthy
        unhealthy=$(compose ps --format json 2>/dev/null | python3 -c "
import sys, json
try:
    services = [json.loads(line) for line in sys.stdin if line.strip()]
    unhealthy = [s['Service'] for s in services if s.get('Health','') not in ('healthy','')]
    print('\n'.join(unhealthy))
except:
    pass
" 2>/dev/null || true)

        if [[ -z "$unhealthy" ]]; then
            ok "All services healthy"
            show_endpoints
            return
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -ne "\r  Waiting... ${elapsed}s (unhealthy: $(echo "$unhealthy" | tr '\n' ', '))"
    done
    warn "Some services may not be healthy yet. Check: docker compose -f $COMPOSE_FILE ps"
    show_endpoints
}

# ─── Status ────────────────────────────────────────────────────
status() {
    compose ps
}

# ─── Logs ──────────────────────────────────────────────────────
logs() {
    local service="${1:-}"
    if [[ -n "$service" ]]; then
        compose logs -f --tail=100 "$service"
    else
        compose logs -f --tail=50
    fi
}

# ─── Down ──────────────────────────────────────────────────────
down() {
    log "Tearing down stack..."
    compose down --remove-orphans
    ok "Stack stopped"
}

# ─── Backup ────────────────────────────────────────────────────
backup() {
    log "Triggering database backup..."
    compose exec -T postgres pg_dump -U "${POSTGRES_USER:-sloughgpt}" "${POSTGRES_DB:-sloughgpt}" | gzip > "backup_$(date +%Y%m%d_%H%M%S).sql.gz"
    ok "Backup saved"
}

# ─── Restore ───────────────────────────────────────────────────
restore() {
    local file="${1:-}"
    if [[ -z "$file" || ! -f "$file" ]]; then
        err "Usage: $0 --restore <backup.sql.gz>"
        exit 1
    fi
    warn "This will OVERWRITE the current database. Continue? [y/N]"
    read -r answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    log "Restoring from $file..."
    gunzip -c "$file" | compose exec -T postgres psql -U "${POSTGRES_USER:-sloughgpt}" "${POSTGRES_DB:-sloughgpt}"
    ok "Restore complete"
}

# ─── Endpoints ─────────────────────────────────────────────────
show_endpoints() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SloughGPT Production Stack"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Web UI:      http://localhost"
    echo "  API:         http://localhost/v1/health"
    echo "  Grafana:     http://localhost:3000 (via nginx if routed)"
    echo "  Prometheus:  http://localhost:9090 (internal)"
    echo ""
    echo "  Logs:        $0 --logs [service]"
    echo "  Status:      $0 --status"
    echo "  Backup:      $0 --backup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ─── Main ──────────────────────────────────────────────────────
check_deps

case "${1:-deploy}" in
    --build)
        check_env
        build
        deploy
        ;;
    --down)
        down
        ;;
    --status)
        status
        ;;
    --logs)
        logs "${2:-}"
        ;;
    --backup)
        backup
        ;;
    --restore)
        restore "${2:-}"
        ;;
    --help|-h)
        echo "Usage: $0 [--build|--down|--status|--logs <svc>|--backup|--restore <file>]"
        ;;
    deploy|"")
        check_env
        deploy
        ;;
    *)
        err "Unknown flag: $1"
        echo "Usage: $0 [--build|--down|--status|--logs <svc>|--backup|--restore <file>]"
        exit 1
        ;;
esac
