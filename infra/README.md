## Infra

Deployment and operations assets.

- `docker/` — Dockerfiles, Compose file, container helpers.
- `k8s/` — Kubernetes, Helm, Grafana, Prometheus.
- `scripts/` — deployment/ops scripts.

**Compose (API only, from repo root):** `docker compose -f infra/docker/docker-compose.yml up -d api` — profiles and stacks are documented in **docs/DEPLOYMENT.md** and **QUICKSTART.md**.

**CI:** **`.github/workflows/ci_cd.yml`** (job **`build`**, context = repo root, **push to main only — PRs do not build images**) publishes exactly three images: **`infra/docker/Dockerfile`** (API), **`apps/web/Dockerfile`** (web), **`infra/docker/Dockerfile.cli`** (CLI). The other Dockerfiles are never published; they are build-checked (no push) by **`.github/workflows/dockerfile-rot.yml`**, and every Dockerfile's build context is validated on each run by **`scripts/check_dockerfile_contexts.py`** (COPY sources must exist _and_ survive `.dockerignore`).
