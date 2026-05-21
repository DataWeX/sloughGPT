## Infra

Deployment and operations assets.

- `docker/` — Dockerfiles, Compose file, container helpers.
- `k8s/` — Kubernetes, Helm, Grafana, Prometheus.
- `scripts/` — deployment/ops scripts.

**Compose (API only, from repo root):** `docker compose -f infra/docker/docker-compose.yml up -d api` — profiles and stacks are documented in **docs/DEPLOYMENT.md** and **QUICKSTART.md**.

**CI:** **`.github/workflows/ci_cd.yml`** builds images from **`infra/docker/Dockerfile`** and **`infra/docker/Dockerfile.dev`** (job **`build`**, context = repo root).
