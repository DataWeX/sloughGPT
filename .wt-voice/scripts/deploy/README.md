# Deployment & ops scripts

Run these from the **repository root** (so relative paths like `infra/docker/docker-compose.yml` resolve correctly):

| Script | Purpose |
|--------|---------|
| `docker-manage.sh` | Docker Compose profiles (api, dev, gpu, …) |
| `deploy.sh` | Production-style build/deploy helpers (uses `git` to find repo root) |
| `deploy-production.sh` | Extended production / k8s-oriented flow |
| `start.sh` | Start API / web / Docker modes (uses `git` to find repo root) |
| `quickstart.sh` | Conda/venv quick setup helper |
| `run_benchmark.sh` | Benchmark driver |
| `check_ci_status.sh` | GitHub Actions status via `gh` CLI |

Examples:

```bash
./scripts/deploy/docker-manage.sh start
./scripts/deploy/start.sh api 8000
```

Install and verification entrypoints remain at the repo root: **`install.sh`**, **`verify.sh`**, **`run.sh`**.
