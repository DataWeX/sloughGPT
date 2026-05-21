# SloughGPT Deployment Guide

**See also:** [Deployment checklist](DEPLOYMENT_CHECKLIST.md) (security, infra, and release gates before production).

Compose examples use **Docker Compose V2** and the stack file at **`infra/docker/docker-compose.yml`** (run all commands from the **repository root**).

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT

# 2. Start with Docker Compose
docker compose -f infra/docker/docker-compose.yml up -d api

# 3. Access the API
curl http://localhost:8000/health
```

## Deployment Options

### 1. Local Development

```bash
# Start API server
docker compose -f infra/docker/docker-compose.yml --profile dev up -d dev

# Access at http://localhost:8000
```

### 2. Production (CPU)

```bash
docker compose -f infra/docker/docker-compose.yml up -d api
```

### 3. Production (GPU)

```bash
docker compose -f infra/docker/docker-compose.yml --profile gpu up -d api-gpu
```

### 4. Full Stack with Monitoring

```bash
docker compose -f infra/docker/docker-compose.yml --profile monitoring up -d prometheus grafana api
```

### 5. With Vector Store

```bash
# ChromaDB (local)
docker compose -f infra/docker/docker-compose.yml --profile vector up -d chromadb api

# Or with Weaviate
docker compose -f infra/docker/docker-compose.yml --profile vector up -d weaviate api
```

### 6. Full Production Stack

```bash
docker compose -f infra/docker/docker-compose.yml up -d api redis prometheus grafana
```

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_PROVIDER` | `in_memory` | Vector store: in_memory, pinecone, weaviate, chromadb |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Embedding: sentence_transformers, openai, huggingface |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `PINECONE_API_KEY` | - | Pinecone API key |
| `WEAVIATE_URL` | - | Weaviate URL |

## Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f infra/k8s/k8s/

# Or use Helm
helm install sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt --create-namespace
```

### GPU Support in K8s

```bash
# Install NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# Deploy with GPU: set resources.api (nvidia.com/gpu) in infra/k8s/helm/sloughgpt/values.yaml, then:
helm upgrade --install sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt --create-namespace
```

## API Endpoints

### Generation
- `POST /generate` - Text generation
- `POST /generate/stream` - Streaming generation
- `POST /chat/stream` - Chat completion

### Vector Store
- `POST /vector/init` - Initialize vector store
- `GET /vector/stats` - Get statistics
- `POST /vector/upsert` - Add documents
- `POST /vector/query` - Semantic search
- `POST /vector/search` - RAG-style search

### Model
- `GET /models` - List models
- `POST /load` - Load model
- `GET /soul` - Get soul profile

## Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health
curl http://localhost:8000/health/detailed

# Readiness
curl http://localhost:8000/health/ready
```

## Scaling

### Horizontal Scaling

```bash
# Scale API replicas
kubectl scale deployment sloughgpt-api --replicas=3

# Or with Docker Compose
docker compose -f infra/docker/docker-compose.yml up -d --scale api=3
```

### Load Balancing

Use Traefik or Nginx as reverse proxy:

```yaml
# traefik.toml
[backends.backend1]
  [[backends.backend1.servers]]
    url = "http://api1:8000"

[[frontends.frontend1.routes]]
  rule = "PathPrefix:/"
  backend = "backend1"
```

## Security

### Enable Authentication

```bash
# Set in .env
ENABLE_AUTH=true
JWT_SECRET_KEY=your-secure-secret
```

### API Keys

```bash
# Generate API key
openssl rand -hex 32

# Add to .env
SLOUGHGPT_API_KEYS=key1,key2
```

## Monitoring

### Prometheus Metrics

```bash
# Access metrics
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:8000/metrics/prometheus
```

### Grafana Dashboards

```
URL: http://localhost:3001
Username: admin
Password: admin
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose -f infra/docker/docker-compose.yml logs api

# Check resource limits
docker stats
```

### Out of Memory

```bash
# Reduce batch size
MODEL_BATCH_SIZE=2

# Use quantization
docker compose --env-file .env.minimal -f infra/docker/docker-compose.yml up -d api
```

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check container GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Production Checklist

- [ ] Set secure JWT_SECRET_KEY
- [ ] Configure API keys
- [ ] Enable HTTPS/SSL
- [ ] Setup monitoring
- [ ] Configure backups
- [ ] Set resource limits
- [ ] Enable authentication
- [ ] Configure rate limiting
- [ ] Setup log aggregation
- [ ] Test failover

## Support

- Issues: https://github.com/iamtowbee/sloughGPT/issues
- Docs: https://github.com/iamtowbee/sloughGPT/docs
