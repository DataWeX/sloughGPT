# SloughGPT Deployment Checklist

## Pre-Deployment

### Security Checklist
- [ ] Generate new API keys (`openssl rand -hex 32`)
- [ ] Generate new JWT secret (`openssl rand -hex 64`)
- [ ] Update `.env` file with production values
- [ ] Enable HTTPS/SSL certificates
- [ ] Configure CORS origins for production domain
- [ ] Review rate limiting settings

### Infrastructure Checklist
- [ ] Verify Kubernetes cluster is accessible
- [ ] Check GPU nodes are available (if using GPU)
- [ ] Verify storage class exists for PVCs
- [ ] Ensure ingress controller is installed
- [ ] Check Prometheus/Grafana operator is installed

### Dependencies Checklist
- [ ] All tests passing locally
- [ ] Docker images built and pushed to registry
- [ ] Helm chart tested in staging

## Docker Deployment

### Using Docker Compose
```bash
# 1. Clone repository
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT

# 2. Copy environment file
cp .env.example .env
# Edit .env with production values

# 3. Start services (repo root; Compose file lives under infra/docker)
docker compose -f infra/docker/docker-compose.yml up -d api

# 4. Check status
docker compose -f infra/docker/docker-compose.yml ps
docker compose -f infra/docker/docker-compose.yml logs -f api
```

### Using Docker (Manual)
```bash
# Build image (repository root as context)
docker build -f infra/docker/Dockerfile -t sloughgpt/api:latest .

# Run container
docker run -d \
  --name sloughgpt-api \
  -p 8000:8000 \
  -e SLOUGHGPT_API_KEY=your-key \
  -e SLOUGHGPT_JWT_SECRET=your-secret \
  -v ./models:/app/models \
  sloughgpt/api:latest
```

## Kubernetes Deployment

### Using Helm
```bash
# 1. Add Helm repository (if published)
helm repo add sloughgpt https://iamtowbee.github.io/sloughGPT
helm repo update

# 2. Create namespace
kubectl create namespace sloughgpt

# 3. Install chart
helm install sloughgpt sloughgpt/sloughgpt \
  --namespace sloughgpt \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=sloughgpt.example.com \
  --set config.apiKey=your-production-api-key \
  --set config.jwtSecret=your-production-jwt-secret

# 4. Check deployment
kubectl get pods -n sloughgpt
kubectl logs -n sloughgpt -l app.kubernetes.io/component=api
```

### Using kubectl (Manifests)
```bash
# 1. Apply manifests (from repository root; review YAMLs before production)
kubectl apply -f infra/k8s/k8s/

# 2. For GPU support, ensure device plugin is installed
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/device-plugin/master/nvidia-device-plugin.yml
```

### Verify Deployment
```bash
# Check all resources
kubectl get all -n sloughgpt

# Check pods are running
kubectl get pods -n sloughgpt -w

# Check service endpoints
kubectl get endpoints -n sloughgpt

# Test health endpoint
curl http://sloughgpt.example.com/health
curl http://sloughgpt.example.com/health/ready
```

## Monitoring Setup

### Prometheus
```bash
# Check Prometheus is scraping metrics
kubectl exec -n monitoring deploy/prometheus-server -- \
  promtool query instant query 'sloughgpt_uptime_seconds'
```

### Grafana
```bash
# Import dashboard from grafana/dashboards/
# Dashboard UID: sloughgpt

# Or import via API
curl -X POST -H "Content-Type: application/json" \
  -d @grafana/dashboards/sloughgpt-dashboard.yaml \
  http://admin:admin@grafana.example.com/api/dashboards
```

## Post-Deployment

### Health Checks
- [ ] `GET /health` returns 200
- [ ] `GET /health/ready` returns 200 with model loaded
- [ ] `GET /health/live` returns 200
- [ ] `GET /metrics` returns Prometheus metrics

### Security Verification
- [ ] Rate limiting is working
- [ ] JWT authentication works
- [ ] Audit logs are being recorded
- [ ] Security headers are present

### Load Testing
```bash
# Test with curl
for i in {1..100}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    http://sloughgpt.example.com/health
done

# Test with Apache Bench (if available)
ab -n 1000 -c 10 http://sloughgpt.example.com/health
```

## Rollback Procedures

### Docker Compose
```bash
# Rollback to previous version
docker compose -f infra/docker/docker-compose.yml down
git checkout v1.0.0
docker compose -f infra/docker/docker-compose.yml up -d api
```

### Kubernetes
```bash
# Rollback deployment
kubectl rollout undo deployment/sloughgpt-api -n sloughgpt

# Check rollback status
kubectl rollout status deployment/sloughgpt-api -n sloughgpt
```

## Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status
kubectl describe pod -n sloughgpt <pod-name>

# Check events
kubectl get events -n sloughgpt --sort-by='.lastTimestamp'
```

#### Model Not Loading
```bash
# Check PVC is bound
kubectl get pvc -n sloughgpt

# Check model files exist
kubectl exec -it -n sloughgpt deploy/sloughgpt-api -- ls -la /app/models
```

#### GPU Not Available
```bash
# Check nvidia-device-plugin is running
kubectl get pods -n nvidia-gpu-operator

# Check node has GPU
kubectl describe node <node-name> | grep nvidia
```

#### High Memory Usage
```bash
# Check pod resource usage
kubectl top pods -n sloughgpt

# Adjust limits in values.yaml
resources:
  limits:
    memory: "8Gi"
```

## Production Checklist

### Security
- [ ] API keys rotated
- [ ] SSL/TLS enabled
- [ ] CORS configured for specific origins
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Secrets stored in Kubernetes secrets

### Scaling
- [ ] HPA configured and tested
- [ ] Load balancer health checks working
- [ ] Database connection pooling configured

### Backup
- [ ] Regular backups scheduled
- [ ] Backup restoration tested
- [ ] Model weights backed up

### Documentation
- [ ] Runbook created
- [ ] On-call contacts documented
- [ ] Architecture diagram updated
