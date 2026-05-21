#!/bin/bash
# SloughGPT Production Deployment Script
# Complete production deployment with monitoring and scaling

set -e

echo "🚀 SloughGPT Production Deployment"
echo "=================================="

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
REGION=${3:-us-west-2}
NAMESPACE=${4:-sloughgpt}

echo "📋 Configuration:"
echo "   Environment: $ENVIRONMENT"
echo "   Version: $VERSION"
echo "   Region: $REGION"
echo "   Namespace: $NAMESPACE"
echo

# Check dependencies
echo "🔍 Checking dependencies..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed"
        exit 1
    else
        echo "✅ $1 found"
    fi
}

check_command docker
check_command kubectl
check_command helm

# Docker build
echo "🏗️ Building Docker images..."

# Build main API image (monorepo: Dockerfile lives under infra/docker/)
docker build -t sloughgpt/api:$VERSION -f infra/docker/Dockerfile .
echo "✅ Built API image: sloughgpt/api:$VERSION"

# Tag for registry
REGISTRY=${REGISTRY:-your-registry.com}
docker tag sloughgpt/api:$VERSION $REGISTRY/sloughgpt/api:$VERSION

echo "✅ Image tagged for registry: $REGISTRY/sloughgpt/api:$VERSION"

# Kubernetes deployment
echo "☸️ Deploying to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMaps and Secrets
echo "📝 Applying configurations..."

kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: sloughgpt-config
  namespace: $NAMESPACE
data:
  ENVIRONMENT: "$ENVIRONMENT"
  VERSION: "$VERSION"
  REGION: "$REGION"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://sloughgpt:password@postgres:5432/sloughgpt"
  REDIS_URL: "redis://redis:6379"
  MODEL_CACHE_SIZE: "1000"
  MAX_CONCURRENT_REQUESTS: "100"
---
apiVersion: v1
kind: Secret
metadata:
  name: sloughgpt-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  DATABASE_PASSWORD: "$(echo -n 'password' | base64)"
  JWT_SECRET: "$(echo -n 'your-jwt-secret-key' | base64)"
  API_SECRET: "$(echo -n 'your-api-secret-key' | base64)"
EOF

echo "✅ Configurations applied"

# Deploy database
echo "🗄️ Deploying database..."

kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: sloughgpt
        - name: POSTGRES_USER
          value: sloughgpt
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sloughgpt-secrets
              key: DATABASE_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd
EOF

echo "✅ Database deployed"

# Deploy Redis cache
echo "🟥 Deploying Redis..."

kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
EOF

echo "✅ Redis deployed"

# Deploy main API
echo "🌐 Deploying SloughGPT API..."

kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sloughgpt-api
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sloughgpt-api
  template:
    metadata:
      labels:
        app: sloughgpt-api
    spec:
      containers:
      - name: api
        image: $REGISTRY/sloughgpt/api:$VERSION
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: REDIS_URL
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: sloughgpt-secrets
              key: JWT_SECRET
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: LOG_LEVEL
        - name: MODEL_CACHE_SIZE
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: MODEL_CACHE_SIZE
        - name: MAX_CONCURRENT_REQUESTS
          valueFrom:
            configMapKeyRef:
              name: sloughgpt-config
              key: MAX_CONCURRENT_REQUESTS
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sloughgpt-api
  namespace: $NAMESPACE
spec:
  selector:
    app: sloughgpt-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sloughgpt-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.sloughgpt.com
    secretName: sloughgpt-tls
  rules:
  - host: api.sloughgpt.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sloughgpt-api
            port:
              number: 80
EOF

echo "✅ API deployed"

# Deploy monitoring
echo "📊 Deploying monitoring..."

kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
        command:
        - '/bin/prometheus'
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=200h'
        - '--web.enable-lifecycle'
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-pvc
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: $NAMESPACE
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'sloughgpt'
        static_configs:
          - targets: ['sloughgpt-api:8000']
        metrics_path: /metrics/prometheus
        scrape_interval: 5s
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin123"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: $NAMESPACE
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
EOF

echo "✅ Monitoring deployed"

# Deploy Horizontal Pod Autoscaler
echo "📈 Configuring auto-scaling..."

kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sloughgpt-api-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sloughgpt-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      selectPolicy: Max
EOF

echo "✅ Auto-scaling configured"

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."

echo "   Waiting for database..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

echo "   Waiting for Redis..."
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

echo "   Waiting for API..."
kubectl wait --for=condition=available deployment/sloughgpt-api -n $NAMESPACE --timeout=300s

echo "   Waiting for monitoring..."
kubectl wait --for=condition=available deployment/prometheus -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=available deployment/grafana -n $NAMESPACE --timeout=300s

echo "✅ All deployments are ready!"

# Get external IPs
echo "🌐 Getting service endpoints..."

API_IP=$(kubectl get service sloughgpt-api -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
GRAFANA_IP=$(kubectl get service grafana -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
PROM_CLUSTER_IP=$(kubectl get service prometheus -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')

echo "🎉 Deployment Complete!"
echo "======================="
echo ""
echo "🌐 SloughGPT API: http://api.sloughgpt.com"
echo "📊 Grafana Dashboard: http://$GRAFANA_IP:3000"
echo "🔍 Prometheus (ClusterIP): http://$PROM_CLUSTER_IP:9090 — from your machine: kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090"
echo ""
echo "👤 Grafana Admin: admin / admin123"
echo ""
echo "🔍 API Health: http://api.sloughgpt.com/health"
echo "📈 API metrics (JSON): http://api.sloughgpt.com/metrics"
echo "📈 API metrics (Prometheus text): http://api.sloughgpt.com/metrics/prometheus"
echo ""

# Cleanup temporary resources
echo "🧹 Cleaning up..."

# Remove completed jobs (if any)
kubectl delete jobs --field-selector=status.successful=1 -n $NAMESPACE

echo "✅ Cleanup completed"

echo ""
echo "🔧 Post-deployment commands:"
echo "   # Check deployment status"
echo "   kubectl get pods -n $NAMESPACE"
echo ""
echo "   # View logs"
echo "   kubectl logs -f deployment/sloughgpt-api -n $NAMESPACE"
echo ""
echo "   # Scale deployment"
echo "   kubectl scale deployment sloughgpt-api --replicas=5 -n $NAMESPACE"
echo ""
echo "   # Update deployment"
echo "   kubectl set image deployment/sloughgpt-api sloughgpt-api=$REGISTRY/sloughgpt/api:new-version -n $NAMESPACE"
echo ""
echo "🎯 Production deployment successful!"