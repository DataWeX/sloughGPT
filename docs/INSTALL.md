# SloughGPT Installation Script

**Canonical repository:** [github.com/iamtowbee/sloughGPT](https://github.com/iamtowbee/sloughGPT). For a maintained local setup, start with **QUICKSTART.md** (`python3 -m pip install -e ".[dev]"` from the repo root, then `./verify.sh`). With **Node** installed, optional repo-root **`npm install`** enables **`npm run dev:stack`** (API + web) and **`npm run test:repo-root`** (root **`package.json`** contract); **`make dev-stack`** / **`make test-repo-root`** are equivalent shell entrypoints.

## 🚀 Automated Installation

This script provides a quick way to set up SloughGPT and resolve common dependency issues.

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/iamtowbee/sloughGPT/main/install.sh | bash
```

## Manual Installation Steps

### 1. Fix NumPy Compatibility

```bash
# Fix NumPy 2.x compatibility issue
python3 -m pip install "numpy<2.0"
python3 -m pip install --force-reinstall torch torchvision torchaudio
```

### 2. Install Core Dependencies

```bash
# Install without torch first to avoid conflicts
python3 -m pip install fastapi uvicorn sqlalchemy alembic redis
python3 -m pip install pydantic "python-jose[cryptography]" "passlib[bcrypt]"
python3 -m pip install python-multipart psutil prometheus-client
python3 -m pip install aiosmtplib pytest pytest-asyncio

# Then install torch with compatibility
python3 -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### 3. Install Additional Dependencies

```bash
# Install remaining dependencies
python3 -m pip install scikit-learn pandas numpy matplotlib seaborn
python3 -m pip install jupyter ipykernel notebook
python3 -m pip install plotly dash streamlit
python3 -m pip install aiofiles aiohttp websockets
```

### 4. Verify Installation

```bash
# After: python3 -m pip install -e .  (from the cloned repo root)
python3 -c "
import sys
try:
    import domains
    print('✅ domains package import OK')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

# CLI entrypoint (console script from pyproject.toml)
sloughgpt --help
```

## Docker Installation (Recommended)

### Using Pre-built Docker Image

```bash
# Image tag/registry may vary; Helm default is ghcr.io/iamtowbee/sloughgpt (see infra/k8s/helm/sloughgpt/values.yaml)
docker pull ghcr.io/iamtowbee/sloughgpt:latest

# Run with environment variables
docker run -d \
  --name sloughgpt-api \
  -p 8000:8000 \
  -e DATABASE_URL=sqlite:///sloughgpt.db \
  -e JWT_SECRET_KEY=your-secret-key \
  ghcr.io/iamtowbee/sloughgpt:latest
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT

# Build Docker image (compose/Dockerfiles live under infra/docker/)
docker build -f infra/docker/Dockerfile -t sloughgpt:local .

# Run with custom configuration
docker run -d \
  --name sloughgpt-local \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  sloughgpt:local
```

## Kubernetes Installation

### Helm Chart

```bash
# Chart source and values: infra/k8s/helm/sloughgpt/ (see README there)
helm repo add sloughgpt https://iamtowbee.github.io/sloughGPT
helm repo update

# Install with default values
helm install sloughgpt sloughgpt/sloughgpt --namespace sloughgpt --create-namespace

# Install with custom values
helm install sloughgpt sloughgpt/sloughgpt -f values.yaml --namespace sloughgpt --create-namespace
```

### Kubernetes Manifests

```bash
# Apply all manifests (from repository root)
kubectl apply -f infra/k8s/k8s/

# Check deployment status
kubectl get pods -n sloughgpt
kubectl get services -n sloughgpt
```

## Environment Configuration

### Required Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=sqlite:///sloughgpt.db

# Security
JWT_SECRET_KEY=your-256-bit-secret-key-here
BCRYPT_ROUNDS=12

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
# Web UI (local dev): Next.js in apps/web — typically http://localhost:3000

# Logging
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true

# Cost Management
DEFAULT_MONTHLY_BUDGET=1000
COST_ALERT_THRESHOLD=0.8
EOF
```

### Production Environment

```bash
# Production settings
DATABASE_URL=postgresql://user:password@localhost:5432/sloughgpt
REDIS_URL=redis://localhost:6379

# Strong security
JWT_SECRET_KEY=$(openssl rand -hex 32)
BCRYPT_ROUNDS=14

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/cert.pem
SSL_KEY_PATH=/etc/ssl/key.pem

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error

**Problem**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2`

**Solution**:
```bash
python3 -m pip install "numpy<2.0"
python3 -m pip install --force-reinstall torch torchvision torchaudio
```

#### 2. Import Error for get_database_manager

**Problem**: `cannot import name 'get_database_manager'`

**Solution**: The function is missing from core module, but this is expected during setup.

#### 3. Torch Not Found

**Problem**: `No module named 'torch'`

**Solution**:
```bash
# Install with specific version
python3 -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Or use conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 4. Permission Denied

**Problem**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x install.sh verify.sh run.sh
# Optional: repo-root CLI wrapper
chmod +x cli.py
```

### Verification Steps

```bash
# 1. Check Python version
python3 --version  # Should be 3.9+

# 2. Check critical imports
python3 -c "
try:
    import fastapi, torch, sqlalchemy, redis
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"

# 3. Test basic functionality (after python3 -m pip install -e . from repo root)
python3 -c "
try:
    import domains
    print('✅ domains package OK')
except Exception as e:
    print(f'❌ Import error: {e}')
"

# 4. Check ports (API vs Next.js dev server)
lsof -i :8000  # Should be free before starting API
lsof -i :3000  # Should be free before npm run dev (web)
```

## Platform-Specific Instructions

### macOS

```bash
# Install with Homebrew
brew install python@3.9

# Install with pip
python3.9 -m pip install --upgrade pip
python3.9 -m pip install "numpy<2.0"
python3.9 -m pip install torch torchvision torchaudio
```

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.9 python3.9-dev
sudo apt-get install build-essential

# Install SloughGPT
python3.9 -m pip install sloughgpt
```

### Windows

```bash
# Use PowerShell (use `py -3 -m pip` if `python3` is not on PATH)
python3 -m pip install "numpy<2.0"
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or use conda
conda create -n sloughgpt python=3.9
conda activate sloughgpt
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Next Steps

After successful installation:

1. **Start API server**: `python3 apps/api/server/main.py` (or `cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000`)
2. **Web UI (dev)**: `cd apps/web && npm run dev`
3. **Check health**: `curl -s http://localhost:8000/health`
4. **API docs**: http://localhost:8000/docs

## Support

- **Documentation**: https://docs.sloughgpt.ai
- **Issues**: https://github.com/iamtowbee/sloughGPT/issues
- **Community**: https://community.sloughgpt.ai
- **Enterprise Support**: enterprise@sloughgpt.ai

---

**🚀 SloughGPT Enterprise AI Framework - Installation Complete!**