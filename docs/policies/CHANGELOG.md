# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-19

### Added

#### Core Features
- **Local Model Training** - Train NanoGPT from scratch or fine-tune HuggingFace models
- **Production Inference** - High-performance inference engine with streaming
- **Quantization** - FP16, INT8, INT4 support for memory-efficient inference
- **Experiment Tracking** - MLflow-style metrics and parameter logging
- **Model Export** - Export to Torch, ONNX, SafeTensors, .sou formats
- **Benchmarking** - Performance metrics and model comparison
- **CLI Interface** - Train, generate, benchmark commands

#### Industry Standard Optimizations
- Mixed Precision (FP16/BF16) - 2-3x speedup
- Gradient Checkpointing - 50% memory savings
- torch.compile - 1.5-2x speedup
- Flash Attention (NVIDIA/AMD) - 2-4x speedup
- Training presets for auto, high-end GPU, mid-range GPU, Apple Silicon, CPU

#### API Security
- JWT Authentication with HS256 signing
- API Key validation with hash comparison
- Rate limiting (60 requests/minute)
- Input validation and sanitization (XSS prevention)
- Audit logging for all security events
- Security headers (HSTS, CSP, X-Frame-Options)
- WebSocket authentication

#### API Endpoints
- **Health**: `/health`, `/health/live`, `/health/ready`, `/health/detailed`
- **Auth**: `/auth/token`, `/auth/verify`, `/auth/refresh`
- **Rate Limiting**: `/rate-limit/status`, `/rate-limit/check`
- **Cache**: `/cache/stats`, `/cache` (DELETE)
- **Metrics**: `/metrics`, `/metrics/prometheus`
- **Security**: `/security/audit`, `/security/keys`
- **Inference**: `/inference/generate`, `/inference/generate/stream`, `/inference/batch`
- **Training**: `/train`, `/training/jobs`, `/training/start`
- **Benchmark**: `/benchmark/run`, `/benchmark/compare`
- **Models**: `/models`, `/models/load`, `/model/export`

#### Infrastructure
- **Kubernetes**: Deployment, Service, Ingress, HPA, PVC, RBAC, NetworkPolicy
- **Helm Chart**: Production-ready deployment with configurable values
- **Docker Compose**: API, GPU, Dev, Model Server, Prometheus, Grafana, Redis
- **Monitoring**: Prometheus ServiceMonitor, Grafana Dashboard, Alerting Rules

### Changed
- Updated API documentation with authentication examples
- Enhanced error handling with custom exception handlers
- Improved streaming responses with proper SSE format
- Updated health probes for Kubernetes readiness/liveness

### Fixed
- macOS MPS hanging issues with `DYLD_INSERT_LIBRARIES=""` fix
- CI/CD pipeline errors with proper dependencies
- Setup script for macOS compatibility

## [0.9.0] - 2026-03-18

### Added
- Initial project structure with domain-based architecture
- Basic inference engine
- Training pipeline
- Quantization support
- Web UI with Next.js

## [0.8.0] - 2026-03-17

### Added
- CLI commands (train, generate, benchmark)
- HuggingFace model support
- LoRA fine-tuning support
- ONNX export

## [0.7.0] - 2026-03-16

### Added
- Basic API server with FastAPI
- Streaming endpoints
- WebSocket support
- Docker support

## [0.6.0] - 2026-03-15

### Added
- NanoGPT implementation
- Shakespeare dataset training
- Basic benchmarking
- Experiment tracking

## [0.5.0] - 2026-03-14

### Added
- Initial project setup
- Basic project structure
- README and documentation

## [0.1.0] - 2024-03-18
- Converted repo to monorepo layout
- Added wrappers and symlinks for legacy paths
- Added basic CI, linting, and tests
