# SloughGPT Developer Guide

## 🎯 Overview

This guide covers development practices, contribution guidelines, and technical details for working with the SloughGPT OOP monorepo architecture.

## 🏗️ Architecture Overview

### Domain-Driven Design

SloughGPT uses a domain-driven architecture where each domain represents a bounded context with its own:

- **Models**: Business logic and entities
- **Services**: Domain services and application logic  
- **Interfaces**: Contracts between domains
- **Infrastructure**: External dependencies and persistence

### Domain Relationships

```
┌─────────────────┐
│   UI Domain        │
├─────────────────┤  │
├─────────────────┤  │
│ Enterprise Core  │  │
├─────────────────┤  │
└─────────────────┘  │
   Integration    │
      Layer        │
└─────────────────┘  │
   Cognitive       │
      Domain        │
└─────────────────┘
    Infrastructure  │
       Domain       │
└─────────────────┘
      Shared         │
     Components     │
└─────────────────┘
```

### Key Conventions

- **Routers** in `apps/api/server/routers/` — one per domain, thin wrapper around domain logic
- **Domains** in `packages/core-py/domains/` — business logic, no framework imports
- **Controllers** on frontend (`apps/web/lib/*-controller.ts`) — axios-based API wrappers
- **No PyTorch in SloNet** — pure NumPy autograd for the custom training pipeline

## 🛠️ Development Setup

### Prerequisites

- **Python**: 3.9+ with type hints support
- **Git**: For version control
- **Docker**: For containerized development
- **Node.js**: 20 (match repo root **`.nvmrc`** for the web app)
- **Make**: For build automation

### Environment Setup

```bash
# Clone repository
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
python3 -m pip install -e ".[dev]"
```

### API + web (local)

Run the FastAPI app and the Next.js dev server in **one terminal** or two. Full options are in **QUICKSTART.md**.

```bash
# One terminal (API :8000 + web :3000)
./scripts/dev-stack.sh
# or: make dev-stack
# or: npm install && npm run dev:stack   # repo root; same shell script

# Contract tests for repo root package.json (after npm install at repo root)
npm run test:repo-root
# or: make test-repo-root
```

### Development Workflow

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
   - Follow code style and architecture patterns
   - Add comprehensive tests
   - Update documentation
   - Ensure type safety

3. **Run Tests**
```bash
pytest tests/ -q -k "your_keyword"
```

4. **Commit Changes**
```bash
git add .
git commit -m "feat: add your feature description"
```

5. **Create Pull Request**
   - Ensure the **CI/CD** GitHub Actions workflow passes
   - Request code review
   - Update documentation

## 🧪 Code Standards

### Python Code Style

We follow [PEP 8](https://pep8.org/) and project-specific conventions:

```python
# Import order
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Class definitions
class ExampleService(BaseService):
    """Example service following OOP principles."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize service with configuration."""
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return results."""
        # Implementation
        pass
```

### Type Hints

All public interfaces must have comprehensive type hints:

```python
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class CognitiveRequest:
    """Request for cognitive processing."""
    content: str
    context: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
```

### Documentation Strings

Use comprehensive docstrings for all public modules, classes, and functions:

```python
def process_memory(
    memory_data: Dict[str, Any],
    memory_type: str = "episodic",
    options: Optional[Dict[str, Any]] = None
) -> str:
    """Process memory data and return storage confirmation.

    Args:
        memory_data: Dictionary containing memory content and metadata
        memory_type: Type of memory (episodic, semantic, procedural, working)
        options: Optional configuration options

    Returns:
        Confirmation string with memory ID

    Raises:
        ValidationError: If input data is invalid
        StorageError: If storage operation fails

    Example:
        >>> result = process_memory(
        ...memory_data...,
        memory_type="episodic"
        options={"importance": 0.8}
        )
        >>> print(result)
        'Memory stored with ID: mem_123456'
    """
```

## 🧪 Testing

Tests are flat in `tests/` (no subdirectories). Run with:

```bash
# All tests
python3 -m pytest tests/ -q

# Skip slow tests
python3 -m pytest tests/ -m "not slow"

# A single test file
python3 -m pytest tests/test_knowledge_memory.py -v

# Frontend tests
cd apps/web && npx vitest run

# TypeScript type check
cd apps/web && npx tsc --noEmit
```

[tool.coverage.run]
source = ["domains"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/site-packages/*",
]
```

### Running Tests

```bash
# Run all tests (from repository root)
python3 -m pytest tests/ -q

# Run a subset
python3 -m pytest tests/test_api.py -v

# Run with coverage (domains live under packages/core-py)
python3 -m pytest tests/ --cov=domains --cov-report=html

# Run integration-focused tests
python3 -m pytest tests/test_integration.py -v
```

## 📊 Monitoring & Debugging

### Logging

Use structured logging with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

class CognitiveService:
    def __init__(self):
        self.logger = logging.getLogger(f"man.{self.__class__.__name__}")

    async def process_request(self, request):
        self.logger.info(f"Processing request: {request.id}")
        try:
            result = await self._do_process(request)
            self.logger.info(f"Request {request.id} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error processing request {request.id}: {e}")
            raise
```

### Metrics Collection

All domains support comprehensive metrics collection:

```python
from domains.shared.monitoring import MetricsCollector

# Track performance
metrics = MetricsCollector()

# Track custom metrics
await metrics.track_metric("cognitive_processing_time", processing_time)
await metrics.track_counter("api_requests_total")
```

### Debugging

#### Local Development
```python
# Enable debug mode
import os
os.environ['SLO_DEBUG'] = '1'

# Use Python debugger
import pdb; pdb.set_trace()

# Enable async debugging
import asyncio
asyncio.run(main())
```

#### Remote Debugging

```python
# Use debugpy for remote debugging
python3 -m debugpy --listen 5678 --wait-for-client your-app.py
```

## 🚀 Performance Optimization

### Async Patterns

Always use async/await for I/O operations:

```python
import asyncio
from typing import List

class OptimizedProcessor:
    async def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process items concurrently for better performance."""
        # Create tasks for all items
        tasks = [self._process_single(item) for item in items]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed tasks
        return [r for r in results if not isinstance(r, Exception)]

    async def _process_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        # Implementation
        pass
```

### Connection Pooling

Database and cache connections use pooling:

```python
from domains.infrastructure.database import DatabaseManager

# Initialize with connection pooling
db_manager = DatabaseManager(pool_size=20)
await db_manager.initialize()

# Connections are automatically pooled
for _ in range(100):
    result = await db_manager.execute_query("SELECT * FROM table")
```

### Caching Strategy

Implement multi-level caching:

```python
class CacheStrategy:
    async def get_data(self, key: str) -> Optional[Any]:
        # L1: In-memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]

        # L2: Redis cache
        if await self.redis_cache.exists(key):
            return await self.redis_cache.get(key)

        # L3: Database cache
        return await self.database_cache.get(key)

    async def set_data(self, key: str, value: Any, ttl: int = 3600):
        # Set in all cache levels with different TTLs
        await self.set_memory_cache(key, value, ttl=60)
        await self.set_redis_cache(key, value, ttl=300)
        await self.set_database_cache(key, value, ttl=86400)
```

## 🔒 Security Best Practices

### Input Validation

Always validate and sanitize input:

```python
from pydantic import BaseModel, validator
from typing import Optional

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserService:
    async def create_user(self, request: UserCreateRequest) -> User:
        # Pydantic handles validation automatically
        validated_data = UserCreateRequest(**request.dict())
        return await self._create_user(validated_data)
```

### Security Headers

Implement comprehensive security headers:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trusted-origin.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    max_age=3600
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    return response
```

### Authentication

Implement JWT-based authentication:

```python
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class JWTManager:
    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            return {}
```

## 📦 Deployment

### Development Deployment

```bash
# API (FastAPI; default port 8000)
python3 apps/api/server/main.py
# or, with reload:
# cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000

# Web UI (separate terminal; port 3000)
cd apps/web && npm install && npm run dev
```

### Production Deployment

```bash
# Docker Compose (from repository root)
docker compose -f infra/docker/docker-compose.yml up -d api

# Kubernetes (raw manifests under infra/k8s/k8s/)
kubectl apply -f infra/k8s/k8s/

# Health check
curl http://localhost:8000/health
```

### Environment Configuration

Use environment-specific configurations:

```python
# config/development.py
DEBUG = True
DATABASE_URL = "postgresql://localhost/sloughgpt_dev"
REDIS_URL = "redis://localhost:6379"
LOG_LEVEL = "DEBUG"

# config/production.py
DEBUG = False
DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL")
LOG_LEVEL = "INFO"
```

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health
curl http://localhost:8000/health/detailed
```

## 📚 Documentation

Docs live flat in `docs/`. Auto-generated API docs at `http://localhost:8000/docs` (OpenAPI/Swagger).

### Writing Documentation

1. Module docstring at top of every `.py` file
2. `Args`/`Returns`/`Side effects` on every public function
3. Update `AGENTS.md` if changing development workflows
4. Update `docs/routers.md` if adding/removing API endpoints

## 🤝 Contributing

### Contribution Process

1. **Fork and Clone**: Fork the repository and clone locally
2. **Create Branch**: Create a feature branch from main
3. **Develop**: Make your changes following our standards
4. **Test**: Add comprehensive tests for new features
5. **Document**: Update documentation as needed
6. **Submit**: Create a pull request with clear description

### Pull Request Guidelines

- **Clear Title**: Summarize the change concisely
- **Detailed Description**: Explain what and why
- **Testing**: Ensure all tests pass
- **Documentation**: Update relevant docs
- **Screenshots**: Include UI changes if applicable
- **Breaking Changes**: Clearly document any breaking changes

### Code Review Process

- **Automated Checks**: The CI/CD workflow runs automated tests and linting
- **Peer Review**: At least one maintainer must review
- **Security Review**: Security-focused review for sensitive changes
- **Architecture Review**: Ensure consistency with domain architecture
- **Performance Review**: Consider performance implications

### Community Guidelines

- **Be Respectful**: Treat all contributors with respect
- **Be Constructive**: Focus on what works best for the project
- **Be Collaborative**: Welcome contributions from all community members
- **Be Patient**: Understand that maintainers have limited time
- **Be Inclusive**: Create a welcoming environment for all

---

## 🚀 Quick Start Development

### Initial Setup

```bash
# Clone and set up
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your settings
```

### Running Tests

```bash
# Full suite
python3 -m pytest tests/ -q

# Skip slow tests
python3 -m pytest tests/ -m "not slow"

# Frontend
cd apps/web && npx vitest run && npx tsc --noEmit
```

### Development servers

```bash
# API (http://localhost:8000 — OpenAPI at /docs)
python3 apps/api/server/main.py

# Web UI (http://localhost:3000)
cd apps/web && npm run dev
```

---

**Happy coding! 🎉**
