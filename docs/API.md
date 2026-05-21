# SloughGPT API Documentation

## Overview

The SloughGPT API provides comprehensive RESTful endpoints for interacting with the AI system's cognitive capabilities, infrastructure services, and enterprise features.

## 🏛️ Architecture

### API Structure

```
/api/
├── health/              # Health check and system status
├── models/               # Available AI models and configurations
├── status/               # System and service status
├── docs/                  # API documentation (this page)
├── chat/                 # Chat and conversation management
├── conversations/        # Conversation history and management
├── users/                # User management endpoints
├── cognitive/            # Cognitive architecture endpoints
├── infrastructure/         # Infrastructure management
├── enterprise/            # Enterprise features (cost, monitoring)
└── integration/            # Cross-domain integration
```

### Core Endpoints

### Health Check
```http
GET /api/health
```

Returns system health status including:
- Overall system status
- Individual domain health
- Database connectivity
- Service dependencies
- Performance metrics

### Models
```http
GET /api/models
```

Returns available AI models:
- Model configurations
- Capabilities and features
- Performance characteristics
- Usage recommendations

### Status
```http
GET /api/status
```

Returns comprehensive system status:
- Active domains and services
- Resource utilization
- Recent performance metrics
- Error rates and issues

## 🧠 Cognitive Architecture API

### Memory Management
```http
POST /api/cognitive/memory/store
GET  /api/cognitive/memory/retrieve/{id}
GET  /api/cognitive/memory/search
DELETE /api/cognitive/memory/{id}
```

Memory operations for:
- Storing episodic, semantic, and procedural memories
- Retrieving memories by ID or content
- Searching memories with filters
- Memory consolidation management

### Reasoning Engine
```http
POST /api/cognitive/reasoning/reason
GET /api/cognitive/reasoning/strategies
GET /api/cognitive/reasoning/path/{id}
```

Reasoning operations:
- Executing different reasoning strategies
- Getting reasoning paths and explanations
- Strategy configuration and optimization
- Confidence assessment and uncertainty quantification

### Metacognitive Monitoring
```http
GET /api/cognitive/metacognition/status
POST /api/cognitive/metacognition/trigger-reflection
GET /api/cognitive/metacognition/insights
```

Metacognitive operations:
- Real-time cognitive process monitoring
- Reflection triggering and management
- Cognitive state assessment
- Performance optimization recommendations

## 🔧 Infrastructure API

### Database Management
```http
GET /api/infrastructure/database/status
POST /api/infrastructure/database/query
GET /api/infrastructure/database/stats
```

Database operations:
- Multi-database health monitoring
- Query execution with result formatting
- Performance metrics and optimization
- Connection pool management

### Caching System
```http
POST /api/infrastructure/cache/{key}
GET /api/infrastructure/cache/{key}
DELETE /api/infrastructure/cache/{key}
GET /api/infrastructure/cache/stats
```

Cache operations:
- Multi-level cache management (memory, Redis, disk)
- TTL-based expiration
- LRU eviction and cache warming
- Performance monitoring

### Deployment Management
```http
POST /api/infrastructure/deployment/deploy
GET /api/infrastructure/deployment/status/{id}
POST /api/infrastructure/deployment/rollback/{id}
GET /api/infrastructure/deployment/history
```

Deployment operations:
- Multi-environment deployment (dev, staging, prod)
- Health checks and rollback capabilities
- Blue-green and canary deployments
- Deployment history and analytics

## 🏢 Enterprise API

### Authentication & Security
```http
POST /api/enterprise/auth/login
POST /api/enterprise/auth/logout
POST /api/enterprise/auth/register
POST /api/enterprise/auth/refresh
GET /api/enterprise/auth/profile
POST /api/enterprise/auth/change-password
POST /api/enterprise/auth/forgot-password
```

Authentication features:
- JWT-based authentication with refresh tokens
- Multi-factor authentication support
- Session management with timeout
- Password policies and security validation
- Account recovery and security events

### User Management
```http
GET  /api/enterprise/users/
POST /api/enterprise/users/
GET  /api/enterprise/users/{id}
PUT  /api/enterprise/users/{id}
DELETE /api/enterprise/users/{id}
GET  /api/enterprise/users/{id}/permissions
POST /api/enterprise/users/{id}/permissions
```

User management operations:
- User CRUD operations with RBAC
- Role and permission management
- Bulk operations and imports
- User analytics and engagement metrics
- Profile and preference management

### Cost Optimization
```http
GET  /api/enterprise/cost/current
GET  /api/enterprise/cost/report
GET  /api/enterprise/cost/forecast
POST /api/enterprise/coptimize/optimize
GET  /api/enterprise/cost/thresholds
```

Cost optimization features:
- Real-time cost tracking per operation
- Resource usage monitoring and analysis
- Budget management with configurable thresholds
- Cost forecasting and trend analysis
- Optimization recommendations and alerts

## 💬 Chat & Communication API

### Chat Interface
```http
POST /api/chat/send
GET  /api/chat/conversations/{id}
GET  /api/chat/conversations
POST /api/chat/conversations
DELETE /api/chat/conversations/{id}
GET  /api/chat/history/{user_id}
POST /api/chat/clear/{user_id}
```

Chat operations:
- Real-time messaging with AI responses
- Conversation management with persistence
- Multi-user chat support
- Message history and search
- Typing indicators and read receipts

### Conversation Management
```http
GET  /api/conversations/
POST  /api/conversations/
GET  /api/conversations/{id}
PUT  /api/conversations/{id}
DELETE /api/conversations/{id}
GET  /api/conversations/{id}/messages
POST  /api/conversations/{id}/messages
```

Conversation features:
- Conversation lifecycle management
- Message threading and context
- Sharing and collaboration
- Conversation search and filtering
- Export and archive capabilities

## 🔗 Integration API

### Event Bus
```http
POST /api/integration/events/publish
POST /api/integration/events/subscribe
DELETE /api/integration/events/unsubscribe/{id}
GET /api/integration/events/history
```

Event system features:
- Publish-subscribe event pattern
- Event filtering and routing
- Event persistence and replay
- Dead letter queue handling
- Event schema validation

### Service Registry
```http
POST /api/integration/services/register
DELETE /api/integration/services/{id}
GET /api/integration/services/discover
GET /api/integration/services/health/{id}
GET /api/integration/services/load-balancer/{type}
```

Service discovery capabilities:
- Dynamic service registration and discovery
- Health monitoring and load balancing
- Service lifecycle management
- Automatic failover and recovery
- Service mesh integration

## 📊 Monitoring API

### Metrics Collection
```http
GET  /api/monitoring/metrics
POST /api/monitoring/metrics/track
GET  /api/monitoring/metrics/series
```

Monitoring features:
- System and application metrics
- Custom metric collection and tracking
- Time series data storage and analysis
- Real-time metric dashboards
- Configurable alerts and notifications

### Health Monitoring
```http
GET  /api/monitoring/health/system
GET  /api/monitoring/health/domains
GET  /api/monitoring/health/services
POST /api/monitoring/health/check
```

Health monitoring includes:
- System resource monitoring
- Domain-specific health checks
- Service dependency monitoring
- Proactive health assessment
- Automated issue detection

## 📝 Usage Examples

### Authentication
```bash
# Login
curl -X POST http://localhost:8000/api/enterprise/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Response
{
  "access_token": "eyJ0eXA...",
  "refresh_token": "eyJ0eXA...",
  "user": {...},
  "expires_in": 3600
}
```

### Chat with AI
```bash
# Send message
curl -X POST http://localhost:8000/api/chat/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, SloughGPT!"}'

# Response
{
  "message_id": "msg_123456",
  "response": "Hello! I'm SloughGPT, your AI assistant...",
  "timestamp": "2024-01-01T12:00:00Z",
  "conversation_id": "conv_789"
}
```

### Cognitive Operations
```bash
# Store memory
curl -X POST http://localhost:8000/api/cognitive/memory/store \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Important information to remember",
    "memory_type": "episodic",
    "importance": 0.8
  }'

# Reason
curl -X POST http://localhost:8000/api/cognitive/reasoning/reason \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "All humans are mortal",
    "strategy": "deductive",
    "context": {"domain": "philosophy"}
  }'
```

## 🚨 Authentication

All API endpoints require authentication except for:
- `/api/health` - System health check
- `/api/docs` - API documentation
- `/api/models` - Public model information

### JWT Token Authentication

Include the JWT token in the Authorization header:
```bash
Authorization: Bearer <your_jwt_token>
```

### Rate Limiting

API requests are rate-limited to prevent abuse:
- Default: 100 requests per minute
- Configurable per-endpoint limits
- Automatic rate limit headers in responses
- Exponential backoff for exceeded limits

## 📋 Response Formats

### Success Response
```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_abc123"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {...}
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_abc123"
}
```

### Validation Errors
- **400 Bad Request**: Invalid request format or parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource does not exist
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side error

## 🔧 SDKs and Libraries

### Python SDK
```python
from sloughgpt_sdk import SloughGPTClient

# Initialize client (package: packages/sdk-py/sloughgpt_sdk)
client = SloughGPTClient(base_url="http://localhost:8000", api_key="your-api-key")

# Use client methods per SDK README (chat, inference, etc.)
```

### JavaScript / TypeScript SDK
```javascript
import { SloughGPTClient } from '@sloughgpt/typescript-sdk';

// Initialize client (package: packages/sdk-ts/typescript-sdk)
const client = new SloughGPTClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key',
});

// Use client methods per SDK README
```

## 🚀 WebSocket Support

### Chat WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

// Authenticate
ws.send(JSON.stringify({
    type: 'authenticate',
    token: 'your-jwt-token'
}));

// Send message
ws.send(JSON.stringify({
    type: 'message',
    content: 'Hello via WebSocket!'
}));

// Real-time responses
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 📚 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Interactive**: http://localhost:8000/docs (Swagger UI)
- **JSON**: http://localhost:8000/api/openapi.json
- **YAML**: http://localhost:8000/api/openapi.yaml

## 🧪 Testing the API

### Running Tests
```bash
# From repository root (dev extras: pytest, pytest-cov, …)
python3 -m pip install -e ".[dev]"

# API-related tests live under tests/
python3 -m pytest tests/test_api.py tests/server/ -q

# With coverage
python3 -m pytest tests/ --cov=domains --cov-report=html
```

### Test Coverage
- Unit Tests: Individual endpoint testing
- Integration Tests: Cross-domain integration testing
- End-to-End Tests: Complete workflow testing
- Performance Tests: Load and stress testing
- Security Tests: Vulnerability assessment

## 📞 Support

### Getting Help
- **Documentation**: [API Documentation](https://docs.sloughgpt.ai/api)
- **Issues**: [GitHub Issues](https://github.com/iamtowbee/sloughGPT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/iamtowbee/sloughGPT/discussions)
- **Community**: [SloughGPT Community](https://community.sloughgpt.ai)

### Reporting Issues
When reporting API issues, please include:
- HTTP method and endpoint
- Request headers and body
- Response status and headers
- Error messages and stack traces
- Steps to reproduce the issue
- Expected vs actual behavior

---

## 🔗 Explore the API

Start exploring the SloughGPT API today and unlock powerful AI capabilities through our comprehensive REST interface!