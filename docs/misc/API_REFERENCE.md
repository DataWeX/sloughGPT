# SloughGPT API Reference

Comprehensive API documentation for the SloughGPT AI system.

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health & Status](#health--status)
3. [Model Management](#model-management)
4. [Conversation Management](#conversation-management)
5. [Chat & Inference](#chat--inference)
6. [Dataset Management](#dataset-management)
7. [Training](#training)
8. [Multi-Modal API](#multi-modal-api)
9. [System Management](#system-management)
10. [WebSocket API](#websocket-api)

---

## Authentication

### JWT Authentication

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Generate JWT Token:**
```bash
POST /auth/token
Content-Type: application/json

{
  "username": "user",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "<jwt_token>",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Health & Status

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 12345.67,
  "timestamp": "2026-03-04T10:30:00Z",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "cache": "healthy",
    "model_service": "healthy"
  }
}
```

### System Metrics

```bash
GET /metrics
```

**Response:**
```json
{
  "cpu_percent": 45.2,
  "memory_percent": 67.8,
  "memory_used_mb": 8192.5,
  "memory_total_mb": 12288.0,
  "disk_percent": 23.4,
  "disk_used_gb": 45.6,
  "disk_total_gb": 200.0,
  "network_sent_mb": 1024.3,
  "network_recv_mb": 2048.7,
  "timestamp": "2026-03-04T10:30:00Z"
}
```

---

## Model Management

### List Models

```bash
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "id": "gpt-3.5-turbo",
      "name": "GPT-3.5 Turbo",
      "provider": "OpenAI",
      "status": "available",
      "description": "Fast, cost-effective language model",
      "context_length": 4096,
      "pricing": {
        "prompt": 0.0015,
        "completion": 0.002
      }
    },
    {
      "id": "nanogpt",
      "name": "NanoGPT",
      "provider": "Local",
      "status": "available",
      "description": "Custom trained GPT model",
      "context_length": 512,
      "pricing": {
        "prompt": 0.0,
        "completion": 0.0
      }
    }
  ]
}
```

### Get Model Details

```bash
GET /models/{model_id}
```

**Response:**
```json
{
  "id": "nanogpt",
  "name": "NanoGPT",
  "provider": "Local",
  "status": "available",
  "description": "Custom trained GPT model",
  "context_length": 512,
  "parameters": {
    "vocab_size": 500,
    "n_embed": 128,
    "n_layer": 3,
    "n_head": 4
  },
  "requirements": {
    "memory_mb": 256,
    "cpu_cores": 1,
    "gpu_memory_mb": 0
  }
}
```

---

## Conversation Management

### Create Conversation

```bash
POST /conversations
Content-Type: application/json

{
  "name": "Customer Support"
}
```

**Response:**
```json
{
  "id": "conv_abc123",
  "name": "Customer Support",
  "created_at": "2026-03-04T10:30:00Z",
  "updated_at": "2026-03-04T10:30:00Z",
  "message_count": 0
}
```

### List Conversations

```bash
GET /conversations
```

**Response:**
```json
{
  "conversations": [
    {
      "id": "conv_abc123",
      "name": "Customer Support",
      "created_at": "2026-03-04T10:30:00Z",
      "updated_at": "2026-03-04T10:30:00Z",
      "message_count": 5
    }
  ]
}
```

### Get Conversation

```bash
GET /conversations/{conversation_id}
```

**Response:**
```json
{
  "id": "conv_abc123",
  "messages": [
    {
      "role": "user",
      "content": "Hello, I need help with my order.",
      "timestamp": "2026-03-04T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help! Could you provide your order number?",
      "timestamp": "2026-03-04T10:30:01Z"
    }
  ],
  "metadata": {
    "name": "Customer Support",
    "created_at": "2026-03-04T10:30:00Z",
    "updated_at": "2026-03-04T10:30:01Z"
  }
}
```

---

## Chat & Inference

### Send Chat Message

```bash
POST /chat
Content-Type: application/json

{
  "message": "What's the weather like today?",
  "model": "gpt-3.5-turbo",
  "conversation_id": "conv_abc123",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "conversation_id": "conv_abc123",
  "message": {
    "role": "assistant",
    "content": "The weather today is sunny with a high of 75°F and low of 55°F. Perfect for outdoor activities!",
    "timestamp": "2026-03-04T10:30:02Z"
  },
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 25,
    "total_tokens": 33
  }
}
```

### Streaming Chat

```bash
POST /chat/stream
Content-Type: application/json
Accept: text/event-stream

{
  "message": "Write a poem about spring",
  "model": "nanogpt",
  "temperature": 0.8,
  "max_tokens": 50
}
```

**Response (streamed):**
```
data: {"type": "chunk", "content": "R", "accumulated": "R"}
data: {"type": "chunk", "content": "o", "accumulated": "Ro"}
data: {"type": "chunk", "content": "s", "accumulated": "Ros"}
...
data: {"type": "done", "content": "Roses bloom in spring's embrace...", "usage": {"prompt_tokens": 6, "completion_tokens": 45, "total_tokens": 51}}
```

### Generate Text

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "model": "nanogpt",
  "max_length": 100,
  "temperature": 0.8,
  "top_k": 40
}
```

**Response:**
```json
{
  "text": "Once upon a time in a faraway land, there lived a young princess who dreamed of adventure...",
  "model": "nanogpt",
  "tokens_generated": 85,
  "processing_time_ms": 125.4
}
```

---

## Dataset Management

### List Datasets

```bash
GET /datasets
```

**Response:**
```json
{
  "datasets": [
    {
      "name": "customer_reviews",
      "path": "/app/datasets/customer_reviews",
      "size": 1048576,
      "created_at": "2026-03-04T10:30:00Z",
      "updated_at": "2026-03-04T10:30:00Z",
      "has_train": true,
      "has_val": true,
      "has_meta": true,
      "description": "Customer review dataset for sentiment analysis"
    }
  ]
}
```

### Create Dataset

```bash
POST /datasets
Content-Type: application/json

{
  "name": "product_descriptions",
  "content": "Product A: High quality widget...\nProduct B: Premium gadget...",
  "description": "Product description dataset"
}
```

**Response:**
```json
{
  "name": "product_descriptions",
  "path": "/app/datasets/product_descriptions",
  "size": 1024,
  "created_at": "2026-03-04T10:30:00Z",
  "updated_at": "2026-03-04T10:30:00Z",
  "has_train": false,
  "has_val": false,
  "has_meta": true,
  "description": "Product description dataset"
}
```

### Get Dataset

```bash
GET /datasets/{name}
```

**Response:**
```json
{
  "name": "product_descriptions",
  "path": "/app/datasets/product_descriptions",
  "size": 1024,
  "created_at": "2026-03-04T10:30:00Z",
  "updated_at": "2026-03-04T10:30:00Z",
  "has_train": false,
  "has_val": false,
  "has_meta": true,
  "description": "Product description dataset"
}
```

---

## Training

### Start Training

```bash
POST /training
Content-Type: application/json

{
  "dataset_name": "customer_reviews",
  "model_id": "nanogpt",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.001,
  "vocab_size": 500,
  "n_embed": 128,
  "n_layer": 3,
  "n_head": 4
}
```

**Response:**
```json
{
  "id": "train_abc123",
  "status": "running",
  "dataset_name": "customer_reviews",
  "model_id": "nanogpt",
  "progress": 0.0,
  "current_epoch": 0,
  "total_epochs": 3,
  "loss": 0.0,
  "started_at": "2026-03-04T10:30:00Z",
  "completed_at": null,
  "error": null,
  "config": {
    "dataset_name": "customer_reviews",
    "model_id": "nanogpt",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.001,
    "vocab_size": 500,
    "n_embed": 128,
    "n_layer": 3,
    "n_head": 4
  }
}
```

### Get Training Job

```bash
GET /training/{job_id}
```

**Response:**
```json
{
  "id": "train_abc123",
  "status": "completed",
  "dataset_name": "customer_reviews",
  "model_id": "nanogpt",
  "progress": 100.0,
  "current_epoch": 3,
  "total_epochs": 3,
  "loss": 0.123,
  "started_at": "2026-03-04T10:30:00Z",
  "completed_at": "2026-03-04T10:35:00Z",
  "error": null,
  "metrics": {
    "final_loss": 0.123,
    "accuracy": 0.95,
    "training_time_seconds": 300,
    "samples_processed": 10000
  }
}
```

---

## Multi-Modal API

### Image Captioning

```bash
POST /multimodal/caption
Content-Type: multipart/form-data

{
  "image": (file),  // Image file
  "model": "vit-caption",
  "max_length": 30,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "caption": "A person riding a bicycle on a city street",
  "confidence": 0.92,
  "model": "vit-caption",
  "processing_time_ms": 150.2
}
```

### Image Classification

```bash
POST /multimodal/classify
Content-Type: multipart/form-data

{
  "image": (file),  // Image file
  "model": "clip-resnet",
  "categories": ["cat", "dog", "bird", "car"]
}
```

**Response:**
```json
{
  "classification": "cat",
  "confidence": 0.87,
  "probabilities": {
    "cat": 0.87,
    "dog": 0.08,
    "bird": 0.03,
    "car": 0.02
  },
  "model": "clip-resnet",
  "processing_time_ms": 200.5
}
```

### Visual Question Answering

```bash
POST /multimodal/vqa
Content-Type: multipart/form-data

{
  "image": (file),  // Image file
  "question": "What color is the car?",
  "model": "clip-vqa"
}
```

**Response:**
```json
{
  "answer": "The car is red",
  "confidence": 0.91,
  "model": "clip-vqa",
  "processing_time_ms": 180.3
}
```

---

## System Management

### System Info

```bash
GET /info
```

**Response:**
```json
{
  "python_version": "3.10.6",
  "platform": "darwin",
  "cpu_count": 8,
  "total_memory_gb": 32.0,
  "disk_total_gb": 500.0,
  "conversations_count": 15,
  "datasets_count": 8,
  "training_jobs_count": 3
}
```

### Statistics

```bash
GET /stats
```

**Response:**
```json
{
  "conversations": {
    "total": 15,
    "total_messages": 342
  },
  "datasets": {
    "total": 8,
    "total_size_bytes": 5242880
  },
  "training": {
    "total": 3,
    "completed": 2,
    "running": 1,
    "pending": 0
  },
  "cache": {
    "total_keys": 45,
    "expired_keys": 5,
    "valid_keys": 40
  },
  "system": {
    "cpu_percent": 25.3,
    "memory_percent": 45.6,
    "disk_percent": 12.8
  }
}
```

### Reset Data

```bash
POST /reset
```

**Response:**
```json
{
  "status": "reset",
  "message": "All conversations and training jobs cleared"
}
```

---

## WebSocket API

### Connect

```bash
GET /ws
Upgrade: websocket
Connection: Upgrade
```

### Message Types

#### Ping/Pong

```json
{
  "type": "ping",
  "timestamp": "2026-03-04T10:30:00Z"
}
```

**Response:**
```json
{
  "type": "pong",
  "timestamp": "2026-03-04T10:30:00Z"
}
```

#### Subscribe to Updates

```json
{
  "type": "subscribe",
  "channels": ["training", "metrics", "conversations"]
}
```

**Response:**
```json
{
  "type": "subscribed",
  "channels": ["training", "metrics", "conversations"]
}
```

#### Training Updates

```json
{
  "type": "training_update",
  "job_id": "train_abc123",
  "status": "running",
  "progress": 45.2,
  "current_epoch": 2,
  "total_epochs": 3,
  "loss": 0.156
}
```

#### System Metrics

```json
{
  "type": "metrics",
  "cpu_percent": 35.2,
  "memory_percent": 65.8,
  "gpu_memory_percent": 45.0,
  "timestamp": "2026-03-04T10:30:00Z"
}
```

---

## Error Handling

### Standard Error Format

```json
{
  "error": "ModelNotFoundError",
  "message": "Model 'nonexistent-model' not found",
  "code": 404,
  "timestamp": "2026-03-04T10:30:00Z"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid input data
- `401`: Unauthorized - Authentication required
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error
- `503`: Service Unavailable - Service temporarily unavailable

---

## Rate Limiting

**Default Limits:**
- 100 requests per minute per IP
- 1000 requests per hour per API key

**Headers Returned:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1709522400
```

---

## Pagination

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

**Response Format:**
```json
{
  "items": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total_items": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

---

## Versioning

**API Versioning:**
- Current version: `v2`
- URL format: `/api/v2/{endpoint}`
- Backward compatibility: Maintained for previous versions

---

## Support

For API support:
- Documentation: https://github.com/iamtowbee/sloughGPT#readme
- Issues: https://github.com/iamtowbee/sloughGPT/issues
- Email: dev@sloughgpt.ai (see **pyproject.toml** authors)

---

*Last updated: March 2026*
*API version: 2.0*
*Document version: 1.0*