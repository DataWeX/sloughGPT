#!/usr/bin/env python3
"""
Generate API documentation from the FastAPI server
"""

import sys
sys.path.insert(0, "..")

from pathlib import Path

def main():
    print("=" * 70)
    print("SloughGPT API Documentation")
    print("=" * 70)
    
    # Base URL for examples
    base = "http://localhost:8000"
    
    print("\n📍 BASE URL")
    print("-" * 70)
    print(f"  Development: {base}")
    print(f"  Production: https://api.sloughgpt.example.com")
    
    print("\n\n📍 HEALTH ENDPOINTS")
    print("-" * 70)
    endpoints = [
        ("GET", "/health", "Basic health check"),
        ("GET", "/health/live", "Kubernetes liveness probe"),
        ("GET", "/health/ready", "Kubernetes readiness probe"),
        ("GET", "/health/detailed", "Detailed health with system info"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 AUTHENTICATION ENDPOINTS")
    print("-" * 70)
    endpoints = [
        ("POST", "/auth/token", "Create JWT token from API key"),
        ("POST", "/auth/verify", "Verify JWT token"),
        ("POST", "/auth/refresh", "Refresh JWT token"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 INFERENCE ENDPOINTS")
    print("-" * 70)
    endpoints = [
        ("POST", "/generate", "Generate text (local model)"),
        ("POST", "/generate/stream", "Streaming text generation"),
        ("POST", "/inference/generate", "Generate using inference engine"),
        ("POST", "/inference/generate/stream", "Streaming inference"),
        ("POST", "/inference/batch", "Batch processing (up to 50 prompts)"),
        ("GET",  "/inference/stats", "Inference engine statistics"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 RATE LIMITING & CACHE")
    print("-" * 70)
    endpoints = [
        ("GET",  "/rate-limit/status", "Rate limit configuration"),
        ("GET",  "/rate-limit/check", "Check current usage"),
        ("GET",  "/cache/stats", "Cache statistics"),
        ("DELETE", "/cache", "Clear cache"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:8} {path:30} - {desc}")
    
    print("\n\n📍 METRICS & MONITORING")
    print("-" * 70)
    endpoints = [
        ("GET", "/metrics", "JSON metrics"),
        ("GET", "/metrics/prometheus", "Prometheus format metrics"),
        ("GET", "/security/audit", "Audit logs (requires auth)"),
        ("GET", "/security/keys", "Security configuration"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 TRAINING ENDPOINTS")
    print("-" * 70)
    endpoints = [
        ("POST", "/train", "Start training job"),
        ("GET",  "/training/jobs", "List training jobs"),
        ("GET",  "/training/jobs/{id}", "Get job status"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 MODELS ENDPOINTS")
    print("-" * 70)
    endpoints = [
        ("GET",  "/models", "List available models"),
        ("POST", "/models/load", "Load a model"),
        ("GET",  "/model/export/formats", "Export formats"),
        ("POST", "/model/export", "Export model"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:30} - {desc}")
    
    print("\n\n📍 EXAMPLE REQUESTS")
    print("-" * 70)
    examples = [
        ("Health check", f"curl {base}/health"),
        ("Generate text", f"curl -X POST {base}/generate -H 'Content-Type: application/json' -d '{{\"prompt\": \"Hello\", \"max_new_tokens\": 50}}'"),
        ("Batch process", f"curl -X POST {base}/inference/batch -H 'Content-Type: application/json' -d '{{\"prompts\": [\"Hello\", \"Hi\"], \"max_new_tokens\": 20}}'"),
        ("Get metrics", f"curl {base}/metrics/prometheus"),
    ]
    for title, cmd in examples:
        print(f"\n  {title}:")
        print(f"    {cmd}")
    
    print("\n\n📍 ENVIRONMENT VARIABLES")
    print("-" * 70)
    env_vars = [
        ("SLOUGHGPT_API_KEY", "API key for authentication"),
        ("SLOUGHGPT_JWT_SECRET", "Secret for JWT signing"),
        ("SLOUGHGPT_ENV", "Environment (development/production)"),
        ("RATE_LIMIT_REQUESTS_PER_MINUTE", "Rate limit (default: 60)"),
    ]
    for var, desc in env_vars:
        print(f"  {var:40} - {desc}")
    
    print("\n" + "=" * 70)
    print("For full API docs, visit: http://localhost:8000/docs")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
