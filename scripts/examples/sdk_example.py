#!/usr/bin/env python3
"""
SloughGPT SDK Example
Demonstrates usage of the SloughGPT Python SDK.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sloughgpt_sdk import (
    SloughGPTClient,
    ChatMessage,
    GenerateRequest,
)


def example_basic_generation(client):
    """Basic text generation example."""
    print("\n=== Basic Generation ===")
    
    result = client.generate(
        prompt="The capital of France is",
        max_new_tokens=20,
        temperature=0.7
    )
    print(f"Prompt: The capital of France is")
    print(f"Generated: {result.generated_text}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Time: {result.inference_time_ms:.2f}ms")


def example_chat(client):
    """Chat completion example."""
    print("\n=== Chat Completion ===")
    
    messages = [
        ChatMessage.system("You are a helpful AI assistant."),
        ChatMessage.user("What is machine learning?"),
    ]
    
    result = client.chat(messages)
    print(f"User: What is machine learning?")
    print(f"Assistant: {result.message.content}")


def example_batch(client):
    """Batch generation example."""
    print("\n=== Batch Generation ===")
    
    prompts = [
        "Hello, how are you?",
        "What is Python?",
        "Tell me a fact about space.",
    ]
    
    result = client.batch_generate(prompts, max_new_tokens=50)
    print(f"Processed {result.total_prompts} prompts")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    
    for i, r in enumerate(result.results):
        print(f"\n{i+1}. Prompt: {prompts[i][:30]}...")
        print(f"   Response: {r.generated_text[:50]}...")


def example_streaming(client):
    """Streaming generation example."""
    print("\n=== Streaming Generation ===")
    
    print("Generating: ", end="", flush=True)
    for token in client.generate_stream(
        prompt="Once upon a time in a distant galaxy",
        max_new_tokens=50,
        temperature=0.8
    ):
        print(token, end="", flush=True)
    print()


def example_models(client):
    """List and inspect models."""
    print("\n=== Available Models ===")
    
    models = client.list_models()
    for model in models[:5]:
        print(f"  - {model.id}: {model.source or 'unknown'}")


def example_datasets(client):
    """List available datasets."""
    print("\n=== Available Datasets ===")
    
    datasets = client.list_datasets()
    for ds in datasets[:5]:
        print(f"  - {ds.id}")


def example_metrics(client):
    """Check API metrics."""
    print("\n=== API Metrics ===")
    
    try:
        metrics = client.metrics()
        print(f"  Total Requests: {metrics.requests_total}")
        print(f"  Successful: {metrics.requests_success}")
        print(f"  Failed: {metrics.requests_failed}")
        print(f"  Cache Hits: {metrics.cache_hits}")
        print(f"  Cache Misses: {metrics.cache_misses}")
    except Exception as e:
        print(f"  Metrics not available: {e}")


def example_health(client):
    """Check API health."""
    print("\n=== Health Check ===")
    
    health = client.health()
    print(f"  Status: {health.status}")
    print(f"  Version: {health.version}")
    print(f"  Model Loaded: {health.model_loaded}")
    print(f"  Device: {health.device}")


def example_system_info(client):
    """Get system information."""
    print("\n=== System Info ===")
    
    info = client.info()
    print(f"  Version: {info.version}")
    print(f"  PyTorch: {info.pytorch_version}")
    print(f"  CUDA Available: {info.cuda_available}")
    if info.cuda:
        print(f"  GPU: {info.cuda.get('device', 'N/A')}")


def main():
    """Run all examples."""
    base_url = os.environ.get("SLOUGHGPT_API_URL", "http://localhost:8000")
    print(f"Connecting to: {base_url}")
    
    client = SloughGPTClient(base_url=base_url, timeout=60)
    
    try:
        example_health(client)
        example_system_info(client)
        example_basic_generation(client)
        example_chat(client)
        example_batch(client)
        example_streaming(client)
        example_models(client)
        example_datasets(client)
        example_metrics(client)
        
        print("\n=== All examples completed! ===")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the SloughGPT API server is running:")
        print("  uvicorn server.main:app --port 8000")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
