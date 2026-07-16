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


def example_souls(client):
    """List and switch souls."""
    print("\n=== Souls ===")
    try:
        souls = client.list_souls()
        for s in souls:
            print(f"  - {s.get('name')}: {s.get('description', '')[:60]}")
        current = client.get_current_soul()
        print(f"  Current: {current.get('name', 'default')}")
    except Exception as e:
        print(f"  Souls not available: {e}")


def example_knowledge(client):
    """List knowledge base."""
    print("\n=== Knowledge ===")
    try:
        items = client.list_knowledge()
        if items:
            for k in items[:3]:
                print(f"  - {k.get('content', '')[:60]}")
        else:
            print("  (empty)")
        topics = client.get_knowledge_topics()
        print(f"  Topics: {topics}")
    except Exception as e:
        print(f"  Knowledge not available: {e}")


def example_tokenizer(client):
    """Check tokenizer stats."""
    print("\n=== Tokenizer ===")
    try:
        stats = client.get_tokenizer_stats()
        print(f"  Vocab size: {stats.get('vocab_size', '?')}")
    except Exception as e:
        print(f"  Tokenizer not available: {e}")


def example_system(client):
    """Get system metrics."""
    print("\n=== System ===")
    try:
        metrics = client.get_system_metrics()
        print(f"  CPU: {metrics.get('cpu_percent', '?')}%")
        print(f"  Memory: {metrics.get('memory_percent', '?')}%")
        print(f"  Disk: {metrics.get('disk_percent', '?')}%")
        print(f"  Uptime: {metrics.get('uptime_seconds', '?')}s")
    except Exception as e:
        print(f"  System metrics not available: {e}")


def example_workflow(client):
    """Check workflow status."""
    print("\n=== Workflow ===")
    try:
        wf = client.get_workflow_status()
        print(f"  Status: {wf.get('status', '?')}")
        print(f"  Active: {wf.get('active', '?')}")
        print(f"  Feedback count: {wf.get('feedback_count', '?')}")
    except Exception as e:
        print(f"  Workflow not available: {e}")


def example_auto_train(client):
    """List auto-train checkpoints."""
    print("\n=== Auto-Train ===")
    try:
        ckpts = client.list_auto_train_checkpoints()
        if ckpts:
            for c in ckpts[:3]:
                print(f"  - {c.get('name', '?')}")
        else:
            print("  (no checkpoints)")
    except Exception as e:
        print(f"  Auto-train not available: {e}")


def main():
    """Run all examples."""
    base_url = os.environ.get("SLO_API_URL", "http://localhost:8000")
    print(f"Connecting to: {base_url}")

    client = SloughGPTClient(base_url=base_url, timeout=60)

    try:
        example_health(client)
        example_system_info(client)
        example_basic_generation(client)
        example_chat(client)
        example_streaming(client)
        example_models(client)
        example_datasets(client)
        example_metrics(client)
        example_souls(client)
        example_knowledge(client)
        example_tokenizer(client)
        example_system(client)
        example_workflow(client)
        example_auto_train(client)

        print("\n=== All examples completed! ===")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the SloughGPT API server is running:")
        print("  uvicorn server.main:app --port 8000")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
