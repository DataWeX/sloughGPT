#!/usr/bin/env python3
"""
SloughGPT SDK CLI
Command-line interface for the SloughGPT SDK.

Usage:
    sloughgpt-cli --help
    sloughgpt-cli health
    sloughgpt-cli generate "Hello, how are you?"   # or: sloughgpt-cli gen "..."
    sloughgpt-cli chat "What is Python?"
    sloughgpt-cli models
"""

import argparse
import json
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sloughgpt_sdk import SloughGPTClient, ChatMessage
except ImportError:
    print("Error: sloughgpt-sdk not found. Install with: pip install sloughgpt-sdk")
    sys.exit(1)


def format_json(data) -> str:
    """Format data as colored JSON."""
    return json.dumps(data, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="SloughGPT SDK CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url", "-u",
        default=os.environ.get("SLO_API_URL", "http://localhost:8000"),
        help="API base URL"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("SLO_API_KEY"),
        help="API key for authentication"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt2",
        help="Default model to use"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-T",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    health = subparsers.add_parser("health", help="Check API health")
    info = subparsers.add_parser("info", help="Get system info")

    gen = subparsers.add_parser(
        "generate",
        aliases=["gen"],
        help='Generate text via the API (alias: "gen")',
    )
    gen.add_argument("prompt", help="Prompt for generation")
    gen.add_argument("--stream", "-s", action="store_true", help="Stream output")

    chat = subparsers.add_parser("chat", help="Chat completion")
    chat.add_argument("message", help="User message")
    chat.add_argument("--system", help="System prompt")

    models = subparsers.add_parser("models", help="List available models")
    models.add_argument("--hf", action="store_true", help="Include HuggingFace models")

    datasets = subparsers.add_parser("datasets", help="List available datasets")

    metrics = subparsers.add_parser("metrics", help="Get API metrics")
    metrics.add_argument("--prometheus", action="store_true", help="Prometheus format")

    registry = subparsers.add_parser("registry", help="Model registry (live server)")
    reg_subparsers = registry.add_subparsers(dest="reg_action", help="Registry actions")

    reg_list = reg_subparsers.add_parser("list", help="List registered models")

    reg_info = reg_subparsers.add_parser("info", help="Get model info")
    reg_info.add_argument("model_id", help="Model ID")

    reg_best = reg_subparsers.add_parser("best", help="Get best model")

    reg_stats = reg_subparsers.add_parser("stats", help="Get registry statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        client = SloughGPTClient(
            base_url=args.url,
            api_key=args.api_key,
            timeout=60,
        )

        if args.command == "health":
            result = client.health()
            if args.json:
                print(format_json(result.raw))
            else:
                print(f"Status: {result.status}")
                print(f"Version: {result.version}")
                print(f"Model Loaded: {result.model_loaded}")
                if result.model_name:
                    print(f"Model: {result.model_name}")
                print(f"Device: {result.device}")

        elif args.command == "info":
            result = client.info()
            if args.json:
                print(format_json(result.raw))
            else:
                print(f"Version: {result.version}")
                print(f"PyTorch: {result.pytorch_version}")
                print(f"CUDA Available: {result.cuda_available}")
                if result.cuda:
                    print(f"GPU: {result.cuda.get('device', 'N/A')}")
                print(f"CPU Cores: {result.cpu_count}")

        elif args.command == "generate":
            if args.stream:
                print("Generating (stream): ", end="", flush=True)
                for token in client.generate_stream(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                ):
                    print(token, end="", flush=True)
                print()
            else:
                result = client.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                if args.json:
                    print(format_json(result.raw_response))
                else:
                    print(result.generated_text)

                if args.verbose:
                    print(f"\nTokens: {result.tokens_generated}")
                    print(f"Time: {result.inference_time_ms:.2f}ms")

        elif args.command == "chat":
            messages = []
            if args.system:
                messages.append(ChatMessage.system(args.system))
            messages.append(ChatMessage.user(args.message))

            result = client.chat(
                messages,
                model=args.model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            if args.json:
                print(format_json(result.raw_response))
            else:
                print(result.message.content)

        elif args.command == "models":
            models_list = client.list_models()
            if args.json:
                print(format_json([m.raw for m in models_list]))
            else:
                for m in models_list:
                    print(f"  {m.id}")
                    if m.source:
                        print(f"    Source: {m.source}")
                    if m.description:
                        print(f"    {m.description}")

        elif args.command == "datasets":
            datasets_list = client.list_datasets()
            if args.json:
                print(format_json([d.raw for d in datasets_list]))
            else:
                for d in datasets_list:
                    print(f"  {d.id}")
                    if d.description:
                        print(f"    {d.description}")

        elif args.command == "metrics":
            if args.prometheus:
                print(client.metrics_prometheus())
            else:
                result = client.metrics()
                if args.json:
                    print(format_json(result.raw))
                else:
                    print(f"Total Requests: {result.requests_total}")
                    print(f"Successful: {result.requests_success}")
                    print(f"Failed: {result.requests_failed}")
                    print(f"Cache Hits: {result.cache_hits}")
                    print(f"Cache Misses: {result.cache_misses}")
                    print(f"Avg Response Time: {result.avg_response_time_ms:.2f}ms")

        elif args.command == "registry":
            if args.reg_action == "list":
                models = client.list_registry_models()
                if not models:
                    print("No models found.")
                else:
                    for m in models:
                        print(f"\n{m.get('model_id', m.get('id', 'unknown'))}")
                        print(f"   Status: {m.get('status', 'unknown')}")
                        print(f"   Type: {m.get('model_type', 'unknown')}")
                        if args.json:
                            print(format_json(m))

            elif args.reg_action == "info":
                model = client.get_registry_model(args.model_id)
                if model:
                    print(format_json(model) if args.json else f"{model.get('model_id', args.model_id)}")
                    if not args.json:
                        for k, v in model.items():
                            if k in ("model_id",):
                                continue
                            print(f"  {k}: {v}")
                else:
                    print(f"Model not found: {args.model_id}")

            elif args.reg_action == "best":
                result = client.get_registry_best()
                print(format_json(result) if args.json else result)

            elif args.reg_action == "stats":
                stats = client.get_registry_stats()
                print(format_json(stats) if args.json else stats)

            else:
                registry.print_help()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
