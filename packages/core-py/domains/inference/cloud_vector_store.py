"""
Cloud Vector Store Connector - Pinecone Only

Provides easy setup for Pinecone vector store.

Usage:
    python -m domains.inference.cloud_vector_store --setup --api-key YOUR_KEY
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import logging

logger = logging.getLogger("man.cloud_vector_store")


async def setup_pinecone(
    api_key: str,
    index: str = "sloughgpt",
    dimension: int = 768,
    environment: str = "us-east-1"
):
    """Setup Pinecone vector store."""
    from domains.inference.vector_stores.pinecone_store import PineconeVectorStore
    from domains.inference.vector_store import VectorEntry

    logger.info("Connecting to Pinecone index: %s", index, extra={"tag": "INF"})
    store = PineconeVectorStore(
        api_key=api_key,
        index_name=index,
        dimension=dimension,
        environment=environment,
    )

    connected = await store.connect()
    if connected:
        logger.info("Connected to Pinecone", extra={"tag": "INF"})

        entries = [
            VectorEntry(
                id="sample_1",
                vector=[0.1] * dimension,
                text="This is a sample document for testing",
                metadata={"type": "test", "created_by": "cloud_vector_store"},
            ),
        ]

        count = await store.upsert(entries)
        logger.info("Upserted %d test documents", count, extra={"tag": "INF"})

        results = await store.query(vector=[0.1] * dimension, top_k=1)
        logger.info("Query returned %d results", len(results), extra={"tag": "INF"})

        await store.disconnect()
        return True
    else:
        logger.info("Failed to connect to Pinecone", extra={"tag": "INF"})
        return False


async def test_pinecone():
    """Test Pinecone connection."""
    logger.info("=" * 60, extra={"tag": "INF"})
    logger.info("TESTING PINECONE VECTOR STORE", extra={"tag": "INF"})
    logger.info("=" * 60, extra={"tag": "INF"})

    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        if await setup_pinecone(api_key):
            logger.info("   Pinecone: Connected and tested", extra={"tag": "INF"})
    else:
        logger.info("   Pinecone: PINECONE_API_KEY not set", extra={"tag": "INF"})
        logger.info("To set up Pinecone:", extra={"tag": "INF"})
        logger.info("   1. Get API key from https://app.pinecone.io", extra={"tag": "INF"})
        logger.info("   2. export PINECONE_API_KEY='your-api-key'", extra={"tag": "INF"})
        logger.info("   3. python -m domains.inference.cloud_vector_store --setup", extra={"tag": "INF"})

    logger.info("=" * 60, extra={"tag": "INF"})


def main():
    parser = argparse.ArgumentParser(description="Pinecone Vector Store Setup")
    parser.add_argument("--setup", action="store_true", help="Setup Pinecone")
    parser.add_argument("--api-key", help="Pinecone API key")
    parser.add_argument("--index", default="sloughgpt", help="Index name")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    parser.add_argument("--environment", default="us-east-1", help="Pinecone environment")

    args = parser.parse_args()

    if args.setup or args.api_key:
        api_key = args.api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("Error: Pinecone API key required (--api-key or PINECONE_API_KEY)")
            sys.exit(1)
        asyncio.run(setup_pinecone(api_key, args.index, args.dimension, args.environment))

    elif not sys.argv[1:]:
        parser.print_help()
        print("\n\nQuick test:")
        print("  export PINECONE_API_KEY='your-key'")
        print("  python -m domains.inference.cloud_vector_store --setup")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
