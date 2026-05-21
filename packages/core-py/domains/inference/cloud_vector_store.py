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


async def setup_pinecone(
    api_key: str,
    index: str = "sloughgpt",
    dimension: int = 768,
    environment: str = "us-east-1"
):
    """Setup Pinecone vector store."""
    from domains.inference.vector_store import PineconeVectorStore, VectorEntry
    
    print(f"Connecting to Pinecone index: {index}")
    store = PineconeVectorStore(
        api_key=api_key,
        index_name=index,
        dimension=dimension,
        environment=environment,
    )
    
    connected = await store.connect()
    if connected:
        print("✓ Connected to Pinecone")
        
        entries = [
            VectorEntry(
                id="sample_1",
                vector=[0.1] * dimension,
                text="This is a sample document for testing",
                metadata={"type": "test", "created_by": "cloud_vector_store"},
            ),
        ]
        
        count = await store.upsert(entries)
        print(f"✓ Upserted {count} test documents")
        
        results = await store.query(vector=[0.1] * dimension, top_k=1)
        print(f"✓ Query returned {len(results)} results")
        
        await store.disconnect()
        return True
    else:
        print("✗ Failed to connect to Pinecone")
        return False


async def test_pinecone():
    """Test Pinecone connection."""
    print("\n" + "="*60)
    print("TESTING PINECONE VECTOR STORE")
    print("="*60 + "\n")
    
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        if await setup_pinecone(api_key):
            print("   ✓ Pinecone: Connected and tested\n")
    else:
        print("   ⊘ Pinecone: PINECONE_API_KEY not set")
        print("\nTo set up Pinecone:")
        print("   1. Get API key from https://app.pinecone.io")
        print("   2. export PINECONE_API_KEY='your-api-key'")
        print("   3. python -m domains.inference.cloud_vector_store --setup")
    
    print("="*60)


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
