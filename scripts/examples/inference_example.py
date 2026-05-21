#!/usr/bin/env python3
"""
Example: Use the inference engine
"""

import sys
sys.path.insert(0, "..")

def main():
    print("=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    try:
        from domains.inference.engine import InferenceEngine
        
        print("\nInitializing engine...")
        engine = InferenceEngine()
        
        print("\nGenerating text...")
        result = engine.generate_single(
            prompt="The quick brown fox",
            max_new_tokens=20,
            temperature=0.8
        )
        
        print(f"\nResult: {result}")
        print("\nInference complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nStart the API server first:")
        print("  python3 apps/api/server/main.py")
        print("\nOr use the CLI:")
        print("  python3 cli.py generate 'The quick brown fox'   # or: cli.py gen '...'")

if __name__ == "__main__":
    main()
