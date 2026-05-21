"""
Demonstration: How the ConversationEngine integrates with existing infrastructure.
"""

import sys
import os
import tempfile
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def demo_conversation_engine():
    print("=" * 60)
    print("DEMO: ConversationEngine Integration")
    print("=" * 60)

    # Load conversation engine - absolute path
    ce_path = Path(
        "/Users/mac/sloughGPT/packages/core-py/domains/infrastructure/conversation_engine.py"
    )
    spec = spec_from_file_location("conversation_engine", ce_path)
    ce_module = module_from_spec(spec)
    spec.loader.exec_module(ce_module)

    # Create engine with temp file
    temp_file = tempfile.mktemp(suffix=".chatdb")
    engine = ce_module.ConversationEngine(temp_file)

    print("\n1. Simulating chat session...")
    print("-" * 40)

    # Simulate: User sends message
    msg1 = engine.add_message(
        session_id="chat_abc123", role="user", content="Hello, how are you?", model="gpt2"
    )
    print(f"   User: Hello, how are you?")
    print(f"   -> Saved to DB (id: {msg1.id[:20]}...)")

    # Simulate: Model generates response
    msg2 = engine.add_message(
        session_id="chat_abc123",
        role="assistant",
        content="I'm doing great! How can I help you today?",
        model="gpt2",
        tokens=10,
    )
    print(f"   Assistant: I'm doing great! How can I help you today?")
    print(f"   -> Saved to DB (id: {msg2.id[:20]}...)")

    # Continue conversation
    engine.add_message("chat_abc123", "user", "What's the weather?", "gpt2")
    engine.add_message(
        "chat_abc123", "assistant", "I'm an AI so I don't have access to real weather data.", "gpt2"
    )

    print("\n2. Building context for next prompt...")
    print("-" * 40)

    context = engine.to_context_string(session_id="chat_abc123", limit=5)
    print("   Context string for model:")
    for line in context.split("\n")[:6]:
        print(f"   {line}")

    print("\n3. Extracting training pairs...")
    print("-" * 40)

    pairs = engine.get_training_pairs()
    print(f"   Found {len(pairs)} (prompt, response) pairs")
    for i, p in enumerate(pairs[:2]):
        print(f"   Pair {i + 1}:")
        print(f"     Prompt: {p['prompt'][:40]}...")
        print(f"     Response: {p['response'][:40]}...")

    print("\n4. Database stats...")
    print("-" * 40)

    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Cleanup
    os.remove(temp_file)

    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)
    print("""
How it works with existing infrastructure:
  
  1. Frontend sends message with session_id
  2. /chat/stream saves user message to conversation DB
  3. Model generates response
  4. Response saved to conversation DB  
  5. Next message: _build_context_prompt() fetches history
  6. History prepended to prompt for better context
  
No breaking changes - it's all additive integration!
""")


if __name__ == "__main__":
    demo_conversation_engine()
