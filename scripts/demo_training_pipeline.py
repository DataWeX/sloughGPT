"""
Demo: How to use training pipeline and train the model.
"""

import json
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


# Load modules
def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module(
    "pipeline",
    Path("/Users/mac/sloughGPT/packages/core-py/domains/infrastructure/training_pipeline.py"),
).TrainingDataPipeline(data_dir="/tmp/demo_training")

print("=" * 60)
print("DEMO: Training Pipeline for Model Fine-tuning")
print("=" * 60)

# 1. Add conversations (simulating chat)
print("\n1. Adding sample conversations...")
conv1 = pipeline.add_conversation(
    session_id="chat_1",
    user_message="What is Python?",
    assistant_message="Python is a high-level programming language known for its simplicity and readability.",
    model="gpt2",
)
print(f"   Added: {conv1.user_message[:30]}...")

conv2 = pipeline.add_conversation(
    session_id="chat_1",
    user_message="How do I learn fast?",
    assistant_message="Practice daily, focus on fundamentals, build projects, and teach others.",
    model="gpt2",
)
print(f"   Added: {conv2.user_message[:30]}...")

conv3 = pipeline.add_conversation(
    session_id="chat_2",
    user_message="Explain AI",
    assistant_message="AI is when machines can learn from data and make decisions like humans.",
    model="gpt2",
)

# 2. Add feedback (simulating user rating)
print("\n2. Adding feedback (quality scores)...")
pipeline.add_feedback(conv1.id, "up")  # Good response
pipeline.add_feedback(conv2.id, "up")  # Good response
# conv3 has no feedback = neutral (0.5 quality)

# 3. Check training pairs with quality scores
print("\n3. Training pairs with quality scores:")
pairs = pipeline.get_training_pairs(min_quality=0.0)
for p in pairs:
    quality = "⭐" if p.quality_score >= 0.8 else ("⚠️" if p.quality_score >= 0.3 else "❌")
    print(f"   {quality} Quality {p.quality_score}: {p.prompt[:25]}...")

# 4. Export for training
print("\n4. Exporting training data...")
filepath = pipeline.export_training_data(min_quality=0.5, format="jsonl")
print(f"   Saved to: {filepath}")

# 5. Show exported format
print("\n5. Exported JSONL format (ready for training):")
with open(filepath) as f:
    for line in f:
        print(f"   {line.strip()}")

# 6. Stats
print("\n6. Pipeline Statistics:")
stats = pipeline.get_stats()
for k, v in stats.items():
    print(f"   {k}: {v}")

print("\n" + "=" * 60)
print("HOW TO TRAIN YOUR MODEL:")
print("=" * 60)
print("""
OPTION 1: Via API
  curl -X POST http://localhost:8000/training/from-conversations \\
       -d '{"min_quality": 0.5, "epochs": 3, "use_lora": true}'

OPTION 2: Use exported JSONL directly:
  1. File is at: data/exports/latest.jsonl
  2. Use with your training pipeline:
     from domains.training import train_from_dataset
     train_from_dataset("data/exports/latest.jsonl")

OPTION 3: Fine-tune with LoRA:
  trainer = SloughGPTTrainer(config)
  trainer.train(dataset_path="data/exports/latest.jsonl")

The JSONL format is compatible with:
- HuggingFace transformers Trainer
- SloughGPTTrainer
- PyTorch DataLoader
""")

print("\nDone!")
