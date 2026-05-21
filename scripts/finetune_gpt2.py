"""Fine-tune GPT-2 on a small conversation dataset for chat."""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api", "server"))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune")

CONVERSATIONS = [
    {"messages": [
        {"role": "user", "content": "Hi, how are you?"},
        {"role": "assistant", "content": "Hi! I'm doing great, thanks for asking! How can I help you today?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial intelligence is the simulation of human intelligence by computer systems. It includes tasks like learning, reasoning, problem-solving, and understanding language. AI can be narrow (designed for specific tasks) or general (human-level intelligence across domains)."},
    ]},
    {"messages": [
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side! Oh wait, you've heard that one? OK here's another: What do you call a fake noodle? An impasta!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have real-time access to weather data, but I can help you find information about weather patterns, climate, or how to check your local forecast!"},
    ]},
    {"messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! Welcome! I'm your AI assistant. Feel free to ask me anything - I'm here to help!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can help answer questions, explain concepts, generate creative writing, and have conversations on a wide range of topics. Just let me know what you need!"},
    ]},
    {"messages": [
        {"role": "user", "content": "Tell me about yourself"},
        {"role": "assistant", "content": "I'm an AI assistant trained to be helpful, harmless, and honest. I can discuss many topics and I'm always learning. What would you like to talk about?"},
    ]},
    {"messages": [
        {"role": "user", "content": "How does machine learning work?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning from rewards and penalties)."},
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks for your help!"},
        {"role": "assistant", "content": "You're welcome! I'm glad I could help. If you have any more questions, don't hesitate to ask. Have a great day!"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a popular, high-level programming language known for its readability and versatility. It's widely used in web development, data science, AI, machine learning, and automation. Guido van Rossum created it in 1991, and its design philosophy emphasizes code readability."},
    ]},
]

def format_conversation(conv):
    """Format a conversation for GPT-2 training."""
    parts = []
    for msg in conv["messages"]:
        if msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"Assistant: {msg['content']}")
    return "\n".join(parts) + "\n"

def main():
    model_name = "gpt2"
    output_dir = "models/gpt2-finetuned-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    texts = [format_conversation(c) for c in CONVERSATIONS]
    logger.info(f"Training on {len(texts)} conversations")

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]})

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=10,
        prediction_loss_only=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.config.save_pretrained(output_dir)
    logger.info("Done! Model saved.")

if __name__ == "__main__":
    main()
