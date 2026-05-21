#!/usr/bin/env python3
"""
Auto-Train Script - Train baby model from scratch.
Usage: python3 scripts/auto_train.py [--steps N] [--lr LR] [--save path]
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class BabyModel(nn.Module):
    """Simple baby model created from scratch."""
    def __init__(self, vocab_size=46, n_embed=128, n_layer=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embed, nhead=4, batch_first=True)
            for _ in range(n_layer)
        ])
        self.output = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x), x
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(generated)
                logits = logits[:, -1, :] / temperature
                if top_k:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == 0:
                    break
        return generated


class AutoTrainer:
    def __init__(
        self,
        vocab_size: int = 46,
        n_embed: int = 128,
        n_layer: int = 2,
        learning_rate: float = 0.001,
    ):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        
        print(f"Creating baby model from scratch: vocab={vocab_size}, embed={n_embed}, layers={n_layer}")
        self.model = BabyModel(vocab_size, n_embed, n_layer)
        
        chars = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
        chars = ["<PAD>", "<UNK>"] + chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Baby model params: {params}")
        self.criterion = nn.CrossEntropyLoss()
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {param_count:,}")
    
    def encode(self, text: str) -> list:
        return [self.stoi.get(c, 0) for c in text[:64]][:64]
    
    def decode(self, ids: list) -> str:
        return "".join([self.itos.get(t, "?") for t in ids])
    
    def generate(self, input_ids: list, max_new: int = 30) -> list:
        """Generate response from input."""
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new):
                idx_cond = input_tensor[:, -self.model.block_size:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / 1.5
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, idx_next], dim=1)
        
        return input_tensor[0].tolist()
    
    def train_step(self, input_text: str, target_text: str) -> float:
        """Train on one example."""
        input_ids = self.encode(input_text)
        target_ids = self.encode(target_text)
        
        # Pad
        while len(input_ids) < 32:
            input_ids.append(0)
        while len(target_ids) < 32:
            target_ids.append(0)
        
        input_tensor = torch.tensor(input_ids[:32], dtype=torch.long).unsqueeze(0)
        target_tensor = torch.tensor(target_ids[:32], dtype=torch.long).unsqueeze(0)
        
        self.model.train()
        logits, _ = self.model(input_tensor)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def run(
        self,
        steps: int = 100,
        print_every: int = 10,
    ):
        """Run training loop."""
        # Simpler prompts/corrections for faster training
        prompts = ["hi", "hello", "hi there", "hey", "greetings"]
        corrections = ["hello", "hello there", "greetings", "hi friend", "well hello"]
        
        print("\n" + "="*60)
        print(" 🤖 AUTO-TRAIN CONVERSATION")
        print("="*60 + "\n")
        
        for step in range(steps):
            # Pick random prompt
            prompt = prompts[step % len(prompts)]
            correct = corrections[step % len(corrections)]
            
            # Generate baby response (babble)
            input_ids = self.encode(prompt)
            if len(input_ids) < 8:
                input_ids = [0] * 8
            
            baby_output = self.generate(input_ids, max_new=10)
            baby_text = self.decode(baby_output)
            
            # Train on (prompt -> correction)
            loss = self.train_step(prompt, correct)
            
            # Print as conversation
            print(f"👤 Teacher (step {step+1}): {prompt}")
            print(f"👶 Baby:          {baby_text[:50]}")
            print(f"✨ Correction:    {correct}")
            print(f"   → Loss: {loss:.4f}")
            print("-" * 40)
        
        print("\n" + "="*60)
        print(" ✅ Training complete!")
        print("="*60)
        return self.model
    
    def save(self, path: str):
        """Save model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "stoi": self.stoi,
            "itos": self.itos,
        }, save_path)
        print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-train baby model")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save", type=str, default="models/auto-training/baby.pt", help="Save path")
    parser.add_argument("--embed", type=int, default=384, help="Embedding size")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    args = parser.parse_args()
    
    trainer = AutoTrainer(
        n_embed=args.embed,
        n_layer=args.layers,
        learning_rate=args.lr,
    )
    
    trainer.run(steps=args.steps)
    trainer.save(args.save)


if __name__ == "__main__":
    main()