#!/usr/bin/env python3
"""Generate a pre-trained benchmark model for comparisons.

Trains a medium SloNet model on Shakespeare text and saves as .soul checkpoint.
This checkpoint is used by benchmark_model_comparison.py --mode native.

Usage:
    python scripts/generate_benchmark_model.py
    python scripts/generate_benchmark_model.py --epochs 200 --output models/bench_shakespeare.soul
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

import numpy as np
from domains.training.slonet import (
    SloNet, SloEmbedding, SloLSTM, SloAdam,
    cross_entropy, tensor, no_grad, _sample_from_logits,
    save_checkpoint_npz,
)
from domains.inference.slo_format import SloProfile, save_soul


# ── Shakespeare training data ────────────────────────────────────────────

SHAKESPEARE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment,
With this regard their currents turn awry,
And lose the name of action.-Soft you now!
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.
""".strip() * 2


def train_benchmark_model(epochs: int = 150, output_path: str = "models/bench_shakespeare.soul"):
    """Train a proper SloNet model and save as .soul checkpoint."""
    chars = sorted(set(SHAKESPEARE_TEXT))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    vocab_size = len(chars) + 1

    def encode(text):
        return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)

    # Model config
    n_embed = 64
    n_hidden = 128
    n_layers = 1

    print(f"Training benchmark model: vocab={vocab_size}, embed={n_embed}, hidden={n_hidden}, layers={n_layers}")
    print(f"Training data: {len(SHAKESPEARE_TEXT)} chars, {epochs} epochs")

    net = SloNet(
        layers=[
            SloEmbedding(vocab_size, n_embed),
            SloLSTM(vocab_size, n_embed, n_hidden, num_layers=n_layers, dropout=0.0),
        ],
        soul_name="bench-shakespeare",
    )
    lstm = net.layers[1]
    opt = SloAdam(lr=0.001)
    data = encode(SHAKESPEARE_TEXT)
    chunk = 128
    losses = []

    t0 = time.time()
    for ep in range(epochs):
        order = np.random.permutation(max(1, len(data) - chunk))
        ep_loss = 0.0
        steps = 0
        for pos in order[:20]:
            x = tensor(data[pos:pos + chunk].reshape(1, -1), requires_grad=True)
            y = tensor(data[pos + 1:pos + chunk + 1].reshape(1, -1))
            h = lstm.init_hidden()
            logits, _ = lstm.forward(x, h)
            loss = cross_entropy(logits, y.reshape(-1))
            ep_loss += float(loss.data)
            steps += 1
            loss.backward()
            opt.step(lstm.parameters())
            lstm.zero_grad()
        avg_loss = ep_loss / max(steps, 1)
        losses.append(avg_loss)
        if ep % 25 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {ep:3d}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    # Save checkpoint as .soul
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    profile = SloProfile(
        name="bench-shakespeare",
        system_prompt="You are a helpful assistant trained on Shakespeare.",
        lineage="benchmark",
    )
    profile.metadata["charset"] = "".join(chars)
    profile.metadata["vocab_size"] = vocab_size
    profile.metadata["n_embed"] = n_embed
    profile.metadata["n_hidden"] = n_hidden
    profile.metadata["n_layer"] = n_layers
    profile.metadata["training"] = {
        "epochs": epochs,
        "final_loss": float(losses[-1]) if losses else None,
        "data": "shakespeare",
    }
    save_soul(net, output_path, soul_profile=profile)
    print(f"\nSaved benchmark model to {output_path}")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output", default="models/bench_shakespeare.soul")
    args = parser.parse_args()
    train_benchmark_model(args.epochs, args.output)
