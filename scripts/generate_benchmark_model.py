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

All the world's a stage,
And all the men and women merely players:
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.
And then the whining school-boy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side;
His youthful hose, well saved, a world too wide
For his shrunk shank; and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest--
For Brutus is an honourable man;
So are they all, all honourable men--
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.
He hath brought many captives home to Rome,
Whose ransoms did the general coffers fill:
Did this in Caesar seem ambitious?
When that the poor have cried, Caesar hath wept:
Ambition should be made of sterner stuff:
Yet Brutus says he was ambitious;
And Brutus is an honourable man.
You all did see that on the Lupercal
I thrice presented him a kingly crown,
Which he did thrice refuse: was this ambition?
Yet Brutus says he was ambitious;
And, sure, he is an honourable man.
I speak not to disprove what Brutus spoke,
But here I am to speak what I know.
You all did love him once, not without cause:
What cause withholds you then, to mourn for him?
O judgment! thou art fled to brutish beasts,
And men have lost their reason.-Bear with me;
My heart is in the coffin there with Caesar,
And I must pause till it come back to me.

If music be the food of love, play on;
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.
That strain again! it had a dying fall:
O, it came o'er my ear like the sweet sound
That breathes upon a bank of violets,
Stealing and giving odour.-Enough; no more:
'Tis not so sweet now as it was before.
O spirit of love! how quick and fresh art thou,
That, notwithstanding thy capacity
Receiveth as the sea, nought enters there,
Of what validity and pitch soe'er,
But falls into abatement and low price,
Even in a minute: so full of shape is madness
That in his vile apprehension of it
He makes a hecatomb of reason,
And kills recreant reason.
What is love? 'tis not hereafter;
Present mirth hath present laughter;
What's to come is still unsure:
In delay there lies no plenty;
Then come kiss me, sweet and twenty,
Youth's a stuff will not endure.
""".strip() * 5


def train_benchmark_model(epochs: int = 150, output_path: str = "models/bench_shakespeare.soul"):
    """Train a proper SloNet model and save as .soul checkpoint."""
    chars = sorted(set(SHAKESPEARE_TEXT))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    vocab_size = len(chars) + 1

    def encode(text):
        return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)

    # Model config: 64 embed, 128 hidden, 1 layer (smaller for fast training)
    n_embed = 64
    n_hidden = 128
    n_layers = 1

    print(f"Training benchmark model: vocab={vocab_size}, embed={n_embed}, hidden={n_hidden}, layers={n_layers}")
    print(f"Training data: {len(SHAKESPEARE_TEXT)} chars, {epochs} epochs")

    net = SloNet(
        layers=[
            SloEmbedding(vocab_size, n_embed),
            SloLSTM(n_embed, n_hidden, n_hidden, num_layers=n_layers, dropout=0.0),
        ],
        soul_name="bench-shakespeare",
    )
    lstm = net.layers[1]
    opt = SloAdam(lr=0.005)
    data = encode(SHAKESPEARE_TEXT)
    chunk = 128
    losses = []

    t0 = time.time()
    for ep in range(epochs):
        order = np.random.permutation(max(1, len(data) - chunk))
        ep_loss = 0.0
        steps = 0
        for pos in order[:20]:  # 20 steps per epoch (faster)
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

    # Save checkpoint
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    profile = SloProfile(
        name="bench-shakespeare",
        soul_traits={"warmth": 0.5, "creativity": 0.5, "curiosity": 0.5, "confidence": 0.5},
        system_prompt="You are a helpful assistant.",
        lineage="benchmark",
        metadata={
            "charset": "".join(chars),
            "model_config": {
                "n_embed": n_embed,
                "n_hidden": n_hidden,
                "n_layer": n_layers,
                "vocab_size": vocab_size,
            },
            "training": {
                "epochs": epochs,
                "final_loss": float(losses[-1]) if losses else None,
                "data": "shakespeare",
            },
        },
    )
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
