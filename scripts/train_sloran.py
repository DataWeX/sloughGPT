"""Train SloRAN on Shakespeare — char-level language model.

Usage:
    python scripts/train_sloran.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'core-py'))
from domains.training.sloran import SloRAN
from domains.training.slonet import Tensor, cross_entropy
import numpy as np

# ── Config ─────────────────────────────────────────────────────────
DATA_PATH = 'datasets/shakespeare/input.txt'
VOCAB_SIZE = 65   # chars in Shakespeare
D_MODEL = 128
N_LAYERS = 4
N_SLOTS = 8
D_STATE = 64
BLOCK_SIZE = 64
DROPOUT = 0.0
LR = 3e-3
EPOCHS = 500
BATCH_SIZE = 32
EVAL_EVERY = 50
GEN_EVERY = 100

def load_data(path):
    with open(path) as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, len(chars), stoi, itos

def get_batch(data, batch_size, block_size):
    n = len(data)
    starts = np.random.randint(0, n - block_size - 1, size=batch_size)
    x = np.stack([data[s:s+block_size] for s in starts])
    y = np.stack([data[s+1:s+block_size+1] for s in starts])
    return x, y

def main():
    data, vocab_size, stoi, itos = load_data(DATA_PATH)
    print(f'Data: {len(data)} chars, vocab={vocab_size}')

    m = SloRAN(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_slots=N_SLOTS,
        d_state=D_STATE,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    )
    print(f'Model: {sum(p.data.size for p in m.parameters()):,} params')

    for epoch in range(1, EPOCHS + 1):
        x_np, y_np = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
        xi = Tensor(x_np)
        t = Tensor(y_np)

        logits, loss = m.forward(xi, targets=t)
        loss.backward()

        lr = LR * (1 - epoch / EPOCHS) if epoch < EPOCHS * 0.8 else LR * 0.1
        for p in m.parameters():
            p.data -= lr * p.grad.data
            p.grad = None

        if epoch % 10 == 0:
            print(f'epoch {epoch:4d}  loss {float(loss.data):.4f}  lr {lr:.2e}')

        if epoch % EVAL_EVERY == 0:
            # eval loss on a held-out batch
            x_eval, y_eval = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            logits_e, loss_e = m.forward(Tensor(x_eval), targets=Tensor(y_eval))
            print(f'  eval loss {float(loss_e.data):.4f}')

        if epoch % GEN_EVERY == 0:
            prompt = np.array([[stoi['O']]])  # "O Romeo, Romeo"
            out = m.generate(prompt, max_new_tokens=128, temperature=0.8, top_k=10)
            gen = ''.join(itos[i] for i in out)
            print(f'  ── gen ──\n  {gen[:256]}\n  ────────')

if __name__ == '__main__':
    main()
