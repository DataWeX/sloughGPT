"""
Inference Quality Benchmark — measures generation quality across configs.

Metrics:
  - Perplexity on held-out validation set
  - Repetition rate (fraction of repeated bigrams/trigrams)
  - Diversity (type-token ratio, distinct n-gram count)
  - Output entropy (how uniformly distributed tokens are)

Usage:
    python scripts/benchmark_inference_quality.py
"""

import sys; sys.path.insert(0, "packages/core-py")
import time, json, math, os, itertools
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from domains.training.slonet import (
    SloNet, SloEmbedding, SloLSTM, SloAdam,
    cross_entropy, tensor, zeros, no_grad,
    _sample_from_logits,
)

TRAIN_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
    "the five boxing wizards jump quickly. "
    "sphinx of black quartz judge my vow. "
    "two driven jocks help fax my big quiz. "
    "five quacking zephyrs jolt my wax bed. "
    "the jay pig fox zebra and my wolves quack. "
    "blowzy red vixens fight for a quick jump. "
    "cozy lummox gives smart squid who asks for job pen. "
    "a quick movement of the enemy will jeopardize six gunboats. "
    "all questions asked by five watch experts amazed the judge. "
    "jack quietly moved up front and seized the big ball of wax. "
    "the quick brown fox jumps over the lazy dog repeatedly. "
    "we promptly judged antique ivory buckles for the next prize. "
) * 5

VAL_TEXT = (
    "crazy fredrick bought many very exquisite opal jewels. "
    "sixty zippers were quickly picked from the woven jute bag. "
)

CHARS = sorted(set(TRAIN_TEXT + VAL_TEXT))
STOI = {c: i+1 for i, c in enumerate(CHARS)}
ITOS = {i+1: c for i, c in enumerate(CHARS)}
PAD_ID = 0
VOCAB_SIZE = len(CHARS) + 1


def char_encode(text: str) -> np.ndarray:
    return np.array([STOI.get(c, PAD_ID) for c in text], dtype=np.int64)


def char_decode(ids: np.ndarray) -> str:
    ids = ids.flatten()
    valid = ids[ids != PAD_ID]
    return "".join(ITOS.get(int(i), "?") for i in valid)


def train_model(epochs=60) -> Tuple[SloNet, SloLSTM]:
    net = SloNet(
        layers=[SloEmbedding(VOCAB_SIZE, 32), SloLSTM(VOCAB_SIZE, 32, 64, num_layers=1, dropout=0.0)],
        soul_name="qual_bench",
    )
    lstm = net.layers[1]
    opt = SloAdam(lr=0.01)
    data = char_encode(TRAIN_TEXT)
    chunk = 32
    losses = []
    for ep in range(epochs):
        order = np.random.permutation(max(1, len(data) - chunk))
        ep_loss = 0.0
        steps = 0
        for pos in order[:30]:
            x = tensor(data[pos:pos+chunk].reshape(1, -1), requires_grad=True)
            y = tensor(data[pos+1:pos+chunk+1].reshape(1, -1))
            h = lstm.init_hidden()
            logits, _ = lstm.forward(x, h)
            loss = cross_entropy(logits, y.reshape(-1))
            ep_loss += float(loss.data)
            steps += 1
            loss.backward()
            opt.step(lstm.parameters())
            lstm.zero_grad()
        losses.append(ep_loss / max(steps, 1))
        if ep > 20 and losses[-1] > np.median(losses[-10:]):
            break
    return net, lstm


def compute_perplexity(lstm: SloLSTM, text: str) -> float:
    ids = char_encode(text)
    chunk = 64
    total_nll = 0.0
    count = 0
    with no_grad():
        for pos in range(0, len(ids) - chunk, chunk//2):
            x = tensor(ids[pos:pos+chunk].reshape(1, -1))
            y = tensor(ids[pos+1:pos+chunk+1].reshape(1, -1))
            h = lstm.init_hidden()
            logits, _ = lstm.forward(x, h)
            loss = cross_entropy(logits, y.reshape(-1))
            total_nll += float(loss.data) * (y.data.shape[0] * y.data.shape[1] if y.data.ndim > 1 else y.data.shape[0])
            count += y.data.shape[0] * y.data.shape[1] if y.data.ndim > 1 else y.data.shape[0]
    return math.exp(total_nll / max(count, 1))


@no_grad()
def generate_text(lstm: SloLSTM, prompt: str, max_tokens: int = 100, **gen_kwargs) -> str:
    ids = char_encode(prompt).flatten().tolist()
    for _ in range(max_tokens):
        seq = np.array([ids[-64:]], dtype=np.int64)
        logits_t, _ = lstm.forward(tensor(seq), lstm.init_hidden())
        data = logits_t.data
        # handle 2D (batch, vocab) or 3D (batch, seq, vocab)
        if data.ndim == 3:
            logits_2d = data[:, -1:, :]
        else:
            logits_2d = data.reshape(1, 1, -1)
        nid = _sample_from_logits(
            logits_2d,
            generated_ids=np.array(ids[len(prompt):], dtype=np.int64),
            **gen_kwargs,
        )
        ids.append(nid)
        if nid == PAD_ID:
            break
    return char_decode(np.array(ids))


def repetition_rate(text: str, n: int = 2) -> float:
    ng = [text[i:i+n] for i in range(len(text)-n)]
    if not ng:
        return 0.0
    return 1.0 - len(set(ng)) / len(ng)


def diversity(text: str) -> Dict:
    tokens = list(text)
    ttr = len(set(tokens)) / max(len(tokens), 1)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    return {
        "type_token_ratio": round(ttr, 4),
        "unique_bigrams": len(set(bigrams)),
        "unique_trigrams": len(set(trigrams)),
        "bigram_rep_rate": round(1.0 - len(set(bigrams))/max(len(bigrams),1), 4),
    }


@dataclass
class QualityResult:
    config_name: str
    perplexity: float
    repetition_2: float
    repetition_3: float
    type_token_ratio: float
    unique_bigrams: int
    unique_trigrams: int
    output_length: int
    entropy: float


@dataclass
class Config:
    name: str
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def output_entropy(lstm: SloLSTM, prompt: str, num_samples: int = 5, num_tokens: int = 50) -> float:
    """Estimate output diversity entropy — higher = more varied generations."""
    texts = []
    with no_grad():
        for _ in range(num_samples):
            t = generate_text(lstm, prompt, max_tokens=num_tokens, temperature=0.9, top_k=40, repetition_penalty=1.1)
            texts.append(t)
    all_tokens = "".join(texts)
    if not all_tokens:
        return 0.0
    from collections import Counter
    freq = Counter(all_tokens)
    probs = np.array([f / len(all_tokens) for f in freq.values()])
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def main():
    print("=" * 60)
    print("Inference Quality Benchmark")
    print("=" * 60)
    print()
    print("Training model...", end=" ", flush=True)
    t0 = time.time()
    net, lstm = train_model()
    print(f"done ({time.time()-t0:.1f}s)")
    print()

    ppl = compute_perplexity(lstm, VAL_TEXT)
    print(f"Validation Perplexity: {ppl:.2f}")
    print()

    configs = [
        Config(name="greedy", temperature=1e-7),
        Config(name="temp=0.5", temperature=0.5),
        Config(name="temp=0.8", temperature=0.8),
        Config(name="temp=1.0", temperature=1.0),
        Config(name="temp=0.8+topk40", temperature=0.8, top_k=40),
        Config(name="temp=0.8+topp0.9", temperature=0.8, top_p=0.9),
        Config(name="temp=0.8+topk40+topp0.9", temperature=0.8, top_k=40, top_p=0.9),
        Config(name="temp=0.8+rp1.2", temperature=0.8, repetition_penalty=1.2),
        Config(name="temp=0.8+rp1.2+fp0.1", temperature=0.8, repetition_penalty=1.2, frequency_penalty=0.1),
        Config(name="temp=0.8+topk40+rp1.2", temperature=0.8, top_k=40, repetition_penalty=1.2),
        Config(name="temp=0.8+topk40+topp0.9+rp1.2", temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2),
        Config(name="temp=0.9+topk40+rp1.1", temperature=0.9, top_k=40, repetition_penalty=1.1),
    ]

    prompt = "the quick brown fox "
    results = []
    for cfg in configs:
        gen_kwargs = {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "frequency_penalty": cfg.frequency_penalty,
            "presence_penalty": cfg.presence_penalty,
        }
        text = generate_text(lstm, prompt, max_tokens=80, **gen_kwargs)
        rep2 = repetition_rate(text, 2)
        rep3 = repetition_rate(text, 3)
        div = diversity(text)
        ent = output_entropy(lstm, prompt[:10], num_samples=3, num_tokens=40)

        results.append(QualityResult(
            config_name=cfg.name,
            perplexity=ppl,
            repetition_2=rep2,
            repetition_3=rep3,
            type_token_ratio=div["type_token_ratio"],
            unique_bigrams=div["unique_bigrams"],
            unique_trigrams=div["unique_trigrams"],
            output_length=len(text),
            entropy=ent,
        ))
        print(f"  [{cfg.name:40s}] rep2={rep2:.3f}  rep3={rep3:.3f}  ttr={div['type_token_ratio']:.3f}  ent={ent:.3f}")
        print(f"  {'':40s}  output: {text[:80]}...")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':40s} {'Rep2':>6s} {'Rep3':>6s} {'TTR':>6s} {'Ent':>6s} {'Bigr':>5s} {'Trgr':>5s}")
    print("-" * 80)
    best = min(results, key=lambda r: r.repetition_2 + r.repetition_3 * 2 - r.entropy * 0.1 + r.type_token_ratio * 0.1)
    for r in results:
        marker = " ←" if r is best else ""
        print(f"{r.config_name:40s} {r.repetition_2:6.3f} {r.repetition_3:6.3f} {r.type_token_ratio:6.3f} {r.entropy:6.3f} {r.unique_bigrams:5d} {r.unique_trigrams:5d}{marker}")
    print()

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "perplexity": ppl,
        "results": [asdict(r) for r in results],
        "best_config": best.config_name,
    }
    os.makedirs("data/eval_results", exist_ok=True)
    with open("data/eval_results/inference_quality.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to data/eval_results/inference_quality.json")
    print(f"Best config: {best.config_name}")


if __name__ == "__main__":
    main()
