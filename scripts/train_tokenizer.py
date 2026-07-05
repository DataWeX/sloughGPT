"""
SloBPE Tokenizer CLI

Train, inspect, and test BPE tokenizers on text files.
Uses TokenizerManager from the core domain.

Usage:
    python train_tokenizer.py train --file data.txt --vocab-size 1024 --output my_tokenizer.json
    python train_tokenizer.py inspect --tokenizer my_tokenizer.json
    python train_tokenizer.py encode --tokenizer my_tokenizer.json --text "hello world"
    python train_tokenizer.py sample --tokenizer my_tokenizer.json
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api", "server"))

from domains.training.tokenizer_manager import get_tokenizer_manager


def cmd_train(args):
    """Train a BPE tokenizer on text files."""
    texts = []
    for path in args.files:
        print(f"Loading {path}...", file=sys.stderr)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        print(f"  {len(lines)} lines, {len(text)} chars", file=sys.stderr)
        texts.extend(lines)

    if not texts:
        print("Error: no text loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Training BPE: vocab_size={args.vocab_size}, min_freq={args.min_frequency}...", file=sys.stderr)
    start = time.perf_counter()
    mgr = get_tokenizer_manager()
    stats = mgr.train(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, lowercase=args.lowercase)
    elapsed = time.perf_counter() - start

    print(f"Trained in {elapsed:.2f}s", file=sys.stderr)
    print(f"  Vocab: {stats['vocab_size']} tokens", file=sys.stderr)
    print(f"  Chars: {stats['base_chars']} | Subwords: {stats['merged_subwords']} | Merges: {stats['total_merges_learned']}", file=sys.stderr)

    output = args.output
    if not output:
        name = os.path.splitext(os.path.basename(args.files[0]))[0]
        output = f"{name}_bpe.json"

    mgr.save(output)
    print(f"Saved to {output} ({os.path.getsize(output)} bytes)", file=sys.stderr)

    # Test sample
    test = "the quick brown fox jumps over the lazy dog"
    ids = mgr.tokenize(test)
    rec = mgr.detokenize(ids)
    print(f"\nSample: \"{test}\"", file=sys.stderr)
    print(f"  Tokens: {len(ids)} (chars: {len(test)})", file=sys.stderr)
    print(f"  Ratio: {len(test)/max(len(ids),1):.1f}x compression", file=sys.stderr)
    print(f"  Roundtrip: {'OK' if test == rec else 'FAIL'}", file=sys.stderr)


def cmd_inspect(args):
    """Show tokenizer vocabulary and stats."""
    mgr = _load_manager(args.tokenizer)
    tok = mgr.get_tokenizer()
    stats = mgr.stats()

    print(f"Vocabulary: {stats['vocab_size']} tokens")
    print(f"  Base chars:   {stats['base_chars']}")
    print(f"  Subwords:     {stats['merged_subwords']}")
    print(f"  Special:      {stats['special_tokens']}")
    print(f"  Merge rules:  {stats['total_merges']}")

    tok.show_vocab(args.top_n)
    if args.show_merges:
        tok.show_merges(args.top_n)


def cmd_encode(args):
    """Encode text to token IDs."""
    mgr = _load_manager(args.tokenizer)
    ids = mgr.tokenize(args.text)
    tok = mgr.get_tokenizer()
    tokens = [tok.itos[i] for i in ids]
    print(f"Text:   {args.text}")
    print(f"IDs:    {ids}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(ids)} tokens ({len(args.text)} chars, {len(args.text)/max(len(ids),1):.1f}x)")


def cmd_decode(args):
    """Decode token IDs back to text."""
    mgr = _load_manager(args.tokenizer)
    ids = [int(x) for x in args.ids.split(",")]
    text = mgr.detokenize(ids)
    print(f"IDs:  {ids}")
    print(f"Text: {text}")


def cmd_sample(args):
    """Show sample tokenizations for common words."""
    mgr = _load_manager(args.tokenizer)
    tok = mgr.get_tokenizer()
    words = ["the", "and", "to", "of", "a", "in", "that", "is",
             "was", "he", "for", "it", "with", "as", "his", "on",
             "hello", "world", "machine", "learning", "neural", "network",
             "artificial", "intelligence", "transformer"]

    print(f"{'Word':<20} {'Tokens':<30} {'Count'}")
    print("-" * 60)
    for w in words:
        ids = mgr.tokenize(w)
        tokens = [tok.itos[i] for i in ids]
        tok_str = " | ".join(tokens)
        print(f"{w:<20} {tok_str:<30} {len(ids)}")


def _load_manager(path: str):
    """Load a saved tokenizer into a fresh manager."""
    mgr = get_tokenizer_manager()
    mgr.load(path)
    print(f"Loaded tokenizer: vocab={mgr.vocab_size}", file=sys.stderr)
    return mgr


def main():
    parser = argparse.ArgumentParser(description="SloBPE Tokenizer CLI")
    sub = parser.add_subparsers(dest="command", help="Command")

    p_train = sub.add_parser("train", help="Train a BPE tokenizer")
    p_train.add_argument("files", nargs="+", help="Text files to train on")
    p_train.add_argument("--vocab-size", type=int, default=1024, help="Target vocabulary size")
    p_train.add_argument("--min-frequency", type=int, default=2, help="Minimum pair frequency")
    p_train.add_argument("--output", "-o", help="Output file path")
    p_train.add_argument("--no-lowercase", dest="lowercase", action="store_false", help="Don't lowercase text")
    p_train.set_defaults(func=cmd_train)

    p_inspect = sub.add_parser("inspect", help="Show tokenizer info")
    p_inspect.add_argument("tokenizer", help="Path to saved tokenizer JSON")
    p_inspect.add_argument("--top-n", type=int, default=40, help="Number of entries to show")
    p_inspect.add_argument("--show-merges", action="store_true", help="Show merge rules")
    p_inspect.set_defaults(func=cmd_inspect)

    p_encode = sub.add_parser("encode", help="Encode text to IDs")
    p_encode.add_argument("tokenizer", help="Path to saved tokenizer JSON")
    p_encode.add_argument("text", help="Text to encode")
    p_encode.set_defaults(func=cmd_encode)

    p_decode = sub.add_parser("decode", help="Decode IDs to text")
    p_decode.add_argument("tokenizer", help="Path to saved tokenizer JSON")
    p_decode.add_argument("ids", help="Comma-separated token IDs")
    p_decode.set_defaults(func=cmd_decode)

    p_sample = sub.add_parser("sample", help="Show sample tokenizations")
    p_sample.add_argument("tokenizer", help="Path to saved tokenizer JSON")
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
