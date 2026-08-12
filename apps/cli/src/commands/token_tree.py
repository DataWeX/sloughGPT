"""
Token-tree commands - train, encode, decode, and query a TokenTree tokenizer.

All commands are thin wrappers over the core ``domains.training.token_tree``
module (infrastructure before endpoints): training materializes BPE merges as
a tree, and every query handler (encode/decode/similar/lineage) descends the
tree or its Point-generated embeddings.
"""
import sys
from pathlib import Path

from core.printer import printer

from domains.training.token_tree import TokenTree


def _resolve_corpus_file(path_or_name: str) -> Path:
    """Resolve a corpus value (explicit path or dataset name) to a file.

    Args:
        path_or_name: explicit corpus path, or a bare name resolved against
            ``datasets/<name>/input.txt`` relative to the repo root.

    Returns:
        Resolved :class:`Path` to a readable corpus file.

    Side effects:
        - Prints an error and exits(2) when nothing resolves.
    """
    p = Path(path_or_name)
    if p.is_file():
        return p
    name = path_or_name.strip("/")
    for candidate in (Path("datasets") / name / "input.txt", Path(name)):
        if candidate.is_file():
            return candidate
    available = sorted(d.name for d in Path("datasets").glob("*") if d.is_dir())
    hint = f" Available datasets: {', '.join(available)}." if available else ""
    printer.error(f"Corpus not found: {path_or_name}.{hint}")
    sys.exit(2)


def _load_tree(tree_path: str) -> TokenTree:
    """Load a TokenTree from a saved base path (``<path>.meta.json``).

    Args:
        tree_path: base path previously passed to ``TokenTree.save``.

    Returns:
        The loaded :class:`TokenTree`.

    Side effects:
        - Prints an error and exits(2) when the meta sidecar is missing.
    """
    meta = Path(str(tree_path) + ".meta.json")
    if not meta.exists():
        printer.error(f"No token tree found at {tree_path} (missing {meta})")
        sys.exit(2)
    return TokenTree.load(tree_path)


def _resolve_token(tree: TokenTree, token: str) -> int:
    """Resolve a token id or literal token string to a vocabulary id.

    Args:
        tree: the token tree.
        token: an integer id, or a literal vocabulary token (special tokens
            and "</w>" are kept whole when they appear as written).

    Returns:
        vocabulary id.

    Side effects:
        - Prints an error and exits(2) on unresolvable input.
    """
    try:
        return int(token)
    except ValueError:
        pass
    if token.startswith("<") or token.endswith("</w>"):
        candidates = [token]
    else:
        candidates = [
            token + "</w>",
            " " + token + "</w>",
            token,
            " " + token,
        ]
    for candidate in candidates:
        if candidate in tree.stoi:
            return tree.stoi[candidate]
    printer.error(f"Token not in vocabulary: {token!r}")
    sys.exit(2)


def _print_stats(tree: TokenTree) -> None:
    """Print a compact stats block for a token tree.

    Args:
        tree: the token tree.

    Returns:
        None.
    """
    stats = tree.stats()
    printer.header(f"TokenTree ({stats['vocab_size']} tokens)")
    printer.status("trained", str(stats["trained"]), "ok" if stats["trained"] else "warn")
    printer.key_value("Merges", str(stats["num_merges"]))
    printer.key_value("Base tokens", str(stats["num_base_tokens"]))
    printer.key_value("Embed dim", str(stats["embed_dim"]))
    printer.key_value("Embedding points", str(stats["embedding_points"]))
    printer.key_value("Embedding compression", f"{stats['embedding_compression_ratio']}x")


def cmd_token_tree_train(args) -> None:
    """Train a TokenTree from a corpus and save it.

    Args:
        args: SimpleNamespace with ``corpus``, ``vocab_size``, ``embed_dim``,
            ``min_freq``, ``output``.

    Side effects:
        - Reads the corpus, trains, and writes ``<output>.meta.json`` +
          ``<output>.points.json``.
    """
    corpus = _resolve_corpus_file(args.corpus)
    text = corpus.read_text(encoding="utf-8", errors="replace")
    printer.step(f"Training TokenTree on {corpus.name} ({len(text) / 1e6:.1f} MB)...")

    tree = TokenTree().train(
        text,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        min_frequency=args.min_freq,
    )
    _print_stats(tree)

    meta_path, points_path = tree.save(args.output)
    printer.success(f"Saved {meta_path.name} + {points_path.name}")


def cmd_token_tree_encode(args) -> None:
    """Encode text into token ids (reads ``args.text`` or stdin).

    Args:
        args: SimpleNamespace with ``tree``, ``text``.

    Side effects:
        - Prints an id/token table.
    """
    tree = _load_tree(args.tree)
    text = args.text if args.text is not None else sys.stdin.read()
    ids = tree.encode(text)
    rows = []
    for i, tid in enumerate(ids):
        token = tree.itos.get(tid, "?")
        display = token.replace("</w>", "") if token not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>") else token
        rows.append([str(i), str(tid), display])
    printer.header(f"Encoding ({len(ids)} tokens)")
    printer.table(["#", "id", "token"], rows)
    printer.key_value("Round-trips", str(tree.decode(ids) == text.lower()))


def cmd_token_tree_decode(args) -> None:
    """Decode a comma-separated list of token ids back to text.

    Args:
        args: SimpleNamespace with ``tree``, ``ids`` (csv string).

    Side effects:
        - Prints the reconstructed text.
    """
    tree = _load_tree(args.tree)
    ids = [int(x) for x in args.ids.replace(" ", "").split(",") if x != ""]
    printer.header("Decoded")
    printer.info(tree.decode(ids))


def cmd_token_tree_stats(args) -> None:
    """Print training statistics for a saved token tree.

    Args:
        args: SimpleNamespace with ``tree``.

    Side effects:
        - Prints the stats block.
    """
    _print_stats(_load_tree(args.tree))


def cmd_token_tree_similar(args) -> None:
    """Find nearest-neighbor tokens via Point-generated embeddings.

    Args:
        args: SimpleNamespace with ``tree``, ``token``, ``top_k``.

    Side effects:
        - Prints a ranked similarity table.
    """
    tree = _load_tree(args.tree)
    token_id = _resolve_token(tree, args.token)
    if not tree.embedding_points():
        printer.error("No embedding points in this tree (trained with embed-dim 0)")
        sys.exit(2)
    query = tree.itos.get(token_id, str(token_id)).replace("</w>", "")
    results = tree.similar(token_id, top_k=args.top_k)
    printer.header(f"Nearest neighbors of {query!r}")
    rows = [
        [tree.itos.get(tid, "?").replace("</w>", ""), str(tid), f"{sim:.4f}"]
        for tid, sim in results
    ]
    printer.table(["token", "id", "similarity"], rows)


def cmd_token_tree_lineage(args) -> None:
    """Render a token's merge lineage down to its character leaves.

    Args:
        args: SimpleNamespace with ``tree``, ``token``.

    Side effects:
        - Prints the ASCII merge tree.
    """
    tree = _load_tree(args.tree)
    token_id = _resolve_token(tree, args.token)
    printer.header(f"Merge lineage of {tree.itos.get(token_id, '?')!r}")
    printer.info(tree.show_tree(token_id))
    printer.key_value("Leaves", " ".join(tree.decompose(token_id)))
