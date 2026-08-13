"""
Token-tree commands - train, encode, decode, and query a TokenTree tokenizer.

All commands are thin wrappers over the core ``domains.training.token_tree``
module (infrastructure before endpoints): training materializes BPE merges as
a tree, and every query handler (encode/decode/similar/lineage/embedding/
path) descends the tree or its Point-generated embeddings.
"""
import sys
from pathlib import Path

import numpy as np

from core.printer import printer

from domains.training.token_tree import TokenTree
from domains.training.token_tree_manager import get_token_tree_manager


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

    Delegates to ``TokenTree.resolve_token`` (core) and converts the
    resulting ``KeyError`` into a CLI error exit.

    Args:
        tree: the token tree.
        token: an integer id, or a literal vocabulary token.

    Returns:
        vocabulary id.

    Side effects:
        - Prints an error and exits(2) on unresolvable input.
    """
    try:
        return tree.resolve_token(token)
    except KeyError:
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


def cmd_token_tree_vocab(args) -> None:
    """List a paged slice of the vocabulary with flags and frequencies.

    Entries are printed in id order (special tokens, base characters, then
    merge tokens) so a fixed page size walks the whole vocabulary.

    Args:
        args: SimpleNamespace with ``tree``, ``offset``, ``limit``.

    Side effects:
        - Prints a vocab table with ``special``/``merged`` markers.
    """
    tree = _load_tree(args.tree)
    out = tree.vocab_entries(offset=args.offset, limit=args.limit)
    lo = args.offset + 1
    hi = min(args.offset + len(out["entries"]), out["total"])
    printer.header(f"Vocabulary ({out['total']} tokens)")
    rows = []
    for e in out["entries"]:
        flags = []
        if e["is_special"]:
            flags.append("special")
        if e["is_merged"]:
            flags.append("merged")
        rows.append([
            str(e["id"]),
            e["token"].replace("</w>", "") or e["token"],
            str(e["freq"]),
            "+".join(flags) if flags else "",
        ])
    printer.table(["id", "token", "freq", "flags"], rows)
    if out["entries"]:
        printer.info(f"Showing {lo}–{hi} of {out['total']}")


def cmd_token_tree_embedding(args) -> None:
    """Inspect a token's generated embedding vector.

    Reports dimensionality, L2 norm, and the largest-magnitude dimensions,
    mirroring the manager's ``embedding_info`` contract.

    Args:
        args: SimpleNamespace with ``tree``, ``token``, ``top_k``.

    Side effects:
        - Prints the embedding summary table.
    """
    tree = _load_tree(args.tree)
    token_id = _resolve_token(tree, args.token)
    vec = tree.embedding(token_id)
    if vec is None:
        printer.error("No embeddings in this tree (trained with embed-dim 0)")
        sys.exit(2)
    top_idx = np.argsort(-np.abs(vec))[: max(args.top_k, 1)]
    query = tree.itos.get(token_id, str(token_id)).replace("</w>", "")
    printer.header(f"Embedding of {query!r}")
    printer.key_value("id", str(token_id))
    printer.key_value("dim", str(vec.shape[0]))
    printer.key_value("L2 norm", f"{float(np.linalg.norm(vec)):.4f}")
    printer.key_value("Embedding points", str(tree.embedding_points()))
    printer.key_value("Compression", f"{tree.embedding_compression_ratio():.2f}x")
    rows = [
        [str(int(i)), f"{float(vec[i]):+.4f}"]
        for i in top_idx
    ]
    printer.table(["dim", "value"], rows)


def cmd_token_tree_path(args) -> None:
    """Trace the greedy trie walk the encoder takes over text.

    Prints each longest-prefix query step (remaining suffix, matched token,
    id, consumed chars) and the resulting token ids, mirroring the manager's
    ``path`` contract.

    Args:
        args: SimpleNamespace with ``tree``, ``text``.

    Side effects:
        - Prints the per-step table and final ids.
    """
    tree = _load_tree(args.tree)
    text = args.text if args.text is not None else sys.stdin.read()
    steps = tree.trace_path(text)
    printer.header(f"Path ({len(steps)} steps)")
    rows = []
    for s in steps:
        display = tree.itos.get(s["id"], "?").replace("</w>", "")
        rows.append([s["remaining"], display, str(s["id"]), str(s["consumed"])])
    printer.table(["remaining", "token", "id", "consumed"], rows)
    ids = [s["id"] for s in steps]
    printer.key_value("Ids", " ".join(str(i) for i in ids))
    printer.key_value("Round-trips", str(tree.decode(ids) == text.lower()))


def cmd_token_tree_matrix(args) -> None:
    """Summarize the full embedding matrix.

    Prints the matrix shape, row-norm distribution, live/dead token counts,
    and the most/least energetic tokens, mirroring the manager's
    ``matrix_summary`` contract.

    Args:
        args: SimpleNamespace with ``tree``, ``top_k``.

    Side effects:
        - Prints the matrix summary block.
    """
    tree = _load_tree(args.tree)
    stats = tree.embedding_matrix_stats(top_n=args.top_k)
    if stats["matrix"] is None:
        printer.error("No embeddings in this tree (trained with embed-dim 0)")
        sys.exit(2)
    rows, cols = stats["matrix"]
    printer.header(f"Embedding matrix ({rows} x {cols})")
    printer.key_value("L2 norm min", f"{stats['norm_min']:.4f}")
    printer.key_value("L2 norm mean", f"{stats['norm_mean']:.4f}")
    printer.key_value("L2 norm max", f"{stats['norm_max']:.4f}")
    printer.key_value(
        "Tokens", f"{stats['live_tokens']} live, {stats['dead_tokens']} dead"
    )

    def energy_rows(key: str):
        return [
            [tok.replace("</w>", ""), str(tid), f"{norm:.4f}"]
            for tok, tid, norm in stats[key]
        ]

    printer.header("Most energetic")
    printer.table(
        ["token", "id", "norm"], energy_rows("most_energetic")
    )
    printer.header("Least energetic")
    printer.table(
        ["token", "id", "norm"], energy_rows("least_energetic")
    )


def cmd_token_tree_compare(args) -> None:
    """Diff two saved token trees (by name) side by side.

    Delegates to the manager's ``compare`` so the current tree is untouched.
    Prints per-side stats, token/merge overlap counts, and the top shared and
    exclusive tokens by corpus frequency.

    Args:
        args: SimpleNamespace with ``a``, ``b``, ``top_n``.

    Side effects:
        - Prints the comparison block; exits(2) on error.
    """
    try:
        out = get_token_tree_manager().compare(args.a, args.b, top_n=args.top_n)
    except (FileNotFoundError, ValueError) as e:
        printer.error(str(e))
        sys.exit(2)

    a, b = out["a"], out["b"]
    printer.header(f"Compare {a['name']!r} vs {b['name']!r}")

    def stat_line(side: dict) -> str:
        s = side["stats"]
        return (
            f"vocab {s['vocab_size']} · merges {s['num_merges']} · "
            f"base {s['num_base_tokens']} · embed_dim {s['embed_dim']} · "
            f"points {s['embedding_points']}"
        )

    printer.key_value("A", f"{a['name']} — {stat_line(a)}")
    printer.key_value("B", f"{b['name']} — {stat_line(b)}")

    printer.header("Vocabulary overlap")
    printer.key_value("Shared tokens", str(out["shared_tokens"]))
    printer.key_value("Only in A", str(out["only_a_tokens"]))
    printer.key_value("Only in B", str(out["only_b_tokens"]))
    printer.key_value("Shared merges", str(out["shared_merges"]))
    printer.key_value("Only in A merges", str(out["only_a_merges"]))
    printer.key_value("Only in B merges", str(out["only_b_merges"]))

    def token_table(key: str, title: str) -> None:
        rows = [[t.replace("</w>", ""), str(f)] for t, f in out[key]]
        if rows:
            printer.header(title)
            printer.table(["token", "freq"], rows)

    token_table("shared_examples", "Top shared tokens")
    token_table("only_a_examples", f"Top tokens only in {a['name']}")
    token_table("only_b_examples", f"Top tokens only in {b['name']}")


def cmd_token_tree_merges(args) -> None:
    """List the most frequent BPE merge rules of a saved tree.

    When ``args.query`` is given, filters rules whose parts contain the
    substring (case-insensitive), keeping their global frequency rank.

    Args:
        args: SimpleNamespace with ``tree``, ``top_n``, ``query``.

    Side effects:
        - Prints a ranked merge table.
    """
    tree = _load_tree(args.tree)
    if args.query:
        data = tree.search_merges(query=args.query, limit=args.top_n)
    else:
        data = tree.top_merges(top_n=args.top_n)
    rows = [
        [
            str(m["rank"]),
            f"{m['left']!r} + {m['right']!r}",
            m["token"].replace("</w>", "") or m["token"],
            str(m["count"]),
        ]
        for m in data
    ]
    printer.header(f"Merges ({len(data)} shown)")
    printer.table(["rank", "pair", "token", "count"], rows)


def cmd_token_tree_saved(args) -> None:
    """List saved token trees (from the manager's save dir), newest first.

    Args:
        args: SimpleNamespace (unused).

    Side effects:
        - Prints a table of saved trees with vocab/merge counts.
    """
    saved = get_token_tree_manager().list_saved()
    if not saved:
        printer.info("No saved token trees.")
        return
    rows = []
    for t in saved:
        rows.append([
            t["name"],
            str(t["vocab_size"]),
            str(t["num_merges"]),
            t["path"],
        ])
    printer.header("Saved token trees")
    printer.table(["name", "vocab", "merges", "path"], rows)


def cmd_token_tree_save(args) -> None:
    """Save the current tree (or a tree loaded from ``args.tree``) by name.

    Args:
        args: SimpleNamespace with ``name`` and optional ``tree`` (base path).

    Side effects:
        - Writes ``<name>.meta.json`` + ``<name>.points.json`` into the
          manager's save dir; exits(2) on invalid name or missing tree.
    """
    mgr = get_token_tree_manager()
    try:
        if args.tree:
            meta_path = Path(str(args.tree) + ".meta.json")
            if not meta_path.exists():
                printer.error(f"No token tree found at {args.tree} (missing {meta_path})")
                sys.exit(2)
            mgr.adopt(TokenTree.load(args.tree))
        out = mgr.save(args.name)
    except ValueError as e:
        printer.error(str(e))
        sys.exit(2)
    printer.success(f"Saved {out['name']!r} ({out['vocab_size']} vocab, {out['num_merges']} merges)")


def cmd_token_tree_load(args) -> None:
    """Load a saved tree by name and make it the manager's current tree.

    Args:
        args: SimpleNamespace with ``name``.

    Side effects:
        - Replaces the manager's current tree; exits(2) on missing name.
    """
    mgr = get_token_tree_manager()
    try:
        out = mgr.load(args.name)
    except (FileNotFoundError, ValueError) as e:
        printer.error(str(e))
        sys.exit(2)
    printer.success(
        f"Loaded {out['name']!r} ({out['vocab_size']} vocab, {out['num_merges']} merges)"
    )


def cmd_token_tree_delete(args) -> None:
    """Delete a saved tree by name.

    Args:
        args: SimpleNamespace with ``name``.

    Side effects:
        - Removes the tree's sidecar files; exits(2) on missing name.
    """
    mgr = get_token_tree_manager()
    try:
        deleted = mgr.delete_saved(args.name)
    except ValueError as e:
        printer.error(str(e))
        sys.exit(2)
    if not deleted:
        printer.error(f"No saved token tree named {args.name!r}")
        sys.exit(2)
    printer.success(f"Deleted {args.name!r}")
