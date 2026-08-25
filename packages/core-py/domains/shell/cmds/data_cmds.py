"""Data-related shell commands: datasets, checkpoints, finetuned, knowledge, remember, recall, tokenizer.

Extracted from monolithic REPL; follows the cmds/ protocol:
    def run(argv, out, api, env) -> int
"""
from __future__ import annotations

help = "Manage datasets, knowledge, checkpoints, and more"
names = ["datasets", "checkpoints", "finetuned", "knowledge", "remember", "recall", "tokenizer"]


def _format_error(e: Exception, cmd: str = "") -> str:
    """Format an exception into a user-friendly error message."""
    from domains.shell.error import format_error
    return format_error(e, cmd, color=False)


def run(argv: list[str], out, api, env: dict) -> int:
    cmd = argv[0] if argv else "datasets"
    args = argv[1:]

    handlers = {
        "datasets": _datasets,
        "checkpoints": _checkpoints,
        "finetuned": _finetuned,
        "knowledge": _knowledge,
        "remember": _remember,
        "recall": _recall,
        "tokenizer": _tokenizer,
    }

    handler = handlers.get(cmd, _datasets)
    return handler(args, out, api)


def _datasets(args, out, api):
    try:
        data = api.datasets()
    except Exception as e:
        out.write(_format_error(e, "datasets"))
        return 1
    if not data:
        out.write("No datasets found.")
        return 0
    for ds in data:
        name = ds.get("name", "?")
        samples = ds.get("samples", "?")
        size = ds.get("size", 0)
        if isinstance(size, (int, float)) and size > 0:
            if size >= 1048576:
                size_str = f"{size / 1048576:.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
        else:
            size_str = "-"
        out.write(f"  {name:20s}  {samples} samples  {size_str}")
    return 0


def _checkpoints(args, out, api):
    try:
        data = api.checkpoints()
    except Exception as e:
        out.write(_format_error(e, "checkpoints"))
        return 1
    if not data:
        out.write("No checkpoints found.")
        return 0
    for cp in data:
        name = cp.get("name", "?")
        loss = cp.get("loss", "?")
        mtype = cp.get("model_type", "?")
        out.write(f"  {name:20s}  loss={loss}  type={mtype}")
    return 0


def _finetuned(args, out, api):
    if not args:
        try:
            data = api.finetuned_models()
        except Exception as e:
            out.write(_format_error(e, "finetuned"))
            return 1
        if not data:
            out.write("No fine-tuned models found.")
            return 0
        for m in data:
            name = m.get("model_name", "?")
            loss = m.get("final_loss", "?")
            epochs = m.get("epochs", "?")
            size = m.get("size_bytes", 0)
            if isinstance(size, (int, float)) and size >= 1048576:
                size_str = f"{size / 1048576:.1f} MB"
            else:
                size_str = f"{size} B"
            out.write(f"  {name:20s}  loss={loss}  epochs={epochs}  {size_str}")
        return 0

    sub = args[0]
    if sub == "load":
        if len(args) < 2:
            out.write("Usage: finetuned load <name>")
            return 1
        name = args[1]
        try:
            result = api.load_finetuned(name)
        except Exception as e:
            out.write(_format_error(e, "finetuned load"))
            return 1
        status = result.get("status", "error")
        if status == "loaded":
            out.write(f"Loaded: {name}")
            return 0
        out.write(f"Failed: {result.get('error', status)}")
        return 1

    if sub in ("rm", "delete", "del"):
        if len(args) < 2:
            out.write(f"Usage: finetuned {sub} <name>")
            return 1
        name = args[1]
        try:
            result = api.delete_finetuned(name)
        except Exception as e:
            out.write(_format_error(e, "finetuned rm"))
            return 1
        status = result.get("status", "error")
        if status == "deleted":
            out.write(f"Deleted: {name}")
            return 0
        out.write(f"Failed: {status}")
        return 1

    out.write(f"Unknown subcommand: {sub}")
    return 1


def _knowledge(args, out, api):
    if args:
        return _knowledge_search(args, out, api)
    try:
        stats = api.knowledge_stats()
    except Exception as e:
        out.write(_format_error(e, "knowledge"))
        return 1
    total = stats.get("total_items", 0)
    topics = stats.get("topics", {})
    out.write(f"Knowledge: {total} items")
    if topics:
        for topic, count in topics.items():
            out.write(f"  {topic}: {count}")
    return 0


def _knowledge_search(args, out, api):
    query = " ".join(args)
    try:
        results = api.list_knowledge(query)
    except Exception as e:
        out.write(_format_error(e, "knowledge search"))
        return 1
    if not results:
        out.write(f"No results for '{query}'.")
        return 0
    for item in results:
        content = item.get("content", "?")
        out.write(f"  - {content}")
    return 0


def _remember(args, out, api):
    if not args:
        out.write("Usage: remember <fact>")
        return 1
    fact = " ".join(args)
    try:
        result = api.add_knowledge(fact)
    except Exception as e:
        out.write(_format_error(e, "remember"))
        return 1
    status = result.get("status", "error")
    if status == "stored":
        out.write("Remembered.")
        return 0
    out.write(f"Could not store: {status}")
    return 0


def _recall(args, out, api):
    if not args:
        try:
            stats = api.knowledge_stats()
        except Exception as e:
            out.write(_format_error(e, "recall"))
            return 1
        total = stats.get("total_items", 0)
        if total == 0:
            out.write("No facts stored.")
            return 0
        out.write(f"{total} facts stored. Use 'recall <query>' to search.")
        return 0
    return _knowledge_search(args, out, api)


def _tokenizer(args, out, api):
    try:
        stats = api.tokenizer_stats()
    except Exception as e:
        out.write(_format_error(e, "tokenizer"))
        return 1
    if "error" in stats:
        out.write(f"Tokenizer error: {stats['error']}")
        return 0
    vocab = stats.get("vocab_size", "?")
    merges = stats.get("merges", "?")
    out.write(f"Vocab: {vocab}  Merges: {merges}")
    return 0
