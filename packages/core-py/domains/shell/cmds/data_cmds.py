"""datasets / knowledge / remember / recall / checkpoints / finetuned / tokenizer — data & training commands."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands

help = "List datasets, knowledge, checkpoints, or fine-tuned models"
names = ["datasets", "knowledge", "remember", "recall", "checkpoints", "finetuned", "tokenizer"]


def _dict_val(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k, default)
        if v:
            return str(v)
    return default


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    cmd = argv[0] if argv else "datasets"

    if cmd == "datasets":
        with out.spinner("Fetching datasets") as s:
            datasets = api.datasets()
        s.ok("Datasets loaded")
        if not datasets:
            out.print("  No datasets available")
            return 0
        rows = []
        for d in datasets:
            name = d.get("name", "?")
            samples = d.get("samples", 0)
            sz = d.get("size", 0)
            sz_str = f"{sz / 1048576:.1f}M" if sz else ""
            rows.append([name, str(samples), sz_str])
        out.table(rows, ["Dataset", "Samples", "Size"])
        return 0

    if cmd == "knowledge":
        query = " ".join(argv[1:]) if len(argv) > 1 else ""
        if query:
            with out.spinner("Searching knowledge") as s:
                results = api.list_knowledge(query)
            s.ok("Search complete")
            if not results:
                out.print("  No results")
                return 0
            for r in results[:20]:
                out.print(f"  \u2022 {r.get('content', '')[:120]}")
            return 0
        with out.spinner("Fetching knowledge stats") as s:
            stats = api.knowledge_stats()
        s.ok("Knowledge stats loaded")
        count = stats.get("total_items", 0)
        if count == 0:
            out.print("  Knowledge base is empty")
            out.print("  Use: remember <fact>  to add a fact")
            return 0
        out.print(f"  Knowledge base: {count} fact(s)")
        topics = stats.get("topics", {})
        if topics:
            out.print(f"  Topics: {', '.join(sorted(topics.keys()))}")
        return 0

    if cmd == "remember":
        content = " ".join(argv[1:]) if len(argv) > 1 else ""
        if not content:
            out.print("  Usage: remember <fact>")
            out.print("    remember this project uses FastAPI")
            return 1
        with out.spinner("Storing fact") as s:
            result = api.add_knowledge(content)
        if isinstance(result, dict) and result.get("status") == "stored":
            topic = result.get("topic", "general")
            preview = content[:80].replace("\n", "\\n")
            s.ok(f"Stored fact [{topic}]")
            out.print(f"  {preview}...")
        else:
            s.fail("Failed to store")
            out.print(f"  Error: {result}")
        return 0

    if cmd == "recall":
        query = " ".join(argv[1:]) if len(argv) > 1 else ""
        if not query:
            with out.spinner("Fetching knowledge stats") as s:
                stats = api.knowledge_stats()
            s.ok("Knowledge stats loaded")
            count = stats.get("total_items", 0)
            if count == 0:
                out.print("  Knowledge base is empty")
                return 0
            out.print(f"  Knowledge base: {count} fact(s)")
            topics = stats.get("topics", {})
            if topics:
                out.print(f"  Topics: {', '.join(sorted(topics.keys()))}")
            out.print("  Use: recall <query>  to search")
            return 0
        with out.spinner("Searching knowledge") as s:
            results = api.list_knowledge(query)
        s.ok("Search complete")
        if not results:
            out.print("  No matching facts")
            return 0
        for r in results[:10]:
            topic = r.get("topic", "")
            content = r.get("content", "")[:120]
            out.print(f"  [{topic}] {content}")
        return 0

    if cmd == "checkpoints":
        with out.spinner("Fetching checkpoints") as s:
            cps = api.checkpoints()
        s.ok("Checkpoints loaded")
        if not cps:
            out.print("  No checkpoints")
            return 0
        rows = []
        for cp in cps:
            rows.append([
                cp.get("name", ""),
                f"{cp.get('loss', '\u2014')}",
                cp.get("model_type", ""),
            ])
        out.table(rows, ["Checkpoint", "Loss", "Type"])
        return 0

    if cmd == "finetuned":
        with out.spinner("Fetching fine-tuned models") as s:
            models = api.finetuned_models()
        s.ok("Fine-tuned models loaded")
        if not models:
            out.print("  No fine-tuned models")
            return 0
        rows = []
        for m in models:
            name = m.get("model_name", "")
            loss = m.get("final_loss", "\u2014")
            ep = m.get("epochs", 0)
            sz_bytes = m.get("size_bytes", 0)
            sz_str = f"{sz_bytes / 1048576:.0f}M"
            rows.append([name, f"{loss}", f"{ep}ep", sz_str])
        out.table(rows, ["Model", "Loss", "Epochs", "Size"])
        return 0

    if cmd == "tokenizer":
        with out.spinner("Fetching tokenizer stats") as s:
            stats = api.tokenizer_stats()
        s.ok("Tokenizer stats loaded")
        if isinstance(stats, dict) and "error" not in stats:
            for k, v in stats.items():
                out.print(f"  {k}: {v}")
        else:
            out.json(stats)
        return 0

    return 0
