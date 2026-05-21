"""Eval Dashboard — browse LoRA evaluation results and benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rich.text import Text
from rich.table import Table
from rich.box import SIMPLE

from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


PAGE_SIZE = 10


class EvalScreen(Screen):
    name = "eval"
    bindings = [
        Binding(["r"], "refresh", "eval"),
        Binding(["n"], "next page", "__page_down"),
        Binding(["p"], "prev page", "__page_up"),
    ]

    def __init__(self):
        super().__init__()
        self.page: int = 0

    @staticmethod
    def _scan_eval_results(repo_root: Path) -> List[dict]:
        eval_dir = repo_root / "data" / "eval_results"
        if not eval_dir.is_dir():
            return []
        results = []
        seen = set()
        for f in sorted(eval_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix != ".json" or f.name == "slonet_benchmark.json":
                continue
            base_stem = f.stem.replace("_detail", "")
            if base_stem in seen:
                continue
            seen.add(base_stem)
            if "_detail" not in f.name:
                continue
            try:
                data = json.loads(f.read_text())
                summary = data.get("summary", data)
                kind = "baseline" if "baseline" in f.name else "aggregated" if "aggregated" in f.name else "other"
                results.append({
                    "name": base_stem, "kind": kind,
                    "perplexity": summary.get("perplexity"),
                    "bleu": summary.get("bleu"),
                    "tokens_per_sec": summary.get("tokens_per_sec"),
                    "personality_score": summary.get("personality_score"),
                    "timestamp": summary.get("timestamp", ""),
                    "prompts": summary.get("prompts", 0),
                })
            except Exception:
                pass
        return results

    @staticmethod
    def _scan_adapter_eval_txt(repo_root: Path) -> List[dict]:
        ua_dir = repo_root / "data" / "user_adapters"
        if not ua_dir.is_dir():
            return []
        results = []
        for f in sorted(ua_dir.glob("*_eval.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                text = f.read_text()
                verdict = ""
                ppl, bleu = None, None
                for line in text.splitlines():
                    if line.startswith("VERDICT:"):
                        verdict = line.replace("VERDICT:", "").strip()
                    if "Perplexity:" in line:
                        parts = line.split()
                        for p in parts:
                            if "→" in p:
                                ppl = p.split("→")[-1].strip()
                    if "BLEU:" in line:
                        parts = line.split()
                        for p in parts:
                            if "→" in p:
                                bleu = p.split("→")[-1].strip()
                results.append({"name": f.stem.replace("_eval", ""), "verdict": verdict, "ppl": ppl, "bleu": bleu})
            except Exception:
                pass
        return results[:10]

    def render(self, session: TuiSession) -> str:
        self.render_header("Eval Dashboard", "data/eval_results  ·  data/user_adapters")

        bm_path = session.repo_root / "data" / "eval_results" / "slonet_benchmark.json"
        if bm_path.is_file():
            try:
                bm = json.loads(bm_path.read_text())
                s = bm.get("summary", {})
                passed, total = s.get("passed", 0), s.get("total", 0)
                c = Color.SUCCESS if passed == total else Color.ERROR
                CONSOLE.print(f"  [{c}]●[/]  SloNet Benchmark:  [{Color.WHITE}]{passed}/{total}[/] tests passed")
                CONSOLE.print()
            except Exception:
                pass

        all_results = self._scan_eval_results(session.repo_root)
        total = len(all_results)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.page >= pages:
            self.page = pages - 1

        start = self.page * PAGE_SIZE
        results = all_results[start:start + PAGE_SIZE]

        if results:
            table = Table(show_header=True, box=SIMPLE, border_style=Color.BORDER, padding=(0, 1))
            table.add_column("Run", style=Color.WHITE, width=18)
            table.add_column("Kind", style=Color.MUTED, width=12)
            table.add_column("PPL", style=Color.MUTED, width=8)
            table.add_column("BLEU", style=Color.MUTED, width=8)
            table.add_column("tok/s", style=Color.MUTED, width=7)
            table.add_column("Pers", style=Color.MUTED, width=6)
            for r in results:
                kc = Color.PRIMARY if r["kind"] == "baseline" else Color.SECONDARY if r["kind"] == "aggregated" else Color.MUTED
                table.add_row(
                    r["name"][:18], f"[{kc}]{r['kind']}[/]",
                    f"{r['perplexity']:.1f}" if r["perplexity"] else "-",
                    f"{r['bleu']:.1f}" if r["bleu"] else "-",
                    f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "-",
                    f"{r['personality_score']:.2f}" if r["personality_score"] else "-",
                )
            CONSOLE.print(table)
            if total > PAGE_SIZE:
                CONSOLE.print(Text(f"  Page {self.page + 1}/{pages}  ({total} total)", style=Color.MUTED))
            CONSOLE.print()

        adapters = self._scan_adapter_eval_txt(session.repo_root)
        if adapters:
            CONSOLE.print(Text("  Adapter Eval Reports", style=f"bold {Color.PRIMARY}"))
            CONSOLE.print()
            for a in adapters:
                vc = Color.SUCCESS if "IMPROVED" in a["verdict"] else Color.WARNING if "MIXED" in a["verdict"] else Color.MUTED
                detail = ""
                if a["ppl"]:
                    detail += f"  PPL→{a['ppl']}"
                if a["bleu"]:
                    detail += f"  BLEU→{a['bleu']}"
                CONSOLE.print(f"    [{Color.MUTED}]▪[/]  {a['name']:<28} [{vc}]{a['verdict']}[/]{detail}")
            CONSOLE.print()

        self.render_footer()
        return self._handle_input(pages)

    def _handle_input(self, pages: int) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action

            val = next((b.action for b in self.bindings if key in b.keys), None)

            if val == "__page_down" and self.page + 1 < pages:
                self.page += 1
            elif val == "__page_up" and self.page > 0:
                self.page -= 1
            elif key == readchar.key.ESC:
                return "home"
            else:
                return "eval"

            return "eval"
