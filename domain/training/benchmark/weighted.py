"""Weighted programmable benchmark — OSBench-like scoring.

Each run produces raw metrics (latency, recall, quality, memory...).
A BenchProgram defines weights + programmable logic to compute a single
0-100 score. Logic is safe-evaluated via `asteval` subset (no import/exec).

Example yaml:
  metrics:
    - name: latency_p99
      weight: 0.4
      norm: lower-better  # lower is better
      lo: 0
      hi: 500  # ms
    - name: recall_at_5
      weight: 0.3
      norm: higher-better
    - name: quality
      weight: 0.3
      norm: higher-better
  logic: "0.4*norm_latency_p99 + 0.3*recall_at_5 + 0.3*quality | if recall_at_5 < 0.9 then 0"
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricWeight:
    name: str
    weight: float
    norm: str = "higher-better"  # or "lower-better"
    lo: float | None = None
    hi: float | None = None

    def normalize(self, value: float) -> float:
        """Map raw value to 0-1 via lo/hi clamp."""
        if self.lo is None or self.hi is None:
            # no normalization — assume 0-1 already
            return max(0.0, min(1.0, float(value)))
        lo, hi = self.lo, self.hi
        if hi == lo:
            return 0.0
        if self.norm == "lower-better":
            # lower is better: invert
            n = (hi - float(value)) / (hi - lo)
        else:
            n = (float(value) - lo) / (hi - lo)
        return max(0.0, min(1.0, n))


@dataclass
class BenchProgram:
    metrics: list[MetricWeight] = field(default_factory=list)
    logic: str = ""  # e.g. "0.4*norm_latency + 0.3*recall"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchProgram:
        metrics = [MetricWeight(**m) for m in data.get("metrics", [])]
        logic = data.get("logic", "")
        # auto-generate logic from weights if not provided
        if not logic and metrics:
            parts = [f"{m.weight}*norm_{m.name}" for m in metrics]
            logic = " + ".join(parts)
        return cls(metrics=metrics, logic=logic)

    @classmethod
    def from_yaml(cls, path: Path) -> BenchProgram:
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [
                {"name": m.name, "weight": m.weight, "norm": m.norm, "lo": m.lo, "hi": m.hi}
                for m in self.metrics
            ],
            "logic": self.logic,
        }

    def score(self, results: dict[str, float]) -> float:
        """Compute 0-100 score from raw results.

        Steps:
          1. Normalize each metric via MetricWeight
          2. Evaluate `logic` with vars {metric: raw, norm_metric: normalized}
          3. Apply conditional suffix `| if <expr> then <value>` (optional)
          4. Clamp 0-100
        """
        # Build normalized map
        norm_map: dict[str, float] = {}
        for m in self.metrics:
            raw = float(results.get(m.name, 0.0))
            norm_map[f"norm_{m.name}"] = m.normalize(raw)

        # Combine raw + norm for logic evaluation
        env: dict[str, float] = {}
        env.update({k: float(v) for k, v in results.items()})
        env.update(norm_map)

        # Handle conditional suffix: "expr | if cond then val"
        logic = self.logic.strip()
        cond_val: float | None = None
        cond_expr: str | None = None
        if "| if" in logic:
            base, cond_part = logic.split("| if", 1)
            logic = base.strip()
            # cond_part: " recall < 0.9 then 0"
            if "then" in cond_part:
                cond_expr, then_expr = cond_part.split("then", 1)
                cond_expr = cond_expr.strip()
                then_expr = then_expr.strip()
                try:
                    if _safe_eval(cond_expr, env):
                        cond_val = float(_safe_eval(then_expr, env))
                except Exception as e:
                    logger.debug("BenchProgram cond eval %s: %s", cond_expr, e)

        if cond_val is not None:
            return max(0.0, min(100.0, cond_val * 100 if cond_val <= 1 else cond_val))

        if not logic:
            # fallback: weighted average of normalized
            total = sum(m.weight for m in self.metrics) or 1
            weighted = (
                sum(norm_map.get(f"norm_{m.name}", 0) * m.weight for m in self.metrics) / total
            )
            return max(0.0, min(100.0, weighted * 100))

        try:
            val = _safe_eval(logic, env)
            # if result 0-1, scale to 0-100 for common case
            if 0 <= val <= 1:
                val *= 100
            return max(0.0, min(100.0, float(val)))
        except Exception as e:
            logger.warning("BenchProgram logic eval failed %r: %s", logic, e)
            return 0.0


def _safe_eval(expr: str, env: dict[str, float]) -> Any:
    """Safe eval: only arithmetic, comparison, boolean, numbers, vars."""
    tree = ast.parse(expr, mode="eval")
    # Allow only safe nodes
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"disallowed node {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Name) and node.id not in env and node.id not in ("True", "False"):
            # missing var -> 0
            env[node.id] = 0.0
    code = compile(tree, "<bench>", "eval")
    return eval(code, {"__builtins__": {}}, env)


# ── Runner helper ───────────────────────────────────────────────────
def score_benchmarks(program: BenchProgram, results: dict[str, float]) -> dict[str, Any]:
    """Score and return breakdown."""
    total = program.score(results)
    breakdown: dict[str, float] = {}
    for m in program.metrics:
        raw = float(results.get(m.name, 0))
        norm = m.normalize(raw)
        breakdown[m.name] = {"raw": raw, "norm": norm, "weight": m.weight}
    return {"score": total, "breakdown": breakdown, "logic": program.logic}
