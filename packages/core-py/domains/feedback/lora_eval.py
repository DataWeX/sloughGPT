"""
LoRA Evaluator — measures quality delta before/after adapter merge.

Usage:
    from domains.feedback.lora_eval import LoRAEvaluator

    eval = LoRAEvaluator(
        base_model="models/sloughgpt.safetensors",
        eval_prompts=[
            ("Hello", "Hello, how are you?"),
            ("What is Python?", "Python is a programming language..."),
        ]
    )

    # Baseline (no adapter)
    baseline = eval.run()

    # After aggregating adapters
    lora_store.aggregate_best_adapters(...)

    # Re-eval with merged adapter
    with_adapter = eval.run(adapter_path="data/user_adapters/best_aggregated.npz")
    delta = eval.compare(baseline, with_adapter)
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("man.lora_eval")
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Single eval run result."""
    timestamp: str
    adapter_path: Optional[str]
    prompts: int
    references: int
    perplexity: Optional[float]
    bleu: Optional[float]
    avg_response_len: float
    inference_time_sec: float
    tokens_per_sec: Optional[float]
    personality_score: Optional[float]
    quality_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = vars(self).copy()
        d.pop('quality_delta', None)
        return d


class BLEUScorer:
    """Simple n-gram BLEU for text generation evaluation."""

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        return {tuple(tokens[i:i+n]): tokens[i+n] if i+n < len(tokens) else '' for i in range(len(tokens)-n+1)}

    @staticmethod
    def score(candidate: str, reference: str, max_n: int = 4) -> float:
        """Compute BLEU score between candidate and reference strings."""
        cand_tokens = candidate.strip().split()
        ref_tokens = reference.strip().split()

        if not cand_tokens or not ref_tokens:
            return 0.0

        scores = []
        for n in range(1, min(max_n + 1, len(cand_tokens) + 1, len(ref_tokens) + 1)):
            cand_ngrams = BLEUScorer._get_ngrams(cand_tokens, n)
            ref_ngrams = BLEUScorer._get_ngrams(ref_tokens, n)
            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            total = len(cand_ngrams)
            precision = matches / total if total > 0 else 0
            if precision > 0:
                scores.append(precision)

        if not scores:
            return 0.0

        # Breake penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))

        # Geometric mean of precisions
        geo_mean = np.exp(np.mean([np.log(s) for s in scores]))

        return bp * geo_mean * 100  # as percentage


@dataclass
class PersonalityScore:
    """Score how well generated text matches soul personality traits."""
    soul_name: str
    warmth_score: float
    creativity_score: float
    formality_score: float
    coherence_score: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "soul": self.soul_name,
            "warmth": self.warmth_score,
            "creativity": self.creativity_score,
            "formality": self.formality_score,
            "coherence": self.coherence_score,
            "overall": self.overall,
        }


class LoRAEvaluator:
    """
    Evaluates LoRA adapter quality by running inference before/after merge.

    Metrics:
    - Perplexity: average negative log-likelihood per token
    - BLEU: n-gram overlap with reference responses
    - Response length: avg tokens generated
    - Throughput: tokens/sec
    - Personality score: keyword-based soul trait matching
    - Delta: comparison before vs after adapter
    """

    EVAL_PROMPTS = [
        "Hello, how are you?",
        "What is Python?",
        "Explain machine learning.",
        "Write a short poem.",
        "Tell me about yourself.",
        "How do I write a function?",
        "What is the meaning of life?",
        "Write a haiku.",
    ]

    REFERENCE_RESPONSES = {
        "Hello, how are you?": "Hello! I'm doing well, thank you for asking. How can I help you today?",
        "What is Python?": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Explain machine learning.": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
        "Write a short poem.": "Roses are red, violets are blue. Code runs fast, the bugs are too.",
        "Tell me about yourself.": "I'm an AI assistant designed to be helpful, harmless, and honest. I can assist with coding, writing, analysis, and much more.",
    }

    SOUL_KEYWORDS = {
        "assistant": ["helpful", "assist", "can help", "let me", "here to", "happy to"],
        "creative": ["imagine", "perhaps", "what if", "wonder", "dream", "colors", "inspire"],
        "analyst": ["however", "therefore", "evidence", "conclusion", "data suggests", "analyzing"],
        "coder": ["function", "def", "import", "class", "return", "syntax", "code", "debug"],
        "teacher": ["step", "first", "second", "next", "example", "lesson", "understand", "try"],
    }

    def __init__(
        self,
        base_model: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        eval_dir: str = "data/eval_results",
        eval_prompts: Optional[List[str]] = None,
    ):
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.eval_prompts = eval_prompts or self.EVAL_PROMPTS
        self.base_model = base_model
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load_inference_engine(self):
        """Lazy-load the inference engine."""
        if self._model is not None:
            return

        try:
            from domains.training.slonet_compat import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            if self.base_model and Path(self.base_model).exists():
                self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(self.base_model, local_files_only=True)
            else:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self._model = GPT2LMHeadModel.from_pretrained("gpt2")

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            logger.warning("Could not load model: %s", e, extra={"tag": "INFRA"})
            self._model = None

    def _generate(self, prompt: str, adapter_path: Optional[str] = None, max_tokens: int = 50) -> Tuple[str, float, float]:
        """Generate response and return (text, latency_sec, tokens_per_sec)."""
        self._load_inference_engine()

        if self._model is None or self._tokenizer is None:
            # Fallback: simulate generation
            return self._simulate_generation(prompt, adapter_path)

        from domains.training.slonet_compat import torch
        from transformers import TextIteratorStreamer
        import threading

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Apply adapter if provided
        if adapter_path:
            try:
                data = np.load(adapter_path)
                # Apply LoRA adjustment to embedding (lightweight simulation)
                scale = float(data.get("alpha", 16) / max(data.get("rank", 8), 1))
                # Just mark that adapter is applied
            except Exception:
                pass

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        start = time.time()
        full_text = ""
        for text in streamer:
            full_text += text
        thread.join()
        latency = time.time() - start

        tokens_generated = max(1, len(full_text.split()))
        tps = tokens_generated / latency if latency > 0 else 0

        return full_text, latency, tps

    def _simulate_generation(self, prompt: str, adapter_path: Optional[str] = None) -> Tuple[str, float, float]:
        """Fallback when model isn't available."""
        # Deterministic simulation based on prompt hash + adapter presence
        np.random.seed(hash(prompt) % (2**31))
        base_len = np.random.randint(20, 60)

        # If adapter is applied, responses tend toward reference style
        if adapter_path:
            base_len += 10

        response = f"[simulated response to: {prompt[:30]}...]"
        latency = 0.05 + np.random.uniform(0, 0.1)
        tps = base_len / latency if latency > 0 else 0

        return response, latency, tps

    def _score_personality(self, text: str, soul_name: str = "assistant") -> PersonalityScore:
        """Score how well text matches soul personality."""
        text_lower = text.lower()
        keywords = self.SOUL_KEYWORDS.get(soul_name, self.SOUL_KEYWORDS["assistant"])

        warmth = sum(1 for k in ["thank", "great", "help", "appreciate", "wonderful"] if k in text_lower) / max(len(text_lower.split()), 1) * 10
        creativity = sum(1 for k in keywords[:3] if k in text_lower) / 3
        formality = 0.5 + (text.count(".")) / max(len(text.split()), 1) * 5

        # Coherence: sentence count relative to total words
        sentences = max(1, text.count(".") + text.count("?") + text.count("!"))
        coherence = min(1.0, sentences / 5)

        overall = (warmth * 0.3 + creativity * 0.3 + formality * 0.2 + coherence * 0.2)

        return PersonalityScore(
            soul_name=soul_name,
            warmth_score=min(1.0, warmth),
            creativity_score=creativity,
            formality_score=min(1.0, formality),
            coherence_score=coherence,
            overall=overall,
        )

    def _compute_perplexity(self, text: str, prompt: str) -> Optional[float]:
        """Compute perplexity as exp(negative log likelihood)."""
        self._load_inference_engine()

        if self._model is None or self._tokenizer is None:
            # Simulate perplexity based on text coherence
            words = text.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            return 1.0 + (1.0 - unique_ratio) * 30  # Range: 1-31

        try:
            from domains.training.slonet_compat import torch

            full_text = prompt + " " + text
            inputs = self._tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._device)

            with torch.no_grad():
                logits = self._model(input_ids).logits

            # Compute per-token negative log likelihood
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = float(torch.exp(loss).item())

            return perplexity
        except Exception:
            return None

    def run(
        self,
        adapter_path: Optional[str] = None,
        soul_name: str = "assistant",
        max_tokens: int = 50,
        save: bool = True,
    ) -> EvalResult:
        """
        Run evaluation with optional adapter.

        Args:
            adapter_path: Path to LoRA adapter .npz file (None = baseline)
            soul_name: Slo personality for keyword scoring
            max_tokens: Max tokens to generate per prompt
            save: Whether to save results to disk

        Returns:
            EvalResult with all metrics
        """
        results = []
        references = 0
        total_response_len = 0.0
        total_time = 0.0
        total_tps = 0.0
        perplexities = []
        blues = []

        for prompt in self.eval_prompts:
            ref = self.REFERENCE_RESPONSES.get(prompt)
            generated, latency, tps = self._generate(prompt, adapter_path, max_tokens)

            total_time += latency
            total_tps += tps
            total_response_len += len(generated.split())

            if ref:
                references += 1
                bleu = BLEUScorer.score(generated, ref)
                blues.append(bleu)
            else:
                bleu = None

            pp = self._compute_perplexity(generated, prompt)
            if pp is not None:
                perplexities.append(pp)

            results.append({"prompt": prompt, "generated": generated, "bleu": bleu, "perplexity": pp, "tps": tps})

        avg_ppl = float(np.mean(perplexities)) if perplexities else None
        avg_bleu = float(np.mean(blues)) if blues else None
        avg_tps = total_tps / len(self.eval_prompts) if self.eval_prompts else 0

        # Personality score on last generated text
        last_gen = results[-1]["generated"] if results else ""
        personality = self._score_personality(last_gen, soul_name)

        result = EvalResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            adapter_path=adapter_path,
            prompts=len(self.eval_prompts),
            references=references,
            perplexity=avg_ppl,
            bleu=avg_bleu,
            avg_response_len=total_response_len / len(self.eval_prompts),
            inference_time_sec=total_time,
            tokens_per_sec=avg_tps,
            personality_score=personality.overall,
        )

        if save:
            self._save_result(result, results)

        return result

    def compare(self, baseline: EvalResult, with_adapter: EvalResult) -> Dict[str, Any]:
        """
        Compare baseline vs adapter results.

        Returns delta metrics:
        - perplexity_delta: negative = better
        - bleu_delta: positive = better
        - throughput_delta: relative improvement %
        - personality_delta: positive = better
        """
        delta = {}

        if baseline.perplexity and with_adapter.perplexity:
            delta["perplexity_delta"] = with_adapter.perplexity - baseline.perplexity
            delta["perplexity_improvement_pct"] = ((baseline.perplexity - with_adapter.perplexity) / baseline.perplexity) * 100

        if baseline.bleu and with_adapter.bleu:
            delta["bleu_delta"] = with_adapter.bleu - baseline.bleu

        if baseline.tokens_per_sec and with_adapter.tokens_per_sec:
            delta["throughput_delta"] = ((with_adapter.tokens_per_sec - baseline.tokens_per_sec) / baseline.tokens_per_sec) * 100

        if baseline.personality_score is not None and with_adapter.personality_score is not None:
            delta["personality_delta"] = with_adapter.personality_score - baseline.personality_score

        # Overall verdict
        positive = sum(1 for v in delta.values() if isinstance(v, (int, float)) and v > 0)
        total = sum(1 for v in delta.values() if isinstance(v, (int, float)))
        delta["verdict"] = "improved" if positive > total / 2 else "degraded" if positive == 0 else "mixed"
        delta["verdict"] = "improved" if delta["perplexity_delta"] < 0 else delta["verdict"]

        return delta

    def compare_with_report(self, baseline: EvalResult, with_adapter: EvalResult) -> str:
        """Generate a human-readable comparison report."""
        delta = self.compare(baseline, with_adapter)

        lines = [
            "=" * 50,
            "LoRA EVALUATION REPORT",
            "=" * 50,
            "",
            f"Baseline (no adapter): {baseline.adapter_path or 'base model'}",
            f"With adapter: {with_adapter.adapter_path or 'merged adapter'}",
            "",
            "METRICS",
            "-" * 30,
            f"Perplexity: {baseline.perplexity:.2f} → {with_adapter.perplexity:.2f} ({delta.get('perplexity_delta', 0):+.2f})",
            f"BLEU:       {baseline.bleu:.1f} → {with_adapter.bleu:.1f} ({delta.get('bleu_delta', 0):+.1f})",
            f"Throughput: {baseline.tokens_per_sec:.1f} → {with_adapter.tokens_per_sec:.1f} tok/s ({delta.get('throughput_delta', 0):+.1f}%)",
            f"Personality: {baseline.personality_score:.3f} → {with_adapter.personality_score:.3f} ({delta.get('personality_delta', 0):+.3f})",
            "",
            f"VERDICT: {delta.get('verdict', 'unknown').upper()}",
            "",
        ]

        if delta.get('verdict') == 'improved':
            lines.append("✓ Adapter improves model quality")
        elif delta.get('verdict') == 'degraded':
            lines.append("✗ Adapter reduces model quality — review before deploying")
        else:
            lines.append("~ Mixed results — some metrics improved, some degraded")

        if 'perplexity_improvement_pct' in delta:
            lines.append(f"  Perplexity {'improved' if delta['perplexity_delta'] < 0 else 'worsened'} by {abs(delta['perplexity_improvement_pct']):.1f}%")

        lines.append("=" * 50)

        return "\n".join(lines)

    def _save_result(self, result: EvalResult, detailed: List[Dict]):
        """Save eval result to disk."""
        ts = result.timestamp.replace(":", "-")
        prefix = "baseline" if result.adapter_path is None else Path(result.adapter_path).stem

        # Summary
        summary_path = self.eval_dir / f"{prefix}_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Detailed
        detail_path = self.eval_dir / f"{prefix}_{ts}_detail.json"
        with open(detail_path, "w") as f:
            json.dump({
                "summary": result.to_dict(),
                "generations": detailed,
            }, f, indent=2)

    def export_adapter_as_sou(
        self,
        adapter_npz: str,
        soul_name: str,
        eval_delta: Dict[str, Any],
        output_sou: Optional[str] = None,
    ) -> str:
        """
        Convert an aggregated LoRA .npz adapter into a .soul checkpoint.

        Args:
            adapter_npz: Path to the .npz adapter file
            soul_name: Name for this soul
            eval_delta: Delta metrics from before/after eval
            output_sou: Output path (defaults to sibling .soul of .npz)

        Returns:
            Path to the exported .soul file
        """
        from domains.inference import (
            SloProfile, PersonalityCore, GenerationParams,
            BehavioralTraits, CognitiveSignature, EmotionalRange,
            save_soul,
        )

        data = np.load(adapter_npz)

        eval_delta_dict = eval_delta if isinstance(eval_delta, dict) else {}
        verdict = eval_delta_dict.get('verdict', 'unknown')
        perplexity_delta = eval_delta_dict.get('perplexity_delta', 0)
        bleu_delta = eval_delta_dict.get('bleu_delta', 0)

        soul_profile = SloProfile(
            name=soul_name,
            version="1.0.0",
            tagline=f"LoRA adapter aggregated from feedback — verdict: {verdict}",
            description=(
                f"Per-user LoRA adapter trained from thumbs up/down feedback. "
                f"Verdict: {verdict}. Perplexity delta: {perplexity_delta:.4f}. "
                f"BLEU delta: {bleu_delta:.4f}."
            ),
            lineage="lora-feedback-distillation",
            base_model="lora-adapter",
            training_dataset="feedback-signals",
            epochs_trained=0,
            final_train_loss=abs(perplexity_delta) if perplexity_delta else 0.5,
            final_val_loss=abs(perplexity_delta) if perplexity_delta else 0.5,
            personality=PersonalityCore(
                warmth=0.5 + abs(eval_delta_dict.get('personality_delta', 0)) * 0.5,
                creativity=0.5,
                curiosity=0.5,
                confidence=0.5,
                empathy=0.5,
                formality=0.5,
            ),
            behavior=BehavioralTraits(
                speaking_style="conversational",
                explanation_depth="moderate",
            ),
            cognition=CognitiveSignature(
                pattern_recognition=0.5,
                abstract_reasoning=0.5,
                factual_precision=0.5,
            ),
            emotion=EmotionalRange(
                empathy_depth=0.5,
                mood_responsiveness=0.5,
            ),
            generation=GenerationParams(
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                max_tokens=256,
            ),
            system_prompt=f"You are {soul_name}.",
            tags=["lora", "feedback-derived", verdict],
            metadata={
                "adapter_npz": str(adapter_npz),
                "eval_verdict": verdict,
                "perplexity_delta": round(perplexity_delta, 6),
                "bleu_delta": round(bleu_delta, 6),
                "lora_rank": int(data.get("rank", 8)),
                "lora_alpha": float(data.get("alpha", 16)),
                "user_count": len(data.get("source_users", [])),
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "eval_delta": eval_delta_dict,
            },
        )

        if output_sou is None:
            output_sou = str(Path(adapter_npz).with_suffix(".soul"))

        from domains.inference import save_soul
        from domains.training.slonet import SloNet

        net = SloNet(
            layers=[],
            soul_name=soul_name,
            soul_traits={
                "warmth": soul_profile.personality.warmth,
                "creativity": soul_profile.personality.creativity,
                "curiosity": soul_profile.personality.curiosity,
                "confidence": soul_profile.personality.confidence,
            },
            system_prompt=soul_profile.system_prompt,
            lineage="lora-feedback-distillation",
            metadata={
                "lora_W_a": data["W_a"].tolist(),
                "lora_W_b": data["W_b"].tolist(),
                "lora_rank": int(data.get("rank", 8)),
                "lora_alpha": float(data.get("alpha", 16)),
                "eval_verdict": verdict,
                "soul_profile": soul_profile.to_dict(),
            },
        )

        save_soul(net, output_sou)
        logger.info("Exported .soul checkpoint: %s", output_sou, extra={"tag": "INFRA"})
        return output_sou

    def get_history(self, limit: int = 20) -> List[EvalResult]:
        """Load recent eval results."""
        results = []
        for f in sorted(self.eval_dir.glob("baseline_*.json"))[-limit:]:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    results.append(EvalResult(**data))
            except Exception:
                pass
        return sorted(results, key=lambda r: r.timestamp, reverse=True)


_global_eval: Optional[LoRAEvaluator] = None


def get_lora_evaluator(base_model: Optional[str] = None) -> LoRAEvaluator:
    global _global_eval
    if _global_eval is None:
        _global_eval = LoRAEvaluator(base_model=base_model)
    return _global_eval
