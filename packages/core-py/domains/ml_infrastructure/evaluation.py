"""
Model Evaluation - Metrics and Evaluation Utilities

Production-grade evaluation framework with:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Regression metrics (MSE, MAE, RMSE, R2)
- NLP metrics (BLEU, ROUGE, METEOR, perplexity)
- Cross-validation
- Model comparison
- Statistical tests
"""

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger("sloughgpt.evaluation")


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    support: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "confusion_matrix": self.confusion_matrix,
            "per_class_metrics": self.per_class_metrics,
            "support": self.support,
        }


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    explained_variance: float
    residual_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "explained_variance": self.explained_variance,
            "residual_std": self.residual_std,
        }


@dataclass
class NLPMetrics:
    """NLP-specific evaluation metrics."""

    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None
    perplexity: Optional[float] = None
    word_error_rate: Optional[float] = None
    char_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bleu": self.bleu,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
            "perplexity": self.perplexity,
            "word_error_rate": self.word_error_rate,
            "char_accuracy": self.char_accuracy,
        }


class MetricsCalculator:
    """Calculate various ML metrics."""

    @staticmethod
    def classification_metrics(
        y_true: List[Any],
        y_pred: List[Any],
        y_prob: Optional[List[float]] = None,
        labels: Optional[List[Any]] = None,
    ) -> ClassificationMetrics:
        """Calculate classification metrics."""

        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        n_classes = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        confusion = [[0] * n_classes for _ in range(n_classes)]
        for true, pred in zip(y_true, y_pred):
            if true in label_to_idx and pred in label_to_idx:
                confusion[label_to_idx[true]][label_to_idx[pred]] += 1

        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        per_class_prec = {}
        per_class_rec = {}
        per_class_f1 = {}
        support = {}

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for idx, label in enumerate(labels):
            tp = confusion[idx][idx]
            fp = sum(confusion[r][idx] for r in range(n_classes)) - tp
            fn = sum(confusion[idx][c] for c in range(n_classes)) - tp
            sup = sum(confusion[idx])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_prec[str(label)] = precision
            per_class_rec[str(label)] = recall
            per_class_f1[str(label)] = f1
            support[str(label)] = sup

            total_tp += tp
            total_fp += fp
            total_fn += fn

        macro_precision = sum(per_class_prec.values()) / n_classes
        macro_recall = sum(per_class_rec.values()) / n_classes
        macro_f1 = sum(per_class_f1.values()) / n_classes

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )

        auc_roc = None
        auc_pr = None

        if y_prob and n_classes == 2:
            auc_roc = MetricsCalculator._calculate_auc_roc(y_true, y_prob, labels)
            auc_pr = MetricsCalculator._calculate_auc_pr(y_true, y_prob, labels)

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=micro_precision,
            recall=micro_recall,
            f1_score=micro_f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            confusion_matrix=confusion,
            per_class_metrics={
                "precision": per_class_prec,
                "recall": per_class_rec,
                "f1": per_class_f1,
            },
            support=support,
        )

    @staticmethod
    def _calculate_auc_roc(y_true: List[Any], y_prob: List[float], labels: List[Any]) -> float:
        """Calculate AUC-ROC."""
        if len(labels) != 2:
            return None

        pos_label = labels[1]
        pairs = [(t, p) for t, p in zip(y_true, y_prob)]
        pairs.sort(key=lambda x: x[1], reverse=True)

        n_pos = sum(1 for t, _ in pairs if t == pos_label)
        n_neg = len(pairs) - n_pos

        if n_pos == 0 or n_neg == 0:
            return None

        tp = fp = 0
        auc = 0

        for true, prob in pairs:
            if true == pos_label:
                tp += 1
                auc += fp / (n_pos * n_neg)
            else:
                fp += 1

        auc += (n_pos - tp) * n_neg / (n_pos * n_neg)

        return auc

    @staticmethod
    def _calculate_auc_pr(y_true: List[Any], y_prob: List[float], labels: List[Any]) -> float:
        """Calculate AUC-PR."""
        if len(labels) != 2:
            return None

        pos_label = labels[1]
        pairs = [(t, p) for t, p in zip(y_true, y_prob)]
        pairs.sort(key=lambda x: x[1], reverse=True)

        n_pos = sum(1 for t, _ in pairs if t == pos_label)

        if n_pos == 0:
            return None

        tp = prec_sum = auc = 0

        for true, _ in pairs:
            if true == pos_label:
                tp += 1
            precision = tp / (
                tp + (sum(1 for t, _ in pairs[: tp + (1 if true != pos_label else 0)]) - tp)
            )
            prec_sum += precision

        return prec_sum / n_pos if n_pos > 0 else None

    @staticmethod
    def regression_metrics(y_true: List[float], y_pred: List[float]) -> RegressionMetrics:
        """Calculate regression metrics."""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        n = len(y_true)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = math.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        var_true = np.var(y_true)
        explained_variance = 1 - (np.var(y_true - y_pred) / var_true) if var_true > 0 else 0

        residual_std = np.std(y_true - y_pred)

        return RegressionMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            explained_variance=explained_variance,
            residual_std=residual_std,
        )

    @staticmethod
    def bleu_score(references: List[List[str]], hypothesis: List[str], n: int = 4) -> float:
        """Calculate BLEU score."""

        scores = []

        for ref_list, hyp in zip(references, hypothesis):
            hyp_tokens = hyp.split()
            if not hyp_tokens:
                scores.append(0.0)
                continue

            precisions = []

            for i in range(1, n + 1):
                ref_ngrams = [tuple(ref_list[j : j + i]) for j in range(len(ref_list) - i + 1)]
                hyp_ngrams = [tuple(hyp_tokens[j : j + i]) for j in range(len(hyp_tokens) - i + 1)]

                if not hyp_ngrams:
                    precisions.append(0)
                    continue

                ref_counts = Counter(ref_ngrams)
                hyp_counts = Counter(hyp_ngrams)

                matches = sum((ref_counts & hyp_counts).values())
                precisions.append(matches / len(hyp_ngrams) if hyp_ngrams else 0)

            if all(p == 0 for p in precisions):
                scores.append(0.0)
                continue

            brevity_penalty = min(1.0, math.exp(1 - len(ref_list[0]) / (len(hyp_tokens) + 1e-10)))
            geo_mean = math.exp(sum(math.log(p + 1e-10) for p in precisions) / n)

            scores.append(brevity_penalty * geo_mean)

        return np.mean(scores) if scores else 0.0

    @staticmethod
    def rouge_l(references: List[List[str]], hypothesis: List[str]) -> float:
        """Calculate ROUGE-L (Longest Common Subsequence)."""

        def lcs(ref: List[str], hyp: List[str]) -> int:
            m, n = len(ref), len(hyp)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref[i - 1] == hyp[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        scores = []

        for ref_list, hyp in zip(references, hypothesis):
            hyp_tokens = hyp.split()
            if not hyp_tokens:
                scores.append(0.0)
                continue

            lcs_len = lcs(ref_list, hyp_tokens)
            precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0
            recall = lcs_len / len(ref_list) if ref_list else 0

            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))

        return np.mean(scores) if scores else 0.0

    @staticmethod
    def perplexity(log_likelihoods: List[float], vocab_size: int) -> float:
        """Calculate perplexity."""
        if not log_likelihoods:
            return float("inf")

        avg_log_likelihood = np.mean(log_likelihoods)
        perplexity = math.exp(-avg_log_likelihood)

        return perplexity

    @staticmethod
    def word_error_rate(references: List[List[str]], hypothesis: List[str]) -> float:
        """Calculate Word Error Rate."""

        def edit_distance(ref: List[str], hyp: List[str]) -> int:
            m, n = len(ref), len(hyp)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref[i - 1] == hyp[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

            return dp[m][n]

        wer_scores = []

        for ref_list, hyp in zip(references, hypothesis):
            hyp_tokens = hyp.split()
            if not ref_list:
                wer_scores.append(1.0 if hyp_tokens else 0.0)
                continue

            distance = edit_distance(ref_list, hyp_tokens)
            wer_scores.append(distance / len(ref_list))

        return np.mean(wer_scores) if wer_scores else 0.0


class CrossValidator:
    """Cross-validation framework."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: List[Any], y: Optional[List[Any]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate train/test splits."""

        n = len(X)
        indices = np.arange(n)

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_size = n // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n

            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

            splits.append((train_indices.tolist(), test_indices.tolist()))

        return splits

    def evaluate(
        self,
        X: List[Any],
        y: List[Any],
        model_fn: Callable[[List[Any], List[Any]], Any],
        predict_fn: Callable[[Any, List[Any]], List[Any]],
        metric_fn: Callable[[List[Any], List[Any]], Dict[str, float]],
    ) -> Dict[str, Any]:
        """Run cross-validation."""

        splits = self.split(X, y)
        fold_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            model = model_fn(X_train, y_train)
            predictions = predict_fn(model, X_test)

            scores = metric_fn(y_test, predictions)
            fold_scores.append(scores)

        aggregated = {}
        for key in fold_scores[0].keys():
            values = [fs[key] for fs in fold_scores]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

        return {
            "n_splits": self.n_splits,
            "fold_scores": fold_scores,
            "aggregated": aggregated,
        }


class ModelComparator:
    """Compare multiple models statistically."""

    @staticmethod
    def compare(results: Dict[str, Dict[str, float]], metric: str = "accuracy") -> Dict[str, Any]:
        """Compare models based on a metric."""

        model_scores = {}

        for model_name, metrics in results.items():
            if metric in metrics:
                model_scores[model_name] = metrics[metric]

        if not model_scores:
            return {"error": f"Metric '{metric}' not found"}

        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        best_model = sorted_models[0][0]
        best_score = sorted_models[0][1]

        return {
            "best_model": best_model,
            "best_score": best_score,
            "rankings": [
                {"model": m, "score": s, "rank": i + 1} for i, (m, s) in enumerate(sorted_models)
            ],
            "scores": model_scores,
            "improvement_over_baseline": (
                (best_score - model_scores.get("baseline", 0))
                / model_scores.get("baseline", 1)
                * 100
                if "baseline" in model_scores
                else None
            ),
        }

    @staticmethod
    def statistical_test(
        scores_a: List[float], scores_b: List[float], alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Paired t-test to compare two models."""

        from scipy import stats

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "difference": mean_a - mean_b,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha,
            "winner": "a" if mean_a > mean_b else "b" if mean_b > mean_a else "tie",
        }


calculator = MetricsCalculator()


__all__ = [
    "MetricsCalculator",
    "ClassificationMetrics",
    "RegressionMetrics",
    "NLPMetrics",
    "CrossValidator",
    "ModelComparator",
    "calculator",
]
