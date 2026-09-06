"""
Train commands - Training, evaluation, and quick smoke tests.
"""
import os
import sys
import time
import math
import re
from pathlib import Path
from typing import Optional, List

import numpy as np

from domains.logging import get_global

log = get_global()
from utils.progress import ProgressBar
from utils.formatting import format_size, format_time, format_number, truncate


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numeric-stable softmax over the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _resolve_corpus_file(path_or_name: str) -> Path:
    """Resolve a ``--dataset`` value (file path or dataset name) to an existing file.

    Args:
        path_or_name: explicit corpus path, or a bare name resolved against
            ``datasets/<name>/input.txt`` relative to the repo root.

    Returns:
        Resolved :class:`Path` to a readable corpus file.

    Side effects:
        - Prints the available datasets and exits(2) when nothing resolves.
    """
    p = Path(path_or_name)
    if p.is_file():
        return p
    name = path_or_name.strip("/")
    for candidate in (Path("data") / name / "input.txt", Path(name)):
        if candidate.is_file():
            return candidate
    available = sorted(d.name for d in Path("data").glob("*") if d.is_dir())
    hint = f" Available datasets: {', '.join(available)}." if available else ""
    log.error(f"Dataset not found: {path_or_name}.{hint}")
    sys.exit(2)


def _print_train_result(result, model_path: str) -> None:
    """Print a ``TrainResult`` summary block (steps, epochs, losses, model path).

    Args:
        result: a ``TrainResult`` (supports attribute and dict-like access).
        model_path: path of the saved model to display.

    Returns:
        None.
    """
    def _loss(v):
        """Return a 4-decimal string for a finite number-like loss, else None."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        return f"{f:.4f}"

    log.header("Results")
    log.key_value("Steps", str(getattr(result, "global_step", "?")))
    log.key_value("Epochs", str(getattr(result, "epochs_completed", "?")))
    best = _loss(getattr(result, "best_eval_loss", None))
    final = _loss(getattr(result, "final_loss", None))
    if best is not None:
        log.key_value("Best eval loss", best)
    if final is not None:
        log.key_value("Final loss", final)
    log.key_value("Model", str(model_path))


def _print_native_next_steps(checkpoint_dir: str, saved: str) -> None:
    """Print how to use a freshly trained native checkpoint in chat.

    Args:
        checkpoint_dir: directory the checkpoint was saved into.
        saved: full path of the saved ``.soul`` file.

    Returns:
        None.
    """
    log.blank()
    log.header("Next steps")
    log.info("Load this model in chat by pointing the server at it and restarting:")
    log.info(f"  SLO_NATIVE_SOUL_PATH={saved} python3 apps/api/server/main.py")
    if Path(checkpoint_dir).resolve() == Path("models/slonet-native").resolve():
        log.info("(Default dir — the server also auto-discovers .soul models under models/slonet-native/.)")


def _gpt2_teacher_cached() -> bool:
    """Whether the GPT-2 teacher weights + tokenizer are present in the HF cache.

    Returns:
        True when the ``models--gpt2`` cache dir contains ``*.safetensors`` and
        ``tokenizer.json``; False otherwise.
    """
    hf_path = Path.home() / ".cache/huggingface/hub/models--gpt2"
    snaps_dir = hf_path / "snapshots"
    if not snaps_dir.is_dir():
        return False
    snaps = sorted(snaps_dir.glob("*"))
    if not snaps:
        return False
    snap = snaps[0]
    return bool(snap.glob("*.safetensors")) and (snap / "tokenizer.json").exists()


def _split_corpus_text(text: str, min_len: int = 40) -> List[str]:
    """Split a corpus string into samples suitable for contrastive training.

    Contrastive learning needs multiple texts, but a corpus file is one blob.
    Splits on paragraph breaks first, then falls back to overlapping windows
    and finally fixed-size word chunks so a single file yields several samples.

    Args:
        text: raw corpus text.
        min_len: minimum character length to keep a sample.

    Returns:
        List of non-empty text chunks.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_len]
    if len(paras) >= 2:
        return paras
    if len(paras) == 1 and len(paras[0]) >= 2 * min_len:
        win = 2 * min_len
        step = max(1, len(paras[0]) // 8)
        return [paras[0][i:i + win] for i in range(0, len(paras[0]) - win + 1, step)][:50]
    words = text.split()
    chunk = 200
    return [
        " ".join(words[i:i + chunk])
        for i in range(0, len(words), chunk)
        if len(" ".join(words[i:i + chunk])) >= min_len
    ]


def _embedder_retrieval_check(embedder, texts, query=None, top_k=3) -> None:
    """Run a computed retrieval sanity check against a trained embedder.

    Embeds real corpus texts plus a query, ranks texts by cosine similarity
    (embeddings are L2-normalized, so dot product equals cosine), and reports
    the top matches — verifying the saved embedder loads, encodes, and
    retrieves end-to-end. When ``query`` is None the first corpus text is
    used and self-retrieval rank #1 is expected.

    Args:
        embedder: a trained ``SloTextEmbedder``.
        texts: corpus strings used to train (retrieved against).
        query: optional held-out query string; defaults to ``texts[0]``.
        top_k: number of matches to list.

    Returns:
        None.
    """
    sample = [t for t in texts if isinstance(t, str) and t.strip()][:20]
    if len(sample) < 2:
        log.warning("Retrieval check skipped: need at least 2 corpus texts")
        return
    try:
        vecs = np.asarray(embedder.embed_batch(sample))
        query = query or sample[0]
        qvec = np.asarray(embedder.embed(query))
    except Exception as e:
        log.warning(f"Retrieval check failed: {e}")
        return

    sims = vecs @ qvec
    order = np.argsort(-sims)

    log.blank()
    log.header("Retrieval check")
    log.key_value("Query", query[:70] + ("..." if len(query) > 70 else ""))
    for rank in range(min(top_k, len(sample))):
        idx = int(order[rank])
        snippet = sample[idx][:70] + ("..." if len(sample[idx]) > 70 else "")
        log.key_value(f"Match {rank + 1}", f"{sims[idx]:.3f}  {snippet}")

    if query == sample[0]:
        self_rank = int(np.where(order == 0)[0][0])
        margin = float(sims[order[0]] - sims[order[1]]) if len(order) >= 2 else 0.0
        if self_rank == 0 and margin > 0.0:
            log.success(f"Self-retrieval OK (rank #1, margin {margin:.3f})")
        else:
            log.warning(f"Self-retrieval rank #{self_rank + 1} (margin {margin:.3f}) — embeddings may be weak")


def cmd_train(args):
    """Start a training job."""
    # Route to sub-modes
    if getattr(args, "checkpoint_info_path", None):
        args.checkpoint = args.checkpoint_info_path
        return _cmd_checkpoint_info(args)
    if getattr(args, "self_train", False):
        return _cmd_self_train(args)
    if getattr(args, "auto_train_action", None):
        return _cmd_autotrain(args)
    if getattr(args, "watch", False):
        return _cmd_monitor(args)
    if getattr(args, "adapters_action", None):
        return _cmd_user_adapters(args)
    if getattr(args, "feedback_train", False):
        return _cmd_feedback_train(args)
    if getattr(args, "export_feedback", False):
        return _cmd_feedback_export(args)
    if getattr(args, "embed_train", False):
        return cmd_train_embed(args)

    # Original train logic
    from utils.helpers import chat_repository_root
    from utils.helpers import train_export_stem_slug, train_export_default_stem

    if not args.api:
        sys.path.insert(0, ".")

        # Load config
        try:
            from config_loader import get_device, load_config, merge_args_with_config
        except ImportError:
            log.error("config_loader not found")
            sys.exit(1)

        config = load_config(args.config)
        config = merge_args_with_config(config, args)

        train_device = get_device(config.device)

        log.header("SloughGPT Training")
        log.key_value("Dataset", str(config.data.dataset))
        log.key_value("Device", f"{train_device} ({config.device.type})")
        log.key_value("Epochs", str(config.training.epochs))
        log.key_value("Batch Size", str(config.training.batch_size))
        log.key_value("Learning Rate", str(config.training.learning_rate))
        log.key_value("LoRA", f"{config.lora.enabled} (rank={config.lora.rank})")
        log.blank()

        # Setup tracking
        tracker = None
        if config.tracking.enabled:
            from dataclasses import asdict
            from domains.training.tracking import ExperimentTracker, TrackerBackend, TrackingConfig
            from domains.training.wandb_helpers import flatten_for_wandb_config

            backend = (
                TrackerBackend.WANDB
                if config.tracking.backend == "wandb"
                else TrackerBackend.MLFLOW
            )
            run_name = f"run_{config.data.dataset}_{config.training.epochs}ep"
            tracking_config = TrackingConfig(
                backend=backend,
                experiment_name=f"{config.model.name}_training",
                project=config.tracking.project,
                entity=config.tracking.entity,
                run_name=run_name,
                job_type="train",
                tags=["sloughgpt", "cli"],
            )
            tracker = ExperimentTracker(config=tracking_config)
            tracker.start_run(run_name=run_name)
            tracker.log_params(flatten_for_wandb_config(asdict(config)))
            log.success(f"Tracking enabled: {config.tracking.backend}")

        from domains.training.train_pipeline import SloughGPTTrainer

        if getattr(args, "save_stem", None):
            save_stem = train_export_stem_slug(args.save_stem, "export")
        else:
            save_stem = train_export_default_stem(str(config.model.name), str(config.data.dataset))
        save_path = f"{config.checkpoint.save_dir}/{save_stem}"

        log.info(f"Export: {save_stem}")

        trainer = SloughGPTTrainer(
            data_path=config.data.data_path,
            vocab_size=config.model.vocab_size,
            n_embed=config.model.n_embed,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            block_size=config.model.block_size,
            dropout=config.model.dropout,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            lr=config.training.learning_rate,
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.gradient_clip,
            checkpoint_dir=config.checkpoint.trainer_dir,
            checkpoint_interval=config.checkpoint.trainer_interval,
            save_best_only=config.checkpoint.save_best_only,
            max_checkpoints=config.checkpoint.max_checkpoints,
            scheduler_type=config.training.scheduler,
            warmup_steps=config.training.warmup_steps,
            min_lr=config.training.min_lr,
            weight_decay=config.training.weight_decay,
            use_lora=config.lora.enabled,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            soul_name=(config.model.soul_name or "").strip() or config.model.name,
            log_interval=config.training.log_interval,
            eval_interval=config.training.eval_interval,
            device=train_device,
            experiment_tracker=tracker,
        )

        log.info(f"Model: {trainer.model.num_parameters():,} params")
        log.blank()

        # Train
        resume_path = None
        if args.resume and getattr(args, "resume_latest", False):
            log.error("Use either --resume PATH or --resume-latest, not both")
            sys.exit(2)
        if getattr(args, "resume_latest", False):
            log.step(f"Resuming from latest under {config.checkpoint.trainer_dir}")
        elif args.resume:
            log.step(f"Resuming from: {args.resume}")
            resume_path = args.resume

        from utils.training_progress import TrainingProgressBar

        pbar = TrainingProgressBar(
            desc="Training",
            total_steps=config.training.max_steps or None,
        )

        start_time = time.time()
        try:
            result = trainer.train(
                resume=bool(args.resume or getattr(args, "resume_latest", False)),
                resume_path=resume_path,
                on_progress=pbar.update,
            )
        except ValueError as e:
            log.error(str(e))
            sys.exit(2)
        elapsed = time.time() - start_time

        pbar.finish()
        log.blank()
        log.success(f"Training complete ({format_time(elapsed)})")
        save_output_path = f"{save_path}.soul"
        if result is not None:
            _print_train_result(result, save_output_path)

        # Save
        log.step("Saving...")
        trainer.save(save_path)
        log.success(f"Saved: {save_output_path}")

        if tracker:
            tracker.end_run()

        return

    # API training
    import requests
    sys.path.insert(0, ".")
    try:
        from config_loader import load_config, merge_args_with_config
    except ImportError:
        log.error("config_loader not found")
        sys.exit(1)

    config = merge_args_with_config(load_config(args.config), args)
    base_url = f"http://{args.host}:{args.port}"

    display_name = (config.model.soul_name or "").strip() or str(config.model.name)
    payload = {
        "name": display_name[:200],
        "model": str(config.model.name),
        "dataset": str(config.data.dataset),
        "epochs": int(config.training.epochs),
        "batch_size": int(config.training.batch_size),
        "learning_rate": float(config.training.learning_rate),
        "n_embed": int(config.model.n_embed),
        "n_layer": int(config.model.n_layer),
        "n_head": int(config.model.n_head),
        "block_size": int(config.model.block_size),
        "log_interval": int(config.training.log_interval),
        "eval_interval": int(config.training.eval_interval),
        "dropout": float(config.model.dropout),
        "weight_decay": float(config.training.weight_decay),
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "max_grad_norm": float(config.training.gradient_clip),
        "warmup_steps": int(config.training.warmup_steps),
        "min_lr": float(config.training.min_lr),
        "scheduler": str(config.training.scheduler),
        "use_lora": bool(config.lora.enabled),
        "lora_rank": int(config.lora.rank),
        "lora_alpha": int(config.lora.alpha),
        "checkpoint_dir": str(config.checkpoint.trainer_dir),
        "checkpoint_interval": int(config.checkpoint.trainer_interval),
        "save_best_only": bool(config.checkpoint.save_best_only),
        "max_checkpoints": int(config.checkpoint.max_checkpoints),
    }
    if config.training.max_steps is not None:
        payload["max_steps"] = int(config.training.max_steps)
    _dtype = str(config.device.type or "").strip().lower()
    if _dtype and _dtype != "auto":
        payload["device"] = str(config.device.type)

    try:
        response = requests.post(
            f"{base_url}/training/start",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            job_id = data.get("id")
            log.success(f"Training started: {job_id}")

            # Stream progress via SSE
            _stream_api_progress(base_url, job_id)
        else:
            log.error(f"Failed ({response.status_code}): {response.text}")
    except Exception as e:
        log.error(f"API error: {e}")


def cmd_quick(args):
    """Quick smoke test: train a toy model and generate."""
    sys.path.insert(0, ".")

    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig
    from domains.training.performance import get_optimal_device

    log.header("SloughGPT Quick Start")

    device = get_optimal_device()
    log.key_value("Device", str(device))

    config = TrainerConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        max_steps=args.steps if args.steps else 100,
        warmup_steps=max(10, args.steps // 10) if args.steps else 10,
    )

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        ratios = None
        if args.ratios:
            try:
                ratios = [float(r.strip()) for r in args.ratios.split(",")]
                if len(ratios) != len(datasets):
                    ratios = None
            except ValueError:
                ratios = None

        if ratios:
            log.info(f"Loading {len(datasets)} datasets with ratios")
            data_path = list(zip(datasets, ratios))
        else:
            log.info(f"Loading {len(datasets)} datasets: {', '.join(datasets)}")
            data_path = datasets
    else:
        log.info(f"Loading: {args.dataset}")
        data_path = args.dataset

    trainer = SloughGPTTrainer(
        data_path=data_path,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.heads,
        block_size=args.block,
        dropout=0.1,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.steps if args.steps else 100,
        soul_name=getattr(args, "soul_name", "SloughGPT-Quick"),
        config=config,
    )

    PRESETS = {
        "tiny": {"embed": 64, "layers": 2, "heads": 2, "block": 64},
        "small": {"embed": 128, "layers": 4, "heads": 4, "block": 128},
        "medium": {"embed": 256, "layers": 6, "heads": 8, "block": 128},
        "large": {"embed": 512, "layers": 12, "heads": 16, "block": 256},
    }

    if getattr(args, "preset", None) and args.preset in PRESETS:
        p = PRESETS[args.preset]
        trainer.config.n_embed = p["embed"]
        trainer.config.n_layer = p["layers"]
        trainer.config.n_head = p["heads"]
        trainer.config.block_size = p["block"]
        log.info(f"Preset: {args.preset}")

    log.info(f"Model: {trainer.model.num_parameters():,} params")
    log.blank()

    from utils.training_progress import TrainingProgressBar

    pbar = TrainingProgressBar(
        desc="Training",
        total_steps=args.steps if args.steps else 100,
    )

    log.step("Training...")
    trainer.train(on_progress=pbar.update)
    pbar.finish()

    log.blank()
    log.step("Generating...")
    text = trainer.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)

    log.blank()
    log.key_value("Prompt", args.prompt)
    log.key_value("Generated", f"{args.prompt}{text[:100]}...")

    output_base = args.output.replace(".safetensors", "").replace(".soul", "")
    trainer.save(output_base)
    log.success(f"Saved: {output_base}.soul")


def cmd_train_native(args):
    """Train a SloNet model from scratch (pure numpy).

    Drives ``SloughGPTTrainer`` directly with a ``TrainerConfig`` — no
    ``config_loader`` indirection. Checkpoints are auto-saved as ``.soul``
    into ``--checkpoint-dir`` at ``--checkpoint-interval`` steps.

    Args:
        args: Namespace with native flags (dataset, steps, embed, layers,
            heads, block, batch, epochs, lr, weight_decay, scheduler, warmup,
            min_lr, grad_norm, checkpoint_dir, checkpoint_interval,
            max_checkpoints, eval_interval, log_interval, soul_name,
            save_stem, resume, resume_latest, dropout).

    Returns:
        None. Prints progress and the final checkpoint path.

    Side effects:
        - Creates ``--checkpoint-dir`` and writes ``.soul`` checkpoints.
        - Optionally loads a resume checkpoint when ``--resume``/``--resume-latest``.
    """
    sys.path.insert(0, ".")

    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig
    from domains.training.performance import get_optimal_device

    checkpoint_dir = getattr(args, "checkpoint_dir", None) or "models/slonet-native"

    device = getattr(args, "device", None) or "cpu"
    if device in ("auto", None):
        device = str(get_optimal_device())

    config = TrainerConfig(
        vocab_size=0,
        n_embed=getattr(args, "embed", 64),
        n_layer=getattr(args, "layers", 2),
        n_head=getattr(args, "heads", 4),
        block_size=getattr(args, "block", 128),
        dropout=getattr(args, "dropout", 0.1),
        batch_size=getattr(args, "batch", 16),
        epochs=getattr(args, "epochs", 1),
        max_steps=getattr(args, "steps", None),
        learning_rate=getattr(args, "lr", 3e-3),
        weight_decay=getattr(args, "weight_decay", 0.01),
        max_grad_norm=getattr(args, "grad_norm", 1.0),
        scheduler_type=getattr(args, "scheduler", "cosine"),
        warmup_steps=getattr(args, "warmup", 100),
        min_lr=getattr(args, "min_lr", 1e-5),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=getattr(args, "checkpoint_interval", 500),
        save_best_only=getattr(args, "save_best_only", False),
        max_checkpoints=getattr(args, "max_checkpoints", 3),
        log_interval=getattr(args, "log_interval", 50),
        eval_interval=getattr(args, "eval_interval", 250),
        device=device,
    )

    dataset_arg = getattr(args, "dataset", None)
    if not dataset_arg:
        log.error("--dataset (corpus file or name) is required")
        sys.exit(2)
    dataset = str(_resolve_corpus_file(dataset_arg))

    soul_name = getattr(args, "soul_name", "sloughgpt-native")

    log.header("SloNet Native Training")
    log.key_value("Dataset", str(dataset))
    log.key_value("Device", str(device))
    log.key_value("Steps", str(config.max_steps or "epoch-budget"))
    log.key_value("Arch", f"e{config.n_embed} l{config.n_layer} h{config.n_head} b{config.block_size}")
    log.key_value("Batch", str(config.batch_size))
    log.key_value("Learning Rate", str(config.learning_rate))
    log.key_value("Checkpoint Dir", str(checkpoint_dir))
    log.blank()

    tokenizer = None
    tokenizer_kind = getattr(args, "tokenizer", "char") or "char"
    if tokenizer_kind == "token-tree":
        from domains.training.token_tree import TokenTree

        corpus_text = Path(dataset).read_text(encoding="utf-8")
        token_vocab_size = getattr(args, "token_vocab_size", 512) or 512
        log.step(f"Training token tree (vocab={token_vocab_size})...")
        tokenizer = TokenTree().train(
            corpus_text,
            vocab_size=token_vocab_size,
            embed_dim=0,
            verbose=False,
        )
        log.key_value("Tokenizer", f"token-tree ({tokenizer.vocab_size} tokens, {len(tokenizer.merges)} merges)")
        log.blank()
    else:
        log.key_value("Tokenizer", "char-level")

    trainer = SloughGPTTrainer(
        data_path=dataset,
        config=config,
        soul_name=soul_name,
        tokenizer=tokenizer,
    )
    log.info(f"Model: {trainer.training_model.num_parameters():,} params")
    log.blank()

    resume_path = None
    if getattr(args, "resume", None) and getattr(args, "resume_latest", False):
        log.error("Use either --resume PATH or --resume-latest, not both")
        sys.exit(2)
    if getattr(args, "resume_latest", False):
        log.step(f"Resuming from latest under {checkpoint_dir}")
    elif getattr(args, "resume", None):
        resume_path = args.resume
        log.step(f"Resuming from: {resume_path}")

    from utils.training_progress import TrainingProgressBar

    pbar = TrainingProgressBar(
        desc="Training",
        total_steps=config.max_steps,
    )

    start_time = time.time()
    try:
        result = trainer.train(
            resume=bool(getattr(args, "resume", None) or getattr(args, "resume_latest", False)),
            resume_path=resume_path,
            on_progress=pbar.update,
        )
    except ValueError as e:
        log.error(str(e))
        sys.exit(2)
    elapsed = time.time() - start_time
    pbar.finish()
    log.blank()
    log.success(f"Training complete ({format_time(elapsed)})")

    save_stem = getattr(args, "save_stem", None)
    if save_stem:
        save_base = f"{checkpoint_dir}/{save_stem}"
    else:
        save_base = f"{checkpoint_dir}/{soul_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save(save_base)
    saved = f"{save_base}.soul"
    # Prune the trainer's auto-named checkpoints so a completed run leaves
    # exactly one final model file.
    for p in Path(checkpoint_dir).glob("*.soul"):
        if str(p) == saved:
            continue
        p.unlink(missing_ok=True)
        meta = Path(str(p) + ".meta.json")
        meta.unlink(missing_ok=True)
    log.success(f"Saved: {saved}")

    if result is not None:
        _print_train_result(result, saved)

    prompt = getattr(args, "prompt", None)
    if prompt:
        log.blank()
        log.header("Sample generation")
        log.key_value("Prompt", prompt)
        try:
            text = trainer.generate(prompt, max_tokens=150, temperature=0.8)
            log.key_value("Generated", f"{prompt}{text[:200]}...")
        except Exception as e:  # pragma: no cover — best-effort sample
            log.warning(f"Sample generation failed: {e}")

    _print_native_next_steps(checkpoint_dir, saved)


def cmd_eval(args):
    """Evaluate char-level model perplexity from a .soul checkpoint via SloNet."""
    log.header("Model Evaluation")
    log.key_value("Checkpoint", args.checkpoint)

    checkpoint_path = str(args.checkpoint)
    data_path = getattr(args, "data", None) or "datasets/shakespeare/input.txt"

    try:
        from domains.training.lm_eval_char import evaluate_soul_char_lm

        if not Path(data_path).is_file():
            log.warning(f"Data file not found: {data_path}")
            return
        log.info(f"Evaluating on: {data_path}")
        metrics = evaluate_soul_char_lm(checkpoint_path, data_path)
        _print_char_lm_metrics(metrics)

        if getattr(args, "benchmark", False):
            log.blank()
            log.step("Running benchmark...")
            start = time.time()
            for _ in range(10):
                evaluate_soul_char_lm(checkpoint_path, data_path)
            elapsed = time.time() - start
            log.info(f"10 iterations: {format_time(elapsed)}")

    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        sys.exit(1)


def _print_char_lm_metrics(metrics: dict) -> None:
    """Render char-LM eval metrics through the CLI log."""
    log.blank()
    log.key_value("Mean Loss", f"{metrics['mean_loss']:.4f}")
    ppl = metrics["perplexity"]
    ppl_s = f"{ppl:.4f}" if ppl != float("inf") else "inf"
    log.key_value("Perplexity", ppl_s)
    log.key_value("Tokens Scored", format_number(metrics["num_token_positions"]))
    for w in metrics.get("warnings") or []:
        log.warning(w)


# ── Sub-commands merged into `train` ──────────────────────────────

def _cmd_self_train(args):
    """Run self-training loop (model talks to itself)."""
    import subprocess

    script = Path("scripts/self_train.py")
    if not script.exists():
        log.error("scripts/self_train.py not found")
        sys.exit(1)

    forever = getattr(args, "self_forever", None) or getattr(args, "forever", False)
    steps = getattr(args, "self_steps", None) or getattr(args, "steps", 1000)
    model = getattr(args, "self_model", None) or getattr(args, "model", "gpt2")
    temperature = getattr(args, "temperature", 0.8)
    max_tokens = getattr(args, "self_max_tokens", None) or getattr(args, "max_tokens", 50)
    seed = getattr(args, "self_seed", None) or getattr(args, "seed", "Hello")

    cmd = [sys.executable, str(script)]
    if forever:
        cmd.append("--forever")
    else:
        cmd.extend(["--steps", str(steps)])
    cmd.extend(["--model", model])
    cmd.extend(["--temperature", str(temperature)])
    cmd.extend(["--max_tokens", str(max_tokens)])
    cmd.extend(["--seed", seed])

    log.header("Self-Training Loop")
    log.info(f"Running: {' '.join(cmd)}")
    log.info("Press Ctrl+C to stop")
    log.blank()
    subprocess.run(cmd)


def _cmd_autotrain(args):
    """Control auto-training via API."""
    import requests

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    api_url = f"http://{host}:{port}"
    action = getattr(args, "auto_train_action", None) or getattr(args, "action", None)

    log.header(f"Auto-Train ({action})")

    if action == "start":
        resp = requests.post(
            f"{api_url}/training/start",
            json={
                "teacher": getattr(args, "auto_teacher", None) or getattr(args, "teacher", "gpt2"),
                "temperature": getattr(args, "temperature", 0.8),
                "max_steps": getattr(args, "auto_steps", None) or getattr(args, "steps", 1000),
            },
        )
        if resp.ok:
            log.success(f"Started: {resp.json()}")
        else:
            log.error(f"Failed: {resp.json()}")
    elif action == "stop":
        resp = requests.post(f"{api_url}/training/stop")
        if resp.ok:
            log.success(f"Stopped: {resp.json()}")
        else:
            log.error(f"Failed: {resp.json()}")
    elif action == "status":
        resp = requests.get(f"{api_url}/training/status")
        if resp.ok:
            data = resp.json()
            for k, v in data.items():
                log.key_value(k, str(v))
        else:
            log.error(f"Failed: {resp.json()}")


def _cmd_monitor(args):
    """Monitor training jobs."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"

    log.header("Training Monitor")

    while True:
        try:
            response = requests.get(f"{base_url}/training/jobs", timeout=5)
            if response.status_code == 200:
                jobs = response.json()
                if isinstance(jobs, dict) and "jobs" in jobs:
                    jobs = jobs["jobs"]
                if jobs:
                    rows = []
                    for job in jobs:
                        rows.append([job.get("name", "unknown"), job.get("status", "unknown")])
                    log.blank()
                    log.table(["Job", "Status"], rows)
                else:
                    log.info("No active training jobs")
            else:
                log.error(f"HTTP {response.status_code}")
        except Exception as e:
            log.error(str(e))

        if not args.watch:
            break
        time.sleep(args.interval)


def _cmd_user_adapters(args):
    """Manage per-user LoRA adapters."""
    import sys as _sys
    _sys.path.insert(0, ".")

    log.header("User Adapters")

    action = getattr(args, "adapters_action", None) or getattr(args, "action", None)
    user_id = getattr(args, "adapters_user", None) or getattr(args, "user", None)
    users_str = getattr(args, "adapters_users", None) or getattr(args, "users", None)
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()

        if action == "list":
            adapters = store.get_all_adapters()
            stats = store.get_stats()
            log.key_value("Total Users", str(stats["total_users"]))
            log.key_value("Total Size", f'{stats["total_size_mb"]:.2f} MB')
            log.key_value("Avg per User", f'{stats["avg_size_per_user_kb"]:.1f} KB')

            if adapters:
                log.blank()
                log.section("Adapters")
                rows = []
                for a in adapters[:20]:
                    rows.append([a["user_id"][:28], str(a["feedback_count"]), a["updated_at"][:19]])
                log.table(["User ID", "Feedback", "Updated"], rows)
            else:
                log.info("No adapters found")

        elif action == "info":
            adapter = store.get_adapter(user_id)
            if adapter is None:
                log.error(f"No adapter for user: {user_id}")
            else:
                log.key_value("User", adapter.user_id)
                log.key_value("Feedback", str(adapter.feedback_count))
                log.key_value("W_a", str(adapter.W_a.shape))
                log.key_value("W_b", str(adapter.W_b.shape))

        elif action == "delete":
            store.delete_adapter(user_id)
            log.success(f"Deleted adapter for {user_id}")

        elif action == "merge":
            user_ids = users_str.split(",") if users_str else []
            if len(user_ids) < 2:
                log.error("Need at least 2 user IDs: --users user1,user2")
                return
            merged = store.merge_adapters(user_ids)
            log.success(f"Merged {merged['user_count']} adapters")

    except ImportError as e:
        log.error(f"Feedback module: {e}")


def _cmd_feedback_train(args):
    """Prepare training data from feedback."""
    import sys as _sys
    _sys.path.insert(0, ".")

    log.header("Feedback Training Pipeline")

    stats_only = getattr(args, "feedback_stats_only", None) or getattr(args, "stats_only", False)
    feedback_fmt = getattr(args, "feedback_format", None) or getattr(args, "format", "all")
    output_dir = getattr(args, "feedback_output", None) or getattr(args, "output", None)

    try:
        from domains.feedback import create_training_pipeline

        trainer = create_training_pipeline()
        stats = trainer.get_training_stats()
        log.section("Available Data")
        log.key_value("Conversations", str(stats["total_conversations"]))
        log.key_value("Thumbs Up", str(stats["thumbs_up"]))
        log.key_value("Thumbs Down", str(stats["thumbs_down"]))
        log.key_value("DPO Pairs", str(stats["available_dpo_pairs"]))
        log.key_value("SFT Examples", str(stats["available_sft_examples"]))

        if stats_only:
            return

        formats = []
        if feedback_fmt == "all":
            formats = ["dpo", "sft", "reward"]
        else:
            formats = [feedback_fmt]

        output_dir = output_dir or "data/training"
        results = trainer.export_for_alignment(output_dir=output_dir, formats=formats)

        log.blank()
        log.section("Exported Files")
        for fmt, path in results.items():
            log.key_value(fmt, path)

    except ImportError as e:
        log.error(f"Feedback module: {e}")


def _cmd_feedback_export(args):
    """Export feedback data for training."""
    import sys as _sys
    _sys.path.insert(0, ".")

    log.header("Feedback Export")

    output_path = getattr(args, "export_feedback_output", None) or getattr(args, "output", "data/training_feedback.jsonl")
    fmt = getattr(args, "export_feedback_format", None) or getattr(args, "format", "jsonl")

    try:
        from domains.feedback import get_meta_weight_manager

        manager = get_meta_weight_manager()
        if manager is None:
            log.error("Meta-weight system not available")
            return

        stats = manager.get_stats()
        log.section("Current Stats")
        log.key_value("Total Feedback", str(stats["db_stats"]["feedback_total"]))
        log.key_value("Thumbs Up", str(stats["db_stats"]["thumbs_up"]))
        log.key_value("Thumbs Down", str(stats["db_stats"]["thumbs_down"]))

        log.blank()
        log.step(f"Exporting to {output_path}...")
        manager.export_training_data(filepath=output_path, format=fmt)

        import os
        if os.path.exists(output_path):
            with open(output_path) as f:
                lines = sum(1 for _ in f)
            log.success(f"Exported {lines} records")
        else:
            log.warning("File not created")

    except ImportError as e:
        log.error(f"Feedback module: {e}")

def _cmd_checkpoint_info(args):
    """Inspect a .soul checkpoint — show metadata, weight summary, training info."""
    from pathlib import Path
    from domains.training.slonet import import_from_sou

    log.header("Checkpoint Info")

    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        log.error(f"File not found: {ckpt_path}")
        sys.exit(1)

    log.key_value("Path", str(ckpt_path))
    log.key_value("Size", format_size(ckpt_path.stat().st_size))

    try:
        net = import_from_sou(str(ckpt_path))
    except Exception as e:
        log.error(f"Failed to load: {e}")
        sys.exit(1)

    meta = getattr(net, "metadata", None) or {}

    # Soul metadata
    log.section("Soul Metadata")
    log.key_value("Soul Name", getattr(net, "soul_name", "?"))
    if getattr(net, "soul_traits", None):
        log.key_value("Traits", str(net.soul_traits))
    if getattr(net, "system_prompt", None):
        log.key_value("System Prompt", truncate(str(net.system_prompt), 80))
    log.key_value("Step", str(getattr(net, "_step", "?")))
    log.key_value("Created", str(getattr(net, "_created_at", "?")))

    arch_keys = ["vocab_size", "n_embed", "n_layer", "n_head", "block_size", "use_rope"]
    arch = {k: meta.get(k) for k in arch_keys if meta.get(k) is not None}
    if arch:
        log.section("Architecture")
        for k, v in arch.items():
            log.key_value(k, str(v))

    # Model weights summary
    params = list(net.parameters())
    log.section("Model Weights")
    total_params = sum(int(np.prod(p.shape)) for p in params)
    total_bytes = sum(int(p.data.nbytes) for p in params) if params else 0
    log.key_value("Parameters", format_number(total_params))
    log.key_value("Weight Groups", str(len(params)))
    log.key_value("Size (FP32)", format_size(total_bytes))
    for p in params[:10]:
        log.key_value("  weight", f"{list(p.shape)} float32")

    # Training metadata
    log.section("Training Metadata")
    for k, v in (meta.get("training") or {}).items():
        log.key_value(f"  {k}", str(v))
    for k, v in (meta.get("metrics") or {}).items():
        log.key_value(f"  {k}", str(v))
    if not meta.get("training") and not meta.get("metrics"):
        log.info("No training info recorded in metadata")


def cmd_demo(args):
    """Run system demos (RAG, KG, EWC, inference)."""
    import sys
    sys.path.insert(0, ".")

    log.header("SloughGPT Demo")

    from domains.cognitive.rag import ProductionRAG
    from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph
    from domains.training.ewc import EwcContinualLearner

    if args.component in ("all", "rag"):
        log.section("RAG - Document Retrieval")
        rag = ProductionRAG()
        rag.add_document("Python is a programming language created by Guido van Rossum in 1991.")
        results = rag.query("What is Python?")
        log.key_value("Retrieved", str(len(results)))

    if args.component in ("all", "kg"):
        log.section("Knowledge Graph - Fact Verification")
        kg = KnowledgeGraph()
        kg.add_fact("python", "is_a", "programming_language")
        kg.add_fact("python", "created_by", "guido_van_rossum")
        facts = kg.query(subject="python")
        log.key_value("Facts", str(len(facts)))

    if args.component in ("all", "ewc"):
        log.section("EWC - Catastrophic Forgetting Prevention")
        from domains.models import SloughGPTModel
        model = SloughGPTModel(vocab_size=50, n_embed=32, n_layer=2, n_head=2, block_size=16)
        ewc = EwcContinualLearner(model)
        log.key_value("Fisher Params", str(len(ewc.fisher_estimator.fisher_accum)))

    if args.component in ("all", "inference"):
        log.section("Inference - KV Cache")
        log.key_value("Status", "KVCache removed — SloNet handles caching internally")

    log.blank()
    log.success("Demo complete!")


def cmd_rlhf(args):
    """Run RLHF demo."""
    import sys

    sys.path.insert(0, ".")

    log.header("RLHF Demo")
    from domains.training.rlhf import RLHFConfig
    from domains.models import SloughGPTModel

    device = "cpu"
    log.key_value("Device", device)

    model = SloughGPTModel(vocab_size=100, n_embed=64, n_layer=2, n_head=4, block_size=32, dropout=0.0)
    log.key_value("Parameters", f"{model.num_parameters():,}")

    config = RLHFConfig(ppo_epochs=2, clip_epsilon=0.2, entropy_coef=0.01, gamma=1.0, lam=0.95)
    log.key_value("PPO Epochs", str(config.ppo_epochs))
    log.key_value("Clip Epsilon", str(config.clip_epsilon))

    batch_size, seq_len, vocab_size = 4, 16, 100
    rng = np.random.default_rng(0)
    log.step("Running PPO steps...")
    for step in range(min(args.steps, 20)):
        input_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len))
        logits, _ = model(input_ids)
        arr = logits.data
        arr = np.where(np.isfinite(arr), arr, np.zeros_like(arr))
        last = arr[:, -1, :]
        probs = _softmax_np(last)
        reward = probs.max(axis=-1).mean()
        if step % 5 == 0 or step == min(args.steps, 20) - 1:
            log.key_value(f"Step {step}", f"reward={reward:.3f}")

    log.blank()
    log.success("RLHF demo complete!")


def cmd_cloud_setup(args):
    """Setup Pinecone vector store."""
    import sys
    import asyncio
    import os
    sys.path.insert(0, ".")

    log.header("Cloud Setup")
    from domains.inference.vector_stores.pinecone_store import PineconeVectorStore
    from domains.inference.vector_store import VectorEntry, simple_embed

    api_key = args.api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        log.error("Pinecone API key required (--api-key or PINECONE_API_KEY)")
        return

    async def setup():
        log.key_value("Index", args.index)
        log.key_value("Dimension", str(args.dimension))
        store = PineconeVectorStore(api_key=api_key, index_name=args.index, dimension=args.dimension, environment=args.environment)
        await store.connect()
        entries = [VectorEntry(id="test", vector=simple_embed("test document", dimension=args.dimension), text="test document", metadata={"created_by": "cli"})]
        await store.upsert(entries)
        count = await store.count()
        log.success(f"Pinecone: {count} documents indexed")
        await store.disconnect()

    asyncio.run(setup())


def register(subparsers):
    """Register train commands with argparse."""
    # Train (unified)
    train_parser = subparsers.add_parser(
        "train",
        help="Training (normal, self, auto, feedback, adapters, watch)",
    )
    train_parser.add_argument("--dataset", default="shakespeare", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--api", action="store_true", help="Use API training")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--resume-latest", action="store_true", help="Resume latest")
    train_parser.add_argument("--save-stem", type=str, default=None, help="Output filename stem")

    # Self-train mode
    train_parser.add_argument("--self", dest="self_train", action="store_true", help="Self-training loop (model talks to itself)")
    train_parser.add_argument("--self-steps", type=int, default=1000, help="Self-train steps")
    train_parser.add_argument("--self-model", default="gpt2", help="Model for self-train")
    train_parser.add_argument("--self-max-tokens", type=int, default=50, help="Self-train max tokens per gen")
    train_parser.add_argument("--self-seed", default="Hello", help="Self-train starting text")
    train_parser.add_argument("--self-forever", action="store_true", help="Self-train until Ctrl+C")

    # Auto-train mode
    train_parser.add_argument("--auto", dest="auto_train_action", choices=["start", "stop", "status"], help="Auto-training via API")
    train_parser.add_argument("--auto-teacher", default="gpt2", help="Auto-train teacher model")
    train_parser.add_argument("--auto-steps", type=int, default=1000, help="Auto-train max steps")

    # Watch mode
    train_parser.add_argument("--watch", action="store_true", help="Monitor training jobs")
    train_parser.add_argument("--interval", type=int, default=5, help="Watch refresh interval (s)")

    # Adapter management
    train_parser.add_argument("--adapters", dest="adapters_action", choices=["list", "info", "delete", "merge"], help="Manage LoRA adapters")
    train_parser.add_argument("--adapters-user", help="User ID for adapter info/delete")
    train_parser.add_argument("--adapters-users", help="Comma-separated user IDs for adapter merge")

    # Feedback training
    train_parser.add_argument("--from-feedback", dest="feedback_train", action="store_true", help="Prepare training data from feedback")
    train_parser.add_argument("--feedback-format", choices=["all", "dpo", "sft", "reward"], default="all", help="Feedback training format")
    train_parser.add_argument("--feedback-output", help="Output directory for feedback training data")
    train_parser.add_argument("--feedback-stats-only", action="store_true", help="Show feedback stats only")

    # Feedback export
    train_parser.add_argument("--export-feedback", action="store_true", help="Export feedback data")
    train_parser.add_argument("--export-feedback-output", default="data/training_feedback.jsonl", help="Export path")
    train_parser.add_argument("--export-feedback-format", choices=["jsonl", "dpo"], default="jsonl", help="Export format")

    # Checkpoint info
    train_parser.add_argument("--checkpoint-info", dest="checkpoint_info_path", help="Inspect a checkpoint file (.soul)")

    # Embedder training
    train_parser.add_argument("--embed", dest="embed_train", action="store_true", help="Train text embedder on corpus")
    train_parser.add_argument("--corpus", help="Corpus path for embedder training (auto-discovers knowledge/datasets if omitted)")
    train_parser.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension")
    train_parser.add_argument("--vocab-size", type=int, default=4096, help="Vocab size for embedder")
    train_parser.add_argument("--output", dest="embed_output", default=None, help="Embedder save path")

    train_parser.set_defaults(func=cmd_train)

    # Quick
    quick_parser = subparsers.add_parser(
        "quick",
        help="Smoke test: train briefly and generate",
    )
    quick_parser.add_argument("--dataset", "-d", default="datasets/shakespeare/input.txt", help="Corpus file")
    quick_parser.add_argument("--prompt", default="The king", help="Generation prompt")
    quick_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    quick_parser.add_argument("--steps", type=int, default=100, help="Max steps")
    quick_parser.add_argument("--embed", type=int, default=128, help="Embedding size")
    quick_parser.add_argument("--layers", type=int, default=4, help="Transformer layers")
    quick_parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    quick_parser.add_argument("--block", type=int, default=128, help="Context length")
    quick_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    quick_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    quick_parser.add_argument("--max-tokens", type=int, default=100, help="Generated tokens")
    quick_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    quick_parser.add_argument("--output", default="models/quick.soul", help="Output path")
    quick_parser.add_argument("--no-optimize", action="store_true", help="Disable optimizations")
    quick_parser.add_argument("--soul-name", default="SloughGPT-Quick", help="Slo name")
    quick_parser.set_defaults(func=cmd_quick)

    # Eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate model perplexity",
    )
    eval_parser.add_argument("--checkpoint", default="models/sloughgpt.soul", help="Checkpoint path")
    eval_parser.add_argument("--data", default="datasets/shakespeare/input.txt", help="Eval text")
    eval_parser.add_argument("--device", default="cpu", help="Device for scoring")
    eval_parser.add_argument("--no-strict", action="store_true", help="Allow partial load")
    eval_parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    eval_parser.set_defaults(func=cmd_eval)

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run system demos (RAG, KG, EWC)")
    demo_parser.add_argument("--component", choices=["all", "rag", "kg", "ewc", "inference"], default="all", help="Subsystem to demo")
    demo_parser.set_defaults(func=cmd_demo)

    # RLHF
    rlhf_parser = subparsers.add_parser("rlhf", help="Run RLHF demo")
    rlhf_parser.add_argument("--steps", type=int, default=20, help="PPO steps")
    rlhf_parser.set_defaults(func=cmd_rlhf)

    # Cloud setup
    cloud_parser = subparsers.add_parser("cloud", help="Setup Pinecone vector store")
    cloud_parser.add_argument("--api-key", help="Pinecone API key")
    cloud_parser.add_argument("--index", default="sloughgpt", help="Index name")
    cloud_parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    cloud_parser.add_argument("--environment", default="us-east-1", help="Pinecone environment")
    cloud_parser.set_defaults(func=cmd_cloud_setup)

    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training jobs")
    monitor_parser.add_argument("--watch", action="store_true", help="Continuous watch")
    monitor_parser.add_argument("--interval", type=int, default=5, help="Refresh interval (s)")
    monitor_parser.set_defaults(func=lambda a: _cmd_monitor(a))


def cmd_train_embed(args):
    """Train a text embedder on your own corpus using contrastive learning.

    Collects texts from knowledge files and chat history, then trains a
    SloNet transformer encoder to produce 384-dim embeddings.  The trained
    model is saved to data/models/text-embedder.soul and automatically
    used by simple_embed() instead of downloading sentence-transformers.
    """
    import os
    import json
    import glob as glob_mod
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    # ── Collect training texts ────────────────────────────────────────
    texts = []

    corpus = getattr(args, "corpus", None)
    if corpus:
        p = Path(corpus)
        if p.is_file():
            texts.extend(_split_corpus_text(p.read_text(errors="ignore")))
        elif p.is_dir():
            for ext in ("*.txt", "*.md", "*.json"):
                for fp in p.rglob(ext):
                    try:
                        texts.append(fp.read_text(errors="ignore"))
                    except OSError as e:
                        log.warning("Skipping %s: %s", fp, e)
        else:
            log.error(f"Corpus not found: {corpus}")
            return
    else:
        # Auto-discover: knowledge files + chat history
        knowledge_dir = repo_root / "data" / "knowledge"
        if knowledge_dir.exists():
            for fp in knowledge_dir.rglob("*.txt"):
                try:
                    texts.append(fp.read_text(errors="ignore"))
                except OSError as e:
                    log.warning("Skipping %s: %s", fp, e)
            for fp in knowledge_dir.rglob("*.json"):
                try:
                    data = json.loads(fp.read_text(errors="ignore"))
                    if isinstance(data, list):
                        texts.extend(str(x) for x in data if isinstance(x, str))
                    elif isinstance(data, dict):
                        texts.extend(str(v) for v in data.values() if isinstance(v, str))
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Skipping %s: %s", fp, e)

        sessions_dir = repo_root / "data" / "sessions"
        if sessions_dir.exists():
            for fp in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(fp.read_text(errors="ignore"))
                    if isinstance(data, list):
                        for msg in data:
                            if isinstance(msg, dict) and "content" in msg:
                                texts.append(msg["content"])
                    elif isinstance(data, dict) and "messages" in data:
                        for msg in data["messages"]:
                            if isinstance(msg, dict) and "content" in msg:
                                texts.append(msg["content"])
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Skipping %s: %s", fp, e)

        # Datasets directory
        datasets_dir = repo_root / "data"
        if datasets_dir.exists():
            for fp in datasets_dir.rglob("*.txt"):
                try:
                    texts.append(fp.read_text(errors="ignore"))
                except OSError as e:
                    log.warning("Skipping %s: %s", fp, e)
            for fp in datasets_dir.rglob("*.jsonl"):
                try:
                    for line in fp.read_text(errors="ignore").splitlines():
                        texts.append(line)
                except OSError as e:
                    log.warning("Skipping %s: %s", fp, e)

    # Filter empty / tiny texts
    texts = [t.strip() for t in texts if len(t.strip()) > 20]

    if len(texts) < 2:
        log.error("Not enough training data. Provide --corpus or add knowledge files.")
        return

    log.header("Training Text Embedder")
    log.key_value("Texts", str(len(texts)))
    log.key_value("Epochs", str(getattr(args, "epochs", 20)))
    log.key_value("Embed dim", str(getattr(args, "embed_dim", 384)))
    log.blank()

    # ── Test mode: retrieve against the collected corpus ─────────────
    test_query = getattr(args, "test", None)
    if test_query:
        from domains.inference.slo_embedder import SloTextEmbedder
        embedder = SloTextEmbedder.load()
        if embedder is None:
            log.error("No trained embedder found. Run training first: sloughgpt train embed")
            return
        vec = embedder.embed(test_query)
        log.success(f"Embedding for '{test_query}': dim={len(vec)}, norm={sum(x*x for x in vec)**0.5:.4f}")
        _embedder_retrieval_check(embedder, texts, query=test_query)
        return

    # ── Train ─────────────────────────────────────────────────────────
    from domains.inference.slo_embedder import train_embedder

    total_epochs = getattr(args, "epochs", 20)
    from utils.training_progress import TrainingProgressBar
    pbar = TrainingProgressBar(desc="Training embedder", total_steps=total_epochs)

    def progress(epoch, loss, total):
        pbar.update({
            "global_step": epoch,
            "progress_percent": int(epoch * 100 / total) if total else 100,
            "epoch": epoch,
            "epochs": total,
            "train_loss": loss,
        })

    result = train_embedder(
        texts=texts,
        vocab_size=getattr(args, "vocab_size", 4096),
        embed_dim=getattr(args, "embed_dim", 384),
        epochs=getattr(args, "epochs", 20),
        lr=getattr(args, "lr", 3e-4),
        batch_size=getattr(args, "batch_size", 32),
        save_path=getattr(args, "output", None),
        progress_callback=progress,
    )
    pbar.finish()

    log.blank()
    log.success("Embedder trained")
    log.key_value("Final loss", f"{result['final_loss']:.4f}")
    log.key_value("Vocab size", str(result["vocab_size"]))
    log.key_value("Parameters", f"{result['n_params']:,}")
    log.key_value("Saved to", result["save_path"])
    log.blank()
    log.info("The embedder is now used automatically by KnowledgeMemory and vector search.")
    log.info("No sentence-transformers download needed.")

    # ── Retrieval sanity check: prove the saved artifact works ───────
    from domains.inference.slo_embedder import SloTextEmbedder
    embedder = SloTextEmbedder.load(result["save_path"])
    if embedder is None:
        log.warning("Trained embedder could not be reloaded — verify --output path.")
    else:
        _embedder_retrieval_check(embedder, texts)
        quality = getattr(embedder, "quality", None) or {}
        if quality:
            acceptable = getattr(embedder, "acceptable", lambda: False)()
            verdict = (
                "accepted for vector search"
                if acceptable
                else "REJECTED — vector search will use the n-gram fallback"
            )
            log.key_value("Quality gate", verdict)
            log.key_value(
                "Probe pairs",
                f"degenerate={quality.get('degenerate_fraction', 1.0):.2%} "
                f"mean_cos={quality.get('mean_cosine', 1.0):.2f} "
                f"nn_agreement={quality.get('nn_agreement', 0.0):.2f}",
            )
            retrieval = quality.get("retrieval") or {}
            if retrieval:
                better = retrieval.get("better", "n_gram")
                log.key_value(
                    "Retrieval vs n-gram",
                    f"trained MRR={retrieval.get('trained_mrr', 0.0):.2f} "
                    f"vs n-gram MRR={retrieval.get('ngram_mrr', 0.0):.2f} "
                    f"(hit@{retrieval.get('top_k', 3)} "
                    f"trained={retrieval.get('trained_hit', 0.0):.2f} / "
                    f"n-gram={retrieval.get('ngram_hit', 0.0):.2f})",
                )
                log.key_value("Retrieval verdict", f"{better} embedder wins")


def cmd_distill(args):
    """Distill GPT-2 teacher into a smaller SloTransformer student."""
    import threading
    import json

    api_mode = getattr(args, "api", False)
    text_source = getattr(args, "text_source", None)
    file_path = getattr(args, "file", None)

    # Resolve text source
    text = None
    if file_path:
        from pathlib import Path
        p = Path(file_path)
        if not p.exists():
            log.error(f"File not found: {file_path}")
            return
        text = p.read_text(encoding="utf-8")
        log.info(f"Loaded {len(text):,} chars from {file_path}")
    elif text_source:
        from pathlib import Path
        p = Path(text_source)
        if p.is_dir():
            # Try standard dataset files
            for name in ("input.txt", "corpus.jsonl", "train.txt"):
                candidate = p / name
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8")
                    log.info(f"Loaded {len(text):,} chars from {candidate}")
                    break
            if text is None:
                log.error(f"No training data found in {text_source}")
                return
        elif p.is_file():
            text = p.read_text(encoding="utf-8")
            log.info(f"Loaded {len(text):,} chars from {text_source}")
        else:
            log.error(f"Not found: {text_source}")
            return

    if not text or not text.strip():
        log.error("No training text provided")
        return

    # Apply preset
    preset = getattr(args, "preset", None)
    n_embed = getattr(args, "n_embed", 128)
    n_layer = getattr(args, "n_layer", 4)
    n_head = getattr(args, "n_head", 4)
    block_size = getattr(args, "block_size", 128)

    if preset == "tiny":
        n_embed, n_layer, n_head, block_size = 64, 2, 4, 64
    elif preset == "small":
        n_embed, n_layer, n_head, block_size = 128, 4, 4, 128
    elif preset == "medium":
        n_embed, n_layer, n_head, block_size = 256, 6, 8, 256

    log.header("Knowledge Distillation (GPT-2 to Student)")
    log.key_value("Teacher", "gpt2")
    log.key_value("Student", f"{n_embed}d {n_layer}L {n_head}H")
    log.key_value("Context", str(block_size))
    log.key_value("Temperature", str(getattr(args, "temperature", 4.0)))
    log.key_value("Text", f"{len(text):,} chars")
    log.blank()

    # ── API mode ──────────────────────────────────────────────────────
    if api_mode:
        import requests
        base_url = f"http://{args.host}:{args.port}"

        # For API mode, resolve the dataset name the server can find
        dataset_name = text_source or file_path
        if dataset_name:
            from pathlib import Path
            p = Path(dataset_name)
            if p.is_dir():
                dataset_name = p.name
            elif p.is_file():
                # Check if it's inside a known dataset dir
                parts = p.parts
                if "data" in parts:
                    idx = parts.index("data")
                    if idx + 1 < len(parts):
                        dataset_name = parts[idx + 1]
                else:
                    dataset_name = p.stem
        else:
            dataset_name = "custom"

        payload = {
            "teacher_model": "gpt2",
            "dataset": dataset_name,
            "epochs": getattr(args, "epochs", 10),
            "temperature": getattr(args, "temperature", 4.0),
            "name": f"distill-{int(time.time())}",
        }
        try:
            resp = requests.post(f"{base_url}/training/distill", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                job_id = data.get("job_id")
                log.success(f"Distillation started: {job_id}")
                _stream_api_progress(base_url, job_id)
            else:
                log.error(f"Failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            log.error(f"API error: {e}")
        return

    # ── Local mode ────────────────────────────────────────────────────
    if not _gpt2_teacher_cached():
        log.warning("GPT-2 teacher weights are not in the local HuggingFace cache.")
        log.warning("The first run downloads ~500MB from HuggingFace Hub.")
        log.error("Aborting — download the teacher first, then retry:")
        log.info("  huggingface-cli download gpt2 --include '*.safetensors' --include tokenizer.json")
        return

    from domains.training.distill_gpt2 import DistillConfig, distill_gpt2_to_slo

    resume_path = getattr(args, "resume", None)
    if resume_path:
        log.info(f"Resuming from checkpoint: {resume_path}")

    config = DistillConfig(
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=getattr(args, "dropout", 0.1),
        epochs=getattr(args, "epochs", 10),
        lr=getattr(args, "lr", 3e-4),
        batch_size=getattr(args, "batch_size", 8),
        temperature=getattr(args, "temperature", 4.0),
        checkpoint_dir=getattr(args, "checkpoint_dir", "models/auto-training"),
        log_interval=getattr(args, "log_interval", 10),
        resume_checkpoint=resume_path,
    )

    # Progress bar — compute total steps from config
    samples_per_epoch = len(text) // config.block_size
    total_steps = config.epochs * (samples_per_epoch // config.batch_size)
    from utils.training_progress import TrainingProgressBar
    pbar = TrainingProgressBar(desc="Distilling", total_steps=total_steps)

    def on_step(step, loss, epoch):
        pbar.update({
            "global_step": step,
            "progress_percent": int(step * 100 / total_steps) if total_steps else 100,
            "epoch": epoch + 1,
            "epochs": config.epochs,
            "train_loss": loss,
        })

    cancel_event = threading.Event()

    def on_sigint(sig, frame):
        log.blank()
        log.warning("Cancelling distillation...")
        cancel_event.set()

    import signal
    old_handler = signal.signal(signal.SIGINT, on_sigint)

    try:
        start_time = time.time()
        student, metadata = distill_gpt2_to_slo(
            text, config,
            on_step=on_step,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - start_time

        pbar.finish()
        log.blank()
        log.success(f"Distillation complete ({format_time(elapsed)})")
        log.key_value("Checkpoint", metadata.get("checkpoint", "?"))
        log.key_value("Final loss", metadata.get("final_loss", "?"))
        log.key_value("Best loss", metadata.get("best_loss", "?"))
        log.key_value("Epochs", metadata.get("epochs", "?"))
        log.key_value("Steps", metadata.get("steps", "?"))
        log.key_value("Student params", f"{sum(p.data.size for p in student.parameters()):,}")
        ppl = metadata.get("perplexity")
        bleu = metadata.get("bleu_vs_teacher")
        if ppl is not None:
            log.key_value("Perplexity", f"{float(ppl):.2f}")
        if bleu is not None:
            log.key_value("BLEU vs teacher", f"{float(bleu):.1f}%")

        log.blank()
        log.header("Next steps")
        ckpt = metadata.get("checkpoint", "")
        log.info("Load this model in chat by pointing the server at it and restarting:")
        log.info(f"  SLO_NATIVE_SOUL_PATH={ckpt} python3 apps/api/server/main.py")
        log.info("The checkpoint also appears in the training-page checkpoint catalog.")

    except Exception as e:
        log.blank()
        log.error(f"Distillation failed: {e}")
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)


def _stream_api_progress(base_url, job_id):
    """Stream training progress from API via polling `/training/jobs/{job_id}`."""
    import time

    log.info("Streaming progress... (Ctrl+C to detach)")

    bar = ProgressBar(total=100, desc="Training", width=36, show_eta=True, show_speed=False)

    try:
        import requests
        while True:
            try:
                resp = requests.get(f"{base_url}/training/jobs/{job_id}", timeout=5)
                if resp.status_code != 200:
                    log.error(f"Poll failed: {resp.status_code}")
                    return

                job = resp.json()
                status = job.get("status", "unknown")
                progress = job.get("progress", 0)
                epoch = job.get("current_epoch", job.get("epoch", 0))
                epochs = job.get("epochs", 0)
                loss = job.get("train_loss", job.get("loss", 0))
                checkpoint = job.get("checkpoint", "")

                # Update bar with extra info
                bar.desc = f"Epoch {epoch}/{epochs} loss={loss or 0:.4f}"
                bar.set_progress(progress)

                if status in ("completed", "failed", "error"):
                    bar.finish()
                    if status == "completed":
                        log.success("Training completed")
                        if checkpoint:
                            log.key_value("Checkpoint", checkpoint)
                        fl = job.get("train_loss") or job.get("loss")
                        if fl:
                            log.key_value("Final loss", str(fl))
                    else:
                        log.error(f"Training {status}: {job.get('error', 'unknown')}")
                    return

            except KeyboardInterrupt:
                bar.finish()
                log.info("Detached from training (job continues on server)")
                return
            except Exception as e:
                bar.finish()
                log.error(f"Poll error: {e}")
                return

            time.sleep(3)

    except Exception as e:
        log.error(f"Progress stream error: {e}")


def cmd_train_from_sessions(args):
    """Train a SloNet model on API chat logs via the server.

    Posts to /training/from-sessions-start, then streams SSE from
    /training/from-sessions-stream for live progress.
    """
    import json as _json
    import requests

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    base_url = f"http://{host}:{port}"

    log.header("Train from API Logs")

    # Check server health
    try:
        health = requests.get(f"{base_url}/health", timeout=3).json()
        if not health.get("model_loaded"):
            log.warning("No model loaded on server. Training will still work,")
            log.warning("but you need a model to test the checkpoint afterwards.")
    except (requests.RequestException, ValueError) as e:
        log.error("Cannot reach server. Is it running? (make api)")
        return

    # Count available pairs
    try:
        sessions_dir = Path("data/chat_sessions")
        corpus_file = Path("data/api_conversations/corpus.jsonl")
        n_sessions = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0
        n_corpus = 0
        if corpus_file.exists():
            with open(corpus_file) as f:
                n_corpus = sum(1 for _ in f)
        log.info(f"Available data: {n_sessions} sessions, {n_corpus} corpus entries")
        if n_sessions == 0 and n_corpus == 0:
            log.error("No chat data found. Use the chat first to generate training data.")
            return
    except OSError as e:
        log.warning("Could not count training data: %s", e)

    # Start training
    session_ids = None
    if getattr(args, "session_ids", None):
        session_ids = [s.strip() for s in args.session_ids.split(",")]

    payload = {
        "epochs": getattr(args, "epochs", 5),
        "learning_rate": getattr(args, "lr", 3e-4),
        "batch_size": getattr(args, "batch_size", 8),
        "n_embed": getattr(args, "n_embed", 128),
        "n_layer": getattr(args, "n_layer", 4),
        "n_head": getattr(args, "n_head", 4),
        "block_size": getattr(args, "block_size", 128),
        "dropout": getattr(args, "dropout", 0.1),
        "soul_name": getattr(args, "soul_name", "chat-trained"),
        "min_pair_quality": getattr(args, "min_quality", 2.0),
        "max_pairs": getattr(args, "max_pairs", 500),
        "session_ids": session_ids,
    }

    try:
        resp = requests.post(f"{base_url}/training/from-sessions-start", json=payload, timeout=10)
        if not resp.ok:
            log.error(f"Failed to start: {resp.status_code} {resp.text}")
            return
    except Exception as e:
        log.error(f"Failed to start training: {e}")
        return

    log.success("Training started — streaming progress...")

    # Stream SSE progress
    bar = ProgressBar(total=100, desc="Training", width=36, show_eta=True, show_speed=False)
    checkpoint_name = None
    final_loss = None
    num_pairs = None

    try:
        import requests as _req
        with _req.get(
            f"{base_url}/training/from-sessions-stream",
            stream=True,
            timeout=300,
        ) as stream:
            for line in stream.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    event = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue

                status = event.get("status", "")
                phase = event.get("phase", "")
                data = event.get("data", {})
                message = event.get("message", "")

                if phase == "PAIRS" and status == "working":
                    log.info(f"  {message}")
                    bar.desc = "Extracting pairs"
                    bar.set_progress(5)

                elif phase == "TRAIN" and status == "working":
                    step = data.get("step", 0)
                    loss = data.get("loss", 0)
                    epoch = event.get("meta", {}).get("epoch", 0)
                    total_epochs = event.get("meta", {}).get("total_epochs", 0)
                    # Estimate progress: epoch progress + step progress
                    if total_epochs > 0:
                        pct = int(((epoch - 1) / total_epochs) * 90 + 10)
                        bar.desc = f"Epoch {epoch}/{total_epochs} loss={loss:.4f}"
                    bar.set_progress(min(pct, 99))

                elif status == "complete":
                    bar.set_progress(100)
                    bar.finish()
                    checkpoint_name = data.get("checkpoint", "")
                    final_loss = data.get("final_loss")
                    num_pairs = data.get("num_pairs")
                    perplexity = data.get("perplexity")
                    samples = data.get("samples", [])

                    log.blank()
                    log.success("Training complete!")
                    log.key_value("Checkpoint", checkpoint_name)
                    if final_loss is not None:
                        log.key_value("Final loss", f"{final_loss:.4f}")
                    if num_pairs is not None:
                        log.key_value("Training pairs", str(num_pairs))
                    if perplexity is not None:
                        log.key_value("Perplexity", f"{perplexity:.2f}")
                    if samples:
                        log.info("Sample outputs:")
                        for s in samples[:3]:
                            log.info(f"  {s.get('prompt', '')} -> {s.get('response', '')[:80]}")
                    break

                elif status == "error":
                    bar.finish()
                    log.error(f"Training failed: {message}")
                    return

    except KeyboardInterrupt:
        bar.finish()
        log.info("Training cancelled")
        return
    except Exception as e:
        bar.finish()
        log.error(f"Stream error: {e}")
        return

    # Auto-load checkpoint
    if checkpoint_name and getattr(args, "auto_load", False):
        log.info(f"Loading checkpoint into chat: {checkpoint_name}")
        try:
            load_resp = requests.post(
                f"{base_url}/training/checkpoints/{checkpoint_name}/load",
                timeout=30,
            )
            if load_resp.ok:
                log.success(f"Loaded '{checkpoint_name}' — ready to chat!")
            else:
                log.warning(f"Load failed ({load_resp.status_code}): {load_resp.text}")
                log.info(f"Load manually: sloughgpt checkpoint load {checkpoint_name}")
        except Exception as e:
            log.warning(f"Load failed: {e}")
            log.info(f"Load manually: sloughgpt checkpoint load {checkpoint_name}")
    elif checkpoint_name:
        log.info(f"Next step: sloughgpt checkpoint load {checkpoint_name}")

    # JSON output
    if getattr(args, "json_output", False):
        import json as _json2
        result = {
            "checkpoint": checkpoint_name,
            "final_loss": final_loss,
            "num_pairs": num_pairs,
        }
        print(_json2.dumps(result, indent=2))
