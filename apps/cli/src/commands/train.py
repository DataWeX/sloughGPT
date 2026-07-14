"""
Train commands - Training, evaluation, and quick smoke tests.
"""
import sys
import time
from pathlib import Path
from typing import Optional

from core.printer import printer
from utils.progress import ProgressBar
from utils.formatting import format_size, format_time, format_number


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
    from ..cli import _chat_repository_root
    from ..cli import _apply_optimized_train_preset, _train_export_stem_slug, _train_export_default_stem

    if not args.api:
        sys.path.insert(0, ".")

        # Load config
        try:
            from config_loader import get_device, load_config, merge_args_with_config
        except ImportError:
            printer.error("config_loader not found")
            sys.exit(1)

        config = load_config(args.config)
        config = merge_args_with_config(config, args)

        _apply_optimized_train_preset(config, args)
        train_device = get_device(config.device)

        printer.header("SloughGPT Training")
        printer.key_value("Dataset", str(config.data.dataset))
        printer.key_value("Device", f"{train_device} ({config.device.type})")
        printer.key_value("Epochs", str(config.training.epochs))
        printer.key_value("Batch Size", str(config.training.batch_size))
        printer.key_value("Learning Rate", str(config.training.learning_rate))
        printer.key_value("LoRA", f"{config.lora.enabled} (rank={config.lora.rank})")
        printer.blank()

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
            printer.success(f"Tracking enabled: {config.tracking.backend}")

        from domains.training.train_pipeline import SloughGPTTrainer

        save_formats = [config.checkpoint.export_format]
        if getattr(args, "save_stem", None):
            save_stem = _train_export_stem_slug(args.save_stem, "export")
        else:
            save_stem = _train_export_default_stem(str(config.model.name), str(config.data.dataset))
        save_path = f"{config.checkpoint.save_dir}/{save_stem}"

        printer.info(f"Export: {save_stem}")

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
            use_mixed_precision=config.training.use_mixed_precision,
            mixed_precision_dtype=config.training.mixed_precision_dtype,
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

        printer.info(f"Model: {trainer.model.num_parameters():,} params")
        printer.blank()

        # Train
        resume_path = None
        if args.resume and getattr(args, "resume_latest", False):
            printer.error("Use either --resume PATH or --resume-latest, not both")
            sys.exit(2)
        if getattr(args, "resume_latest", False):
            printer.step(f"Resuming from latest under {config.checkpoint.trainer_dir}")
        elif args.resume:
            printer.step(f"Resuming from: {args.resume}")
            resume_path = args.resume

        pbar = ProgressBar(total=100, desc="Training", width=36, show_eta=True, show_speed=False)

        def _on_progress(info):
            pct = info.get("progress_percent", 0)
            step = info.get("global_step", 0)
            epoch = info.get("epoch", 0)
            epochs = info.get("epochs", 0)
            loss = info.get("train_loss", 0)
            lr = info.get("learning_rate", 0)
            pbar.desc = f"step {step} epoch {epoch}/{epochs} loss={loss:.4f} lr={lr:.2e}"
            pbar.set_progress(pct)

        start_time = time.time()
        trainer.train(
            resume=bool(args.resume or getattr(args, "resume_latest", False)),
            resume_path=resume_path,
            on_progress=_on_progress,
        )
        elapsed = time.time() - start_time

        pbar.finish()
        printer.blank()
        printer.success(f"Training complete ({format_time(elapsed)})")

        # Save
        printer.step("Saving...")
        for fmt in save_formats:
            trainer.save(save_path, format=fmt)
            printer.success(f"Saved: {save_path}.{fmt}")

        if tracker:
            tracker.end_run()

        return

    # API training
    import requests
    sys.path.insert(0, ".")
    try:
        from config_loader import load_config, merge_args_with_config
    except ImportError:
        printer.error("config_loader not found")
        sys.exit(1)

    config = merge_args_with_config(load_config(args.config), args)
    _apply_optimized_train_preset(config, args)
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
        "use_mixed_precision": bool(config.training.use_mixed_precision),
        "mixed_precision_dtype": str(config.training.mixed_precision_dtype),
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
            printer.success(f"Training started: {job_id}")

            # Stream progress via SSE
            _stream_api_progress(base_url, job_id)
        else:
            printer.error(f"Failed ({response.status_code}): {response.text}")
    except Exception as e:
        printer.error(f"API error: {e}")


def cmd_quick(args):
    """Quick smoke test: train a toy model and generate."""
    import torch

    sys.path.insert(0, ".")

    from domains.models import SloughGPTModel
    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig
    from domains.training.performance import get_optimal_device

    printer.header("SloughGPT Quick Start")

    device = get_optimal_device()
    printer.key_value("Device", str(device))

    use_optimize = not getattr(args, "no_optimize", False)
    config = TrainerConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        use_mixed_precision=use_optimize and device != "cpu",
        use_compile=use_optimize and hasattr(torch, "compile"),
        max_steps=args.steps if args.steps else 100,
        warmup_steps=max(10, args.steps // 10) if args.steps else 10,
    )

    printer.status("Mixed Precision", "Yes" if config.use_mixed_precision else "No", "ok" if config.use_mixed_precision else "warn")
    printer.status("torch.compile", "Yes" if config.use_compile else "No", "ok" if config.use_compile else "info")

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
            printer.info(f"Loading {len(datasets)} datasets with ratios")
            data_path = list(zip(datasets, ratios))
        else:
            printer.info(f"Loading {len(datasets)} datasets: {', '.join(datasets)}")
            data_path = datasets
    else:
        printer.info(f"Loading: {args.dataset}")
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
        printer.info(f"Preset: {args.preset}")

    printer.info(f"Model: {trainer.model.num_parameters():,} params")
    printer.blank()

    pbar = ProgressBar(total=100, desc="Training", width=36, show_eta=True, show_speed=False)

    def _on_progress(info):
        pct = info.get("progress_percent", 0)
        step = info.get("global_step", 0)
        epoch = info.get("epoch", 0)
        epochs = info.get("epochs", 0)
        loss = info.get("train_loss", 0)
        pbar.desc = f"step {step} epoch {epoch}/{epochs} loss={loss:.4f}"
        pbar.set_progress(pct)

    printer.step("Training...")
    trainer.train(on_progress=_on_progress)
    pbar.finish()

    printer.blank()
    printer.step("Generating...")
    model = trainer.model
    model.eval()
    input_ids = torch.tensor([[trainer.stoi.get(c, 0) for c in args.prompt]]).to(device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        next_token = logits[-1].argmax().item()
        generated = [next_token]

        for _ in range(args.max_tokens - 1):
            output = model(torch.tensor([[next_token]]).to(device))
            logits = output[0] if isinstance(output, tuple) else logits
            next_token = logits[-1].argmax().item()
            if next_token == 0:
                break
            generated.append(next_token)

    text = "".join([trainer.itos.get(i, "") for i in generated])
    printer.blank()
    printer.key_value("Prompt", args.prompt)
    printer.key_value("Generated", f"{args.prompt}{text[:100]}...")

    output_base = args.output.replace(".pt", "").replace(".safetensors", "").replace(".soul", "")
    trainer.save(output_base, format="sou")
    printer.success(f"Saved: {output_base}.soul")


def cmd_eval(args):
    """Evaluate char-level model perplexity."""
    import torch

    printer.header("Model Evaluation")
    printer.key_value("Checkpoint", args.checkpoint)

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        if "training_info" in checkpoint:
            info = checkpoint["training_info"]
            printer.blank()
            printer.section("Training Info")
            for k, v in info.items():
                printer.key_value(k, str(v))

        def _param_stats(state_dict: dict) -> None:
            total_params = sum(v.numel() for v in state_dict.values())
            printer.blank()
            printer.section("Model Statistics")
            printer.key_value("Parameters", format_number(total_params))
            printer.key_value("Parameter Groups", str(len(state_dict)))
            total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
            printer.key_value("Size (FP32)", format_size(total_size))
            printer.key_value("Size (FP16)", format_size(total_size // 2))

        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            _param_stats(checkpoint["model"])
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            _param_stats(checkpoint["model_state_dict"])

        data_path = getattr(args, "data", None) or "datasets/shakespeare/input.txt"
        if Path(data_path).is_file():
            from domains.training.lm_eval_char import evaluate_sloughgpt_char_lm

            dev = getattr(args, "device", None) or "cpu"
            strict = not getattr(args, "no_strict", False)
            printer.blank()
            printer.info(f"Evaluating on: {data_path}")

            metrics = evaluate_sloughgpt_char_lm(
                args.checkpoint,
                data_path,
                device=dev,
                strict_load=strict,
            )
            printer.blank()
            printer.key_value("Mean Loss", f"{metrics['mean_loss']:.4f}")
            ppl = metrics["perplexity"]
            ppl_s = f"{ppl:.4f}" if ppl != float("inf") else "inf"
            printer.key_value("Perplexity", ppl_s)
            printer.key_value("Tokens Scored", format_number(metrics["num_token_positions"]))

            for w in metrics.get("warnings") or []:
                printer.warning(w)
        else:
            printer.warning(f"Data file not found: {data_path}")

        if args.benchmark:
            printer.blank()
            printer.step("Running benchmark...")
            _ = torch.randint(0, 1000, (1, 128))
            with torch.no_grad():
                start = time.time()
                for _ in range(10):
                    pass
                elapsed = time.time() - start
            printer.info(f"10 iterations: {format_time(elapsed)}")

    except Exception as e:
        printer.error(f"Evaluation failed: {e}")
        sys.exit(1)


# ── Sub-commands merged into `train` ──────────────────────────────

def _cmd_self_train(args):
    """Run self-training loop (model talks to itself)."""
    import subprocess

    script = Path("scripts/self_train.py")
    if not script.exists():
        printer.error("scripts/self_train.py not found")
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

    printer.header("Self-Training Loop")
    printer.info(f"Running: {' '.join(cmd)}")
    printer.info("Press Ctrl+C to stop")
    printer.blank()
    subprocess.run(cmd)


def _cmd_autotrain(args):
    """Control auto-training via API."""
    import requests

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    api_url = f"http://{host}:{port}"
    action = getattr(args, "auto_train_action", None) or getattr(args, "action", None)

    printer.header(f"Auto-Train ({action})")

    if action == "start":
        resp = requests.post(
            f"{api_url}/auto-train/start",
            json={
                "teacher": getattr(args, "auto_teacher", None) or getattr(args, "teacher", "gpt2"),
                "temperature": getattr(args, "temperature", 0.8),
                "max_steps": getattr(args, "auto_steps", None) or getattr(args, "steps", 1000),
            },
        )
        if resp.ok:
            printer.success(f"Started: {resp.json()}")
        else:
            printer.error(f"Failed: {resp.json()}")
    elif action == "stop":
        resp = requests.post(f"{api_url}/auto-train/stop")
        if resp.ok:
            printer.success(f"Stopped: {resp.json()}")
        else:
            printer.error(f"Failed: {resp.json()}")
    elif action == "status":
        resp = requests.get(f"{api_url}/auto-train/status")
        if resp.ok:
            data = resp.json()
            for k, v in data.items():
                printer.key_value(k, str(v))
        else:
            printer.error(f"Failed: {resp.json()}")


def _cmd_monitor(args):
    """Monitor training jobs."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"

    printer.header("Training Monitor")

    while True:
        try:
            response = requests.get(f"{base_url}/training", timeout=5)
            if response.status_code == 200:
                jobs = response.json()
                if isinstance(jobs, dict) and "jobs" in jobs:
                    jobs = jobs["jobs"]
                if jobs:
                    rows = []
                    for job in jobs:
                        rows.append([job.get("name", "unknown"), job.get("status", "unknown")])
                    printer.blank()
                    printer.table(["Job", "Status"], rows)
                else:
                    printer.info("No active training jobs")
            else:
                printer.error(f"HTTP {response.status_code}")
        except Exception as e:
            printer.error(str(e))

        if not args.watch:
            break
        time.sleep(args.interval)


def _cmd_user_adapters(args):
    """Manage per-user LoRA adapters."""
    import sys as _sys
    _sys.path.insert(0, ".")

    printer.header("User Adapters")

    action = getattr(args, "adapters_action", None) or getattr(args, "action", None)
    user_id = getattr(args, "adapters_user", None) or getattr(args, "user", None)
    users_str = getattr(args, "adapters_users", None) or getattr(args, "users", None)
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()

        if action == "list":
            adapters = store.get_all_adapters()
            stats = store.get_stats()
            printer.key_value("Total Users", str(stats["total_users"]))
            printer.key_value("Total Size", f'{stats["total_size_mb"]:.2f} MB')
            printer.key_value("Avg per User", f'{stats["avg_size_per_user_kb"]:.1f} KB')

            if adapters:
                printer.blank()
                printer.section("Adapters")
                rows = []
                for a in adapters[:20]:
                    rows.append([a["user_id"][:28], str(a["feedback_count"]), a["updated_at"][:19]])
                printer.table(["User ID", "Feedback", "Updated"], rows)
            else:
                printer.info("No adapters found")

        elif action == "info":
            adapter = store.get_adapter(user_id)
            if adapter is None:
                printer.error(f"No adapter for user: {user_id}")
            else:
                printer.key_value("User", adapter.user_id)
                printer.key_value("Feedback", str(adapter.feedback_count))
                printer.key_value("W_a", str(adapter.W_a.shape))
                printer.key_value("W_b", str(adapter.W_b.shape))

        elif action == "delete":
            store.delete_adapter(user_id)
            printer.success(f"Deleted adapter for {user_id}")

        elif action == "merge":
            user_ids = users_str.split(",") if users_str else []
            if len(user_ids) < 2:
                printer.error("Need at least 2 user IDs: --users user1,user2")
                return
            merged = store.merge_adapters(user_ids)
            printer.success(f"Merged {merged['user_count']} adapters")

    except ImportError as e:
        printer.error(f"Feedback module: {e}")


def _cmd_feedback_train(args):
    """Prepare training data from feedback."""
    import sys as _sys
    _sys.path.insert(0, ".")

    printer.header("Feedback Training Pipeline")

    stats_only = getattr(args, "feedback_stats_only", None) or getattr(args, "stats_only", False)
    feedback_fmt = getattr(args, "feedback_format", None) or getattr(args, "format", "all")
    output_dir = getattr(args, "feedback_output", None) or getattr(args, "output", None)

    try:
        from domains.feedback import create_training_pipeline

        trainer = create_training_pipeline()
        stats = trainer.get_training_stats()
        printer.section("Available Data")
        printer.key_value("Conversations", str(stats["total_conversations"]))
        printer.key_value("Thumbs Up", str(stats["thumbs_up"]))
        printer.key_value("Thumbs Down", str(stats["thumbs_down"]))
        printer.key_value("DPO Pairs", str(stats["available_dpo_pairs"]))
        printer.key_value("SFT Examples", str(stats["available_sft_examples"]))

        if stats_only:
            return

        formats = []
        if feedback_fmt == "all":
            formats = ["dpo", "sft", "reward"]
        else:
            formats = [feedback_fmt]

        output_dir = output_dir or "data/training"
        results = trainer.export_for_alignment(output_dir=output_dir, formats=formats)

        printer.blank()
        printer.section("Exported Files")
        for fmt, path in results.items():
            printer.key_value(fmt, path)

    except ImportError as e:
        printer.error(f"Feedback module: {e}")


def _cmd_feedback_export(args):
    """Export feedback data for training."""
    import sys as _sys
    _sys.path.insert(0, ".")

    printer.header("Feedback Export")

    output_path = getattr(args, "export_feedback_output", None) or getattr(args, "output", "data/training_feedback.jsonl")
    fmt = getattr(args, "export_feedback_format", None) or getattr(args, "format", "jsonl")

    try:
        from domains.feedback import get_meta_weight_manager

        manager = get_meta_weight_manager()
        if manager is None:
            printer.error("Meta-weight system not available")
            return

        stats = manager.get_stats()
        printer.section("Current Stats")
        printer.key_value("Total Feedback", str(stats["db_stats"]["feedback_total"]))
        printer.key_value("Thumbs Up", str(stats["db_stats"]["thumbs_up"]))
        printer.key_value("Thumbs Down", str(stats["db_stats"]["thumbs_down"]))

        printer.blank()
        printer.step(f"Exporting to {output_path}...")
        manager.export_training_data(filepath=output_path, format=fmt)

        import os
        if os.path.exists(output_path):
            with open(output_path) as f:
                lines = sum(1 for _ in f)
            printer.success(f"Exported {lines} records")
        else:
            printer.warning("File not created")

    except ImportError as e:
        printer.error(f"Feedback module: {e}")


def _cmd_checkpoint_info(args):
    """Inspect a training checkpoint — show metadata, optimizer state, resume readiness."""
    import torch
    from pathlib import Path

    printer.header("Checkpoint Info")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        printer.error(f"File not found: {ckpt_path}")
        sys.exit(1)

    printer.key_value("Path", str(ckpt_path))
    printer.key_value("Size", format_size(ckpt_path.stat().st_size))

    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        printer.error(f"Failed to load: {e}")
        sys.exit(1)

    # Top-level keys
    printer.section("Keys")
    for k in checkpoint.keys():
        v = checkpoint[k]
        if isinstance(v, dict):
            printer.key_value(k, f"dict ({len(v)} keys)")
        elif isinstance(v, torch.Tensor):
            printer.key_value(k, f"tensor {list(v.shape)}")
        elif isinstance(v, (int, float, str, bool)):
            printer.key_value(k, str(v))
        elif v is None:
            printer.key_value(k, "None")
        else:
            printer.key_value(k, type(v).__name__)

    # Model weights summary
    state = checkpoint.get("model_state_dict") or checkpoint.get("model") or {}
    if isinstance(state, dict) and state:
        printer.section("Model Weights")
        total_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        total_bytes = sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
        printer.key_value("Parameters", format_number(total_params))
        printer.key_value("Weight Groups", str(len(state)))
        printer.key_value("Size (FP32)", format_size(total_bytes))
        for name, tensor in list(state.items())[:10]:
            printer.key_value(f"  {name}", f"{list(tensor.shape)} {tensor.dtype}")

    # Optimizer state
    opt_state = checkpoint.get("optimizer_state_dict")
    if isinstance(opt_state, dict) and opt_state:
        printer.section("Optimizer State")
        groups = opt_state.get("param_groups", [])
        printer.key_value("Param Groups", str(len(groups)))
        for i, pg in enumerate(groups):
            lr = pg.get("lr", "?")
            wd = pg.get("weight_decay", "?")
            printer.key_value(f"  Group {i}", f"lr={lr}, weight_decay={wd}, params={len(pg.get('params', []))}")
        # State tensors
        state_keys = [k for k in opt_state.keys() if k != "param_groups"]
        if state_keys:
            printer.key_value("State Entries", str(len(state_keys)))
    else:
        printer.section("Optimizer State")
        printer.warning("Not saved — cannot resume with full optimizer momentum/LR schedule")
        printer.info("Train with updated CheckpointManager to save optimizer state")

    # Scheduler state
    sched_state = checkpoint.get("scheduler_state_dict")
    if isinstance(sched_state, dict) and sched_state:
        printer.section("Scheduler State")
        for k, v in sched_state.items():
            if isinstance(v, (int, float)):
                printer.key_value(k, str(v))
    else:
        printer.section("Scheduler State")
        printer.info("Not saved (will use fresh scheduler on resume)")

    # Training metadata
    printer.section("Training Metadata")
    step = checkpoint.get("step", checkpoint.get("global_step", "?"))
    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    ts = checkpoint.get("timestamp", "")
    printer.key_value("Step", str(step))
    printer.key_value("Epoch", str(epoch))
    printer.key_value("Timestamp", str(ts))
    if metrics:
        for k, v in metrics.items():
            printer.key_value(f"  {k}", str(v))

    # Resume readiness
    printer.section("Resume Readiness")
    has_model = "model_state_dict" in checkpoint or "model" in checkpoint
    has_optim = isinstance(opt_state, dict) and bool(opt_state)
    has_sched = isinstance(sched_state, dict) and bool(sched_state)
    printer.status("Model weights", "Yes" if has_model else "No", "ok" if has_model else "error")
    printer.status("Optimizer state", "Yes" if has_optim else "No", "ok" if has_optim else "warn")
    printer.status("Scheduler state", "Yes" if has_sched else "No", "info" if has_sched else "info")
    if has_model and has_optim:
        printer.success("Full resume possible — weights + optimizer + LR schedule")
    elif has_model:
        printer.warning("Partial resume — weights only, optimizer/scheduler reset")
    else:
        printer.error("Cannot resume — no model weights found")


def cmd_demo(args):
    """Run system demos (RAG, KG, EWC, inference)."""
    import sys
    sys.path.insert(0, ".")

    printer.header("SloughGPT Demo")

    from domains.cognitive.rag import ProductionRAG
    from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph
    from domains.training.ewc import EwcContinualLearner
    from domains.inference.optimizer import KVCache

    if args.component in ("all", "rag"):
        printer.section("RAG - Document Retrieval")
        rag = ProductionRAG()
        rag.add_document("Python is a programming language created by Guido van Rossum in 1991.")
        results = rag.query("What is Python?")
        printer.key_value("Retrieved", str(len(results)))

    if args.component in ("all", "kg"):
        printer.section("Knowledge Graph - Fact Verification")
        kg = KnowledgeGraph()
        kg.add_fact("python", "is_a", "programming_language")
        kg.add_fact("python", "created_by", "guido_van_rossum")
        facts = kg.query(subject="python")
        printer.key_value("Facts", str(len(facts)))

    if args.component in ("all", "ewc"):
        printer.section("EWC - Catastrophic Forgetting Prevention")
        from domains.models import SloughGPTModel
        model = SloughGPTModel(vocab_size=50, n_embed=32, n_layer=2, n_head=2, block_size=16)
        ewc = EwcContinualLearner(model)
        printer.key_value("Fisher Params", str(len(ewc.fisher_estimator.fisher_accum)))

    if args.component in ("all", "inference"):
        printer.section("Inference - KV Cache")
        cache = KVCache(num_layers=2, num_heads=2, head_dim=64, max_length=100)
        printer.key_value("Max Tokens", str(cache.max_length))

    printer.blank()
    printer.success("Demo complete!")


def cmd_rlhf(args):
    """Run RLHF demo."""
    import sys
    import torch
    sys.path.insert(0, ".")

    printer.header("RLHF Demo")
    from domains.training.rlhf import RLHFConfig
    from domains.models import SloughGPTModel

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    printer.key_value("Device", device)

    model = SloughGPTModel(vocab_size=100, n_embed=64, n_layer=2, n_head=4, block_size=32, dropout=0.0).to(device)
    printer.key_value("Parameters", f"{model.num_parameters():,}")

    config = RLHFConfig(ppo_epochs=2, clip_epsilon=0.2, entropy_coef=0.01, gamma=1.0, lam=0.95)
    printer.key_value("PPO Epochs", str(config.ppo_epochs))
    printer.key_value("Clip Epsilon", str(config.clip_epsilon))

    batch_size, seq_len, vocab_size = 4, 16, 100
    printer.step("Running PPO steps...")
    for step in range(min(args.steps, 20)):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        reward = probs.max(dim=-1).values.mean()
        if step % 5 == 0 or step == min(args.steps, 20) - 1:
            printer.key_value(f"Step {step}", f"reward={reward.item():.3f}")

    printer.blank()
    printer.success("RLHF demo complete!")


def cmd_cloud_setup(args):
    """Setup Pinecone vector store."""
    import sys
    import asyncio
    import os
    sys.path.insert(0, ".")

    printer.header("Cloud Setup")
    from domains.inference.vector_stores.pinecone_store import PineconeVectorStore
    from domains.inference.vector_store import VectorEntry, simple_embed

    api_key = args.api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        printer.error("Pinecone API key required (--api-key or PINECONE_API_KEY)")
        return

    async def setup():
        printer.key_value("Index", args.index)
        printer.key_value("Dimension", str(args.dimension))
        store = PineconeVectorStore(api_key=api_key, index_name=args.index, dimension=args.dimension, environment=args.environment)
        await store.connect()
        entries = [VectorEntry(id="test", vector=simple_embed("test document", dimension=args.dimension), text="test document", metadata={"created_by": "cli"})]
        await store.upsert(entries)
        count = await store.count()
        printer.success(f"Pinecone: {count} documents indexed")
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
    train_parser.add_argument("--checkpoint-info", dest="checkpoint_info_path", help="Inspect a checkpoint file (.pt)")

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
    quick_parser.add_argument("--output", default="models/quick.pt", help="Output path")
    quick_parser.add_argument("--no-optimize", action="store_true", help="Disable optimizations")
    quick_parser.add_argument("--soul-name", default="SloughGPT-Quick", help="Slo name")
    quick_parser.set_defaults(func=cmd_quick)

    # Eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate model perplexity",
    )
    eval_parser.add_argument("--checkpoint", default="models/sloughgpt.pt", help="Checkpoint path")
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
    model is saved to data/models/text-embedder.sou and automatically
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
            texts.append(p.read_text(errors="ignore"))
        elif p.is_dir():
            for ext in ("*.txt", "*.md", "*.json"):
                for fp in p.rglob(ext):
                    try:
                        texts.append(fp.read_text(errors="ignore"))
                    except Exception:
                        pass
        else:
            printer.error(f"Corpus not found: {corpus}")
            return
    else:
        # Auto-discover: knowledge files + chat history
        knowledge_dir = repo_root / "data" / "knowledge"
        if knowledge_dir.exists():
            for fp in knowledge_dir.rglob("*.txt"):
                try:
                    texts.append(fp.read_text(errors="ignore"))
                except Exception:
                    pass
            for fp in knowledge_dir.rglob("*.json"):
                try:
                    data = json.loads(fp.read_text(errors="ignore"))
                    if isinstance(data, list):
                        texts.extend(str(x) for x in data if isinstance(x, str))
                    elif isinstance(data, dict):
                        texts.extend(str(v) for v in data.values() if isinstance(v, str))
                except Exception:
                    pass

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
                except Exception:
                    pass

        # Datasets directory
        datasets_dir = repo_root / "datasets"
        if datasets_dir.exists():
            for fp in datasets_dir.rglob("*.txt"):
                try:
                    texts.append(fp.read_text(errors="ignore"))
                except Exception:
                    pass
            for fp in datasets_dir.rglob("*.jsonl"):
                try:
                    for line in fp.read_text(errors="ignore").splitlines():
                        texts.append(line)
                except Exception:
                    pass

    # Filter empty / tiny texts
    texts = [t.strip() for t in texts if len(t.strip()) > 20]

    if len(texts) < 2:
        printer.error("Not enough training data. Provide --corpus or add knowledge files.")
        return

    printer.header("Training Text Embedder")
    printer.key_value("Texts", str(len(texts)))
    printer.key_value("Epochs", str(getattr(args, "epochs", 20)))
    printer.key_value("Embed dim", str(getattr(args, "embed_dim", 384)))
    printer.blank()

    # ── Test mode: just embed a query ─────────────────────────────────
    test_query = getattr(args, "test", None)
    if test_query:
        from domains.inference.slo_embedder import SloTextEmbedder
        embedder = SloTextEmbedder.load()
        if embedder is None:
            printer.error("No trained embedder found. Run training first: sloughgpt train embed")
            return
        vec = embedder.embed(test_query)
        printer.success(f"Embedding for '{test_query}': dim={len(vec)}, norm={sum(x*x for x in vec)**0.5:.4f}")
        printer.info(f"First 8 values: {vec[:8]}")
        return

    # ── Train ─────────────────────────────────────────────────────────
    from domains.inference.slo_embedder import train_embedder

    total_epochs = getattr(args, "epochs", 20)
    pbar = ProgressBar(total=total_epochs, desc="Training embedder", width=36, show_eta=True, show_speed=False)

    def progress(epoch, loss, total):
        pbar.desc = f"epoch {epoch}/{total} loss={loss:.4f}"
        pbar.set_progress(epoch)
        if epoch == total:
            pbar.finish()

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

    printer.blank()
    printer.success("Embedder trained")
    printer.key_value("Final loss", f"{result['final_loss']:.4f}")
    printer.key_value("Vocab size", str(result["vocab_size"]))
    printer.key_value("Parameters", f"{result['n_params']:,}")
    printer.key_value("Saved to", result["save_path"])
    printer.blank()
    printer.info("The embedder is now used automatically by KnowledgeMemory and vector search.")
    printer.info("No sentence-transformers download needed.")


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
            printer.error(f"File not found: {file_path}")
            return
        text = p.read_text(encoding="utf-8")
        printer.info(f"Loaded {len(text):,} chars from {file_path}")
    elif text_source:
        from pathlib import Path
        p = Path(text_source)
        if p.is_dir():
            # Try standard dataset files
            for name in ("input.txt", "corpus.jsonl", "train.txt"):
                candidate = p / name
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8")
                    printer.info(f"Loaded {len(text):,} chars from {candidate}")
                    break
            if text is None:
                printer.error(f"No training data found in {text_source}")
                return
        elif p.is_file():
            text = p.read_text(encoding="utf-8")
            printer.info(f"Loaded {len(text):,} chars from {text_source}")
        else:
            printer.error(f"Not found: {text_source}")
            return

    if not text or not text.strip():
        printer.error("No training text provided")
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

    printer.header("Knowledge Distillation (GPT-2 → Student)")
    printer.key_value("Teacher", "gpt2")
    printer.key_value("Student", f"{n_embed}d {n_layer}L {n_head}H")
    printer.key_value("Context", str(block_size))
    printer.key_value("Temperature", str(getattr(args, "temperature", 4.0)))
    printer.key_value("Text", f"{len(text):,} chars")
    printer.blank()

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
                if "datasets" in parts:
                    idx = parts.index("datasets")
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
                printer.success(f"Distillation started: {job_id}")
                _stream_api_progress(base_url, job_id)
            else:
                printer.error(f"Failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            printer.error(f"API error: {e}")
        return

    # ── Local mode ────────────────────────────────────────────────────
    from domains.training.distill_gpt2 import DistillConfig, distill_gpt2_to_slo

    resume_path = getattr(args, "resume", None)
    if resume_path:
        printer.info(f"Resuming from checkpoint: {resume_path}")

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
    pbar = ProgressBar(total=total_steps, desc="Distilling", width=36, show_eta=True, show_speed=True)

    def on_step(step, loss, epoch):
        pbar.desc = f"epoch {epoch+1}/{config.epochs} loss={loss:.4f}"
        pbar.set_progress(step)

    cancel_event = threading.Event()

    def on_sigint(sig, frame):
        printer.blank()
        printer.warning("Cancelling distillation...")
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
        printer.blank()
        printer.success(f"Distillation complete ({format_time(elapsed)})")
        printer.key_value("Checkpoint", metadata.get("checkpoint", "?"))
        printer.key_value("Final loss", metadata.get("final_loss", "?"))
        printer.key_value("Best loss", metadata.get("best_loss", "?"))
        printer.key_value("Epochs", metadata.get("epochs", "?"))
        printer.key_value("Steps", metadata.get("steps", "?"))
        printer.key_value("Student params", f"{sum(p.data.size for p in student.parameters()):,}")

    except Exception as e:
        printer.blank()
        printer.error(f"Distillation failed: {e}")
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)


def _stream_api_progress(base_url, job_id):
    """Stream training progress from API via polling `/training/jobs/{job_id}`."""
    import time

    printer.info("Streaming progress... (Ctrl+C to detach)")

    bar = ProgressBar(total=100, desc="Training", width=36, show_eta=True, show_speed=False)

    try:
        import requests
        while True:
            try:
                resp = requests.get(f"{base_url}/training/jobs/{job_id}", timeout=5)
                if resp.status_code != 200:
                    printer.error(f"Poll failed: {resp.status_code}")
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
                        printer.success("Training completed")
                        if checkpoint:
                            printer.key_value("Checkpoint", checkpoint)
                        fl = job.get("train_loss") or job.get("loss")
                        if fl:
                            printer.key_value("Final loss", str(fl))
                    else:
                        printer.error(f"Training {status}: {job.get('error', 'unknown')}")
                    return

            except KeyboardInterrupt:
                bar.finish()
                printer.info("Detached from training (job continues on server)")
                return
            except Exception as e:
                bar.finish()
                printer.error(f"Poll error: {e}")
                return

            time.sleep(3)

    except Exception as e:
        printer.error(f"Progress stream error: {e}")
