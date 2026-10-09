"""
Train command group — training, evaluation, and monitoring.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register train commands with the CLI group."""

    @cli.group(help="Train, evaluate, and monitor models")
    @click.pass_context
    def train(ctx):
        pass

    @train.command("start", help="Full training pipeline")
    @click.option("--dataset", default="shakespeare", help="Dataset name")
    @click.option("--epochs", default=3, type=int, help="Training epochs")
    @click.option("--batch-size", default=32, type=int, help="Batch size")
    @click.option("--lr", default=0.01, type=float, help="Learning rate")
    @click.option("--api", is_flag=True, help="Use API training")
    @click.option("--resume", default=None, help="Resume from checkpoint")
    @click.option("--resume-latest", is_flag=True, help="Resume latest")
    @click.option("--save-stem", default=None, help="Output filename stem")
    @click.pass_context
    def train_start(ctx, dataset, epochs, batch_size, lr, api, resume, resume_latest, save_stem):
        from commands.train import cmd_train
        kwargs = dict(
            dataset=dataset, epochs=epochs, batch_size=batch_size, lr=lr,
            api=api, resume=resume, resume_latest=resume_latest,
            save_stem=save_stem,
            host=ctx.obj["host"], port=ctx.obj["port"], config=ctx.obj["config"],
        )
        cmd_train(_ns(**kwargs))

    @train.command("native", help="Train a SloNet model from scratch (.soul checkpoints)")
    @click.option("--dataset", default="datasets/tinyshakespeare/input.txt", help="Corpus file or dataset name")
    @click.option("--steps", default=None, type=int, help="Max training steps (default: epoch budget)")
    @click.option("--embed", default=64, type=int, help="Embedding dimension")
    @click.option("--layers", default=2, type=int, help="Transformer layers")
    @click.option("--heads", default=4, type=int, help="Attention heads")
    @click.option("--block", default=128, type=int, help="Context block size")
    @click.option("--batch", default=16, type=int, help="Batch size")
    @click.option("--epochs", default=1, type=int, help="Training epochs")
    @click.option("--lr", default=3e-3, type=float, help="Learning rate")
    @click.option("--weight-decay", default=0.01, type=float, help="Weight decay")
    @click.option("--scheduler", default="cosine", help="LR scheduler (cosine/linear/constant)")
    @click.option("--warmup", default=100, type=int, help="Warmup steps")
    @click.option("--min-lr", default=1e-5, type=float, help="Minimum learning rate")
    @click.option("--grad-norm", default=1.0, type=float, help="Max gradient norm (0 disables clipping)")
    @click.option("--dropout", default=0.1, type=float, help="Dropout")
    @click.option("--checkpoint-dir", default="models/slonet-native", help="Checkpoint directory")
    @click.option("--checkpoint-interval", default=500, type=int, help="Checkpoint interval (steps)")
    @click.option("--max-checkpoints", default=3, type=int, help="Max checkpoints to keep")
    @click.option("--save-best-only", is_flag=True, help="Only keep best-eval checkpoints")
    @click.option("--eval-interval", default=250, type=int, help="Eval interval (steps)")
    @click.option("--log-interval", default=50, type=int, help="Progress log interval (steps)")
    @click.option("--soul-name", default="sloughgpt-native", help="Soul name for the checkpoint")
    @click.option("--save-stem", default=None, help="Output filename stem (default: soul name)")
    @click.option("--save-format", default="soul", type=click.Choice(["soul", "sou", "npz"]), help="DEPRECATED — ignored; SloughGPTTrainer.save() always writes .soul")
    @click.option("--resume", default=None, help="Resume from a .soul/.npz checkpoint path")
    @click.option("--resume-latest", is_flag=True, help="Resume from latest checkpoint in --checkpoint-dir")
    @click.option("--device", default="cpu", help="Device (cpu/auto)")
    @click.option("--tokenizer", default="char", type=click.Choice(["char", "token-tree"]), help="Tokenization strategy for the corpus")
    @click.option("--token-vocab-size", default=512, type=int, help="Token tree vocabulary size (token-tree tokenizer)")
    @click.option("--prompt", default=None, help="Generate a sample from this prompt after training")
    @click.pass_context
    def train_native(ctx, **kwargs):
        from commands.train import cmd_train_native
        kwargs["host"] = ctx.obj["host"]
        kwargs["port"] = ctx.obj["port"]
        cmd_train_native(_ns(**kwargs))

    @train.command("quick", help="Smoke test: train briefly and generate")
    @click.option("--dataset", "-d", default="datasets/shakespeare/input.txt", help="Corpus file")
    @click.option("--prompt", default="The king", help="Generation prompt")
    @click.option("--epochs", default=1, type=int, help="Training epochs")
    @click.option("--steps", default=100, type=int, help="Max steps")
    @click.option("--embed", default=128, type=int, help="Embedding size")
    @click.option("--layers", default=4, type=int, help="Transformer layers")
    @click.option("--heads", default=4, type=int, help="Attention heads")
    @click.option("--block", default=128, type=int, help="Context length")
    @click.option("--batch", default=16, type=int, help="Batch size")
    @click.option("--lr", default=1e-3, type=float, help="Learning rate")
    @click.option("--max-tokens", default=100, type=int, help="Generated tokens")
    @click.option("--temperature", default=0.8, type=float, help="Temperature")
    @click.option("--output", default="models/quick.soul", help="Output path")
    @click.option("--no-optimize", is_flag=True, help="Disable optimizations")
    @click.option("--soul-name", default="SloughGPT-Quick", help="Slo name")
    @click.option("--datasets", help="Comma-separated datasets (overrides --dataset)")
    @click.option("--ratios", help="Comma-separated dataset ratios")
    @click.option("--preset", type=click.Choice(["tiny", "small", "medium", "large"]), help="Model preset")
    @click.pass_context
    def train_quick(ctx, **kwargs):
        from commands.train import cmd_quick
        kwargs["host"] = ctx.obj["host"]
        kwargs["port"] = ctx.obj["port"]
        cmd_quick(_ns(**kwargs))

    @train.command("auto", help="Control auto-training via API")
    @click.argument("action", type=click.Choice(["start", "stop", "status"]))
    @click.option("--teacher", default="gpt2", help="Teacher model")
    @click.option("--temperature", default=0.8, type=float, help="Temperature")
    @click.option("--steps", default=1000, type=int, help="Max steps")
    @click.pass_context
    def train_auto(ctx, action, teacher, temperature, steps):
        from commands.train import _cmd_autotrain
        args = _ns(
            action=action, teacher=teacher, temperature=temperature,
            steps=steps, host=ctx.obj["host"], port=ctx.obj["port"],
        )
        _cmd_autotrain(args)

    @train.command(name="self", help="Model talks to itself")
    @click.option("--steps", default=1000, type=int, help="Training steps")
    @click.option("--model", default="gpt2", help="Teacher model")
    @click.option("--temperature", default=0.8, type=float, help="Temperature")
    @click.option("--max-tokens", default=50, type=int, help="Max tokens per generation")
    @click.option("--seed", default="Hello", help="Starting text")
    @click.option("--forever", is_flag=True, help="Run until Ctrl+C")
    def train_self(steps, model, temperature, max_tokens, seed, forever):
        from commands.train import _cmd_self_train
        args = _ns(
            steps=steps, model=model, temperature=temperature,
            max_tokens=max_tokens, seed=seed, forever=forever,
        )
        _cmd_self_train(args)

    @train.command("eval", help="Evaluate model perplexity")
    @click.option("--checkpoint", default="models/sloughgpt.soul", help="Checkpoint path")
    @click.option("--data", default="datasets/shakespeare/input.txt", help="Eval text")
    @click.option("--benchmark", is_flag=True, help="Run benchmark")
    def train_eval(checkpoint, data, benchmark):
        from commands.train import cmd_eval
        args = _ns(checkpoint=checkpoint, data=data, benchmark=benchmark)
        cmd_eval(args)

    @train.command("monitor", help="Monitor training jobs (delegates to dashboard)")
    @click.option("--watch", is_flag=True, help="Continuous watch (ignored — always live)")
    @click.option("--interval", default=2, type=int, help="Refresh interval (s)")
    @click.pass_context
    def train_monitor(ctx, watch, interval):
        from commands.monitor import monitor as _monitor_cmd
        ctx.invoke(_monitor_cmd, interval=float(interval), host=ctx.obj["host"], port=ctx.obj["port"], output_json=False, no_clear=False)

    @train.command("rlhf", help="Run RLHF demo")
    @click.option("--steps", default=20, type=int, help="PPO steps")
    def train_rlhf(steps):
        from commands.train import cmd_rlhf
        args = _ns(steps=steps)
        cmd_rlhf(args)

    @train.command("demo", help="Run system demos (RAG, KG, EWC)")
    @click.option("--component", type=click.Choice(["all", "rag", "kg", "ewc", "inference"]), default="all")
    def train_demo(component):
        from commands.train import cmd_demo
        args = _ns(component=component)
        cmd_demo(args)

    @train.command("cloud", help="Setup Pinecone vector store")
    @click.option("--api-key", help="Pinecone API key")
    @click.option("--index", default="sloughgpt", help="Index name")
    @click.option("--dimension", default=768, type=int, help="Vector dimension")
    @click.option("--environment", default="us-east-1", help="Pinecone environment")
    def train_cloud(api_key, index, dimension, environment):
        from commands.train import cmd_cloud_setup
        args = _ns(api_key=api_key, index=index, dimension=dimension, environment=environment)
        cmd_cloud_setup(args)

    @train.command("embed", help="Train a text embedder on your corpus (no downloads)")
    @click.option("--corpus", default=None, help="Text file or directory to train on (default: knowledge + chat history)")
    @click.option("--epochs", default=20, type=int, help="Training epochs")
    @click.option("--lr", default=3e-4, type=float, help="Learning rate")
    @click.option("--batch-size", default=32, type=int, help="Batch size")
    @click.option("--embed-dim", default=384, type=int, help="Embedding dimension")
    @click.option("--vocab-size", default=4096, type=int, help="Max vocabulary size")
    @click.option("--output", default=None, help="Output checkpoint path")
    @click.option("--test", default=None, help="Test: embed a query string and print top matches")
    def train_embed(corpus, epochs, lr, batch_size, embed_dim, vocab_size, output, test):
        """Train a text embedder on your own data using contrastive learning.

        \b
        Examples:
          sloughgpt train embed                          # train on knowledge + chat history
          sloughgpt train embed --corpus datasets/       # train on a directory of text files
          sloughgpt train embed --corpus my_corpus.txt   # train on a single file
          sloughgpt train embed --test "neural networks" # embed a test query
        """
        from commands.train import cmd_train_embed
        args = _ns(
            corpus=corpus, epochs=epochs, lr=lr, batch_size=batch_size,
            embed_dim=embed_dim, vocab_size=vocab_size, output=output, test=test,
        )
        cmd_train_embed(args)

    @train.command("distill", help="Distill a teacher model into a smaller student")
    @click.argument("text_source", required=False, default=None)
    @click.option("--file", "-f", default=None, help="Text file to train on")
    @click.option("--epochs", default=10, type=int, help="Training epochs")
    @click.option("--lr", default=3e-4, type=float, help="Learning rate")
    @click.option("--batch-size", default=8, type=int, help="Batch size")
    @click.option("--n-embed", default=128, type=int, help="Student embedding size")
    @click.option("--n-layer", default=4, type=int, help="Student layers")
    @click.option("--n-head", default=4, type=int, help="Student attention heads")
    @click.option("--block-size", default=128, type=int, help="Context length")
    @click.option("--temperature", default=4.0, type=float, help="Distillation temperature")
    @click.option("--dropout", default=0.1, type=float, help="Dropout rate")
    @click.option("--checkpoint-dir", default="models/auto-training", help="Save directory")
    @click.option("--log-interval", default=10, type=int, help="Log every N steps")
    @click.option("--preset", type=click.Choice(["tiny", "small", "medium"]), help="Architecture preset")
    @click.option("--api", is_flag=True, help="Use server API instead of local")
    @click.option("--json", "json_output", is_flag=True, help="JSON output")
    @click.option("--resume", default=None, help="Resume from checkpoint path (.soul file)")
    @click.pass_context
    def train_distill(ctx, text_source, file, epochs, lr, batch_size, n_embed, n_layer,
                      n_head, block_size, temperature, dropout, checkpoint_dir,
                      log_interval, preset, api, json_output, resume):
        """Distill GPT-2 into a smaller, faster student model.

        \b
        Examples:
          sloughgpt train distill datasets/shakespeare/input.txt
          sloughgpt train distill -f my_book.txt --epochs 20 --preset small
          sloughgpt train distill datasets/shakespeare/input.txt --api
          sloughgpt train distill datasets/shakespeare/input.txt --n-embed 64 --n-layer 2
          sloughgpt train distill datasets/shakespeare/input.txt --resume models/auto-training/checkpoint.soul
        """
        from commands.train import cmd_distill
        args = _ns(
            text_source=text_source, file=file, epochs=epochs, lr=lr,
            batch_size=batch_size, n_embed=n_embed, n_layer=n_layer,
            n_head=n_head, block_size=block_size, temperature=temperature,
            dropout=dropout, checkpoint_dir=checkpoint_dir,
            log_interval=log_interval, preset=preset, api=api,
            json_output=json_output, host=ctx.obj["host"], port=ctx.obj["port"],
            resume=resume,
        )
        cmd_distill(args)

    @train.command("from-sessions", help="Train on your API chat logs (sessions + response logs)")
    @click.option("--epochs", default=5, type=int, help="Training epochs")
    @click.option("--lr", default=3e-4, type=float, help="Learning rate")
    @click.option("--batch-size", default=8, type=int, help="Batch size")
    @click.option("--n-embed", default=128, type=int, help="Embedding dimension")
    @click.option("--n-layer", default=4, type=int, help="Transformer layers")
    @click.option("--n-head", default=4, type=int, help="Attention heads")
    @click.option("--block-size", default=128, type=int, help="Context block size")
    @click.option("--dropout", default=0.1, type=float, help="Dropout rate")
    @click.option("--soul-name", default="chat-trained", help="Name for the trained soul")
    @click.option("--min-quality", default=2.0, type=float, help="Min pair quality (0-5)")
    @click.option("--max-pairs", default=500, type=int, help="Max training pairs to use")
    @click.option("--session-ids", default=None, help="Comma-separated session IDs (default: all)")
    @click.option("--load", "auto_load", is_flag=True, help="Auto-load checkpoint into chat after training")
    @click.option("--json", "json_output", is_flag=True, help="JSON output")
    @click.pass_context
    def train_from_sessions(ctx, epochs, lr, batch_size, n_embed, n_layer, n_head,
                            block_size, dropout, soul_name, min_quality, max_pairs,
                            session_ids, auto_load, json_output):
        """Train a model on your API chat logs.

        \b
        Examples:
          sloughgpt train from-sessions                          # train with defaults
          sloughgpt train from-sessions --epochs 10 --lr 1e-3    # tune hyperparams
          sloughgpt train from-sessions --load                   # train + load into chat
          sloughgpt train from-sessions --max-pairs 1000         # use more data
        """
        from commands.train import cmd_train_from_sessions
        args = _ns(
            epochs=epochs, lr=lr, batch_size=batch_size,
            n_embed=n_embed, n_layer=n_layer, n_head=n_head,
            block_size=block_size, dropout=dropout,
            soul_name=soul_name, min_quality=min_quality,
            max_pairs=max_pairs, session_ids=session_ids,
            auto_load=auto_load, json_output=json_output,
            host=ctx.obj["host"], port=ctx.obj["port"],
        )
        cmd_train_from_sessions(args)

    return train
