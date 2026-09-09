"""
Tokenizer command group — tokenizer management and text analysis.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register tokenizer commands with the CLI group."""

    @cli.group(help="Tokenizer management and text analysis")
    def tokenizer():
        pass

    @tokenizer.command("tokenize", help="Tokenize text")
    @click.argument("text")
    @click.pass_context
    def tokenizer_tokenize(ctx, text):
        r = api_post(ctx, "/tokenizer/tokenize", json={"text": text})
        if r.status_code != 200:
            log.error(f"Tokenize failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        tokens = data.get("data", data).get("tokens", [])
        log.info(f"Tokens: {tokens}")
        log.info(f"Count: {len(tokens)}")

    @tokenizer.command("detokenize", help="Convert token IDs back to text")
    @click.argument("ids")
    @click.pass_context
    def tokenizer_detokenize(ctx, ids):
        id_list = [int(x.strip()) for x in ids.split(",")]
        r = api_post(ctx, "/tokenizer/detokenize", json={"ids": id_list})
        if r.status_code != 200:
            log.error(f"Detokenize failed: {r.text}")
            return
        data = r.json()
        text = data.get("data", data).get("text", "")
        log.info(f"Text: {text}")

    @tokenizer.command("analyze", help="Analyze token distribution in text")
    @click.argument("text")
    @click.pass_context
    def tokenizer_analyze(ctx, text):
        r = api_post(ctx, "/tokenizer/analyze", json={"texts": [text]})
        if r.status_code != 200:
            log.error(f"Analyze failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        info = data.get("data", data)
        for k, v in info.items():
            if k != "texts":
                log.info(f"  {k}: {v}")

    @tokenizer.command("vocab", help="Show vocabulary")
    @click.option("--limit", "-n", default=20, type=int, help="Max entries")
    @click.pass_context
    def tokenizer_vocab(ctx, limit):
        r = api_get(ctx, f"/tokenizer/vocab?limit={limit}")
        if r.status_code != 200:
            log.error(f"Vocab failed: {r.text}")
            return
        data = r.json()
        vocab = data.get("data", data).get("vocab", {})
        if output_json(ctx, {"vocab": vocab}):
            return
        log.header("Vocabulary")
        for token_id, token_str in vocab.items():
            log.info(f"  {token_id}: {token_str}")

    @tokenizer.command("merges", help="Show BPE merge rules")
    @click.option("--limit", "-n", default=20, type=int, help="Max merges")
    @click.pass_context
    def tokenizer_merges(ctx, limit):
        r = api_get(ctx, f"/tokenizer/merges?limit={limit}")
        if r.status_code != 200:
            log.error(f"Merges failed: {r.text}")
            return
        data = r.json()
        merges = data.get("data", data).get("merges", [])
        if output_json(ctx, {"merges": merges}):
            return
        log.header("BPE Merges")
        for m in merges:
            log.info(f"  {m}")

    @tokenizer.command("train", help="Train tokenizer on texts")
    @click.option("--vocab-size", type=int, default=512, help="Vocabulary size")
    @click.option("--texts", default="", help="Comma-separated training texts")
    @click.pass_context
    def tokenizer_train(ctx, vocab_size, texts):
        text_list = [t.strip() for t in texts.split(",") if t.strip()] if texts else []
        r = api_post(ctx, "/tokenizer/train",
                     json={"vocab_size": vocab_size, "texts": text_list})
        if r.status_code != 200:
            log.error(f"Train failed: {r.text}")
            return
        log.success("Tokenizer trained")

    @tokenizer.command("stats", help="Show tokenizer statistics")
    @click.pass_context
    def tokenizer_stats(ctx):
        r = api_get(ctx, "/tokenizer/stats")
        if r.status_code != 200:
            log.error(f"Stats failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Tokenizer Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    return tokenizer
