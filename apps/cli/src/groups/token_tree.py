"""
Token-tree command group — train, encode, decode, and query a tree tokenizer.
"""

from core.framework import click
from core.helpers import ns as _ns
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register token-tree commands with the CLI group."""

    @cli.group("token-tree", help="Train, encode, decode, and query a tree tokenizer")
    @click.pass_context
    def token_tree(ctx):
        pass

    @token_tree.command("train", help="Train a tree tokenizer from a corpus and save it")
    @click.option("--corpus", "-c", default="datasets/tinyshakespeare/input.txt", help="Corpus file or dataset name")
    @click.option("--vocab-size", "-v", default=512, type=int, help="Target vocabulary size")
    @click.option("--embed-dim", "-e", default=64, type=int, help="Embedding dimension (0 disables embeddings)")
    @click.option("--min-freq", default=2, type=int, help="Minimum pair frequency to merge")
    @click.option("--output", "-o", default="models/slonet-native/token_tree", help="Save base path")
    @click.pass_context
    def token_tree_train(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_train
        cmd_token_tree_train(_ns(**kwargs))

    @token_tree.command("encode", help="Encode text into token ids (reads stdin when no --text)")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--text", default=None, help="Text to encode")
    @click.pass_context
    def token_tree_encode(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_encode
        cmd_token_tree_encode(_ns(**kwargs))

    @token_tree.command("decode", help="Decode comma-separated token ids back to text")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.argument("ids")
    @click.pass_context
    def token_tree_decode(ctx, ids, **kwargs):
        from commands.token_tree import cmd_token_tree_decode
        kwargs["ids"] = ids
        cmd_token_tree_decode(_ns(**kwargs))

    @token_tree.command("stats", help="Show training statistics for a saved tree")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.pass_context
    def token_tree_stats(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_stats
        cmd_token_tree_stats(_ns(**kwargs))

    @token_tree.command("similar", help="Find nearest-neighbor tokens via generated embeddings")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--top-k", "-k", default=5, type=int, help="Number of results")
    @click.argument("token")
    @click.pass_context
    def token_tree_similar(ctx, token, **kwargs):
        from commands.token_tree import cmd_token_tree_similar
        kwargs["token"] = token
        cmd_token_tree_similar(_ns(**kwargs))

    @token_tree.command("lineage", help="Render a token's merge lineage down to its leaves")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.argument("token")
    @click.pass_context
    def token_tree_lineage(ctx, token, **kwargs):
        from commands.token_tree import cmd_token_tree_lineage
        kwargs["token"] = token
        cmd_token_tree_lineage(_ns(**kwargs))

    @token_tree.command("vocab", help="List a paged slice of the vocabulary with flags")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--offset", default=0, type=int, help="Number of leading entries to skip")
    @click.option("--limit", "-n", default=50, type=int, help="Maximum entries to print (0 = no limit)")
    @click.pass_context
    def token_tree_vocab(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_vocab
        cmd_token_tree_vocab(_ns(**kwargs))

    @token_tree.command("embedding", help="Inspect a token's generated embedding vector")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--top-k", "-k", default=8, type=int, help="Largest-magnitude dimensions to show")
    @click.argument("token")
    @click.pass_context
    def token_tree_embedding(ctx, token, **kwargs):
        from commands.token_tree import cmd_token_tree_embedding
        kwargs["token"] = token
        cmd_token_tree_embedding(_ns(**kwargs))

    @token_tree.command("path", help="Trace the greedy trie walk over text (reads stdin when no --text)")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--text", default=None, help="Text to trace")
    @click.pass_context
    def token_tree_path(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_path
        cmd_token_tree_path(_ns(**kwargs))

    @token_tree.command("matrix", help="Summarize the full embedding matrix")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--top-k", "-k", default=8, type=int, help="Most/least energetic tokens to show")
    @click.pass_context
    def token_tree_matrix(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_matrix
        cmd_token_tree_matrix(_ns(**kwargs))

    @token_tree.command("compare", help="Diff two saved token trees by name")
    @click.option("--a", "-a", "a_name", required=True, help="First saved tree name")
    @click.option("--b", "-b", "b_name", required=True, help="Second saved tree name")
    @click.option("--top-k", "-k", default=10, type=int, help="Shared/exclusive token examples per side")
    @click.pass_context
    def token_tree_compare(ctx, a_name, b_name, top_k):
        from commands.token_tree import cmd_token_tree_compare
        cmd_token_tree_compare(_ns(a=a_name, b=b_name, top_n=top_k))

    @token_tree.command("merges", help="List the most frequent BPE merge rules of a saved tree")
    @click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
    @click.option("--top-n", "-n", default=20, type=int, help="Maximum merge rules to show")
    @click.option("--query", "-q", default="", help="Filter rules whose parts contain this substring")
    @click.pass_context
    def token_tree_merges(ctx, **kwargs):
        from commands.token_tree import cmd_token_tree_merges
        cmd_token_tree_merges(_ns(**kwargs))

    @token_tree.command("saved", help="List saved token trees")
    @click.pass_context
    def token_tree_saved(ctx):
        from commands.token_tree import cmd_token_tree_saved
        cmd_token_tree_saved(_ns())

    @token_tree.command("save", help="Save the current tree (or --tree path) under a name")
    @click.option("--name", "-n", "name", required=True, help="Name to save the tree under")
    @click.option("--tree", "-t", default=None, help="Optional saved tree base path to adopt first")
    @click.pass_context
    def token_tree_save(ctx, name, **kwargs):
        from commands.token_tree import cmd_token_tree_save
        cmd_token_tree_save(_ns(name=name, **kwargs))

    @token_tree.command("load", help="Load a saved tree by name and make it current")
    @click.argument("name")
    @click.pass_context
    def token_tree_load(ctx, name):
        from commands.token_tree import cmd_token_tree_load
        cmd_token_tree_load(_ns(name=name))

    @token_tree.command("delete", help="Delete a saved token tree by name")
    @click.argument("name")
    @click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
    @click.pass_context
    def token_tree_delete(ctx, name, dry_run):
        from commands.token_tree import cmd_token_tree_delete
        if dry_run:
            log.info(f"Would delete token tree: {name}")
            return
        cmd_token_tree_delete(_ns(name=name))

    return token_tree
