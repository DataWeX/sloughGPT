# SLO Interactive CLI System

**Status:** This describes a **planned** `slo` / `slo_cli.py` shell. The monorepo’s supported CLI is **`./sloughgpt`** (repo root → **`apps/cli/cli.py`**) or the **`sloughgpt`** console script after **`python3 -m pip install -e ".[dev]"`** — see **QUICKSTART.md**.

The text below is retained as a design sketch for a future interactive wrapper.

## Features

**🎯 Interactive Mode**: Run `slo` for an interactive menu system
**🚀 Command Mode**: Run `slo train small` for direct commands  
**🎨 ANSI Interface**: Native terminal output with colors, tables, and progress bars
**🔍 Auto-discovery**: Automatically finds configs, datasets, and models
**⚡ Smart Prompts**: Interactive selection menus for complex operations

## Usage

### Interactive Mode (planned)
```bash
# Planned future entrypoints (not currently shipped in this repo):
./slo
python3 slo_cli.py
```

### Command Mode (planned)
```bash
./slo help
./slo config list
./slo data prepare shakespeare
./slo train start small
./slo system info
./slo status
```

## Command Categories

### 🔧 Configuration Management
```bash
slo config list           # List all configs
slo config show small     # Show config details
slo config validate small # Validate config
slo config edit small     # Edit config in $EDITOR
```

### 🚀 Training Workflow  
```bash
slo train start small     # Start training
slo train status         # Show training status
slo train logs          # Show training logs
slo train stop          # Stop training
```

### 📊 Data Management
```bash
slo data list            # List available datasets
slo data prepare shakespeare  # Prepare dataset
slo data info shakespeare     # Show dataset info
slo data clean shakespeare    # Clean dataset
```

### 🤖 Model Operations
```bash
slo model list           # List trained models
slo model info model    # Show model details
slo model chat model    # Chat with model (placeholder)
slo model evaluate model # Evaluate model (placeholder)
```

### 🖥️ System Utilities
```bash
slo system info         # Show system information
slo system check       # Check requirements
slo system benchmark    # Run benchmarks
slo clean              # Clean temp files
```

## Benefits

**🧭 User-Friendly**: No need to remember complex commands or file paths
**🎯 Context-Aware**: Auto-discovery of configs, datasets, and models
**🔄 Workflow-Oriented**: Commands follow natural training workflow
**🎨 Visual Feedback**: Native ANSI output with tables, progress bars, and status indicators
**⚡ Efficient**: Tab completion, command history, and keyboard shortcuts
**🔧 Extensible**: Easy to add new commands and features

## Implementation Details

The CLI is built with:
- **Native ANSI** for terminal output (colors, tables, progress bars)
- **Readline** for tab completion and history
- **Modular design** for easy extension
- **Fallback modes** when dependencies aren't available
- **Error handling** with user-friendly messages

This transforms our complex ML workflow into an intuitive, interactive experience that's much more accessible than remembering individual script names and arguments.

---

## Auto-Memory Command Group (current `sloughgpt` CLI)

The shipped `sloughgpt` CLI exposes a `memory` command group that wraps the core auto-memory service (`packages/core-py/domains/memory/`). The group is fail-closed: with `SLO_MEMORY_ENABLED=false` every subcommand reports that memory is disabled.

| Command | Description |
|---------|-------------|
| `sloughgpt memory stats` | Show memory statistics (enabled, total facts, topic buckets, visited URLs). |
| `sloughgpt memory enable` | Turn the memory master switch on at runtime (updates the shared config). |
| `sloughgpt memory disable` | Turn the memory master switch off at runtime — every other subcommand then no-ops. |
| `sloughgpt memory list` | List stored facts, most recent first. |
| `sloughgpt memory search "query"` | Semantic-search stored facts by relevance. |
| `sloughgpt memory store "fact"` | Persist one explicit fact. |
| `sloughgpt memory remember "user msg" --assistant "reply"` | Persist one completed turn (user + assistant) for auto-extraction. |
| `sloughgpt memory consolidate --threshold 0.85` | Merge near-duplicate facts (default threshold from `SLO_MEMORY_CONSOLIDATION_THRESHOLD`). |
| `sloughgpt memory archive [--limit N]` | Show the task-backed provenance archive: path, record counts per task type, and the most recent records (newest first). |
| `sloughgpt memory archive --prune-days 30` | Delete archive records older than the retention window (prompts for confirmation). |
| `sloughgpt memory clear` | Remove all stored memory. |

Example:

```bash
sloughgpt memory stats
sloughgpt memory disable
sloughgpt memory store "The user prefers the code editor Zed"
sloughgpt memory search "editor preferences"
sloughgpt memory archive --limit 10
sloughgpt memory clear
sloughgpt memory enable
```

## Token Tree Command Group (current `sloughgpt` CLI)

The `sloughgpt token-tree` group wraps the tree-based BPE tokenizer (`packages/core-py/domains/training/token_tree.py`). Commands read a saved tree (`.slnp` matrix + `.json` metadata) from `--tree` (`-t`, default `models/slonet-native/token_tree`), matching the `.soul`-embedded tokenizer format.

| Command | Description |
|---------|-------------|
| `sloughgpt token-tree train --corpus ... --vocab-size ... --embed-dim ...` | Train a tree tokenizer from a corpus file or dataset name and save it. |
| `sloughgpt token-tree encode --text "..."` | Encode text into token ids (reads stdin when `--text` is omitted). |
| `sloughgpt token-tree decode "1,2,3"` | Decode comma-separated token ids back to text. |
| `sloughgpt token-tree stats` | Show training statistics (vocab size, merges, embedding points/ratio). |
| `sloughgpt token-tree similar "quick" --top-k 5` | Find nearest-neighbor tokens via generated embeddings. |
| `sloughgpt token-tree lineage "quick"` | Render a token's merge lineage down to its character leaves. |
| `sloughgpt token-tree vocab --offset 0 --limit 50` | List a paged slice of the vocabulary with id/freq/merged flags. |
| `sloughgpt token-tree embedding "quick" --top-k 8` | Inspect a token's generated embedding vector (largest-magnitude dims). |
| `sloughgpt token-tree path --text "..."` | Trace the encoder's greedy trie walk over text (reads stdin when `--text` is omitted). |
| `sloughgpt token-tree matrix --top-k 8` | Summarize the full embedding matrix (shape, L2 norm min/mean/max, live/dead tokens, most/least energetic tokens). |
| `sloughgpt token-tree merges --top-n 20 [--query "qu"]` | List the most frequent BPE merge rules; `--query` filters rules whose parts contain the substring. |
| `sloughgpt token-tree compare A B --top-k 10` | Diff two saved trees: per-side stats, vocab/merge overlap, top shared and exclusive tokens. |
| `sloughgpt token-tree saved` | List saved token trees (name, vocab/merge counts, path), newest first. |
| `sloughgpt token-tree save NAME [--tree PATH]` | Save the current tree (or adopt a tree from `--tree`) under NAME. |
| `sloughgpt token-tree load NAME` | Load a saved tree by name and make it the current tree. |
| `sloughgpt token-tree delete NAME` | Delete a saved token tree by name. |

Example:

```bash
sloughgpt token-tree stats
sloughgpt token-tree encode --text "to be or not to be"
sloughgpt token-tree similar "quick" --top-k 3
sloughgpt token-tree matrix --top-k 5
```

