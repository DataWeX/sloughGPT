# SLO Interactive CLI System

**Status:** This describes a **planned** `slo` / `slo_cli.py` shell. The monorepo’s supported CLI is **`python3 cli.py`** (repo root → **`apps/cli/cli.py`**) or the **`sloughgpt`** console script after **`python3 -m pip install -e ".[dev]"`** — see **QUICKSTART.md**.

The text below is retained as a design sketch for a future interactive wrapper.

## Features

**🎯 Interactive Mode**: Run `slo` for an interactive menu system
**🚀 Command Mode**: Run `slo train small` for direct commands  
**🎨 Rich Interface**: Beautiful tables, progress bars, and colors
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
**🎨 Visual Feedback**: Rich output with tables, progress bars, and status indicators
**⚡ Efficient**: Tab completion, command history, and keyboard shortcuts
**🔧 Extensible**: Easy to add new commands and features

## Implementation Details

The CLI is built with:
- **Rich** for beautiful terminal output
- **Readline** for tab completion and history
- **Modular design** for easy extension
- **Fallback modes** when dependencies aren't available
- **Error handling** with user-friendly messages

This transforms our complex ML workflow into an intuitive, interactive experience that's much more accessible than remembering individual script names and arguments.