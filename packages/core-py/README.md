## Core Python

`packages/core-py/` is on the Python path when you install from the repo root (**`python3 -m pip install -e ".[dev]"`**). It still holds the package's real content: `pyproject.toml`, `conftest.py`, `scripts/`, `config/`, and `benchmarks/`.

Training, inference, models, and infrastructure code now live under the repo-root **`domain/`** package (`domain/<area>/_internal/<module>.py`); the only code still shipped under the plural **`domains`** package is the deferred **`domains.infrastructure.pugqeep`** subtree. The API imports these modules; keep heavy logic here instead of in **`apps/api/server/`** route handlers. Trainer-native **`.soul`** checkpoints embed **`stoi` / `itos` / `chars`** for char-LM eval; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

### Key infrastructure modules

These modules left `packages/core-py` for the repo-root **`domain/`** package — except **pugqeep**, which remains here as **`domains.infrastructure.pugqeep`** while its port is deferred.

| Module | Purpose | Docs |
|--------|---------|------|
| `domain.infrastructure._internal.producer_consumer` | General-purpose bounded work queue with priority, backpressure, consumer thread pools | [PRODUCER_CONSUMER_QUEUE.md](../../docs/PRODUCER_CONSUMER_QUEUE.md) |
| `domains.infrastructure.pugqeep` | Point-Graph-Queue system: compressed data points, model trees, task queues, engine | [PUGQEEP.md](../../docs/PUGQEEP.md) |
| `domain.infrastructure._internal.cancel_manager` | Cancellation for long-running operations | [AGENTS.md](../../AGENTS.md) |
| `domain.infrastructure._internal.model_server` | Model lifecycle, backends, circuit breaker | [AGENTS.md](../../AGENTS.md) |
| `domain.infrastructure._internal.process_guard` | Subprocess crash isolation, auto-restart | [AGENTS.md](../../AGENTS.md) |

### Core domains

All six moved to the repo-root **`domain/`** package.

| Domain | Purpose |
|--------|---------|
| `domain.training` | Training pipelines, executor, distillation |
| `domain.inference` | Vector store, KV cache, providers |
| `domain.feedback` | LoRA eval, per-user adapter |
| `domain.cognitive` | Soul engine, metacognition |
| `domain.infrastructure` | ProcessGuard, ModelServer, TaskQueue, CancelManager |
| `domain.multimodal` | Phoneme encoders, TTS, speech synthesis |

### Phoneme Encoder API

The phoneme encoder system supports 6 languages: English, German, French, Spanish, Italian, and Portuguese.

#### CLI Usage

```bash
# Single text encoding
python -m domain.multimodal._internal.phoneme_encoder_cli "hello world"

# Batch encoding
python -m domain.multimodal._internal.phoneme_encoder_cli --batch "hello" "world" "test"

# Language-specific encoding
python -m domain.multimodal._internal.phoneme_encoder_cli --lang it "ciao mondo"

# Language detection
python -m domain.multimodal._internal.phoneme_encoder_cli --detect "hello world"
```

#### REST API Endpoints

- `POST /multimodal/encode-phonemes` — Single text encoding
- `POST /multimodal/decode-phonemes` — Decode phoneme IDs
- `POST /multimodal/batch-encode-phonemes` — Batch encoding
- `POST /multimodal/score-pronunciation` — Pronunciation scoring
- `POST /multimodal/batch-score-pronunciation` — Batch pronunciation scoring
- `POST /multimodal/detect-language` — Language detection
- `POST /multimodal/synthesize-speech` — TTS synthesis with speed/pitch control
- `POST /multimodal/training` — Pronunciation training feedback

#### Web Interface Features

- **Encode** — Encode text to phonemes with language auto-detection
- **TTS Synthesis** — Synthesize speech from text with speed/pitch control and audio playback
- **Pronunciation Scoring** — Score pronunciation accuracy between target and spoken words
- **Language Detection** — Auto-detect language from text input
- **Pronunciation Training** — Get feedback on pronunciation with tips and suggestions

See **docs/STRUCTURE.md** and **docs/AI_SOFTWARE_ENGINEERING.md**.
