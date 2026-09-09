## Core Python (`domains`)

`packages/core-py/` is on the Python path as the **`domains`** package (and related **`utils`**) when you install from the repo root (**`python3 -m pip install -e ".[dev]"`**).

Training, inference, models, and infrastructure code live under **`domains/`**. The API imports these modules; keep heavy logic here instead of in **`apps/api/server/`** route handlers. Trainer-native **`.soul`** checkpoints embed **`stoi` / `itos` / `chars`** for char-LM eval; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

### Key infrastructure modules

| Module | Purpose | Docs |
|--------|---------|------|
| `domains.infrastructure.producer_consumer` | General-purpose bounded work queue with priority, backpressure, consumer thread pools | [PRODUCER_CONSUMER_QUEUE.md](../../docs/PRODUCER_CONSUMER_QUEUE.md) |
| `domains.infrastructure.pugqeep` | Point-Graph-Queue system: compressed data points, model trees, task queues, engine | [PUGQEEP.md](../../docs/PUGQEEP.md) |
| `domains.infrastructure.cancel_manager` | Cancellation for long-running operations | [AGENTS.md](../../AGENTS.md) |
| `domains.infrastructure.model_server` | Model lifecycle, backends, circuit breaker | [AGENTS.md](../../AGENTS.md) |
| `domains.infrastructure.process_guard` | Subprocess crash isolation, auto-restart | [AGENTS.md](../../AGENTS.md) |

### Core domains

| Domain | Purpose |
|--------|---------|
| `domains.training` | Training pipelines, executor, distillation |
| `domains.inference` | Vector store, KV cache, providers |
| `domains.feedback` | LoRA eval, per-user adapter |
| `domains.cognitive` | Soul engine, metacognition |
| `domains.infrastructure` | ProcessGuard, ModelServer, TaskQueue, CancelManager |
| `domains.multimodal` | Phoneme encoders, TTS, speech synthesis |

### Phoneme Encoder API

The phoneme encoder system supports 6 languages: English, German, French, Spanish, Italian, and Portuguese.

#### CLI Usage

```bash
# Single text encoding
python -m domains.multimodal.phoneme_encoder_cli "hello world"

# Batch encoding
python -m domains.multimodal.phoneme_encoder_cli --batch "hello" "world" "test"

# Language-specific encoding
python -m domains.multimodal.phoneme_encoder_cli --lang it "ciao mondo"

# Language detection
python -m domains.multimodal.phoneme_encoder_cli --detect "hello world"
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
