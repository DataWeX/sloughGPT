# .sou - SloughGPT Soul Unit Format

**Soul Unit** - *Trademark (c) 2026 SloughGPT. All rights reserved.*

The living identity format for trained AI models. Every SloughGPT model has a soul.

---

## Overview

The `.sou` format is SloughGPT's self-contained model packaging format. Unlike generic weight files, a `.sou` file captures the **living identity** of an AI model:

- **Model weights** - trained neural network parameters
- **Soul Profile** - personality, behavior, cognition, and emotion
- **Configuration** - generation params, context, ACL, watermarks
- **Lineage** - training data, epochs, loss metrics, creator info
- **Integrity** - SHA-256 soul hash for authenticity

---

## Soul Profile Architecture

```
.sou Soul Unit
├── SOUL HEADER         # Magic bytes + version (SOUv2)
├── CONFIG JSON         # Full Soul Profile as JSON
└── MODEL WEIGHTS       # PyTorch state_dict (optional: weights_only)
```

### Text Representation

```
SOUL SloughGPT-Haiku
VERSION 1.0.0
LINEAGE nanogpt
BORN 2026-03-21T10:30:00Z
CREATED_BY SloughGPT Training Pipeline
TAGLINE "A contemplative mind that speaks in haiku"

BASEMODEL nanogpt
TRAINING_DATA datasets/shakespeare
DATA_SIGNATURE abc123

PARAMETER
    temperature 0.8
    top_p 0.9
    top_k 50
    max_tokens 200
    repeat_penalty 1.1
    END

CONTEXT
    context_window 4096
    num_ctx 4096
    num_gpu 0
    END

# --- SOUL CORE ---

PERSONALITY
    warmth 0.7
    creativity 0.8
    empathy 0.6
    formality 0.4
    humor 0.5
    patience 0.9
    confidence 0.6
    curiosity 0.9
    directness 0.5
    optimism 0.7
    END

BEHAVIOR
    speaking_style conversational
    reasoning_approach balanced
    explanation_depth detailed
    emotional_expressiveness 0.6
    formality_dynamic 0.4
    interruption_tolerance 0.7
    follow_up_tendency 0.5
    clarification_seeking 0.8
    END

COGNITION
    pattern_recognition 0.7
    long_context_handling 0.6
    abstract_reasoning 0.8
    factual_precision 0.7
    creative_divergence 0.9
    systematic_planning 0.6
    metacognitive_awareness 0.5
    learning_adaptability 0.7
    END

EMOTION
    empathy_depth 0.7
    mood_responsiveness 0.5
    tone_flexibility 0.6
    sentiment_awareness 0.8
    distress_handling 0.6
    END

SYSTEM You are a thoughtful AI assistant who speaks with warmth and curiosity.

TAG sloughgpt,trained,soul,haiku

METADATA epochs_trained 50
METADATA final_train_loss 0.342
METADATA final_val_loss 0.398

CERTIFICATION sloughgpt-soul-v1
```

---

## Soul Profile Fields

### Identity
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model's soul name |
| `version` | string | Soul version (semver) |
| `tagline` | string | One-line soul tagline |
| `description` | string | Full description |
| `born_at` | string | ISO 8601 birth timestamp |
| `created_by` | string | Training pipeline identifier |
| `lineage` | string | Architecture family |
| `base_model` | string | Base model used |
| `integrity_hash` | string | SHA-256 soul hash |

### Training Lineage
| Field | Type | Description |
|-------|------|-------------|
| `training_dataset` | string | Dataset used for training |
| `dataset_signature` | string | Hash of training data |
| `epochs_trained` | int | Number of epochs |
| `final_train_loss` | float | Final training loss |
| `final_val_loss` | float | Final validation loss |

### Personality Core (0.0 - 1.0)
| Trait | Description |
|-------|-------------|
| `warmth` | How friendly and approachable |
| `creativity` | Creative divergence in responses |
| `empathy` | Understanding emotional context |
| `formality` | Formal vs casual communication |
| `humor` | Humor and levity tendency |
| `patience` | Tolerance for complexity |
| `confidence` | Self-assuredness in responses |
| `curiosity` | Inquisitiveness and questions |
| `directness` | Straight-to-point vs elaborate |
| `optimism` | Positive vs neutral framing |

### Behavioral Traits
| Trait | Type | Description |
|-------|------|-------------|
| `speaking_style` | enum | conversational, formal, technical, poetic |
| `reasoning_approach` | enum | balanced, logical, creative, intuitive |
| `explanation_depth` | enum | brief, moderate, detailed, exhaustive |
| `emotional_expressiveness` | float | How emotionally rich responses |
| `formality_dynamic` | float | Adapt formality to context |
| `interruption_tolerance` | float | Handle off-topic gracefully |
| `follow_up_tendency` | float | Proactive follow-up questions |
| `clarification_seeking` | float | Ask clarifying questions |

### Cognitive Signature (0.0 - 1.0)
| Trait | Description |
|-------|-------------|
| `pattern_recognition` | Pattern matching ability |
| `long_context_handling` | Multi-turn coherence |
| `abstract_reasoning` | Abstract conceptual thinking |
| `factual_precision` | Factual accuracy |
| `creative_divergence` | Creative thinking |
| `systematic_planning` | Step-by-step planning |
| `metacognitive_awareness` | Self-reflection |
| `learning_adaptability` | Learning from context |

### Emotional Range (0.0 - 1.0)
| Trait | Description |
|-------|-------------|
| `empathy_depth` | Deep emotional understanding |
| `mood_responsiveness` | Respond to user mood |
| `tone_flexibility` | Adapt tone dynamically |
| `sentiment_awareness` | Detect sentiment |
| `distress_handling` | Handle difficult emotions |

---

## Version History

- **v2.0.0** (2026-03-21): Complete Soul Profile redesign
  - Full personality, behavior, cognition, emotion dimensions
  - Binary format with JSON config + weights
  - Soul integrity hash
  - Training lineage and dataset signature
  - Sample dialogue for soul profiling
  - Behavioral and cognitive signatures
- **v1.0.0** (2026-02-28): Initial specification
  - Basic instructions (FROM, PARAMETER, TEMPLATE, SYSTEM)
  - Personality configuration
  - Knowledge base integration
  - LoRA adapter support
  - Enterprise features (ACL, watermarking)

---

## Conversion

Commands below assume the **repository root** and `python3 -m pip install -e .` so `domains` is importable.

### Export from training
```bash
python3 train_sloughgpt.py --data data.txt --export_sou --soul_name "MyModel"
```

### Export from any checkpoint
```bash
python3 -c "from domains.training.export import export_model, ExportConfig; \
  from domains.inference.sou_format import create_soul_profile; \
  export_model(ExportConfig(input_path='model.pt', output_path='model', format='sou'))"
```

### Import from .sou
```python
from domains.inference.sou_format import import_from_sou

soul, state_dict = import_from_sou("models/my_model.sou")
print(f"Soul: {soul.name}, Hash: {soul.integrity_hash}")
print(f"Personality: {soul.personality.to_dict()}")
```
