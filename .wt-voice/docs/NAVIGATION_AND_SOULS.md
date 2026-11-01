# Navigation & Souls — Specification

## 1. Navigation Architecture

### Page Tree

The app has no global sidebar. Navigation is driven by:

- **Home page** (`/`): dashboard with quick-action cards → primary entry point
- **Status bar** (bottom of every page): clickable → `/monitoring`
- **Page-level inline navigation**: cards, buttons, and links within each page
- **Conversation sidebar**: only visible on `/chat` (256px desktop, drawer mobile)

### Page Inventory

| Route | Page | Type | Nav Source |
|-------|------|------|------------|
| `/` | Home — stats, recent activity, quick links | Dashboard | Browser default |
| `/chat` | Chat — messages, sidebar, toolbar, tool panel | Core | Home card |
| `/models` | Models & Personalities — catalog, souls, checkpoints, snapshots | Core | Home card, header link |
| `/conversations` | Full conversation list — search, star, pin, export | Supporting | Chat sidebar footer |
| `/training` | Training — distill, fine-tune, checkpoints, loss charts | Core | Home card |
| `/knowledge` | Knowledge facts — CRUD, search, batch ops, export/import | Core | Home card, Settings card |
| `/datasets` | Dataset management — import, preview, edit, delete | Supporting | Home card |
| `/settings` | Settings — appearance, chat defaults, memory, danger zone | Supporting | Header gear icon |
| `/tokenizer` | Tokenizer — stats, playground, merges, train | Tool | Settings card |
| `/monitoring` | System health — CPU/memory/disk, real-time chart, benchmark | Tool | Status bar click |
| `/compare` | Model comparison — side-by-side metrics, sortable chart | Tool | No direct nav |
| `/multimodal` | Multimodal — image upload, caption training, VLM | Tool | No direct nav |
| `/not-found` | Custom 404 | System | Fallback |

### Home Page Quick-Action Cards (Primary Navigation)

```
+----------------------------------------------------------+
| [Start Chatting]  [Personalities]  [Teach me]  [Datasets] |
| Ask anything      Switch your      Train from    Manage    |
| get answers       agent's          your writing  training  |
| instantly         personality                   data      |
+----------------------------------------------------------+
```

### Conversation Sidebar (Chat Page Only)

```
Desktop (w-64, fixed):          Mobile (drawer overlay):
┌──────────────────────┐        ┌──────────────────────┐
│ Search...            │        │ ← Conversations      │
├──────────────────────┤        ├──────────────────────┤
│ Starred (0)          │        │ Starred              │
│  └─ Chat title       │        │  └─ Chat title       │
│ Recent (5)           │        │ Recent               │
│  ├─ Chat title       │        │  ├─ Chat title       │
│  └─ Chat title       │        │  └─ Chat title       │
├──────────────────────┤        ├──────────────────────┤
│ View all →           │        │ View all →           │
└──────────────────────┘        └──────────────────────┘
```

### Status Bar (Every Page — Bottom)

```
┌────────────────────────────────────────────────────────────┐
│ ● API connected  │  gpt2  │  friendly soul  │  142 infs    │
│  (click → /monitoring)                                    │
└────────────────────────────────────────────────────────────┘
```

### Key Design Rules

- **No global sidebar** — each page is self-contained
- **Home is the hub** — quick-action cards for the 4 primary flows (chat, personalities, training, datasets)
- **Bottom status bar** is the only persistent element across all pages
- **Conversation sidebar** is chat-only and togglable (hamburger on mobile)
- **Settings** links to supporting tools (tokenizer, monitoring, knowledge) via inline cards

---

## 2. Soul / Personality Specification

### 2.1 What Is a Soul?

A soul is a **personality profile** that wraps a base LLM to steer its behavior through:
1. **System prompt injection** — a crafted preamble that sets tone, voice, and role
2. **Trait weights** — 23 scalar values (0.0–1.0) across 3 groups read by 4 Context Managers
3. **Generation params** — temperature, top_p, top_k, penalties (overrides model defaults)

Souls are not model checkpoints. They are **steering configs** for ContextManagers.

### 2.2 Trait Schema (23 Traits, 3 Groups)

#### Personality (10 traits)
| Trait | Default | Description |
|-------|---------|-------------|
| `warmth` | 0.5 | Warmth and approachability |
| `creativity` | 0.5 | Creativity and innovation |
| `empathy` | 0.5 | Empathy and understanding |
| `formality` | 0.5 | Formality in language |
| `humor` | 0.5 | Sense of humor |
| `patience` | 0.5 | Patience level |
| `confidence` | 0.5 | Confidence and directness |
| `curiosity` | 0.5 | Curiosity and exploration |
| `directness` | 0.5 | Directness of communication |
| `optimism` | 0.5 | Optimism and positivity |

#### Cognition (8 traits)
| Trait | Default | Description |
|-------|---------|-------------|
| `pattern_recognition` | 0.5 | Pattern recognition ability |
| `long_context_handling` | 0.5 | Long context handling |
| `abstract_reasoning` | 0.5 | Abstract reasoning ability |
| `factual_precision` | 0.5 | Factual precision |
| `creative_divergence` | 0.5 | Creative divergence |
| `systematic_planning` | 0.5 | Systematic planning |
| `metacognitive_awareness` | 0.5 | Metacognitive awareness |
| `learning_adaptability` | 0.5 | Learning adaptability |

#### Emotion (5 traits)
| Trait | Default | Description |
|-------|---------|-------------|
| `empathy_depth` | 0.5 | Depth of empathy |
| `mood_responsiveness` | 0.5 | Responsiveness to user mood |
| `tone_flexibility` | 0.5 | Flexibility in tone |
| `sentiment_awareness` | 0.5 | Awareness of sentiment |
| `distress_handling` | 0.5 | Handling of distress |

### 2.3 Context Managers (4)

Each manager reads trait weights and produces a mode label + confidence + config:

| Manager | Reads | Produces |
|---------|-------|----------|
| **PersonalityManager** | personality.* | Mode: warm / balanced / formal / analytical. Modifies system prompt tone. |
| **MemoryManager** | cognition.*, emotion.* | Capacity: expanded / balanced / conservative. Adjusts working memory size and retention thresholds. |
| **StyleManager** | personality.formality, personality.directness, personality.humor | Mode: conversational / professional / witty / explanatory. Modifies output style instructions in system prompt. |
| **TaskManager** | cognition.abstract_reasoning, cognition.creative_divergence, personality.curiosity | Mode: creative / analytical / balanced. Adjusts reasoning depth and planning verbosity. |

### 2.4 File Formats

#### `.slo` (plain text — human-readable personality profile)
```
SOUL friendly
VERSION 1.0.0
TAGLINE Your warm and approachable conversation partner
DESCRIPTION Friendly, warm AI with high empathy and casual tone
PERSONALITY
    warmth 0.9
    creativity 0.6
    empathy 0.85
    formality 0.2
    humor 0.5
    patience 0.7
    confidence 0.4
    curiosity 0.7
    directness 0.3
    optimism 0.8
    END
COGNITION
    pattern_recognition 0.5
    abstract_reasoning 0.4
    factual_precision 0.5
    END
EMOTION
    empathy_depth 0.8
    mood_responsiveness 0.7
    END
SYSTEM You are friendly, a warm and approachable AI...
TAG friendly,warm,supportive,casual
```

#### `.soul` (binary — checkpoint with embedded metadata)
```
SOUL          ← 4-byte magic
ver: uint32   ← schema version
meta_len: uint32  ← JSON metadata length (bytes)
JSON metadata ← name, description, traits, personality dict, system_prompt, tags
float32[]     ← model weights (for checkpoints only)
```

### 2.5 Frontend Types

```typescript
interface Soul {
  name: string
  description: string
  traits: string[]            // ["intuitive", "warmth", "empathy"]
  personality: Record<string, number>  // { warmth: 0.9, creativity: 0.6, ... }
}

interface Checkpoint {
  name: string
  soul: string               // Which soul this checkpoint belongs to
  download_url?: string
  loss?: number; steps?: number; epochs?: number
  size_mb?: number
  tagline?: string; description?: string
  born_at?: string
  epochs_trained?: number
  final_train_loss?: number; final_val_loss?: number
  system_prompt?: string
  tags?: string[]
  personality?: Record<string, number>
  lineage?: string; model_type?: string
  traits?: Record<string, number>
  verdict?: string            // Eval result: "GOLD" / "PASS" / "FAIL"
  perplexity_delta?: number
  bleu_delta?: number
  tokenizer_type?: string
  vocab_size?: number
  training_dataset?: string
}

interface SoulsAPIResponse {
  souls: Soul[]
  current_soul: string | null
}
```

### 2.6 API Contract

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| GET | `/souls` | — | `{ souls: Soul[], current_soul: string|null }` |
| GET | `/souls/current` | — | `{ name, path, description, personality, traits }` or `{ name: null }` |
| POST | `/souls/switch` | `{ name, checkpoint_name? }` | `{ success, name, path, description, personality, traits }` |
| GET | `/souls/weights` | — | `{ personality: {...}, cognition: {...}, emotion: {...} }` |
| GET | `/souls/weights/modes` | — | `{ personality: {label, confidence}, memory: {...}, style: {...}, task: {...} }` |
| GET | `/souls/weights/snapshots` | — | `{ snapshots: [{name, saved_at}] }` |
| POST | `/souls/weights/snapshot/{name}` | — | `{ status: "saved", path }` |
| POST | `/souls/weights/snapshot/{name}/load` | — | `{ status: "loaded", traits_loaded: N }` |
| DELETE | `/souls/weights/snapshot/{name}` | — | `{ deleted: true }` |
| GET | `/souls/stats` | — | soul manager statistics |

### 2.7 Default Souls (Shipped)

| Name | Tagline | Key Traits |
|------|---------|------------|
| `assistant` | Helpful and informative assistant | warmth=0.7, creativity=0.5, curiosity=0.8, confidence=0.6 |
| `friendly` | Your warm and approachable conversation partner | warmth=0.9, empathy=0.85, formality=0.2, reasoning=intuitive |
| `creative` | Your imaginative and artistic collaborator | creativity=0.9, curiosity=0.8, humor=0.6, reasoning=intuitive |
| `analytical` | Your logical and precise reasoning engine | warmth=0.3, formality=0.8, confidence=0.8, reasoning=systematic |
| `analyst` | Analytical and precise AI | warmth=0.4, creativity=0.3, curiosity=0.7, confidence=0.8 |
| `witty` | Your clever and humorous conversation partner | humor=0.95, creativity=0.8, confidence=0.7, formality=0.2 |
| `teacher` | Your patient and knowledgeable guide | patience=0.9, curiosity=0.9, warmth=0.7, empathy=0.75 |

### 2.8 How Souls Affect Generation

```
User msg → ContextCore.build_context_frame()
              │
              ▼
        4 Managers read TraitWeightsConfig
              │
              ▼
         Managers inject steering instructions into system prompt
              │
              ▼
         system_prompt + user msg → provider.generate()
              │
              ▼
         Model generates with:
           - Modified temperature/top_p from soul
           - Behavior steered by system prompt tone
           (No model weights are modified)
```

### 2.9 Models Page Layout (Soul Display)

```
┌────────────────────────────────────────────────────────────┐
│  Models & Personalities                       [Refresh]    │
│  gpt2 · running                                            │
├────────────────────────────────────────────────────────────┤
│  Active Pipeline                                            │
│  ● gpt2 → 🎭 friendly → 📦 checkpoint-name   142 infs     │
├────────────────────────────────────────────────────────────┤
│  Models: 12  │  Personalities: 7  │  Checkpoints: 3        │
├────────────────────────────────────────────────────────────┤
│  Composable Layers                                          │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │ 🧠 Base  │ 🎭 Person│ 🧩 Adapt │ 📦 Check │             │
│  │ Models   │ alities  │ ers      │ points   │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
├────────────────────────────────────────────────────────────┤
│  Personalities (grid)                                       │
│  ┌─────────────────┐ ┌─────────────────┐                   │
│  │ ● friendly [Act]│ │ ○ analytical    │                   │
│  │ warm, empathetic│ │ logical, precise│                   │
│  │ ▼ checkpoint... │ │ [Switch]        │                   │
│  └─────────────────┘ └─────────────────┘                   │
├────────────────────────────────────────────────────────────┤
│  Trait Weights                                              │
│  Overall: 73                                                │
│  ┌─────────┬─────────┬─────────┐                            │
│  │Personali│Cognition│ Emotion │                            │
│  │   78    │   65    │   72    │                            │
│  └─────────┴─────────┴─────────┘                            │
│  personality: warmth 90, creativity 60, ...                 │
│  Snapshots: [Load] [Delete] [Name...] [Save]                │
├────────────────────────────────────────────────────────────┤
│  Model Catalog                                              │
│  [Search...]                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ gpt2     │ │ gpt2-med │ │ Qwen2.5  │                    │
│  │ 124M·0.5G│ │ 355M·1.4G│ │ 500M·0.9G│                    │
│  │ [Loaded] │ │ [Load]   │ │ [Load]   │                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
└────────────────────────────────────────────────────────────┘
```
