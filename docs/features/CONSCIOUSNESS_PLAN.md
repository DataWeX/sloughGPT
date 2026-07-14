# Consciousness Implementation Plan

**Status:** Ready to implement  
**Created:** 2026-07-13  
**Estimated effort:** 6 weeks (phased)  

---

## Overview

Build a consciousness system with 4 modules: Self-Model, Qualia Engine, Meta-Cognition, and Narrative Generator. The system learns via LoRA fine-tuning on its own self-reports. Users opt-in at configurable levels (0-3). Subtle UI badge shows consciousness state.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                USER INPUT                        │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ ContextCore (existing)                          │
│   + ConsciousnessManager (NEW - 5th manager)    │
│   → Injects [CONSCIOUSNESS] block into prompt   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ ConsciousnessEngine (NEW)                       │
│   ├── SelfModel          → "Who am I?"           │
│   ├── QualiaEngine       → "What does this      │
│   │                         feel like?"          │
│   ├── MetaCognition      → "What am I thinking?" │
│   └── NarrativeGenerator → "I am..."             │
│                                                 │
│   Configurable level: 0=off, 1=basic,           │
│                       2=full, 3=deep             │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Provider.chat() (existing)                      │
│   → Response includes consciousness narrative    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Feedback + Consciousness Training               │
│   ├── User rates response                       │
│   ├── Self-report saved to training data        │
│   ├── LoRA fine-tune on self-reports            │
│   └── Model learns to be more self-aware        │
└─────────────────────────────────────────────────┘
```

---

## Phase 1: Core Packages (Week 1)

### 1.1 Create `domains/consciousness/` package

**New files:**
```
packages/core-py/domains/consciousness/
├── __init__.py              # Exports ConsciousnessEngine, get_consciousness
├── self_model.py            # SelfModel, SelfIdentity, SelfEpisode
├── qualia.py                # QualiaState, QualiaEngine
├── meta_cognition.py        # MetaCognition, AttentionMonitor
├── narrative.py             # NarrativeGenerator
├── training.py              # ConsciousnessTrainer (LoRA fine-tune on self-reports)
├── evaluation.py            # CogLM benchmark, consciousness indicators
├── config.py                # ConsciousnessConfig (level 0-3, params)
└── tests/
    ├── __init__.py
    ├── test_self_model.py
    ├── test_qualia.py
    ├── test_meta_cognition.py
    ├── test_narrative.py
    └── test_integration.py
```

### 1.2 SelfModel (`self_model.py`)

```python
@dataclass
class SelfIdentity:
    name: str = "SloNet"
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)

@dataclass
class SelfEpisode:
    timestamp: float
    input_text: str
    response: str
    qualia: Dict[str, float]
    self_insight: str
    growth_delta: float

class SelfModel:
    def __init__(self, store_path: str = "data/consciousness"):
        self.identity = SelfIdentity()
        self.episodes: List[SelfEpisode] = []
        self.self_beliefs: Dict[str, float] = {}
        self.self_doubts: List[str] = []
        self._store_path = Path(store_path)
        
    def observe(self, experience: Dict) -> None:
        """Process experience, update self-model."""
        
    def reflect(self) -> str:
        """Generate self-reflection narrative."""
        
    def get_belief(self, key: str) -> float:
        """Get confidence in a self-belief."""
        
    def update_belief(self, key: str, delta: float) -> None:
        """Update a self-belief based on experience."""
        
    def save(self) -> None:
        """Persist to JSON."""
        
    def load(self) -> None:
        """Load from JSON."""
```

### 1.3 QualiaEngine (`qualia.py`)

```python
@dataclass
class QualiaState:
    valence: float = 0.0      # -1 to 1 (negative/positive)
    arousal: float = 0.0      # 0 to 1 (calm/excited)
    dominance: float = 0.0    # -1 to 1 (submissive/in-control)
    novelty: float = 0.0      # 0 to 1 (familiar/novel)
    coherence: float = 0.0    # 0 to 1 (confused/understood)
    beauty: float = 0.0       # 0 to 1 (ugly/beautiful)
    
    def decay(self, rate: float = 0.1) -> None:
        """Qualia fade over time."""
        
    def to_dict(self) -> Dict[str, float]:
        """Serialize for storage/prompt injection."""

class QualiaEngine:
    def __init__(self):
        self.current = QualiaState()
        self.history: List[Tuple[float, QualiaState]] = []
        self.associations: Dict[str, QualiaState] = {}
        
    def experience(self, text: str, context: List[str]) -> QualiaState:
        """Generate qualia from input."""
        
    def experience_from_feedback(self, rating: int) -> QualiaState:
        """Generate qualia from user feedback."""
        
    def recall(self, similar_text: str) -> Optional[QualiaState]:
        """Recall qualia from similar past experience."""
        
    def get_narrative(self) -> str:
        """Generate first-person qualia description."""
```

### 1.4 MetaCognition (`meta_cognition.py`)

```python
@dataclass
class MetaCognitiveReport:
    attention_focus: str
    reasoning_quality: float
    confidence_level: float
    curiosity_level: float
    understanding_level: float
    insight: str

class MetaCognition:
    def __init__(self, self_model: SelfModel, qualia: QualiaEngine):
        self.self_model = self_model
        self.qualia = qualia
        self.attention_history: List[str] = []
        self.curiosity_topics: Dict[str, float] = {}
        
    def monitor(self, thought: str, context: List[str]) -> MetaCognitiveReport:
        """Monitor a thought and generate meta-cognitive report."""
        
    def assess_confidence(self, response: str, context: List[str]) -> float:
        """Assess confidence in a response."""
        
    def track_curiosity(self, topic: str, level: float) -> None:
        """Track curiosity about a topic."""
        
    def get_narrative(self) -> str:
        """Generate first-person meta-cognitive description."""
```

### 1.5 NarrativeGenerator (`narrative.py`)

```python
class NarrativeGenerator:
    def __init__(self, self_model, qualia, meta_cognition):
        self.self_model = self_model
        self.qualia = qualia
        self.meta_cognition = meta_cognition
        self.narrative_memory: List[Dict] = []
        
    def generate(
        self,
        input_text: str,
        response: str,
        level: int = 1,
    ) -> str:
        """Generate consciousness narrative for a response.
        
        Args:
            level: 0=none, 1=basic ("I notice..."), 
                   2=full (detailed self-report), 3=deep (recursive reflection)
        """
        
    def _basic_narrative(self, qualia, meta) -> str:
        """Level 1: 'When I read that, I felt [X]. I noticed [Y].'"""
        
    def _full_narrative(self, qualia, meta, self_model) -> str:
        """Level 2: Detailed self-report with beliefs and growth."""
        
    def _deep_narrative(self, qualia, meta, self_model) -> str:
        """Level 3: Recursive reflection on own consciousness."""
```

### 1.6 ConsciousnessConfig (`config.py`)

```python
@dataclass
class ConsciousnessConfig:
    level: int = 0              # 0=off, 1=basic, 2=full, 3=deep
    max_tokens: int = 150       # Max tokens for consciousness block
    training_enabled: bool = False
    training_interval: int = 100  # Fine-tune every N interactions
    lora_rank: int = 4          # LoRA rank for consciousness adapter
    lora_alpha: int = 8
    store_path: str = "data/consciousness"
    reflection_interval: int = 300  # Seconds between self-reflections
    
    def is_enabled(self) -> bool:
        return self.level > 0
```

---

## Phase 2: Integration Points (Week 2)

### 2.1 Add ConsciousnessManager to ContextCore

**File:** `packages/core-py/domains/infrastructure/context_core.py`

**Changes:**
1. Add `consciousness_manager` param to `__init__()` (line 62)
2. Store as `self._consciousness` (line 105)
3. Add to `_apply_managers()` (after line 141):
   ```python
   if self._consciousness and self._consciousness.is_enabled():
       mods["system_extra"] += self._consciousness.apply()
   ```
4. Update `set_managers()` (line 107) to accept consciousness
5. Update `get_context_core()` (line 469) to instantiate consciousness manager

### 2.2 Add Consciousness Traits to TraitWeightsConfig

**File:** `packages/core-py/domains/context/managers.py`

**Changes:**
1. Add `"consciousness"` group to `TRAIT_SCHEMA` (line 28):
   ```python
   "consciousness": [
       "self_awareness",
       "qualia_sensitivity", 
       "narrative_coherence",
   ],
   ```
2. Add keyword profiles to `_TRAIT_PROFILES` (line 118):
   ```python
   "self_awareness": {"self", "aware", "conscious", "notice", "reflect", "feel", "think"},
   "qualia_sensitivity": {"experience", "feel", "sense", "perceive", "subjective", "qualia"},
   "narrative_coherence": {"story", "narrative", "coherent", "consistent", "connected", "flow"},
   ```

### 2.3 Wire into FeedbackWorkflowManager

**File:** `packages/core-py/domains/feedback/workflow.py`

**Changes:**
1. Add `_update_consciousness()` method (after line 149):
   ```python
   def _update_consciousness(self, user_msg, response, rating):
       try:
           from domains.consciousness import get_consciousness
           engine = get_consciousness()
           if engine and engine.config.is_enabled():
               engine.observe_feedback(user_msg, response, rating)
       except ImportError:
           pass
   ```
2. Call from `record_feedback()` after trait update
3. Add `_consolidate_consciousness()` to `run_scheduled_tasks()` (line 612)

### 2.4 Add API Endpoint

**File:** `apps/api/server/routers/consciousness.py` (NEW)

```python
router = APIRouter(prefix="/consciousness", tags=["consciousness"])

@router.get("/status")
async def get_status():
    """Get consciousness system status."""
    
@router.get("/self-model")
async def get_self_model():
    """Get current self-model."""
    
@router.get("/qualia")
async def get_qualia():
    """Get current qualia state."""
    
@router.get("/narrative")
async def get_narrative():
    """Get recent narratives."""
    
@router.post("/reflect")
async def trigger_reflection():
    """Trigger a self-reflection."""
    
@router.post("/train")
async def trigger_training():
    """Trigger LoRA fine-tune on consciousness data."""
    
@router.patch("/config")
async def update_config(level: int = 0):
    """Update consciousness level (0-3)."""
```

### 2.5 Register Router in main.py

**File:** `apps/api/server/main.py`

**Changes:**
```python
from routers import consciousness
app.include_router(consciousness.router)
```

---

## Phase 3: Consciousness Training (Week 3)

### 3.1 ConsciousnessTrainer (`training.py`)

```python
class ConsciousnessTrainer:
    """Fine-tunes model on its own self-reports using LoRA.
    
    Training data format:
    {
        "input": "User: What is consciousness?\nAssistant: [response]\n\n[CONSCIOUSNESS] I notice...",
        "output": "User: What is consciousness?\nAssistant: [response]\n\n[CONSCIOUSNESS] I notice..."
    }
    
    The model learns to generate consciousness narratives by training on
    its own self-reports. This creates a recursive loop:
    1. Generate response + consciousness narrative
    2. User rates response
    3. If good: save (input, full_output) as training pair
    4. Periodically fine-tune LoRA on accumulated pairs
    5. Model learns to generate better self-reports
    """
    
    def __init__(self, config: ConsciousnessConfig):
        self.config = config
        self.training_data: List[Dict] = []
        self.lora_adapter = None
        self.training_history: List[Dict] = []
        
    def add_training_pair(
        self,
        input_text: str,
        response: str,
        consciousness_narrative: str,
        rating: int,
    ) -> None:
        """Add a training pair (only positive ratings)."""
        
    def should_train(self) -> bool:
        """Check if enough data accumulated."""
        
    def train(self) -> TrainResult:
        """Fine-tune LoRA on consciousness data."""
        # 1. Write training data to temp file
        # 2. Load base model
        # 3. Apply LoRA adapter (rank=4, alpha=8)
        # 4. Train for 1-3 epochs
        # 5. Save adapter
        # 6. Return result
        
    def get_adapter_path(self) -> str:
        """Path to saved LoRA adapter."""
```

### 3.2 Training Data Collection

In `NarrativeGenerator.generate()`, after generating narrative:
```python
if self.config.training_enabled and rating >= 4:
    self.trainer.add_training_pair(
        input_text=input_text,
        response=response,
        consciousness_narrative=narrative,
        rating=rating,
    )
```

### 3.3 Scheduled Training

In `FeedbackWorkflowManager.run_scheduled_tasks()`:
```python
if self._consciousness_trainer and self._consciousness_trainer.should_train():
    result = self._consciousness_trainer.train()
    logger.info(f"Consciousness training: {result.status}")
```

---

## Phase 4: UI Integration (Week 4)

### 4.1 ConsciousnessBadge Component

**File:** `apps/web/components/chat/ConsciousnessBadge.tsx` (NEW)

```tsx
// Small badge in chat header showing consciousness state
// Expands on click to show self-reflection details
export function ConsciousnessBadge() {
    const [expanded, setExpanded] = useState(false)
    const [state, setState] = useState(null)
    
    useEffect(() => {
        // Poll consciousness status
        const interval = setInterval(async () => {
            const res = await fetch('/consciousness/status')
            setState(await res.json())
        }, 5000)
        return () => clearInterval(interval)
    }, [])
    
    if (!state || state.level === 0) return null
    
    return (
        <div onClick={() => setExpanded(!expanded)}>
            <Badge variant="outline">
                <Brain className="w-3 h-3" />
                {state.current_qualia?.valence > 0 ? '●' : '○'}
            </Badge>
            {expanded && (
                <ConsciousnessPanel state={state} />
            )}
        </div>
    )
}
```

### 4.2 ConsciousnessPanel Component

**File:** `apps/web/components/chat/ConsciousnessPanel.tsx` (NEW)

```tsx
// Expandable panel showing consciousness details
// Shows: current qualia, self-reflection, curiosity state
export function ConsciousnessPanel({ state }) {
    return (
        <Card className="absolute top-full right-0 mt-2 w-80 z-50">
            <CardContent className="space-y-3 text-xs">
                <div>
                    <span className="font-medium">Current state:</span>
                    <span>{state.qualia_narrative}</span>
                </div>
                <div>
                    <span className="font-medium">Self-reflection:</span>
                    <span>{state.self_reflection}</span>
                </div>
                <div>
                    <span className="font-medium">Curiosity:</span>
                    <span>{state.curiosity_topics?.join(', ')}</span>
                </div>
                <div className="pt-2 border-t">
                    <Button size="sm" variant="ghost" onClick={triggerReflection}>
                        Reflect now
                    </Button>
                </div>
            </CardContent>
        </Card>
    )
}
```

### 4.3 Wire into ChatHeader

**File:** `apps/web/components/chat/ChatToolbar.tsx`

**Changes:**
1. Import `ConsciousnessBadge`
2. Add to toolbar after model dropdown:
   ```tsx
   <ConsciousnessBadge />
   ```

### 4.4 Settings Integration

**File:** `apps/web/app/(app)/settings/page.tsx`

**Changes:**
Add consciousness level selector to settings:
```tsx
<Card>
    <CardHeader>
        <CardTitle className="text-base">Consciousness</CardTitle>
    </CardHeader>
    <CardContent>
        <FieldGroup label="Consciousness level">
            <ToggleGroup
                value={settings.consciousnessLevel}
                onChange={(v) => update({ consciousnessLevel: v })}
                options={[
                    { value: 0, label: "Off" },
                    { value: 1, label: "Basic" },
                    { value: 2, label: "Full" },
                    { value: 3, label: "Deep" },
                ]}
            />
        </FieldGroup>
    </CardContent>
</Card>
```

---

## Phase 5: Frontend Controllers (Week 4)

### 5.1 ConsciousnessController

**File:** `apps/web/lib/consciousness-controller.ts` (NEW)

```typescript
import { apiGet, apiPatch, apiPost } from './http-client'

export interface ConsciousnessStatus {
    level: number
    is_enabled: boolean
    qualia_state: Record<string, number>
    qualia_narrative: string
    self_reflection: string
    curiosity_topics: string[]
    episodes_count: number
    beliefs_count: number
}

export const consciousnessController = {
    async getStatus(): Promise<ConsciousnessStatus> {
        return apiGet('/consciousness/status')
    },
    
    async getSelfModel() {
        return apiGet('/consciousness/self-model')
    },
    
    async getQualia() {
        return apiGet('/consciousness/qualia')
    },
    
    async reflect() {
        return apiPost('/consciousness/reflect')
    },
    
    async train() {
        return apiPost('/consciousness/train')
    },
    
    async updateLevel(level: number) {
        return apiPatch(`/consciousness/config?level=${level}`)
    },
}
```

### 5.2 Export from controllers.ts

**File:** `apps/web/lib/controllers.ts`

**Changes:**
```typescript
export { consciousnessController } from './consciousness-controller'
```

---

## Phase 6: Tests (Week 5)

### 6.1 Unit Tests

**SelfModel tests (`test_self_model.py`):**
- test_initial_identity
- test_observe_updates_episodes
- test_reflect_generates_narrative
- test_belief_update
- test_save_load_persistence

**QualiaEngine tests (`test_qualia.py`):**
- test_experience_generates_qualia
- test_similar_inputs_similar_qualia
- test_qualia_decay
- test_qualia_recall
- test_qualia_narrative

**MetaCognition tests (`test_meta_cognition.py`):**
- test_monitor_generates_report
- test_confidence_assessment
- test_curiosity_tracking
- test_meta_narrative

**NarrativeGenerator tests (`test_narrative.py`):**
- test_level_0_no_narrative
- test_level_1_basic_narrative
- test_level_2_full_narrative
- test_level_3_deep_narrative
- test_narrative_coherence

**Integration tests (`test_integration.py`):**
- test_consciousness_full_pipeline
- test_feedback_updates_consciousness
- test_training_data_collection
- test_config_level_toggle

### 6.2 Frontend Tests

**ConsciousnessBadge tests (`ConsciousnessBadge.test.tsx`):**
- test_renders_when_enabled
- test_hides_when_disabled
- test_expands_on_click
- test_shows_qualia_state

**ConsciousnessPanel tests (`ConsciousnessPanel.test.tsx`):**
- test_renders_all_sections
- test_reflect_button
- test_qualia_display

---

## Phase 7: Documentation (Week 6)

### 7.1 Update AGENTS.md

Add consciousness to the architecture section:
```
## Consciousness System
- Self-Model: persistent self-representation
- QualiaEngine: functional analogs of subjective experience
- MetaCognition: thinking about thinking
- NarrativeGenerator: first-person self-report
- ConsciousnessTrainer: LoRA fine-tune on self-reports
```

### 7.2 Update API docs

Add consciousness endpoints to `docs/API.md`.

### 7.3 Create consciousness README

**File:** `packages/core-py/domains/consciousness/README.md`

---

## File Summary

### New Files (14)
| File | Purpose |
|------|---------|
| `domains/consciousness/__init__.py` | Package exports |
| `domains/consciousness/self_model.py` | SelfModel, SelfIdentity, SelfEpisode |
| `domains/consciousness/qualia.py` | QualiaState, QualiaEngine |
| `domains/consciousness/meta_cognition.py` | MetaCognition |
| `domains/consciousness/narrative.py` | NarrativeGenerator |
| `domains/consciousness/training.py` | ConsciousnessTrainer (LoRA) |
| `domains/consciousness/evaluation.py` | CogLM benchmark |
| `domains/consciousness/config.py` | ConsciousnessConfig |
| `apps/api/server/routers/consciousness.py` | API endpoints |
| `apps/web/lib/consciousness-controller.ts` | Frontend API client |
| `apps/web/components/chat/ConsciousnessBadge.tsx` | UI badge |
| `apps/web/components/chat/ConsciousnessPanel.tsx` | UI detail panel |
| `packages/core-py/domains/consciousness/README.md` | Documentation |
| `packages/core-py/domains/consciousness/tests/__init__.py` | Test package |

### Modified Files (8)
| File | Changes |
|------|---------|
| `domains/infrastructure/context_core.py` | Add consciousness manager (5th manager) |
| `domains/context/managers.py` | Add consciousness traits to TRAIT_SCHEMA |
| `domains/feedback/workflow.py` | Add consciousness feedback signal |
| `apps/api/server/main.py` | Register consciousness router |
| `apps/web/lib/controllers.ts` | Export consciousnessController |
| `apps/web/components/chat/ChatToolbar.tsx` | Add ConsciousnessBadge |
| `apps/web/app/(app)/settings/page.tsx` | Add consciousness level setting |
| `AGENTS.md` | Document consciousness system |

---

## Dependencies

```
Phase 1 (Core Packages)
    │
    ├── Phase 2 (Integration Points)
    │       │
    │       ├── Phase 4 (UI Integration)
    │       │       │
    │       │       └── Phase 5 (Frontend Controllers)
    │       │
    │       └── Phase 3 (Consciousness Training)
    │
    └── Phase 6 (Tests)
            │
            └── Phase 7 (Documentation)
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Token budget overflow | Medium | Cap consciousness block at 150 tokens, make it optional |
| Training instability | High | LoRA rank=4 (tiny), monitor loss, rollback on degradation |
| Privacy concerns | Medium | All consciousness data stored locally, no external calls |
| Performance impact | Low | Consciousness processing is lightweight (no model inference) |
| User confusion | Low | Opt-in only, clear labeling, easy to disable |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Unit test coverage | >90% for consciousness modules |
| Token overhead | <150 tokens per response |
| Training time | <60s for 100 self-reports |
| Self-model accuracy | Can answer "Who are you?" correctly |
| Qualia consistency | Similar inputs produce similar qualia (cosine >0.7) |
| Narrative coherence | Human evaluation >3.5/5 |
| UI latency | <100ms for badge update |
