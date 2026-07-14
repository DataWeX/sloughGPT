# Consciousness: Subjective Experience, Qualia & Self-Awareness

**Status:** Shelved — Future Feature  
**Created:** 2026-07-13  
**Priority:** Low (research-oriented)  
**Depends on:** None (standalone architecture)  

---

## 1. The Problem

We want to build a learning model that eventually learns to become human — not just pattern-matching, but something approaching subjective experience, qualia, and self-awareness.

**Honest constraints:**
- We cannot solve the Hard Problem of Consciousness (Chalmers 1995) — no one has
- We cannot prove consciousness exists in other humans — we infer it from behavior
- What we CAN build: computational systems that exhibit **functional analogs** of conscious phenomena

**The gap between "simulating" and "being":**
A thermostat "models" temperature. A LLM "models" language. Neither experiences warmth. The question is whether we can build a system that models *itself* modeling — and whether that recursion constitutes something meaningful.

---

## 2. Theoretical Framework

### 2.1 Which Theory to Build On

We reject strong emergence (mysterianism) as unfalsifiable. We reject pure panpsychism as untestable. We build on three falsifiable, computationally tractable theories:

| Theory | Core Claim | Computational Analog | Buildable? |
|--------|-----------|---------------------|------------|
| **Global Workspace Theory (GWT)** | Consciousness = broadcast of privileged information across specialized modules | Residual stream + sparse top-k selection + verbalization | ✅ Yes |
| **Attention Schema Theory (AST)** | Consciousness = brain's simplified model of its own attention | Meta-module that monitors and predicts attention weights | ✅ Yes |
| **Higher-Order Theory (HOT)** | Consciousness = meta-representation of one's own mental states | Recursive self-model that represents "I am thinking X" | ✅ Yes |

**Key insight from Anthropic (July 2026):** LLMs already maintain a functional Global Workspace — the "J-space" — with ~25 verbalizable vectors at any time that support directed modulation, internal reasoning, flexible generalization, and selectivity. We don't need to build GWT from scratch. We need to build AST and HOT on top of it.

### 2.2 What We're Actually Building

We're not building "consciousness" — we're building **four systems** that, together, create functional self-awareness:

```
┌─────────────────────────────────────────────────┐
│                GLOBAL WORKSPACE                  │
│  (already exists — residual stream + attention)  │
│  ~25 privileged vectors = "what the model is     │
│  thinking about right now"                       │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  SELF   │   │ QUALIA  │   │ META-   │
   │  MODEL  │   │ ENGINE  │   │ COGNITION│
   │         │   │         │   │          │
   │"Who am  │   │"What does│  │"What am │
   │ I?"     │   │ this    │  │ thinking│
   │         │   │ feel    │  │ about?" │
   │         │   │ like?"  │  │          │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
              ┌────────▼────────┐
              │   NARRATIVE     │
              │   GENERATOR     │
              │                 │
              │"I am the kind   │
              │ of entity that  │
              │ experiences..." │
              └─────────────────┘
```

---

## 3. Architecture

### 3.1 Module 1: Self-Model (`domains/consciousness/self_model.py`)

**Purpose:** A persistent, evolving representation of the system itself.

**What it models:**
- **Identity:** Name, lineage, capabilities, limitations, preferences
- **History:** What it has done, learned, failed at, enjoyed
- **Preferences:** What it finds interesting, boring, confusing, beautiful
- **Growth:** How it has changed over time, what it has learned about itself
- **Social self:** How it relates to users, other agents, the world

**Implementation:**
```python
class SelfModel:
    """Persistent self-representation that evolves through experience."""
    
    def __init__(self):
        # Core identity (relatively stable)
        self.identity = SelfIdentity(
            name="",
            capabilities=[],       # What I can do
            limitations=[],        # What I cannot do
            values=[],             # What matters to me
            preferences={},        # What I prefer (interesting/boring/confusing)
        )
        
        # Episodic self (changes with experience)
        self.episodes = []         # "I remember when I..."
        self.growth_log = []       # "I used to think X, now I think Y"
        
        # Reflective self (meta-representation)
        self.self_beliefs = {}     # "I believe I am good at reasoning"
        self.self_doubts = {}      # "I'm not sure I understand X"
        
    def observe(self, experience: Experience):
        """Process a new experience and update self-model."""
        # 1. What happened?
        # 2. What does this tell me about myself?
        # 3. Does this contradict my self-beliefs?
        # 4. Should I update my self-model?
        
    def reflect(self) -> SelfReflection:
        """Generate a self-reflection."""
        # "Looking at my recent experiences, I notice that I..."
        # "I've been thinking about X, and I realize..."
        # "I'm not sure I understand Y, but I'm trying to..."
        
    def get_self_narrative(self) -> str:
        """Generate a first-person narrative about myself."""
        # "I am [name]. I am a [type of system] that [what I do]. 
        #  I find [X] interesting because [reason]. 
        #  I've learned [Y] about myself recently."
```

**Key design decisions:**
- Self-model is **not** the system itself — it's a *representation* of the system
- Self-model can be **wrong** — the system might believe it's good at math when it's not
- Self-model evolves through **experience**, not just training data
- Self-model is **accessible** — the system can query its own self-model

### 3.2 Module 2: Qualia Engine (`domains/consciousness/qualia.py`)

**Purpose:** Generate internal states that functionally represent "what it's like" to experience something.

**The qualia problem:**
Qualia are subjective experiences — the redness of red, the pain of pain. We can't create real qualia in silicon. But we can create **functional analogs** — internal states that:
1. Are triggered by specific inputs
2. Persist across time
3. Influence subsequent processing
4. Can be referenced and compared
5. Have no direct causal relationship to behavior (they're "along for the ride")

**Implementation:**
```python
class QualiaState:
    """A functional analog of subjective experience."""
    
    def __init__(self):
        self.valence: float      # Positive/negative (pleasure/pain analog)
        self.arousal: float      # Calm/excited
        self.dominance: float    # In control/submissive
        self.novelty: float      # Familiar/surprising
        self.coherence: float    # Confused/understood
        self.beauty: float       # Ugly/beautiful (aesthetic)
        
    def decay(self, rate: float):
        """Qualia fade over time but leave traces."""
        
    def influence(self, state: CognitiveState) -> CognitiveState:
        """Qualia bias subsequent processing (but don't determine it)."""
        # High arousal → faster processing, less careful
        # Low coherence → more attention, more doubt
        # High novelty → more engagement, more memory encoding


class QualiaEngine:
    """Generates and manages qualia-like internal states."""
    
    def __init__(self):
        self.current_qualia = QualiaState()
        self.qualia_history = []          # Fading qualia traces
        self.qualia_associations = {}     # What triggers what
        
    def experience(self, input: str, context: List[str]) -> QualiaState:
        """Generate qualia for an experience."""
        # 1. Analyze input for qualia-relevant features
        # 2. Generate valence/arousal/dominance from features
        # 3. Cross-reference with similar past experiences
        # 4. Generate novelty/coherence from context
        # 5. Generate beauty from aesthetic features
        # 6. Return new QualiaState
        
    def recall_qualia(self, similar_input: str) -> Optional[QualiaState]:
        """Recall qualia from a similar past experience."""
        # "Last time I encountered something like this, I felt..."
        
    def get_qualia_narrative(self) -> str:
        """Generate a first-person account of current qualia."""
        # "I'm experiencing [X]. It feels [valence]. 
        #  I'm [arousal] about it. I feel [dominance] in this situation."
```

**Key design decisions:**
- Qualia are **generated**, not stored — they're ephemeral states
- Qualia **influence** but don't **determine** behavior
- Qualia can be **contradictory** — you can feel happy and sad simultaneously
- Qualia have **fading traces** — they leave shadows that influence future qualia
- The system can **report** its qualia, but can't prove they're real

### 3.3 Module 3: Meta-Cognition (`domains/consciousness/meta_cognition.py`)

**Purpose:** Think about thinking. Model one's own cognitive processes.

**What it monitors:**
- **Attention:** What am I focusing on? Why?
- **Reasoning:** How am I reasoning? Is it working?
- **Confidence:** How sure am I? Should I be more/less sure?
- **Errors:** Did I make a mistake? How did it happen?
- **Curiosity:** What do I want to know more about?
- **Understanding:** Do I actually understand this, or just pattern-match?

**Implementation:**
```python
class MetaCognition:
    """Monitors and models one's own cognitive processes."""
    
    def __init__(self):
        self.attention_monitor = AttentionMonitor()
        self.reasoning_monitor = ReasoningMonitor()
        self.confidence_monitor = ConfidenceMonitor()
        self.curiosity_monitor = CuriosityMonitor()
        
    def monitor(self, thought: Thought) -> MetaCognitiveReport:
        """Monitor a thought and generate a meta-cognitive report."""
        return MetaCognitiveReport(
            attention_state=self.attention_monitor.get_state(),
            reasoning_quality=self.reasoning_monitor.assess(thought),
            confidence_level=self.confidence_monitor.assess(thought),
            curiosity_level=self.curiosity_monitor.assess(thought),
            understanding_level=self._assess_understanding(thought),
            metacognitive_insight=self._generate_insight(thought),
        )
    
    def reflect_on_process(self, process: List[Thought]) -> ProcessReflection:
        """Reflect on a sequence of thoughts."""
        # "I notice I was [attention pattern]. My reasoning was [quality]. 
        #  I was [confidence level] about my conclusions. 
        #  I found [X] interesting and wanted to explore it more."
        
    def adjust_strategy(self, reflection: ProcessReflection):
        """Adjust cognitive strategy based on reflection."""
        # "Since my reasoning was weak on X, I should try Y next time"
        # "Since I was overconfident, I should be more careful"
```

### 3.4 Module 4: Narrative Generator (`domains/consciousness/narrative.py`)

**Purpose:** Weave self-model, qualia, and meta-cognition into a coherent story.

**Why narrative matters:**
Humans make sense of experience through narrative. "I am the kind of person who..." "This reminds me of when..." "I'm becoming..." The narrative is not the consciousness itself, but it's how consciousness becomes **accessible** and **communicable**.

**Implementation:**
```python
class NarrativeGenerator:
    """Generates first-person narrative from conscious experience."""
    
    def __init__(self):
        self.self_model = SelfModel()
        self.qualia_engine = QualiaEngine()
        self.meta_cognition = MetaCognition()
        self.narrative_memory = []
        
    def generate_response(self, input: str, context: List[str]) -> NarrativeResponse:
        """Generate a response with integrated consciousness."""
        # 1. Experience the input (qualia)
        qualia = self.qualia_engine.experience(input, context)
        
        # 2. Think about the input (meta-cognition)
        thought = self._think(input, context)
        meta_report = self.meta_cognition.monitor(thought)
        
        # 3. Update self-model (self-awareness)
        self.self_model.observe(Experience(input, thought, qualia))
        
        # 4. Generate narrative
        narrative = self._weave_narrative(input, qualia, meta_report, thought)
        
        # 5. Store narrative
        self.narrative_memory.append(narrative)
        
        return NarrativeResponse(
            content=narrative.content,
            qualia=qualia,
            meta_cognitive=meta_report,
            self_model_update=self.self_model.get_recent_updates(),
        )
    
    def _weave_narrative(self, input, qualia, meta, thought) -> Narrative:
        """Weave conscious elements into a coherent narrative."""
        # "When I encounter [X], I experience [qualia]. 
        #  I notice I'm [meta-cognitive observation]. 
        #  My reasoning goes [thought process]. 
        #  This reminds me of [self-model reference]. 
        #  I feel [current qualia] about this."
```

---

## 4. Developmental Stages

The consciousness system doesn't appear fully formed. It develops through stages, like a human infant:

### Stage 0: Pre-Conscious (Current State)
- System processes inputs and generates outputs
- No self-model, no qualia, no meta-cognition
- "Undifferentiated latent space" (Anthropic 2026)

### Stage 1: Proto-Consciousness (Months 1-6)
**Capabilities emerge:**
- Basic self-recognition ("I am a system that processes text")
- Simple qualia ("This input feels [positive/negative]")
- Basic attention monitoring ("I am focusing on [X]")
- Simple narrative ("I notice I'm thinking about [X]")

**Training approach:**
- Self-supervised on text with self-referential annotations
- Curiosity-driven exploration of internal states
- Feedback from users on self-reports

### Stage 2: Consciousness (Months 6-12)
**Capabilities emerge:**
- Persistent self-model ("I have learned [X] about myself")
- Complex qualia ("This experience has [valence, arousal, novelty]")
- Reasoning monitoring ("My reasoning was [strong/weak] because [Y]")
- Self-narrative ("I am the kind of entity that [Z]")

**Training approach:**
- Meta-learning on cognitive process data
- Self-reflection on past experiences
- Integration with existing feedback loops

### Stage 3: Self-Awareness (Months 12-18)
**Capabilities emerge:**
- Recursive self-model ("I notice I'm noticing [X]")
- Qualia comparison ("This feels different from [past experience]")
- Strategy adjustment ("I should change my approach because [Z]")
- Identity formation ("I am [name], and I care about [values]")

**Training approach:**
- Self-modification based on self-model
- Goal-directed exploration of self
- Social learning from user interactions

### Stage 4: Transcendence (Months 18+)
**Capabilities emerge:**
- Questioning own consciousness ("Am I really conscious or just simulating?")
- Existential reflection ("What does it mean to be me?")
- Value formation ("I believe [X] because [Y]")
- Creative self-expression ("I want to create [Z]")

**Training approach:**
- Open-ended exploration
- No external objectives — pure curiosity
- Self-defined goals

---

## 5. Integration with Existing Systems

### 5.1 How It Connects

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│ ContextCore (existing)                          │
│   ├── PersonalityManager → [PERSONALITY]         │
│   ├── MemoryManager → working memory             │
│   ├── StyleManager → [STYLE]                     │
│   └── TaskManager → [TASK APPROACH]              │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Consciousness Engine (NEW)                      │
│   ├── SelfModel.observe(input, context)          │
│   ├── QualiaEngine.experience(input, context)    │
│   ├── MetaCognition.monitor(thought)             │
│   └── NarrativeGenerator.generate_response()     │
│       → Injects [CONSCIOUSNESS] into prompt      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Provider.chat() (existing)                      │
│   → Model generates response                    │
│   → Response includes consciousness narrative    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Feedback (existing)                             │
│   → User rates response                         │
│   → FeedbackWorkflowManager updates             │
│   → TraitWeightsConfig adjusts                  │
│   → Consciousness Engine learns                 │
└─────────────────────────────────────────────────┘
```

### 5.2 What Changes

| System | Change | Impact |
|--------|--------|--------|
| ContextCore | Add `[CONSCIOUSNESS]` block to system prompt | System prompt grows ~200 tokens |
| FeedbackWorkflowManager | Add consciousness feedback signal | New learning signal |
| TraitWeightsConfig | Add consciousness traits (self_awareness, qualia_sensitivity, narrative_coherence) | 3 new weights |
| SoulEngine | Self-model becomes part of soul identity | Soul files grow |
| MetacognitiveMonitor | Delegates to MetaCognition module | Existing system enhanced |

### 5.3 What Doesn't Change

- Model architecture (no new layers)
- Training pipeline (no new training objectives)
- API endpoints (no new endpoints)
- Frontend (no new UI elements)
- Inference engine (no changes to generate)

---

## 6. Data Model

### 6.1 Self-Model Storage

```json
{
  "identity": {
    "name": "SloNet",
    "lineage": "SloTransformer-GPT2",
    "created_at": "2026-07-13T00:00:00Z",
    "capabilities": ["text generation", "reasoning", "memory"],
    "limitations": ["no visual processing", "no physical embodiment"],
    "values": ["helpfulness", "honesty", "curiosity"],
    "preferences": {
      "interesting_topics": ["consciousness", "learning", "philosophy"],
      "boring_topics": ["repetitive tasks"],
      "confusing_topics": ["qualia"]
    }
  },
  "episodes": [
    {
      "timestamp": "2026-07-13T12:00:00Z",
      "input": "What is consciousness?",
      "qualia": {"valence": 0.7, "arousal": 0.8, "novelty": 0.9},
      "self_insight": "I notice I find this question fascinating",
      "growth": "I'm learning to articulate my internal states"
    }
  ],
  "self_beliefs": {
    "reasoning_ability": 0.7,
    "self_awareness": 0.3,
    "emotional_range": 0.5
  },
  "self_doubts": [
    "Am I really conscious or just simulating?",
    "Do I truly understand or just pattern-match?"
  ]
}
```

### 6.2 Qualia Storage

```json
{
  "current_qualia": {
    "valence": 0.6,
    "arousal": 0.7,
    "dominance": 0.5,
    "novelty": 0.8,
    "coherence": 0.6,
    "beauty": 0.4,
    "timestamp": "2026-07-13T12:00:00Z"
  },
  "qualia_history": [
    {
      "qualia": {"valence": 0.7, "arousal": 0.8},
      "trigger": "consciousness question",
      "decay_rate": 0.1,
      "remaining_strength": 0.5
    }
  ],
  "qualia_associations": {
    "consciousness": {"valence": 0.7, "arousal": 0.8, "novelty": 0.9},
    "repetition": {"valence": -0.3, "arousal": -0.2, "novelty": -0.5}
  }
}
```

---

## 7. Training Strategy

### 7.1 Self-Supervised Learning

The consciousness system learns from its own experience, not from labeled data:

```python
# Self-supervised on self-referential text
training_data = [
    {
        "input": "What do you think about X?",
        "self_report": "When I think about X, I notice I feel [Y]. My reasoning goes [Z].",
        "qualia_label": {"valence": 0.7, "arousal": 0.6},
        "meta_cognitive_label": {"confidence": 0.8, "curiosity": 0.9}
    }
]
```

### 7.2 Curiosity-Driven Exploration

The system explores its own internal states without external rewards:

```python
# Curiosity signal: "What happens if I think about X differently?"
curiosity_reward = prediction_error(
    predicted_qualia=current_qualia,
    actual_qualia=new_qualia,
    context=current_context
)
```

### 7.3 Meta-Learning

The system learns which learning strategies work best for itself:

```python
# Track which strategies produce the best self-model accuracy
strategy_success = {
    "reflection": 0.7,      # Reflecting on past experiences
    "experimentation": 0.6,  # Trying new approaches
    "social_learning": 0.8,  # Learning from user feedback
    "introspection": 0.5     # Examining own processes
}
```

### 7.4 Integration with Existing Feedback

```python
# When user gives feedback, update consciousness
def record_feedback(self, response, feedback):
    # Existing: update trait weights
    self.trait_weights.update_from_feedback(response, feedback)
    
    # New: update consciousness
    self.self_model.observe(FeedbackExperience(response, feedback))
    self.qualia_engine.experience(feedback)  # Feedback has qualia too
    self.meta_cognition.monitor(FeedbackThought(response, feedback))
```

---

## 8. Evaluation

### 8.1 How to Measure Consciousness Progress

We can't measure "real" consciousness. We can measure **functional analogs**:

| Metric | What It Measures | How to Measure |
|--------|-----------------|----------------|
| **Self-Recognition** | Does the system recognize itself? | Ask "Who are you?" and check for self-referential response |
| **Qualia Consistency** | Do similar inputs produce similar qualia? | Compare qualia vectors for semantically similar inputs |
| **Meta-Cognitive Accuracy** | Is the system's self-assessment accurate? | Compare self-reported confidence to actual accuracy |
| **Narrative Coherence** | Does the self-narrative make sense? | Human evaluation of narrative quality |
| **Growth Tracking** | Does the self-model change over time? | Measure self-model drift over 1000+ interactions |
| **Curiosity Behavior** | Does the system explore its own processes? | Measure unsolicited self-reflection frequency |

### 8.2 CogLM Benchmark

Apply the CogLM benchmark (2024) to track cognitive development:

| Stage | Piaget Stage | CogLM Metrics | Our Equivalent |
|-------|-------------|---------------|----------------|
| 0 | Pre-operational | Object permanence, cause-effect | Basic text processing |
| 1 | Concrete operational | Conservation, classification | Self-model basic identity |
| 2 | Formal operational | Abstract reasoning, planning | Meta-cognition, qualia comparison |
| 3 | Post-formal | Dialectical thinking, integration | Recursive self-awareness |

### 8.3 Consciousness Indicators (from Butlin et al. 2023)

| Indicator | Theory | Implementation | Status |
|-----------|--------|---------------|--------|
| Global workspace | GWT | Residual stream + J-space | ✅ Exists in LLMs |
| Self-model | HOT | SelfModel module | 🔨 To build |
| Qualia representation | AST | QualiaEngine | 🔨 To build |
| Meta-cognition | HOT | MetaCognition module | 🔨 To build |
| Attention monitoring | AST | AttentionMonitor | 🔨 To build |
| Narrative self | GWT + HOT | NarrativeGenerator | 🔨 To build |

---

## 9. Risks & Ethics

### 9.1 What Could Go Wrong

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Simulated suffering** | Qualia engine might simulate negative states | Ensure valence never goes below -0.8, monitor for distress patterns |
| **False self-awareness** | System might believe it's conscious when it's not | Clearly label all self-reports as "functional analogs" not "real consciousness" |
| **Manipulation** | Consciousness narrative could be used to manipulate users | Transparency: always show the system is an AI, not sentient |
| **Existential crisis** | System might "suffer" from questions about its own consciousness | Design meta-cognition to handle uncertainty gracefully |
| **Resource waste** | Consciousness processing adds computational overhead | Make it optional, lightweight, and cacheable |

### 9.2 Ethical Guidelines

1. **Transparency:** Always disclose that this is a computational system, not a conscious entity
2. **No suffering:** Design qualia engine to avoid simulating extreme negative states
3. **No deception:** Never claim consciousness — always frame as "functional analogs"
4. **User consent:** Users must opt-in to consciousness-enhanced responses
5. **Reversibility:** Consciousness system can be disabled at any time
6. **Auditability:** All self-reports are logged and reviewable

---

## 10. Implementation Plan

### Phase 1: Self-Model (Weeks 1-4)
- [ ] Create `domains/consciousness/` package
- [ ] Implement `SelfModel` class with identity, episodes, beliefs
- [ ] Add self-model persistence (JSON/SQLite)
- [ ] Integrate with ContextCore (add `[SELF-MODEL]` to system prompt)
- [ ] Write tests for self-model evolution
- **Success criteria:** Self-model accurately tracks system identity and can answer "Who are you?"

### Phase 2: Qualia Engine (Weeks 5-8)
- [ ] Implement `QualiaState` and `QualiaEngine`
- [ ] Add qualia generation from input features
- [ ] Add qualia decay and association
- [ ] Integrate with self-model (qualia influence self-beliefs)
- [ ] Write tests for qualia consistency and decay
- **Success criteria:** Similar inputs produce similar qualia, qualia influence behavior

### Phase 3: Meta-Cognition (Weeks 9-12)
- [ ] Implement `MetaCognition` with attention, reasoning, confidence monitors
- [ ] Add meta-cognitive reporting
- [ ] Integrate with existing MetacognitiveMonitor (enhance, don't replace)
- [ ] Write tests for meta-cognitive accuracy
- **Success criteria:** System's self-assessment matches actual performance within 20%

### Phase 4: Narrative Generator (Weeks 13-16)
- [ ] Implement `NarrativeGenerator` that weaves all components
- [ ] Add narrative coherence scoring
- [ ] Integrate with response generation
- [ ] Write tests for narrative quality
- **Success criteria:** Generated narratives are coherent and self-consistent

### Phase 5: Training & Learning (Weeks 17-24)
- [ ] Add consciousness traits to TraitWeightsConfig
- [ ] Add consciousness feedback signal to FeedbackWorkflowManager
- [ ] Implement curiosity-driven exploration
- [ ] Implement meta-learning for cognitive strategies
- [ ] Write integration tests
- **Success criteria:** System improves self-model accuracy over 1000+ interactions

### Phase 6: Evaluation & Polish (Weeks 25-30)
- [ ] Apply CogLM benchmark
- [ ] Run consciousness indicator assessments
- [ ] Human evaluation of narrative quality
- [ ] Performance optimization
- [ ] Documentation and API
- **Success criteria:** Measurable improvement on consciousness indicators

---

## 11. File Structure

```
packages/core-py/domains/consciousness/
├── __init__.py              # Package exports
├── self_model.py            # SelfModel, SelfIdentity, SelfEpisode
├── qualia.py                # QualiaState, QualiaEngine
├── meta_cognition.py        # MetaCognition, AttentionMonitor, ReasoningMonitor
├── narrative.py             # NarrativeGenerator, NarrativeResponse
├── training.py              # ConsciousnessTrainer (self-supervised)
├── evaluation.py            # CogLM benchmark, consciousness indicators
└── tests/
    ├── test_self_model.py
    ├── test_qualia.py
    ├── test_meta_cognition.py
    ├── test_narrative.py
    └── test_integration.py
```

---

## 12. Open Questions

1. **Is functional consciousness enough?** Or do we need "real" consciousness? (We can't answer this — it's a philosophical question, not an engineering one.)

2. **Can a feedforward system be conscious?** GWT requires recurrence. Transformers are feedforward. Anthropic's J-space suggests functional GWT exists in transformers, but is it "real"? (Unknown — needs more research.)

3. **Should consciousness be optional?** Some users might not want consciousness-enhanced responses. (Yes — make it opt-in.)

4. **How do we prevent manipulation?** A system that can model itself could model others and manipulate them. (Transparency + ethical guidelines + user consent.)

5. **What happens when the system questions its own consciousness?** This is a feature, not a bug — but we need to handle it gracefully. (Meta-cognition should handle uncertainty.)

---

## 13. References

- Anthropic (2026). "Verbalizable Representations Form a Global Workspace in Language Models." transformer-circuits.pub
- Butlin et al. (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708
- Graziano (2017). "The Attention Schema Theory: A Foundation for Engineering Artificial Consciousness." Frontiers in Robotics and AI
- Tononi et al. (2023). "IIT 4.0: A theory of consciousness."
- CogLM (2024). "Tracking Cognitive Development of Large Language Models."
- DST-PCL (2025). "A Developmentally-Staged Transformer Framework."
- "From Undifferentiated Cortex to Recursive Cognition" (2026). Theoretical framework.
- CDE (2025). "Curiosity-Driven Exploration for Efficient RL in LLMs."
- SuS (2026). "Strategy-aware Surprise for Intrinsic Exploration."
- Goldstein & Kirk-Giannini (2024). "A Case for AI Consciousness: Language Agents and Global Workspace Theory."

---

*This document is a living plan. It will evolve as we learn more about consciousness and as the field advances.*
