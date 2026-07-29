# Programmable World Realm — Architecture

## What This Is

A self-contained virtual universe where AI babies are born into a grid.
The environment is not a backdrop — it is a complete programmable computing system.
Babies are not scripted — they perceive, feel, and react.
The world computes on its own. Babies read results.

## The Seven Stages

The world engine is not built in one leap. It is built in stages,
each depending on the last. Each stage is a working system.

```
STAGE 1: BORROWED INTELLIGENCE
  HuggingFace models, inference engine, training pipeline, chat, souls
  Status: DONE
  What it gives us: working AI that can talk, think, generate
  What it lacks: self-sufficiency. We consume other people's intelligence.

STAGE 2: OPERATING SYSTEM
  VM, processes, syscalls, memory allocator, filesystem
  Status: DONE
  What it gives us: programs have a substrate to run on
  What it lacks: a world to exist in

STAGE 3: PERCEPTION
  Path tracer (Cycles), BVH acceleration, GGX BSDF, state tensors,
  neural feature extraction (RenderNeuralDevice)
  Status: DONE
  What it gives us: the machine can see
  What it lacks: something to see

STAGE 4: WORLD ENGINE
  3D spatial grid, cells, entities, perceptron perception,
  energy propagation, diffusion, tick-based simulation
  Status: DONE
  What it gives us: babies exist in a world, perceive, feel, react
  What it lacks: cognition from scratch (babies use random perceptrons)

STAGE 5: COGNITION FROM SCRATCH
  Neural networks trained inside the world engine itself.
  No HuggingFace. No pre-trained models. Pure emergence.
  Status: NOT BUILT
  What it gives us: babies learn to perceive and react from raw experience
  What it lacks: multiple babies, social dynamics

STAGE 6: MULTI-AGENT
  Multiple babies perceiving, competing, cooperating in the same world.
  Communication through world signals, not direct messages.
  Status: NOT BUILT
  What it gives us: territoriality, resource competition, tool use, cooperation
  What it lacks: civilization, culture, evolution

STAGE 7: AUTONOMOUS CIVILIZATION
  Babies modify the world, build structures, teach each other.
  The world becomes the computer, babies are the programs.
  Culture, evolution, artificial life emerge.
  Status: NOT BUILT
  What it gives us: a self-sustaining universe
```

**The critical transition is Stage 4 → Stage 5.**
Right now our agents run HuggingFace models for cognition.
Stage 5 means training neural nets *inside* the world —
agents learn to perceive and act from scratch, no pre-trained weights,
no external data. That's when it stops being a simulation
and starts being a universe.

## Core Idea

```
WORLD = STATE × RULES × TIME

  STATE   — every atom of the world is a number in memory
  RULES   — diffusion, waves, energy conservation (simple physics)
  TIME    — discrete ticks, each tick the world advances one step

AGENTS = BABIES IN THE GRID

  Each agent:
    - has a position in space (3D coordinates)
    - has a body (material, energy)
    - has perceptrons (random weights, no knowledge)
    - reads cells (perception)
    - writes cells (action, arbitrary, no menu)
    - feels energy change (good/bad, no concepts)
    - learns from experience (survival is the teacher)
    - does NOT know the rules (must discover them)

THE WORLD IS THE COMPUTER

  - Space is a grid of computational cells
  - Each cell has state: material, energy, temperature
  - Cells interact with neighbors (diffusion, waves)
  - The world ticks: cells update → agents perceive → agents write → cells update
  - Agents don't orchestrate computation. The world computes. Agents read results.
```

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LAYER                         │
│  Tick loop, baby lifecycle, world state management          │
├─────────────────────────────────────────────────────────────┤
│                    PERCEPTION LAYER                         │
│  Perceptrons on raw input (cells, entities, energy, light)  │
├─────────────────────────────────────────────────────────────┤
│                    FEELING LAYER                            │
│  Energy change = good/bad. No concepts. Just sensation.     │
├─────────────────────────────────────────────────────────────┤
│                    REACTION LAYER                           │
│  Write cells (arbitrary, no menu). World computes result.   │
├─────────────────────────────────────────────────────────────┤
│                    WORLD STATE                              │
│  Spatial grid, materials, energy fields, entity registry    │
├─────────────────────────────────────────────────────────────┤
│                    COMPUTE ENGINE                           │
│  Diffusion, waves, energy conservation — always running     │
└─────────────────────────────────────────────────────────────┘
```

## World State

The world is a 3D spatial grid. Each cell has:

```python
class WorldCell:
    material: int      # what substance occupies this cell (unnamed, 0-7)
    energy: float      # energy level (heat, light, kinetic)
    temperature: float # local temperature
```

The grid is the memory of the world-computer.
Agents read cells. Agents write cells. The world computes between ticks.

### Materials

Materials are numbered 0–7. Agents do not know their names.
They discover materials through behavior: "material 1 flows, conducts heat."

| ID | Behavior (discovered by agents) |
|----|--------------------------------|
| 0 | Fills space, low density |
| 1 | Flows, transparent, conducts heat |
| 2 | Solid, opaque, absorbs energy slowly |
| 3 | Solid, organic, burns |
| 4 | Solid, conducts energy rapidly |
| 5 | Emits energy, heats surroundings |
| 6 | Living material, grows, consumes energy |
| 7 | Transient — wave propagation |

## Entity System

Everything in the world is an entity:

```python
class Entity:
    id: int
    position: np.ndarray      # (x, y, z)
    energy: float
    entity_type: EntityType   # AGENT, OBJECT, LIGHT
```

Simple. No velocity, no rotation, no mass, no charge.
Movement is emergent: the baby writes its material at a new position.
The old position becomes empty. The baby "moved."

### Entity Types

| Type | What | Can Do |
|------|------|--------|
| AGENT | Baby with perceptrons | read cells, write cells |
| OBJECT | Passive physical thing | exists, has material properties |
| LIGHT | Emission source | emits energy, heats surroundings |
| EFFECTOR | World modifier | writes cells (like agent, but no brain) |
| SENSOR | World reader | reads cell states (like agent, but no action) |

## Agent Lifecycle

```
SPAWN → PERCEIVE → FEEL → REACT → WORLD → PERCEIVE → ...
  │        │         │      │       │         │
  │        │         │      │       │         └── new state available
  │        │         │      │       └── world computes (diffusion, waves)
  │        │         │      └── write cells (arbitrary, no action menu)
  │        │         └── energy went up (good) or down (bad)
  │        └── read cells, entities, energy, light
  └── place in world, no knowledge, random weights
```

### Perception

Agents perceive the environment through simple perceptrons.
Raw input (cells, entities, energy) feeds into small neural units that produce features.
No agent chooses how to perceive — the perceptrons are fixed at spawn.

```
Agent → perceptron → nearby_cells[material, energy, temperature]
                    → nearby_entities[position, energy]
                    → agent_body[energy, position]
```

The perceptrons are simple. No complex attention. No transformers.
Just weighted sums and activation functions.
The baby sees the world through tiny windows.

### Cognition

The baby's brain is perceptrons. Random weights. No knowledge.

```
raw input (cells, entities, energy)
  ↓
perceptron (random weights)
  ↓
output: which cells to write, what material, what energy
  ↓
write cells
  ↓
feel: energy went up (good) or down (bad)
  ↓
learn: strengthen what worked, weaken what didn't
```

The baby doesn't "decide." It reacts. Over time, reactions improve.
No memory. No concepts. No language. Just sensation and reaction.

### Action

There is no action menu. The baby writes cells. That's it.

```
WORLD RULES (simple, universal):
  1. Read cells in radius (perception)
  2. Write cells anywhere (action)
  3. That's it.
```

Movement, creation, destruction, communication — all emergent from writing cells.
The baby discovers what it can do by trying everything and keeping what works.

## Communication

Babies don't communicate directly. They write cells.
Other babies perceive those cells. That's communication.

```
Baby A: writes SIGNAL material cells nearby
  Signal cells propagate through diffusion
Baby B: perceives signal cells in its radius
  Doesn't know what they mean
  Must infer from context (what happened before the signal)
```

Language emerges from repeated patterns. If A always writes signal cells near organic cells,
B learns: "signal near organic cells = energy here."

## World Compute Engine

The world computes between ticks. Always running.

```
DIFFUSION: energy flows from high to low between neighboring cells
WAVES: signals propagate outward from sources, decay with distance
ENERGY CONSERVATION: total energy can only decrease (loss)
```

Babies don't orchestrate this. The world computes. Babies read results.
Sometimes babies write cells that trigger new computation.
But the world does the work.

## Parameters

The world has rules. The agents do NOT know them.
Numbers below are design-time defaults — arbitrary, will be tuned.

```python
class WorldParams:
    grid_size: tuple[int, int, int]       # e.g. (64, 32, 64)
    tick_rate: float                       # e.g. 0.1s
    max_entities: int                      # e.g. 128
    diffusion_rate: float                  # how fast energy spreads
    wave_speed: float                      # how fast signals propagate
    energy_loss: float                     # total energy decay per tick
    see_radius: float                      # how far perception reaches
    render_width: int                      # camera pixel width
    render_height: int                     # camera pixel height
    render_samples: int                    # path tracing samples
```

No agent gets this list. No agent gets a manual.
They must discover the rules through observation and experiment.
This is simulation, not a game.

## Perception Rules

### No Privileged Information

| Rule | Constraint |
|------|-----------|
| **No parameter queries** | Agents cannot ask the world for its rules |
| **No cheat codes** | No entity gets special access to internal state |
| **No rules.txt** | Agents receive nothing at spawn except their body |
| **Discovery only** | Rules are discovered through experiment and observation |

### What Perceptrons Can Do

| Rule | Constraint |
|------|-----------|
| **Range** | Limited radius — agents discover how far by hitting the boundary |
| **Resolution** | Cell granularity — distant cells blurred |
| **Latency** | Instant — raw data, no computation |
| **Cost** | Energy per use |
| **Noise** | Gaussian, proportional to distance |

### What Perceptrons Can't Do

| Rule | Constraint |
|------|-----------|
| **No magic** | Perceptrons are fixed at spawn. Can't rewire themselves. |
| **No memory** | Each perception is independent. No episodic memory yet. |
| **No concepts** | No labels, no names, no categories. Just numbers. |

## What Exists Now

### Built and Working

| Component | File | What it does |
|-----------|------|-------------|
| WorldGrid | `simulation.py` | 3D grid — material, energy, temperature, signal. idx/coords, get/set cell, place/write cell, nearby cell queries |
| Cell Update Functions | `simulation.py` | Diffusion (energy high→low), wave propagation, energy conservation (total only decreases) |
| Entity | `simulation.py` | Minimal: id + position + energy + type + alive |
| Perceptron | `simulation.py` | Simple neural unit — sigmoid(Wx + b), delta-rule learning |
| SimBaby | `simulation.py` | Perceive (read cells, body), feel (energy delta), react (write cells), learn (reinforce/weaken), absorb energy from organic |
| SimScene | `simulation.py` | World + babies + entities + cell updates + baby lifecycle |
| Simulation | `simulation.py` | Tick loop: world compute → perceive → feel → react → write → learn → drain. Stop/run/summary. |
| 58 Tests | `test_simulation.py` | WorldGrid (11), CellUpdate (5), Entity (3), Perceptron (4), BabyAction (2), SimBaby (15), SimScene (9), Simulation (9) — all pass |

### Built but Incomplete

| Component | File | What exists | What's missing |
|-----------|------|-------------|----------------|
| CyclesRenderer | `cycles.py` | Path tracer — GGX BSDF, BVH, multi-bounce, state tensors | Not used in world engine (Stage 3 infra) |
| CyclesDevice | `cycles_device.py` | VM device wrapper for renderer | Not used in world engine |
| RenderNeuralDevice | `render_neural.py` | CNN feature extractor on rendered tensors | Not used in world engine |

### Not Built

| Component | What it needs |
|-----------|--------------|
| Cognition From Scratch | Perceptrons trained inside the world — no HuggingFace, no pre-trained weights |
| Multi-Agent | Multiple babies in the same world, competing and cooperating |
| Genetic Algorithm | Babies evolve over generations, weights inherited with mutation |
| Memory | Episodic memory (ring buffer) — comes in Stage 5+ |

## Standalone

This system needs nothing from the existing infrastructure.
No kernel, no VM, no shell, no API server, no HuggingFace, no pre-trained models.

It is:
- numpy (tensor math)
- the grid (world state)
- the perceptrons (baby brains)
- the simulation loop (time)

Everything else is built on top.

## Emergent Behaviors (Expected)

With enough babies and ticks:

1. **Territoriality** — babies claim and defend regions
2. **Resource competition** — energy is finite, babies compete
3. **Communication protocols** — babies develop signaling through cell writes
4. **Cooperation** — babies form groups for mutual benefit
5. **Predation** — babies consume other entities for energy
6. **Structure building** — babies write cell patterns that persist
7. **Culture** — transmitted behaviors evolve over generations

## Engineering Notes

### Design Decisions (Made)

| Decision | Value | Reasoning |
|----------|-------|-----------|
| Grid size | 64×32×64 (~131K cells) | Big enough to have unknown regions. Small enough to explore in ~500 ticks. |
| Agent count | Start 4, scale to 8/16/32 | 4 agents minimum for social dynamics. Territory likely emerges at 6-8 in 64³ grid. |
| Start energy | 100 | Arbitrary. Will be tuned. |
| Passive drain | 0.5/tick | Forces agents to find energy or die. |
| Perception cost | 0.5/tick | Cheap enough to use often. Expensive enough to matter. |
| Render cost | 5/tick | 10× perception. Agents must choose: see often or see well. |
| Materials | Unnamed (0-7) | Agents discover behavior, not names. |
| Perception | Perceptrons on raw input | Simple neural units. No complex attention. |
| Actions | Arbitrary (write cells) | No action menu. Baby writes cells anywhere. Movement is emergent. |

### Energy Economy

```
BABY starts with energy.
  Each tick: energy decreases (passive drain).
  Each perception: energy decreases (perception cost).
  Each write: energy decreases (write cost).
  Baby absorbs energy from nearby cells (organic material).
  Baby dies when energy reaches 0.
  Dead baby's energy returns to grid.
```

Energy is finite. Conservation is enforced. Total energy can only decrease.
This creates survival pressure. Babies that can't find energy die.

### World as Compute Substrate (Hybrid)

The world has two modes: pre-computed state and triggered computation.

**Pre-computed (always available):**
The world has already computed all states. Energy gradients exist. Cell temperatures exist.
Entity positions exist. Everything is already there. The baby just reads state.

**Triggered (by baby's actions):**
When the baby writes cells, the world computes the new state.
Diffusion runs. Waves propagate. Heat spreads.
The baby can trigger new computation — but doesn't have to. State is already there.

The baby doesn't orchestrate computation. The world computes. The baby reads results.
Like a trained neural network: you feed it input, read output. You don't tell it how to compute.

### Cognition Bootstrap

Agents start as babies. Random perceptron weights. No training data. No teacher.
The world itself is the training data.

```
INPUT: raw perception (cells, entities, energy, light)
  ↓
PERCEPTRON: random weights → outputs
  ↓
OUTPUT: which cells to write, what material, what energy
  ↓
WORLD: computes new state
  ↓
FEEDBACK: energy went up (good) or down (bad)
  ↓
LEARNING: strengthen weights that increased energy
          weaken weights that decreased energy
```

No backpropagation. No action menu. No predefined outputs.
The baby tries random things. Some work. Some don't.
What works gets reinforced. What doesn't gets weakened.
Survival is the only teacher.

### Memory

No memory yet. Each perception is independent.
This is Stage 5 — memory comes later.
For now, babies react to immediate sensation only.

### Evolution

Baby dies → offspring born at same position
  → inherits parent's perceptron weights (±10% mutation)
  → starts with less energy (must earn survival)
  → no memory of parent's experiences

Over generations, survival pressure selects for better weights.

### Communication

Agents don't communicate directly. They write cells. Other babies perceive those cells.

```
Baby A: writes SIGNAL material cells nearby
  Signal cells propagate through diffusion
Baby B: perceives signal cells in its radius
  Doesn't know what they mean
  Must infer from context (what happened before the signal)
```

Language emerges from repeated patterns. If A always writes signal cells near organic cells,
B learns: "signal near organic cells = energy here."

## The Baby in the World

An agent is a baby born alone in a grid.
No language. No concepts. No instructions.
It doesn't know gravity exists until it falls.
It doesn't know energy is finite until it runs out.
It doesn't know what "food" is — it only knows: energy went up, that felt good.

```
BABY AGENT
──────────

  Born in the grid. No knowledge.
  Perceives: cells, energy, light, other entities.
  Feels: energy rising (good) or falling (bad).
  Acts: anything. Not pre-defined actions.
       Just "do something and see what happens."

  It touches hot → pain → avoids hot next time.
  It finds sweet → pleasure → seeks sweet again.
  It falls → surprise → learns "up is different from down."

  No instructions. No rules. No vocabulary.
  Just experience. Just sensation. Just survival.

  Over time:
    reactions that increase energy → reinforced
    reactions that decrease energy → weakened
    the baby learns to survive
    without ever understanding why
```

### Hybrid Architecture

The world has two modes: pre-computed state and triggered computation.

```
WORLD STATE (pre-computed, always available):
  cell materials, energies, temperatures
  entity positions, energy
  energy gradients
  → the baby reads this. always. no effort.

WORLD COMPUTE (triggered by baby's actions):
  baby writes cells → diffusion runs
  baby writes signal → waves propagate
  baby places light → heat spreads
  → the baby can trigger new computation.
     but doesn't have to. state is already there.
```

The baby doesn't have to orchestrate computation.
The world has already computed everything.
The baby reads state. Sometimes it writes cells.
The world computes the new state. The baby reads again.

```
feel → react → feel → react → feel → react
```

### Actions Are Arbitrary

Not: MOVE, CREATE, DESTROY, etc.
Just: write to cells. anywhere. any material. any energy.

The baby doesn't know "move."
It knows: "write SELF material where I am" → it appears to move.

The baby doesn't know "create."
It knows: "write STONE material there" → a stone appears.

The baby doesn't know "broadcast."
It knows: "write SIGNAL material everywhere nearby" → a signal propagates.

```
WORLD RULES (simple, universal):
  1. Read cells in radius (perception)
  2. Write cells anywhere (action)
  3. That's it.

  Everything else — movement, creation, destruction, communication —
  is emergent from writing and reading cells.
```

The world doesn't have an action menu.
The world has: "write cells, read cells, that's it."
Everything else is emergent.
