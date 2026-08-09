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
  Status: BUILT
  Mechanism: evolution-driven. A population of baby genomes (perceptron
  weights + movement gates) is re-evaluated each generation in a fixed
  generated world; the fittest harvest more energy and breed. Emergence
  proof: evolved populations out-harvest frozen-random ones (see the
  --emergence benchmark). In-life delta-rule learning remains optional.
  What it gives us: babies learn to perceive, move, and harvest from raw experience
  What it lacks: multiple babies, social dynamics

STAGE 6: MULTI-AGENT
  Multiple babies perceiving, competing, cooperating in the same world.
  Status: BUILT
  Mechanism: multilevel (trait-group) selection. Babies are born into
  tribes; tribes compete by the geometric mean of member energy and breed
  uniformly within the tribe, so an act that keeps a starving tribe-mate
  alive preserves the tribe's score. A kin signal (same-tribe bit) enters
  the entity perceptron, and cooperation targets the neediest neighbor.
  Communication is two-channel: a learnable broadcast signal (Stage 3
  SIGNAL material waves) and an opt-in directed message channel — a baby
  can address one specific neighbor, pay a cost scaled by the gate
  amplitude, and that neighbor perceives it exactly one tick later.
  Emergence proof: group-arm cooperation outlasts the individual arm's
  (see the --social benchmark). In-life delta learning stays disabled —
  selection is the only teacher.
  What it gives us: territoriality, resource competition, tool use, cooperation
  What it lacks: civilization, culture, evolution

STAGE 7: AUTONOMOUS CIVILIZATION
  Babies modify the world, build structures, teach each other.
  The world becomes the computer, babies are the programs.
  Culture, evolution, artificial life emerge.
  Status: NOT BUILT
  What it gives us: a self-sustaining universe
```

**The critical transition was Stage 4 → Stage 5.**
We no longer run HuggingFace models for cognition inside the world.
Stage 5 trains neural nets *inside* the world —
agents learn to perceive, move, and act from scratch, no pre-trained weights,
no external data. Stage 6 adds multiple babies in one world so social
dynamics (competition, cooperation, communication) emerge on top of evolved
individual cognition.

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

Each material's behavior is computed by the world every tick (`cell_update_materials`).

| ID | Behavior (discovered by agents) | Computed |
|----|--------------------------------|----------|
| 0 | Fills space, low density | — baseline (air) |
| 1 | Flows, transparent, conducts heat | Water damps `signal` (`water_signal_dampen`) and relaxes temperature toward ambient (`water_cool_rate`) |
| 2 | Solid, opaque, absorbs energy slowly | — inert floor stone; exhausted ember cools to stone |
| 3 | Solid, organic, burns | Feeds babies (`absorb_energy`); rots slowly (`organic_metabolism`); ignites into ember above `ignition_temp` |
| 4 | Solid, conducts energy rapidly | Extra pairwise energy diffusion on metal cells (`metal_conduction_boost`) |
| 5 | Emits energy, heats surroundings | Burns stored energy: radiates energy to neighbors and heat as temperature (`ember_heat_rate`, `ember_energy_fraction`, `heat_to_temp`); sustains `burn_temp`; exhausted → stone |
| 6 | Living material, grows, consumes energy | Spends energy to turn adjacent air into organic (`living_growth_rate`, `living_growth_cost`, `growth_transfer_fraction`); deterministic, no RNG |
| 7 | Transient — wave propagation | Wave engine: written SIGNAL emits amplitude, waves carry it outward |

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
learn: reinforce gains, weaken losses, scaled by surprise
       (outcomes that diverge from the recent-reward baseline teach hardest)
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

**Directed messaging (Stage 6).** Opt-in (`message_enabled`), one baby addresses one specific
neighbor. The sender's message perceptron reads that neighbor's entity features and, if the gate
clears `message_gate_threshold`, emits the gate value as the amplitude; the scene charges
`message_cost * amplitude` and delivers it to the target's inbox at the start of the next tick —
so the recipient perceives it exactly one tick later as the entity feature at index 5 (visible only
to brains built with `entity_input_dim >= 6`). Signaling costs real energy, so it is a strategic
choice, not free chatter. Unlike the broadcast signal channel, a message has an address and a
purpose: a tribe-mate can whisper a warning to the hungriest neighbor without the whole world
hearing it.

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
    see_cost: float                        # energy per perception
    write_cost: float                      # energy fee per cell written
    write_energy_scale: float              # max energy deposited per cell written
    passive_drain: float                   # energy lost per tick alive
    memory_capacity: int                   # episodic ring-buffer size
    memory_lookback: int                   # episodes averaged as learning baseline
    render_width: int                      # camera pixel width
    render_height: int                     # camera pixel height
    render_samples: int                    # path tracing samples
    # Terrain
    generate_world: bool                   # build terrain on scene creation
    world_seed: int                        # deterministic terrain seed (local RNG)
    # Temperature & combustion
    ambient_temp: float                    # temperature the world relaxes toward
    ambient_cooling: float                 # fraction of temp gap closed per tick
    ignition_temp: float                   # organic ignites above this
    burn_temp: float                       # temperature a live ember sustains
    # Material behaviors
    ember_heat_rate: float                 # fraction of ember fuel emitted per tick
    ember_energy_fraction: float           # share of emission given as energy
    heat_to_temp: float                    # converts emitted heat to temperature
    organic_metabolism: float              # organic rot sink per tick
    living_growth_rate: float              # fraction of living energy spent growing
    living_growth_cost: float              # energy per new organic cell
    growth_transfer_fraction: float        # share moved into the new cell
    metal_conduction_boost: float          # extra diffusion carried by metal
    water_signal_dampen: float             # signal removed at water cells
    water_cool_rate: float                 # water relaxes to ambient per tick
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
| **No infinite memory** | Each perception is independent. The living episodic buffer holds only recent episodes (ring buffer); experience survives death only as the inherited memotype (top-reward episodes, capped at memory_inherit). |
| **No concepts** | No labels, no names, no categories. Just numbers. |

## What Exists Now

### Built and Working

| Component | File | What it does |
|-----------|------|-------------|
| WorldGrid | `simulation.py` | 3D grid — material, energy, temperature, signal. idx/coords, get/set cell, place/write cell, nearby cell queries |
| Cell Update Functions | `simulation.py` | Diffusion (energy high→low), wave propagation, material behaviors (ignition/rot/ember burn/living growth/water damping/metal conduction), ambient temperature relaxation, energy conservation (total only decreases) |
| Material Behaviors | `simulation.py` | Each material's behavior is computed every tick: organic rots and ignites into ember above `ignition_temp`; ember burns its stored fuel, radiating energy to neighbors and heat as temperature, then cools to stone; living material grows into adjacent air (deterministic, no RNG); water damps `signal` and relaxes temperature; metal conducts energy rapidly. All animated by the existing fields (`energy`, `temperature`, `signal`) with knobs in `WorldParams` |
| Terrain Generation | `simulation.py` | `generate_world()` fills an empty grid using a local `default_rng` (deterministic on `(grid_size, world_seed)`, never the global stream — snapshots/restore stay RNG-neutral): stone floor, water pools and organic food patches on the surface, buried ember vents. Enabled via `generate_world=True`; `SimScene.spawn_babies` drops babies on the ground surface instead of inside solid rock |
| Entity | `simulation.py` | Minimal: id + position + energy + type + alive |
| Perceptron | `simulation.py` | Simple neural unit — sigmoid(Wx + b), delta-rule learning. Stage 5 deep brain: optional hidden projection (`brain_hidden_units > 0`, fixed random H/bh, never trained — no backpropagation), readout W/b delta-rule updated on a skip-connected raw+hidden feature vector |
| SimBaby | `simulation.py` | Perceive (read cells, nearby agents, body), feel (energy delta), react (cells perceptron gates what/how much to write), learn (reinforce/weaken body, cells, and entity perceptrons, scaled by surprise vs. the episodic-reward baseline), absorb energy from organic, share energy (cooperate), contest energy (compete), episodic memory (records every learning event, recalls recent experience) |
| SimScene | `simulation.py` | World + babies + entities + cell updates + baby lifecycle + nearby_babies neighbor query |
| Simulation | `simulation.py` | Tick loop: world compute → perceive → feel → react → write → social step (cooperate/contest) → learn → drain. Stop/run/summary with social stats |
| Multi-Agent Social | `simulation.py` | Babies perceive nearby agents (id, type, energy, distance, angle), cooperate by sharing surplus energy, compete by contesting weaker agents — driven by the entity perceptron and learned from energy deltas |
| EpisodicMemory | `memory.py` | Ring-buffer episodic memory — record, recall (k, by reward), mean reward, stats. SimBaby records a feature/action/reward/tick episode each learn; recall_memories() surfaces recent experience (Stage 5+) |
| Evolution Engine | `evolution.py` | Genetic algorithm — Genome (serializable perceptron weights + inherited memotype, incl. `group_id`) crossover/mutate; EvolutionEngine runs generations, scores by energy (0 if dead), elite + tournament selection (both crossover parents are tournament winners); offspring are born with the winning lineage's consolidated memories seeded into their episodic memory. `run_frozen()` is the no-selection baseline. Supports a fixed environment (deterministic terrain via `world_seed`, food pools re-seeded per generation, optional fixed `spawn_positions`) so fitness measures genetic quality, not spawn luck. With `group_weight > 0` selection becomes two-level: tribes compete by the geometric mean of member energy, parents mate uniformly within the chosen tribe, and offspring inherit their tribe |
| Trait-Group Selection | `evolution.py` | Multilevel (trait-group) selection. `_select_groups` scores each tribe by `geometric_mean(member energy)` (a single starving member drags the whole tribe down), picks a tribe by a tournament on those scores, then draws both parents uniformly inside that tribe. `group_weight` blends the tribe score with individual fitness; `group_weight = 0` is pure individual selection. Offspring inherit the tribe id, so cooperative lineages stay together across generations |
| Kin-Signal Social Perception | `simulation.py` | The entity perceptron input has a kin bit (`1.0` if the perceived baby shares the actor's `group_id`), so cooperation and restraint can be learned and directed at tribe-mates. `social_step` targets the **neediest** nearby baby (`min` by energy) — a donor rescues the hungriest tribe-mate, not the nearest, which is what lifts the tribe's geometric mean |
| Social Emergence Proof | `evolution.py`, `world_driver.py` | `benchmark_social()` runs two arms on the SAME grouped world (same terrain, same per-group food pools, same per-tribe spawn territories): pure individual selection vs. trait-group selection. Cooperation is never hardcoded — the entity perceptron must learn to open its cooperate gate. The world is deliberately scarce (`organic_pools=3`, 24 ticks/gen). At seed 1: `ind_coop=0.0000 grp_coop=0.0312 ind_contest=0.1823 grp_contest=0.0260 emerged=True`. CLI: `--social` |
| Movement | `simulation.py` | Born ability to relocate: a `perceptron_move` (cells input dim → 3 sigmoid gates) maps each axis to a step via `sign(gate - 0.5)` (a gate held at 0.5 means stay put); the simulation applies one wrapped grid step per tick and charges `move_cost`. Starving babies stay still below `move_threshold`. Movement is emergent — no pathfinding, no target — and evolvable: evolution selects for policies that wander toward food and stop to absorb it |
| Emergence Proof | `evolution.py`, `world_driver.py` | `benchmark_emergence()` runs two arms — an evolved population vs. fresh random genomes each generation — re-evaluated every generation on the SAME generated world, food pools, and shared spawn. Emergence is the evolved last-generation mean exceeding the frozen one (5/5 seeds at hidden=0 and hidden=4). CLI: `--emergence` |
| Scene Persistence | `simulation.py`, `memory.py` | Save/load the entire world: `WorldGrid.to_dict()`/`from_dict()` (all four arrays), `Entity`, `Perceptron`, `SimBaby` (weights + memory + counters), `EpisodicMemory` (exact ring-buffer state incl. head), `SimScene` (params + tick + entity-id counter). JSON-safe snapshots; restore is RNG-neutral — a resumed run is bit-identical to an uninterrupted run, and new spawns never reuse a restored id |
| Signal Communication | `simulation.py` | First-class broadcast channel: SIGNAL-material writes transfer deposited energy into the `signal` field, waves carry it outward (`wave_speed`, `signal_decay`), `get_nearby_cells` returns the `signal` array, and `_perception_features` exposes it as a learnable 5th cells feature (`cells_input_dim = 5`). Emission is conservation-safe (a transfer, not creation); the channel round-trips through scene persistence |
| Directed Communication | `simulation.py`, `evolution.py` | Opt-in (`message_enabled`) inter-agent messaging channel. Each baby gains a `perceptron_message` (entity-input → 1 gate) that reads a chosen neighbor's features and, if the gate clears `message_gate_threshold`, emits a message whose amplitude is the gate value. Emission is a step-4c act, posted to the scene's pending bus and delivered to the target's inbox at the start of the next tick — a message is perceived exactly one tick after it is sent. The target is the neediest baby within `message_range` (ties broken deterministically, strongest amplitude wins per sender). Cost is `message_cost * amplitude`, so signaling has a real energetic price. The recipient sees the amplitude as the entity feature at index 5 — visible only to brains built with `entity_input_dim >= 6`. Genome serialization carries the message weights only when the channel is on, preserving the exact RNG layout of the locked selection proofs; the CLI exposes it as `--messages` |
| 117 Tests | `test_simulation.py` | WorldGrid (11), CellUpdate (5), Entity (3), Perceptron (4), BabyAction (2), SimBaby (20), SimScene (9), Simulation (9), Social Perception (4), Social Mechanics (6), Social Step (6), Simulation Social (4) — all pass |
| 68 Tests | `test_evolution.py` | Genome serialization/crossover/mutation, engine determinism, elitism, memotype inheritance (consolidation, capping, seeding, generation transition), emergence proof (evolved beats frozen, benchmark determinism, no-selection baseline, shared spawn, surface spawns, identical generation 1 across arms), trait-group selection (dispatches to group mode, single-population ignores groups, geometric-mean tribe ranking, group-id inheritance, grouped-run group means), kin signal (perception carries `group_id`, entity-input shape vs. world dim, dim-4 compat), social acts (cooperate transfers surplus fraction, needs surplus, contest takes from weaker only, no contest vs. stronger, neediest-neighbor targeting), social benchmark (determinism, structure, group coop > individual coop, individual sharing punished, group contests less), directed message (off-by-default guard, gate-gated emission, amplitude = gate value, one-tick-latent delivery to the neediest neighbor, message feature at index 5, dim-5 slicing, genome tensor round-trip, scene pending-bus round-trip, message-brain learning) — all pass |
| 22 Tests | `test_memory.py` | EpisodicMemory buffer semantics (record/recall/ring-wrap/reward), SimBaby wiring, memory fills over ticks, surprise-scaled learning (baseline/surprise/scale, weight-update ratio) — all pass |
| 22 Tests | `test_scene_persistence.py` | WorldGrid/Entity/Perceptron/Memory/Baby/Scene round-trips (lossless, dtype-stable), JSON dump/load, size-mismatch rejection, entity identity after restore, non-baby entity survival, no RNG consumed by restore, resume-matches-continuous-run determinism, id continuity on post-restore spawns, multi-cycle stability — all pass |
| 18 Tests | `test_signal_communication.py` | Signal emission (write/place/air/OOB), perception (nearby signal array, empty read, feature zero/positive, learnable signal row, untouched row without signal), propagation (wave carries to neighbors, reaches distance over ticks), end-to-end loop (B perceives A's broadcast, snapshot survival, full tick loop) — all pass |
| 26 Tests | `test_environment.py` | Material behaviors (organic ignition above `ignition_temp`, full-pipeline combustion, rot metabolism, ember radiation + energy conservation + burnout to stone, living growth + determinism + poverty rest, water signal dampening + cooling, metal conduction faster than baseline + conservation, ambient temperature relaxation), terrain generation (floor/food/water/ember present, deterministic on seed, seed-sensitive, global-RNG neutral), scene integration (opt-in generation, default empty, surface spawn, persistence round-trip, resume-matches-continuous with terrain, babies survive longer with food) — all pass |
| World Driver | `world_driver.py` | Headless observability harness: `WorldDriver` builds a world (empty or deterministic terrain), runs N ticks via the real simulation loop, and reports per-tick energy economy, material populations, and a conservation invariant (grid+entity energy must never increase — a physics regression tripwire). `run_evolution()` delegates to `EvolutionEngine` for genetic sweeps; `--emergence` runs the emergence benchmark; `--social` runs the trait-group-vs-individual cooperation benchmark; `--messages` switches on the directed inter-agent message channel. CLI: `python3 -m domains.shell.world_driver --grid 64,32,64 --seed 42 --ticks 50 --every 5` (or `--evolution --generations N --population N --ticks-per-gen N`, or `--emergence`, or `--social`, or `--messages`). Pure observability — adds no physics, mutates nothing beyond the live scene |
| 22 Tests | `test_world_driver.py` | Material naming (derived from constants, never hardcoded), grid parsing, empty-vs-terrain populations, snapshot shape, per-tick cadence, energy-ledger consistency, conservation monotonicity + violation detection, seed determinism, mean baby energy, evolution summary shape, emergence CLI path, CLI output paths — all pass |

### Built but Incomplete

| Component | File | What exists | What's missing |
|-----------|------|-------------|----------------|
| CyclesRenderer | `cycles.py` | Path tracer — GGX BSDF, BVH, multi-bounce, state tensors | Not used in world engine (Stage 3 infra) |
| CyclesDevice | `cycles_device.py` | VM device wrapper for renderer | Not used in world engine |
| RenderNeuralDevice | `render_neural.py` | CNN feature extractor on rendered tensors | Not used in world engine |

### Not Built

| Component | What it needs |
|-----------|--------------|
| Stage 7 Civilization | Babies modify the world, build structures, teach each other — the world becomes the computer |
| Infinite Long-Term Memory | Beyond the per-baby ring buffer and inherited memotype |

## Standalone

This system needs nothing from the existing infrastructure.
No kernel, no VM, no shell, no API server, no HuggingFace, no pre-trained models.

It is:
- numpy (tensor math)
- the grid (world state)
- the perceptrons (baby brains)
- the simulation loop (time)

Headless observation via the world driver:

```bash
PYTHONPATH=packages/core-py python3 -m domains.shell.world_driver \
    --grid 64,32,64 --seed 42 --babies 4 --ticks 100 --every 10
```

Everything else is built on top.

## Emergent Behaviors (Expected)

With enough babies and ticks:

1. **Territoriality** — babies claim and defend regions
2. **Resource competition** — energy is finite, babies compete
3. **Communication protocols** — babies develop signaling through cell writes. The channel is now live: writing SIGNAL material emits amplitude into the `signal` field, waves broadcast it across the grid, and other babies perceive it as a learnable feature — B can now learn what A's broadcasts mean, not just that they exist. Stage 6 adds the directed channel: a baby can address one specific neighbor directly, pay an energy cost for the amplitude, and that neighbor perceives it exactly one tick later (`--messages`)
4. **Cooperation** — babies form groups for mutual benefit. Demonstrated: under trait-group selection a cooperating population outlasts a free-riding one (`--social` benchmark), driven by the kin signal and neediest-neighbor targeting
5. **Predation** — babies consume other entities for energy
6. **Structure building** — babies write cell patterns that persist (the world itself is now saveable — a scene snapshot preserves every written cell, so structures survive across sessions, not just ticks)
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
  Each write: energy decreases (write cost + deposited energy).
    — the baby funds the cells it writes. Deposits are transfers,
      never creation. Writing material = spending your own energy.
  Baby absorbs energy from nearby cells (organic material).
  Baby dies when energy reaches 0.
  Dead baby's energy returns to grid.
```

Energy is finite. Conservation is enforced. Total energy can only decrease.
Deposit scale is `write_energy_scale` (WorldParams, default 1.0) × the
perceptron's energy gate — a design-time number, tuned for the energy budget.
This creates survival pressure. Babies that can't find energy die.

Material sinks keep the world a competitive place: organic rots
(`organic_metabolism`), ember converts part of its fuel into heat rather than
transferable energy (`ember_energy_fraction`, `heat_to_temp`), and living
material spends energy to grow (`living_growth_cost`). Energy moved between
cells is always a transfer — pairwise diffusion, ember radiation, and growth
all conserve the energy pool; only decay/conversion to heat remove it.

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

Two teachers exist, and they are not equally strong:

1. **Evolution (the measured headline).** A population of baby genomes is
   re-evaluated every generation; the fittest harvest the most energy and
   breed. Selection is the teacher. In this mode in-life delta-rule updates
   are off (`learning_enabled=False`), so the genome's own weights carry the
   entire signal.
2. **In-life delta-rule learning (optional).** Each baby adjusts its own
   readout weights from its energy deltas. This is the classic loop above,
   still present for the interactive world, but under selection it can
   reinforce the self-funding write that drains the baby (a uniform scalar
   error strengthens whatever it most recently did), so it is not used in
   the emergence benchmark.

### Memory

Episodic memory is a ring buffer (Stage 5). Each `learn()` event records
an episode — the perception features, the action taken, the energy reward,
and the tick. Babies can recall the most recent k episodes (or the best
ones by reward) as a reference for behavior.

Memory also drives **surprise-based learning** (predictive coding): the
baseline is the mean reward of recent episodes (memory_lookback), and the
perceptron update rate scales from 0.5x to 1.5x with how much the latest
outcome diverges from that baseline. A bigger-than-expected gain or a
worse-than-expected loss teaches hardest; unsurprising outcomes barely
nudge weights. The sign rule is unchanged — gains reinforce, losses weaken —
so the loop stays compatible with energy conservation.

What still doesn't exist: infinite long-term memory. Each baby's living
buffer holds only its recent episodes, and memory dies with the baby. But
experience now survives death through the lineage: when a genome is bred,
it consolidates the parent's highest-reward episodes (the memotype, capped
at memory_inherit) and the offspring is born with those episodes seeded
into its episodic memory. Generations inherit experience AND weights.

### Evolution

Baby dies → offspring born at same position
  → inherits parent's perceptron weights (±10% mutation)
  → inherits parent's best episodes (memotype — seeds episodic memory)
  → starts with less energy (must earn survival)
  → no memory of the parent's *identity* — only its winning experiences

Over generations, survival pressure selects for better weights, and the
memotype lets each generation start from the previous one's successes:
offspring that inherit "energy went up after writing near organic cells"
begin with a surprise baseline that biases their learning toward what
worked. Mutation reshapes weights, never memories.

**The emergence proof.** `benchmark_emergence()` gives the population a
fixed world to learn in: the same generated terrain, the same food pools
re-seeded every generation, and a shared spawn position. A policy's fitness
is therefore its genes, not its luck. The evolved arm is compared against
the frozen arm — a fresh random population re-evaluated each generation with
no selection. Generation 1 is identical in both arms; from there the evolved
population's mean harvest climbs while the frozen one stays flat and noisy.
Across seeds 1–5 (hidden=0 and hidden=4) the evolved last-generation mean
beats the frozen one on every seed (mean ≈ +30–37 energy). Babies have
learned to perceive, move toward food, and harvest from raw experience —
no pathfinding, no pre-trained weights, no backpropagation.

### The social emergence proof (Stage 6)

**The problem.** Sharing energy is individually irrational. A baby that
opens its cooperate gate hands its surplus to a neighbor and lowers its own
breeding chances, while the free-riding neighbor pays nothing. Under pure
individual selection cooperation is selected against and collapses.

**The fix is the selection layer, not the behavior.** `benchmark_social()`
runs two arms on the same grouped world and compares selection objectives:

- **Individual arm** (`group_weight = 0`): babies are scored by their own
  energy. A donor is out-bred by the free-rider it feeds.
- **Group arm** (`group_weight = 0.5`): babies are born into tribes; tribes
  compete by the **geometric mean** of member energy, and parents are drawn
  uniformly inside the chosen tribe. One starving member drags a whole tribe
  down, so an act that rescues a starving tribe-mate preserves the tribe's
  score — and the donor's cooperative genes spread through uniform within-
  tribe mating.

The babies themselves are not told to cooperate. The entity perceptron
input carries a **kin signal** (`1.0` when the perceived baby shares the
actor's `group_id`), and `social_step` targets the **neediest** nearby baby,
so a cooperative policy is a real learned response — it must open the
cooperate gate, close the contest gate, and aim at the hungriest neighbor.
In-life delta learning is disabled in both arms: selection is the only
teacher.

**The world is deliberately scarce.** `organic_pools=3` food clusters and
`ticks_per_generation=24` mean some babies end a generation with surplus
while others near starvation — so a timely gift measurably raises the
tribe's geometric mean. In a rich world (6 pools, 40 ticks) cooperation
collapsed instead: no one starves, the donor just pays the cost, and the
group arm descends into free-riding like the individual arm.

**Results at seed 1 (12 generations, deterministic):**

| Metric | Individual arm | Group arm |
|--------|----------------|-----------|
| Last-gen cooperate rate | 0.0000 | 0.0312 |
| Last-gen contest rate  | 0.1823 | 0.0260 |
| Best fitness           | 226.0 | 405.6 |

The individual arm's cooperation collapses to zero and its contest rate
stays high — sharing is punished, stealing is rewarded. The group arm keeps
cooperating (up to ~37 acts in a generation) and its contest rate drops
~7×: contesting a tribe-mate steals from the same geometric mean. The
verdict `cooperation_emerged` is `grp_coop > ind_coop` on the final
generation. CLI: `--social` (`--social-pools`, `--social-ticks-per-gen`,
`--group-count`, `--group-weight` to explore the regime).

### Persistence

A scene is a single JSON-safe snapshot: `SimScene.to_dict()` → `from_dict()`.
It captures everything that changes a future tick:

| What | How |
|------|-----|
| World grid | `material`, `energy`, `temperature`, `signal` arrays (flat lists + size) |
| Entity state | id, position, energy, type, alive — restored with object identity intact |
| Baby brains | all three perceptron weight matrices, restored without touching RNG |
| Episodic memory | exact ring-buffer contents AND head, so eviction resumes in the same slot |
| Scene counters | tick, next-entity-id, params |

Restore is deterministic: it consumes no random numbers, so a simulation
snapshot after tick N and resumed produces the identical state as an
uninterrupted run. Spawning babies after a restore continues the id counter
past the largest restored id, so new babies never collide with saved ones.
The cell-update callable is not serializable and resets to the default on
load — hook a custom one back after restore when needed.

### Communication

Agents don't communicate directly. They write cells. Other babies perceive those cells.

```
Baby A: writes SIGNAL material cells nearby
  Writing SIGNAL deposits the energy as wave amplitude in the signal field
  The wave engine propagates the amplitude outward each cell update
Baby B: perceives the signal as its 5th cells feature (mean signal in radius)
  Doesn't know what it means
  Must infer from context (what happened before the signal)
```

Language emerges from repeated patterns. If A always writes signal cells near organic cells,
B learns: "signal near organic cells = energy here."

**Directed messages (Stage 6).** An opt-in second channel. With `message_enabled`, a sender picks
the neediest nearby baby within `message_range` and its message perceptron reads that baby's entity
features; if the gate clears `message_gate_threshold`, the sender emits the gate value as an
amplitude, pays `message_cost * amplitude`, and the scene delivers the message to the target's
inbox at the start of the next tick. The recipient sees it as the entity feature at index 5 —
visible only to brains built with `entity_input_dim >= 6`. Same inference story: the target doesn't
know what the message means and must learn from context, but now the message has an address.

The signal channel is first-class and learnable:

| Piece | What it does |
|-------|--------------|
| Emission | `write_cell`/`place_material` with `MATERIAL_SIGNAL` transfer the deposited energy into `grid.signal` at that cell — the wave engine then broadcasts it |
| Waves | `cell_update_waves` carries amplitude to neighbors each tick (`wave_speed` × `(1 - signal_decay)`), so a written signal spreads across the grid over time |
| Perception | `get_nearby_cells` returns the `signal` array; `_perception_features` exposes it as the 5th cells feature (`min(mean(signal), 1.0)`) |
| Learning | `cells_input_dim` is now 5 — every Perceptron/Genome/EpisodicMemory archetype resizes automatically; the signal row of the cells perceptron is reinforced when signal presence correlates with reward |
| Persistence | the `signal` array already round-trips through scene snapshots, so live broadcasts survive save/load |

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
