# Programmable World Realm — Architecture

## What This Is

A self-contained virtual universe where AI babies are born into a grid.
The environment is not a backdrop — it is a complete programmable computing system.
Babies are not scripted — they perceive, feel, and react.
The world computes on its own. Babies read results.

## The Seven Stages (plus Stages 8 & 9)

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
  Status: BUILT (final wave — infinite long-term memory / world reservoir)
  Mechanism: opt-in (`teaching_enabled`). A surplus baby whose teach
  perceptron gate clears `teach_gate_threshold` teaches the neediest
  tribe-mate in range (cultural transmission is in-group): the student's
  behavior weights blend toward the teacher's and a capped number of the
  teacher's best episodes are copied into the student's memory. The teacher
  pays `teach_cost * amplitude` and the cost lands in the same tick's net
  reward, so the teach perceptron is shaped by the honest outcome of the
  lesson. Teaching amplifies lineage quality — good or bad — so the culture
  arm of `benchmark_culture` beats its vertical-only control on 3/5 seeds
  (locked). The world reservoir (`memory_enabled`) completes the picture:
  one append-only `WorldMemory` spans every generation — dead babies and
  survivors deposit their best episodes, newborns are seeded from the
  reservoir's best episodes on top of their memotype, and nothing is ever
  evicted, so lived experience survives death AND crosses lineages. Off by
  default so the locked selection proofs keep their exact energy flow and
  genome layout.
  What it gives us: structures, lateral culture, and world-level memory —
  the full "world becomes the computer" milestone. No stage roadmap items
  remain.
```

**Stage 8 — PREDATION (predator-prey dynamics).**
An opt-in lethal interaction channel added after Stage 7. `predation_enabled`
gives each baby a `perceptron_predation` (entity-input → 1 gate) that decides
whether to hunt the weakest nearby baby within `predation_range` when the gate
clears `predation_gate_threshold`. A strike is lethal and transfers the prey's
full energy to the predator (a transfer, never creation — energy still
conserves), and costs `predation_cost`; the gate is shaped by the honest
same-tick net reward, so hunting pays while prey energy exceeds the strike
cost and self-limits as prey grows scarce. Weights draw last from a dedicated
RNG stream, so the four behavior brains stay bit-identical with the channel
off — the locked selection proofs keep their exact genome layout and energy
flow. CLI: `--predation`.

**Stage 9 — TERRITORIALITY (claim / defend / raid).**
An opt-in spatial channel added after Stage 8. `territoriality_enabled` makes
the region a two-sided resource and gives each baby a `perceptron_territory`
(entity-input → 1 gate) that reads a trespasser's features and, when the gate
clears `defend_gate_threshold`, defends its tribe's region. Territory is
CLAIMED by building nests (Stage 7): a tribe's region is the ground within
`territory_radius` of the nearest nest it owns, so defense only triggers while
the defender stands on its own ground. The value being defended is real shared
energy: a hungry baby (below start energy) standing on FOREIGN ground — within
`territory_radius` of a rival tribe's nearest nest — can RAID that bank, a
one-way drain at `nest_draw_rate` per tick (capped by the gap back to start
energy and by the bank), so an unguarded nest is siphoned and a defended one
is kept. A cleared defend gate evicts the nearest foreign baby within
`defend_range`: the trespasser is softly shoved `defend_push` cells away from
the defender (a pure relocation, never a kill or a stranding) and
`defend_take_fraction` of its energy transfers to the defender — a toll that
scales with what the trespasser carries, so evicting a rich raider pays and
the gate has an honest gradient to learn it. The defender pays `defend_cost`,
a transfer that lands in the same tick's net reward, so the gate is shaped by
the true outcome. Raid and defense are transfers, never creation — the
conservation invariant holds. The territory perceptron is built from fixed
zeros when the channel is off, and genome weights draw from a dedicated RNG
stream, so the four behavior brains stay bit-identical with the channel off —
the locked selection proofs keep their exact genome layout and energy flow.
CLI: `--territory`.

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
    # Durable structures (nests, Stage 7)
    structure_enabled: bool                # nests built, fed, and drawn (opt-in)
    nest_radius: float                     # feed writes must land within this of a nest
    nest_seed_energy: float                # energy needed to seed a new nest
    nest_draw_rate: float                  # max energy a starving baby draws per tick
    nest_use_radius: float                 # draw range around the nest
    nest_decay: float                      # fraction of bank lost per tick
    max_nests: int                         # world-wide nest cap
    # Cultural transmission (teaching, Stage 7)
    teaching_enabled: bool                 # lateral teaching between living agents (opt-in)
    teach_cost: float                      # energy spent per unit of lesson amplitude
    teach_range: float                     # max distance for a lesson
    teach_gate_threshold: float            # perceptron gate must clear to teach
    teach_weight_blend: float              # fraction of the weight gap closed per lesson
    teach_memotype_cap: int                # best episodes copied per lesson (1 default, 0 = none)
    # World-level long-term memory (Stage 7)
    memory_enabled: bool                   # scene carries an append-only world reservoir (opt-in)
    memory_deposit: int                    # best episodes a baby deposits into the reservoir
    memory_seed: int                       # best reservoir episodes seeded into a newborn
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
| **No infinite memory** | Each perception is independent. The living episodic buffer holds only recent episodes (ring buffer). Experience survives death two ways: the inherited memotype (top-reward episodes, capped at memory_inherit) and, when the world-memory channel is on, the append-only world reservoir that never evicts. |
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
| SimBaby | `simulation.py` | Perceive (read cells, nearby agents, body), feel (energy delta), react (cells perceptron gates what/how much to write), learn (reinforce/weaken body, cells, and entity perceptrons, scaled by surprise vs. the episodic-reward baseline), absorb energy from organic, share energy (cooperate), contest energy (compete), teach tribe-mates (opt-in), episodic memory (records every learning event, recalls recent experience) |
| SimScene | `simulation.py` | World + babies + entities + cell updates + baby lifecycle + nearby_babies neighbor query |
| Simulation | `simulation.py` | Tick loop: world compute → perceive → feel → react → write → social step (cooperate/contest) → teach (opt-in) → learn (honest net tick delta) → drain. Stop/run/summary with social stats |
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
| Durable Structures (Nests) | `simulation.py`, `world_driver.py` | Opt-in (`structure_enabled`) durable structures. A baby's cell-write deposit near a nest feeds its bank; a substantial deposit far from any nest seeds a new nest owned by the writer's tribe; a baby under its start energy can draw from its own tribe's nearest nest (a starvation buffer). Fed/seed writes leave the cell as rubble with zero energy, so a deposit is a transfer into the bank — never creation — and the world's conservation invariant holds. Nests decay every tick (`nest_decay`) and empty ones erode away; the world-wide nest cap (`max_nests`) bounds territory. Nests are visible to the entity brain as `EntityType.OBJECT` entries keyed by stored energy (`is_nest`), so territoriality and resource pooling are evolvable. Scene snapshots carry nests and their id counter; the CLI exposes it as `--structures`. Off by default so the locked selection proofs keep their exact energy flow and genome layout |
| Cultural Transmission (Teaching) | `simulation.py`, `evolution.py`, `world_driver.py` | Opt-in (`teaching_enabled`) lateral learning between living agents. Each baby gains a `perceptron_teach` (entity-input → 1 gate, constructed RNG-neutral so the four behavior brains' draw order is unchanged) that reads a chosen tribe-mate's features. When the gate clears `teach_gate_threshold` a lesson is given with the gate value as amplitude: the student's behavior weights blend toward the teacher's by `teach_weight_blend * amplitude`, and up to `teach_memotype_cap` (default 1) of the teacher's best episodes are copied into the student's memory. The target is the neediest baby within `teach_range` among same-tribe neighbors (cultural transmission is in-group, mirroring the tribe-scoped nest draw); the teacher pays `teach_cost * amplitude`, which lands in the same tick's net reward so the teach perceptron is shaped by the honest outcome of the lesson. Episode bulk copies raised the student's reward baseline and dampened its own learning, so the cap is minimal by default and `0` disables episode transfer entirely. Genome serialization carries the teach weights only when the channel is on. CLI: `--culture` |
| Infinite Long-Term Memory (World Reservoir) | `memory.py`, `simulation.py`, `evolution.py`, `world_driver.py` | Opt-in (`memory_enabled`) world-level long-term memory. The engine carries one `WorldMemory` reservoir that spans every generation and never evicts (append-only — growth is bounded by the deposit cadence, not a capacity). Dead babies deposit their best episodes in-sim; every survivor deposits at the generation boundary; each newborn is seeded with the reservoir's best episodes (`memory_seed`) on top of its memotype — so lived experience survives death AND crosses lineages, the collective counterpart to the parent→child memotype. Episodes are stamped with the donor's tribe and id; `run()` reports `memory_size` and `memory_seeds_total` per generation. Off by default so the locked selection proofs keep their exact energy flow and genome layout. The reservoir round-trips through scene snapshots. CLI: `--memory` |
| Predator-Prey (Predation) | `simulation.py`, `evolution.py`, `world_driver.py` | Opt-in (`predation_enabled`) lethal interaction channel. Each baby gains a `perceptron_predation` (entity-input → 1 gate, weights drawn last from a dedicated RNG stream so the four behavior brains' draw order is unchanged) that reads a chosen neighbor's features and, when the gate clears `predation_gate_threshold`, hunts the **weakest** nearby baby within `predation_range`. `hunt()` is lethal: the prey's full energy transfers to the predator (a transfer, never creation — the conservation invariant holds) and the prey dies; the predator pays `predation_cost`, which lands in the same tick's honest net reward so the gate is shaped by the true outcome — hunting pays while prey energy exceeds the strike cost and self-limits as prey runs scarce. The strike executes in the tick's predation step (after social, before absorption); dead prey is swept at the end of the tick. The kin feature lets a tribe learn not to eat its own members while hunting rival tribes. Genome/scene serialization carries the predation weights only when the channel is on. CLI: `--predation` |
| Predation Emergence Proof | `evolution.py`, `world_driver.py` | `benchmark_predation()` runs two arms on the SAME grouped world (same terrain, same per-group food pools, same per-tribe spawn territories, same initial core genomes — the predation weights draw from a dedicated stream so the behavior brains are bit-identical): predation disabled vs. enabled. Predation is never hardcoded — the perceptron must learn to open its gate only when the hunt pays. Verdict: predation-arm final average fitness beats the control arm's. CLI: `--predation` |
| Territoriality (Defense + Raid) | `simulation.py`, `evolution.py`, `world_driver.py` | Opt-in (`territoriality_enabled`) spatial defense channel. Territory is CLAIMED by building nests (Stage 7): a tribe's region is the ground within `territory_radius` of the nearest nest it owns. The region is two-sided: a hungry baby on foreign ground (within `territory_radius` of a rival's nearest nest) can RAID that bank at `nest_draw_rate` per tick — the one-way drain defense protects. Each baby gains a `perceptron_territory` (entity-input → 1 gate, weights drawn from a dedicated RNG stream so the four behavior brains' draw order is unchanged) that reads a trespasser's features; standing on its own tribe's territory, a gate that clears `defend_gate_threshold` evicts the nearest foreign baby within `defend_range`. The eviction is a soft shove — `defend_push` cells away from the defender (a pure relocation, never a kill or a stranding) — plus a toll: `defend_take_fraction` of the trespasser's energy transfers to the defender (capped non-lethal, so evicting a rich raider pays), and the defender pays `defend_cost`, which lands in the same tick's honest net reward so the gate is shaped by the true outcome — defending pays while a trespasser carries more than the eviction costs and self-limits as trespassers run scarce. Raid and defense are transfers, never creation — the conservation invariant holds. The kin feature keeps a tribe from evicting or raiding its own members. Genome/scene serialization carries the territory weights only when the channel is on. CLI: `--territory` |
| Territoriality Emergence Proof | `evolution.py`, `world_driver.py` | `benchmark_territoriality()` runs two arms on the SAME grouped world (same terrain, same per-group food pools, same per-tribe spawn territories, same initial core genomes — the territory weights draw from a dedicated stream so the behavior brains are bit-identical): territoriality disabled vs. enabled. Territoriality is never hardcoded — the territory perceptron must learn to open its gate only when the eviction pays, and the RAID side of the channel makes defense protect real shared energy. Verdict: territoriality-arm final average fitness beats the control arm's. The run reports `defend_rate`, `defenses`, `defend_energy_moved`, `raids`, and `raid_energy_moved`. CLI: `--territory` |
| 38 Tests | `test_territoriality.py` | Territory brain (off-by-default locked-proof guard, gate-below-threshold no-defend, zero-init gate on the threshold, open gate full strength), defend mechanics (fraction toll + shove displacement, conservation safety, noop on dead trespasser / without a brain, push-away shove), raiding (channel-off no-op, hunger-gated, never touches own tribe's bank, requires standing on foreign ground, transfers energy from the foreign nest, capped by gap and bank, targets the nearest foreign nest, summary records raids, off channel runs no raids, exact raider energy flow), scene (off-by-default runs no defense, evicts the nearest foreign trespasser, requires standing on own territory, ignores tribe-mates, range limits evictions, eviction is relocation never kill — geometric toll across repeated evictions, defender pays `defend_cost`, summary totals match the tick log), persistence (serialization round-trip preserves the territory brain), learning (delta-rule shapes the gate from the defense outcome, determinism with territoriality on), evolution (genome round-trip, dedicated stream keeps shared draws identical, run history carries territory + raid fields, run-off-default has no territory brains, benchmark structure + verdict keys carry raids, benchmark determinism) — all pass |
| 23 Tests | `test_predation.py` | Predation brain (off-by-default locked-proof guard, gate-below-threshold no-hunt, zero-init gate on the threshold, open gate full strength), hunt mechanics (energy transfer + kill, conservation safety, noop on dead prey / without a brain), scene (off-by-default runs no predation, strikes the neediest weaker prey with full energy accounting incl. passive drain, strike requires strictly weaker prey, range limits strikes, prey removed next step, summary totals match the tick log), brain persistence (serialization round-trip preserves the predation brain), learning (delta-rule shapes the gate from the hunt outcome, determinism with predation on), evolution (genome round-trip, dedicated stream keeps shared draws identical, run history carries predation fields, run-off-default has no predation brains, benchmark structure + verdict keys, benchmark determinism) — all pass |
| 117 Tests | `test_simulation.py` | WorldGrid (11), CellUpdate (5), Entity (3), Perceptron (4), BabyAction (2), SimBaby (20), SimScene (9), Simulation (9), Social Perception (4), Social Mechanics (6), Social Step (6), Simulation Social (4) — all pass |
| 109 Tests | `test_evolution.py` | Genome serialization/crossover/mutation, engine determinism, elitism, memotype inheritance (consolidation, capping, seeding, generation transition), emergence proof (evolved beats frozen, benchmark determinism, no-selection baseline, shared spawn, surface spawns, identical generation 1 across arms), trait-group selection (dispatches to group mode, single-population ignores groups, geometric-mean tribe ranking, group-id inheritance, grouped-run group means), kin signal (perception carries `group_id`, entity-input shape vs. world dim, dim-4 compat), social acts (cooperate transfers surplus fraction, needs surplus, contest takes from weaker only, no contest vs. stronger, neediest-neighbor targeting), social benchmark (determinism, structure, group coop > individual coop, individual sharing punished, group contests less), directed message (off-by-default guard, gate-gated emission, amplitude = gate value, one-tick-latent delivery to the neediest neighbor, message feature at index 5, dim-5 slicing, genome tensor round-trip, scene pending-bus round-trip, message-brain learning), durable structures/nests (off-by-default guard, nearest-nest group filtering, feed near bank / seed far away, seed energy threshold, world-wide nest cap, deposit-draw conservation, draw gating by hunger/rate/tribe, decay + empty-nest pruning, nest perception, disabled invisibility, nest serialization round-trip + id continuity, legacy snapshot without nests, structures emerging end-to-end in a simulation), delta-rule sign (loss weakens, gain reinforces, neutral no-op), teaching (off-by-default guard, surplus gate, gate threshold, weight blend + episode cap, highest-reward episode copy, neediest same-tribe targeting in a simulation, teach-brain serialization), culture benchmark (determinism, structure, only-culture-arm teaches, locked 3/5 verdict across seeds 1–5, culture wins on an emerged seed), world long-term memory (off-by-default guard, enabled engine carries reservoir, reservoir grows across generations, newborns seeded, deposits stamped with tribe+donor, determinism with memory on, deposit cap) — all pass |
| 42 Tests | `test_memory.py` | EpisodicMemory buffer semantics (record/recall/ring-wrap/reward), SimBaby wiring, memory fills over ticks, surprise-scaled learning (baseline/surprise/scale, weight-update ratio), WorldMemory reservoir (append-only never evicts, record stamps group/donor, consolidate top-reward, zero-cap no-op, recall by reward/recent/group, stats, lossless serialization), scene wiring (off by default, reservoir on demand, deposit consolidation, newborn seeding from reservoir, no-reservoir no-seed guard, scene snapshot preserves reservoir, dead-baby deposit before the sweep, deposit cap on death) — all pass |
| 22 Tests | `test_scene_persistence.py` | WorldGrid/Entity/Perceptron/Memory/Baby/Scene round-trips (lossless, dtype-stable), JSON dump/load, size-mismatch rejection, entity identity after restore, non-baby entity survival, no RNG consumed by restore, resume-matches-continuous-run determinism, id continuity on post-restore spawns, multi-cycle stability — all pass |
| 18 Tests | `test_signal_communication.py` | Signal emission (write/place/air/OOB), perception (nearby signal array, empty read, feature zero/positive, learnable signal row, untouched row without signal), propagation (wave carries to neighbors, reaches distance over ticks), end-to-end loop (B perceives A's broadcast, snapshot survival, full tick loop) — all pass |
| 26 Tests | `test_environment.py` | Material behaviors (organic ignition above `ignition_temp`, full-pipeline combustion, rot metabolism, ember radiation + energy conservation + burnout to stone, living growth + determinism + poverty rest, water signal dampening + cooling, metal conduction faster than baseline + conservation, ambient temperature relaxation), terrain generation (floor/food/water/ember present, deterministic on seed, seed-sensitive, global-RNG neutral), scene integration (opt-in generation, default empty, surface spawn, persistence round-trip, resume-matches-continuous with terrain, babies survive longer with food) — all pass |
| World Driver | `world_driver.py` | Headless observability harness: `WorldDriver` builds a world (empty or deterministic terrain), runs N ticks via the real simulation loop, and reports per-tick energy economy, material populations, and a conservation invariant (grid+entity energy must never increase — a physics regression tripwire). `run_evolution()` delegates to `EvolutionEngine` for genetic sweeps; `--emergence` runs the emergence benchmark; `--social` runs the trait-group-vs-individual cooperation benchmark; `--messages` switches on the directed inter-agent message channel; `--structures` switches on durable nest building; `--culture` runs the teaching-vs-vertical-only cultural transmission benchmark; `--memory` runs the world-reservoir-vs-memotype-only long-term memory benchmark; `--predation` runs the predator-prey-on-vs-off benchmark (Stage 8); `--territory` runs the territoriality-on-vs-off defense+raid benchmark (Stage 9, reports `defenses`, `defend_rate`, `defend_energy_moved`, `raids`, `raid_energy_moved`). CLI: `python3 -m domains.shell.world_driver --grid 64,32,64 --seed 42 --ticks 50 --every 5` (or `--evolution --generations N --population N --ticks-per-gen N`, or `--emergence`, or `--social`, or `--messages`, or `--structures`, or `--culture`, or `--memory`, or `--predation`, or `--territory`). Pure observability — adds no physics, mutates nothing beyond the live scene |
| 22 Tests | `test_world_driver.py` | Material naming (derived from constants, never hardcoded), grid parsing, empty-vs-terrain populations, snapshot shape (incl. `nests` + `nest_energy` keys), per-tick cadence, energy-ledger consistency (incl. nest energy in the total), conservation monotonicity + violation detection, seed determinism, mean baby energy, evolution summary shape, emergence CLI path, CLI output paths — all pass |

### Built but Incomplete

| Component | File | What exists | What's missing |
|-----------|------|-------------|----------------|
| CyclesRenderer | `cycles.py` | Path tracer — GGX BSDF, BVH, multi-bounce, state tensors | Not used in world engine (Stage 3 infra) |
| CyclesDevice | `cycles_device.py` | VM device wrapper for renderer | Not used in world engine |
| RenderNeuralDevice | `render_neural.py` | CNN feature extractor on rendered tensors | Not used in world engine |

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

 1. **Territoriality** — babies claim, defend, and raid regions. Built (Stage 9, opt-in `--territory`): territory is claimed by building nests (Stage 7) — a tribe's region is the ground within `territory_radius` of the nearest nest it owns, and the nest bank is the shared value worth fighting over. The region is two-sided: a hungry baby on foreign ground can RAID a rival tribe's nearest nest at `nest_draw_rate` per tick (a one-way drain, capped by hunger and the bank), so an unguarded nest is siphoned and a defended one is kept. Standing on its own ground, a baby whose `perceptron_territory` gate clears `defend_gate_threshold` evicts the nearest foreign baby within `defend_range`: the trespasser is shoved `defend_push` cells away (a pure relocation, never a kill) and `defend_take_fraction` of its energy transfers to the defender — a toll that scales with what the trespasser carries, so evicting a rich raider pays. The defender pays `defend_cost`, and the honest same-tick net reward shapes the gate (the territory brain's weights draw last from a dedicated RNG stream, so the four behavior brains stay bit-identical with the channel off). The territoriality benchmark (`benchmark_territoriality`) runs defense+raid-on vs defense+raid-off arms on the same grouped world; selection shapes the gate — the kin feature lets a tribe defend its own region while leaving foreign ground alone
2. **Resource competition** — energy is finite, babies compete
3. **Communication protocols** — babies develop signaling through cell writes. The channel is now live: writing SIGNAL material emits amplitude into the `signal` field, waves broadcast it across the grid, and other babies perceive it as a learnable feature — B can now learn what A's broadcasts mean, not just that they exist. Stage 6 adds the directed channel: a baby can address one specific neighbor directly, pay an energy cost for the amplitude, and that neighbor perceives it exactly one tick later (`--messages`)
4. **Cooperation** — babies form groups for mutual benefit. Demonstrated: under trait-group selection a cooperating population outlasts a free-riding one (`--social` benchmark), driven by the kin signal and neediest-neighbor targeting
5. **Predation** — babies consume other entities for energy. Built (Stage 8, opt-in `--predation`): a `perceptron_predation` (entity-input → 1 gate) decides whether to hunt the weakest nearby baby within `predation_range` when the gate clears `predation_gate_threshold`. A strike is lethal — the prey's full energy transfers to the predator (a transfer, not creation, so the world still conserves energy) and the prey dies; the predator pays `predation_cost`, so the gate is shaped by the honest same-tick net reward: hunting pays while prey energy exceeds the strike cost, and self-limits as prey grows scarce. The predation brain's weights draw last from a dedicated RNG stream so the four behavior brains stay bit-identical with the channel off. The predator-prey benchmark (`benchmark_predation`) runs predation-on vs predation-off arms on the same grouped world; selection shapes the balance — the kin feature lets a tribe learn not to eat its own while hunting rival tribes
6. **Structure building** — babies write cell patterns that persist (the world itself is now saveable — a scene snapshot preserves every written cell, so structures survive across sessions, not just ticks). Stage 7 makes structure *durable and shared*: opt-in nest banks hold tribal energy (`--structures`), feeding/seeding/drawing are conservation-safe transfers, and the cap on nests turns building into territorial strategy
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
2. **In-life delta-rule learning (optional, honest reward).** Each baby
   adjusts its own readout weights from its **net tick delta** — the full
   energy flow across the tick (perception cost, movement, writes, social
   transfers, teaching cost, absorption, nest draw, passive drain), measured
   at step 8b after every gain and drain has landed. A baby that reaches food
   genuinely feels the gain; one that wastes energy feels the loss. The
   delta-rule direction comes from the error sign and the learning rate is
   always positive — two defects had to be fixed for this to work: the reward
   used to collapse to the uniform −0.5 see_cost (absorption was measured
   after the delta was captured), and the "weaken" branch double-flipped its
   sign (`error * lr` came out positive) so losses actually reinforced the
   bad behavior and drove the perceptrons to saturation. With both fixed,
   honest rewards make the world thrive under learning, and teaching (below)
   builds on them. Emergence still runs `learning_enabled=False` — under pure
   selection the genome's own weights carry the entire signal.

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

Beyond the living buffer and the lineage, the world itself now remembers.
When the world-memory channel is on (`memory_enabled`), the simulation
carries a single append-only reservoir (`WorldMemory`) that spans the whole
run and never evicts: every dead baby deposits its best episodes in-sim,
every survivor deposits at the generation boundary, and every newborn is
seeded with the reservoir's best episodes (`memory_seed`) on top of its
memotype — so lived experience survives death AND crosses lineages, not
just parent→child. The reservoir is the collective counterpart to the
memotype; growth is bounded only by the deposit cadence, not a capacity.
The reservoir serializes into the scene snapshot like any other state.

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

**Durable structures / nests (Stage 7).** An opt-in third channel that turns cell writes into
shared, persistent resources. With `structure_enabled`, a baby's deposit write (the `write` action)
is routed: if it lands within `nest_radius` of an existing nest it feeds that nest's energy bank
and the cell is left as zero-energy rubble (a **transfer** into the bank — never creation, so the
conservation invariant holds); if it lands far from any nest and the deposit is at least
`nest_seed_energy`, it seeds a **new nest** owned by the writer's tribe. A baby below its start
energy can draw up to `nest_draw_rate` per tick from its own tribe's nearest nest within
`nest_use_radius`. Nests decay by `nest_decay` each tick, erode away when their bank empties, and
the world-wide `max_nests` cap bounds territory. Nests are perceived as `EntityType.OBJECT`
entities keyed by stored energy (`is_nest`), so brains can learn to defend, share, and build near
them. Snapshot round-trip includes every nest and the id counter, so structures survive sessions
and new spawns never reuse a nest id.

| Piece | What it does |
|-------|--------------|
| Feed | deposit within `nest_radius` → energy moves into the nearest nest's bank; the written cell is rubble with zero energy |
| Seed | deposit ≥ `nest_seed_energy` far from any nest → a new `Nest` owned by the writer's `group_id`; cell left as rubble |
| Draw | baby below its start energy → transfers up to `nest_draw_rate` from its tribe's nearest nest within `nest_use_radius` |
| Decay | each tick, `stored_energy *= (1 - nest_decay)`; a nest below the pruning floor erodes away |
| Cap | `max_nests` nests world-wide; a deposit that would seed past the cap stays an ordinary cell deposit |
| Perception | `nearest_nest`/nearby-entity dicts expose nests as OBJECT entities with `energy` + `is_nest` + owner `group_id` |
| Conservation | feed/seed/draw are all transfers (cell energy → bank, or bank → baby); total grid+entity+nest energy never increases |
| Persistence | `nests` + `next_nest_id` round-trip through scene snapshots; a legacy snapshot without them defaults to none |
| CLI | `--structures` enables the channel; tick table gains a `nest_energy` column, energy ledger total includes nests |

**Cultural transmission / teaching (Stage 7).** An opt-in fourth channel that lets learned behavior
spread *laterally* between living agents, not just vertically through genomes. With
`teaching_enabled`, each baby gains a `perceptron_teach` (entity-input → 1 gate) that reads a
chosen tribe-mate's features; when the gate clears `teach_gate_threshold`, a lesson is given at the
gate value's amplitude. The student's behavior weights blend toward the teacher's by
`teach_weight_blend * amplitude`, and up to `teach_memotype_cap` (default 1) of the teacher's
best-reward episodes are copied into the student's memory — a culture *with* a memory, not just
weights. The teacher pays `teach_cost * amplitude`, which lands in the same tick's net reward, so
the teach perceptron itself is shaped by the honest outcome of its lesson. Target selection mirrors
the nests' tribe scoping: the **neediest** same-tribe baby within `teach_range`.

The locked benchmark `benchmark_culture` compares an arm with teaching on (`culture`) against one
with vertical inheritance only (`control`) on identical worlds and selection pressure. Verdict at
the default config (`teach_memotype_cap=1`): **3/5 seeds emerged, mean Δ +9.24** — seeds 2, 3, 5
win (Δ +42.6, +22.5, +84.6), seeds 1 and 4 lose (Δ −10.4, −93.0). Teaching amplifies lineage
quality in both directions: seed 4's culture lineage drifted into aggression and teaching spread
the trait to tribe-mates. Culture is a real lateral channel, not a guarantee on every world.

| Piece | What it does |
|-------|--------------|
| Perceptron | `perceptron_teach` (entity-input → 1 gate), constructed RNG-neutral so the four behavior brains' draw order is unchanged |
| Gate | lesson fires only when the gate clears `teach_gate_threshold`; gate value becomes the lesson amplitude |
| Blend | student behavior weights move toward the teacher's by `teach_weight_blend * amplitude` per lesson |
| Memory | `teach_memotype_cap` (default 1) of the teacher's best episodes copied into the student's memory; `0` disables episode transfer |
| Target | the neediest same-tribe baby within `teach_range` — cultural transmission is in-group |
| Cost | teacher pays `teach_cost * amplitude`, landing in the same tick's net reward (honest step-8b reward) |
| Default | `teaching_enabled=False` so the locked selection proofs keep their exact genome layout and RNG order |
| Serialization | genome tensor carries teach weights only when the channel is on |
| CLI | `--culture` runs the teaching-vs-vertical-only benchmark (3/5 seeds emerged at cap=1) |

**Infinite long-term memory / world reservoir (Stage 7).** The memotype moves experience only
parent→child. The world reservoir is the collective counterpart: with `memory_enabled`, the engine
carries a single `WorldMemory` reservoir that every generation's scene shares. Dead babies deposit
their best episodes in-sim; every survivor deposits at the generation boundary; and each newborn is
seeded with the reservoir's best episodes (`memory_seed`) on top of its memotype — so lived
experience survives death AND crosses lineages. The reservoir is **append-only and never evicts**:
growth is bounded by the deposit cadence, not a fixed capacity. Episodes are honest (features →
action → reward as the baby actually learned them), stamped with the donor's tribe and id.

| Piece | What it does |
|-------|--------------|
| Reservoir | `WorldMemory` — append-only `WorldEpisode` list; `record`, `consolidate` (top-`memory_deposit` from a baby's ring buffer), `recall` (by reward or recent, optionally per tribe), `mean_reward`, `stats`, `to_dict`/`from_dict` |
| Deposit (death) | the simulation's dead-baby cleanup calls `scene.deposit_memory` before the body leaves the world |
| Deposit (boundary) | the evolution engine deposits every survivor after each generation's run |
| Seed | `scene.add_baby` seeds the newborn with the reservoir's `memory_seed` best episodes on top of the memotype `apply_to` recorded first |
| Survival | the reservoir spans the whole engine run, so generation N+1 sees generation 1's deposits |
| Stamps | each episode carries `group_id` and `donor_id`, so tribal experience is attributable and recallable per tribe |
| Default | `memory_enabled=False` so the locked selection proofs keep their exact energy flow and genome layout |
| Persistence | the reservoir round-trips through scene snapshots (lossless `to_dict`/`from_dict`) |
| Reporting | `run()` history rows carry `memory_size` (reservoir length) and `memory_seeds` (episodes seeded this generation); the result carries `memory_size` and `memory_seeds_total` |
| CLI | `--memory` runs the reservoir-vs-memotype-only benchmark with per-generation reservoir size and seeds |

The locked benchmark `benchmark_memory` compares an arm with the world reservoir on (`memory`)
against vertical memotype inheritance only (`control`) on identical worlds, learning enabled in both
arms. Verdict at the default config: **2/5 seeds emerged, mean Δ +5.12** — seeds 2 and 5 win
(Δ +51.6, +147.3), seeds 1, 3, 4 lose (Δ −39.0, −53.0, −81.3). Like teaching, the reservoir is an
honest lateral channel: it preserves whatever experience actually happened — good or bad — so on the
losing seeds it faithfully propagated drifted lineages' dead ends. It is a genuine collective
counterpart to the memotype, not a guarantee on every world.

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
