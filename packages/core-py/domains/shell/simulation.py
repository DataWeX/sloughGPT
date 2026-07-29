"""
Simulation engine — baby agents in a 3D grid world.

The world is a 3D spatial grid where every cell has:
  material (unnamed, 0-7), energy, temperature.

Agents are babies:
  - born alone, no knowledge, random perceptron weights
  - perceive: read nearby cells, detect nearby entities, read own body
  - feel: energy going up (good) or down (bad)
  - act: write cells anywhere (arbitrary, no action menu)
  - learn: strengthen what increased energy, weaken what decreased

The world computes on its own (diffusion, waves, energy conservation).
Babies read results. The world ticks: perceive → feel → react → world compute.

No agent knows the rules. They discover them through experiment.
"""

from __future__ import annotations

import time
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("slo.sim")


# ── Constants ────────────────────────────────────────────────────────────────

NUM_MATERIALS = 8

MATERIAL_AIR = 0
MATERIAL_WATER = 1
MATERIAL_STONE = 2
MATERIAL_ORGANIC = 3
MATERIAL_METAL = 4
MATERIAL_EMBER = 5
MATERIAL_LIVING = 6
MATERIAL_SIGNAL = 7


class EntityType(IntEnum):
    AGENT = 0
    OBJECT = 1
    LIGHT = 2
    EFFECTOR = 3
    SENSOR = 4


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass
class WorldParams:
    """World rules. Agents do NOT know these — they discover them."""
    grid_size: tuple[int, int, int] = (64, 32, 64)
    tick_rate: float = 0.1

    see_radius: float = 5.0
    see_cost: float = 0.5
    write_cost: float = 0.1
    passive_drain: float = 0.5
    energy_loss: float = 0.02
    diffusion_rate: float = 0.1
    wave_speed: float = 1.0
    signal_decay: float = 0.15

    start_energy: float = 100.0
    start_agents: int = 4
    max_entities: int = 128

    # Perception input dimensions (for perceptrons)
    cells_input_dim: int = 4  # material, energy, temperature, distance
    body_input_dim: int = 3   # energy, position x, position y
    entity_input_dim: int = 4  # type, energy, distance, angle


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class WorldCell:
    """Per-cell state — the memory of the world-computer."""
    material: int = MATERIAL_AIR
    energy: float = 0.0
    temperature: float = 20.0


class WorldGrid:
    """
    3D spatial grid — the world state.

    Stored as parallel numpy arrays for vectorized updates.
    Each cell has material, energy, temperature.
    """

    def __init__(self, size: tuple[int, int, int] = (64, 32, 64)):
        self.size = size
        self.nx, self.ny, self.nz = size
        self.total = self.nx * self.ny * self.nz
        self.reset()

    def reset(self):
        """Initialize grid to defaults — air at room temperature."""
        self.material = np.full(self.total, MATERIAL_AIR, dtype=np.int32)
        self.energy = np.zeros(self.total, dtype=np.float32)
        self.temperature = np.full(self.total, 20.0, dtype=np.float32)
        self.signal = np.zeros(self.total, dtype=np.float32)

    def idx(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to flat index. Wraps at boundaries."""
        return (
            (x % self.nx) * self.ny * self.nz
            + (y % self.ny) * self.nz
            + (z % self.nz)
        )

    def coords(self, flat_idx: int) -> tuple[int, int, int]:
        """Convert flat index to 3D coordinates."""
        x = flat_idx // (self.ny * self.nz)
        y = (flat_idx % (self.ny * self.nz)) // self.nz
        z = flat_idx % self.nz
        return x, y, z

    def get_cell(self, x: int, y: int, z: int) -> WorldCell:
        """Read a single cell."""
        i = self.idx(x, y, z)
        return WorldCell(
            material=int(self.material[i]),
            energy=float(self.energy[i]),
            temperature=float(self.temperature[i]),
        )

    def set_cell(self, x: int, y: int, z: int, cell: WorldCell):
        """Write a single cell."""
        i = self.idx(x, y, z)
        self.material[i] = cell.material
        self.energy[i] = cell.energy
        self.temperature[i] = cell.temperature

    def place_material(self, x: int, y: int, z: int, material: int,
                       energy: float = 0.0, temperature: float = 20.0):
        """Place material at a cell."""
        i = self.idx(x, y, z)
        self.material[i] = material
        self.energy[i] = energy
        self.temperature[i] = temperature

    def write_cell(self, x: int, y: int, z: int, material: int,
                   energy: float = 0.0) -> bool:
        """
        Write a cell — the fundamental baby action.
        Returns True if the write succeeded.
        """
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            return False
        i = self.idx(x, y, z)
        self.material[i] = material
        self.energy[i] = energy
        return True

    def get_nearby_cells(self, cx: int, cy: int, cz: int,
                         radius: float) -> dict[str, np.ndarray]:
        """
        Read cells within radius of a point.
        Returns flattened arrays of cell properties.
        """
        r = int(np.ceil(radius))
        xs, ys, zs, dists = [], [], [], []

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    d = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if d <= radius:
                        xs.append(cx + dx)
                        ys.append(cy + dy)
                        zs.append(cz + dz)
                        dists.append(d)

        if not xs:
            return {
                "material": np.array([], dtype=np.int32),
                "energy": np.array([], dtype=np.float32),
                "temperature": np.array([], dtype=np.float32),
                "distance": np.array([], dtype=np.float32),
                "count": 0,
            }

        flat_idx = np.array([self.idx(x, y, z) for x, y, z in zip(xs, ys, zs)])
        dists_arr = np.array(dists, dtype=np.float32)

        return {
            "material": self.material[flat_idx].copy(),
            "energy": self.energy[flat_idx].copy(),
            "temperature": self.temperature[flat_idx].copy(),
            "distance": dists_arr,
            "count": len(flat_idx),
        }

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.energy))

    @property
    def total_signal(self) -> float:
        return float(np.sum(self.signal))


# ── Cell Update Functions (world-computer) ───────────────────────────────────

def cell_update_diffusion(grid: WorldGrid, params: WorldParams):
    """Energy diffusion between neighboring cells. High → low."""
    nx, ny, nz = grid.size
    rate = params.diffusion_rate
    energy = grid.energy.copy()
    temp = grid.temperature.copy()

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = grid.idx(x, y, z)
                neighbors = [
                    grid.idx(x - 1, y, z), grid.idx(x + 1, y, z),
                    grid.idx(x, y - 1, z), grid.idx(x, y + 1, z),
                    grid.idx(x, y, z - 1), grid.idx(x, y, z + 1),
                ]
                for ni in neighbors:
                    de = (energy[ni] - energy[i]) * rate
                    energy[i] += de
                    energy[ni] -= de
                    dt = (temp[ni] - temp[i]) * rate * 0.1
                    temp[i] += dt
                    temp[ni] -= dt

    grid.energy = energy.astype(np.float32)
    grid.temperature = temp.astype(np.float32)


def cell_update_waves(grid: WorldGrid, params: WorldParams):
    """Wave propagation through signal field. Spread outward, decay with distance."""
    signal = grid.signal.copy()
    speed = params.wave_speed
    decay = params.signal_decay

    for x in range(grid.nx):
        for y in range(grid.ny):
            for z in range(grid.nz):
                i = grid.idx(x, y, z)
                if signal[i] <= 0:
                    continue
                neighbors = [
                    grid.idx(x - 1, y, z), grid.idx(x + 1, y, z),
                    grid.idx(x, y - 1, z), grid.idx(x, y + 1, z),
                    grid.idx(x, y, z - 1), grid.idx(x, y, z + 1),
                ]
                for ni in neighbors:
                    transfer = signal[i] * speed * (1.0 - decay)
                    signal[ni] += transfer
                    signal[i] -= transfer

    grid.signal = signal.astype(np.float32)


def cell_update_energy_conservation(grid: WorldGrid, params: WorldParams):
    """Enforce energy conservation — total energy can only decrease."""
    if params.energy_loss > 0:
        grid.energy *= (1.0 - params.energy_loss)
    grid.energy = np.maximum(grid.energy, 0.0)
    grid.temperature = np.clip(grid.temperature, -273.15, 1000.0)


def cell_update_default(grid: WorldGrid, params: WorldParams):
    """Default cell update: diffusion → waves → conservation."""
    cell_update_diffusion(grid, params)
    cell_update_waves(grid, params)
    cell_update_energy_conservation(grid, params)


# ── Entity ───────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A physical thing in the world. Minimal: position + energy."""
    id: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy: float = 100.0
    entity_type: EntityType = EntityType.OBJECT
    alive: bool = True

    def distance_to(self, other: Entity) -> float:
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_point(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - point))


# ── Perception ───────────────────────────────────────────────────────────────

@dataclass
class Perception:
    """What a baby perceives — nearby cells, entities, own body."""
    nearby_cells: dict[str, np.ndarray] = field(default_factory=dict)
    nearby_entities: list[dict[str, Any]] = field(default_factory=list)
    agent_body: dict[str, Any] = field(default_factory=dict)
    time_ms: float = 0.0


# ── Action (arbitrary write) ────────────────────────────────────────────────

@dataclass
class CellWrite:
    """A single cell write — the fundamental baby action."""
    x: int = 0
    y: int = 0
    z: int = 0
    material: int = MATERIAL_AIR
    energy: float = 0.0


@dataclass
class BabyAction:
    """What a baby does — writes any number of cells anywhere."""
    writes: list[CellWrite] = field(default_factory=list)


# ── Perceptron ───────────────────────────────────────────────────────────────

class Perceptron:
    """
    A simple neural unit — weighted sum + activation.
    Random weights at birth. No knowledge.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self.b = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: sigmoid(Wx + b)."""
        return self._sigmoid(x @ self.W + self.b)

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """
        Simple weight update — reinforce what worked, weaken what didn't.
        error is positive (good) or negative (bad).
        """
        pred = self.forward(x)
        # Simple delta rule: weight update proportional to error * input
        grad = np.outer(x, error * pred * (1.0 - pred))
        self.W += grad * lr
        self.b += error * pred * (1.0 - pred) * lr

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


# ── SimBaby ──────────────────────────────────────────────────────────────────

class SimBaby:
    """
    A baby agent in the world.

    Born alone. No knowledge. Random weights.
    Perceives → feels → reacts → learns.
    """

    _next_id = 0

    def __init__(self, position: np.ndarray | None = None,
                 initial_energy: float = 100.0,
                 params: WorldParams | None = None):
        SimBaby._next_id += 1
        self.params = params or WorldParams()

        self.entity = Entity(
            id=SimBaby._next_id,
            position=position if position is not None else np.array([
                np.random.randint(0, max(self.params.grid_size[0], 1)),
                np.random.randint(0, max(self.params.grid_size[1], 1)),
                np.random.randint(0, max(self.params.grid_size[2], 1)),
            ], dtype=np.float64),
            energy=initial_energy,
            entity_type=EntityType.AGENT,
        )

        # Perceptrons — random weights, no knowledge
        cells_input_dim = self.params.cells_input_dim
        body_input_dim = self.params.body_input_dim
        entity_input_dim = self.params.entity_input_dim
        self.perceptron_cells = Perceptron(cells_input_dim, 3)
        self.perceptron_body = Perceptron(body_input_dim, 2)
        self.perceptron_entity = Perceptron(entity_input_dim, 2)

        self._last_perception: Perception | None = None
        self._last_action: BabyAction | None = None
        self._total_ticks = 0
        self._previous_energy = initial_energy

    @property
    def position(self) -> np.ndarray:
        return self.entity.position

    @property
    def energy(self) -> float:
        return self.entity.energy

    def perceive(self, world: WorldGrid) -> Perception:
        """
        Read nearby cells, detect entities, read own body.
        Energy cost is deducted.
        """
        t0 = time.time()
        gx, gy, gz = int(self.position[0]), int(self.position[1]), int(self.position[2])

        cells = world.get_nearby_cells(gx, gy, gz, self.params.see_radius)

        # Add noise proportional to distance
        if cells["count"] > 0:
            noise_scale = cells["distance"] / (self.params.see_radius + 1e-8)
            cells["energy"] += np.random.normal(0, 0.05 * noise_scale).astype(np.float32)
            cells["temperature"] += np.random.normal(0, 0.5 * noise_scale).astype(np.float32)

        # Simple body readout
        body = {
            "position": self.position.tolist(),
            "energy": self.energy,
        }

        elapsed = (time.time() - t0) * 1000

        p = Perception(
            nearby_cells=cells,
            agent_body=body,
            time_ms=elapsed,
        )
        self._last_perception = p
        return p

    def feel(self, previous_energy: float) -> float:
        """
        Feel energy change. Positive = good, negative = bad.
        No concepts. Just sensation.
        """
        return self.energy - previous_energy

    def react(self, perception: Perception, energy_delta: float) -> BabyAction:
        """
        Produce an action based on perception and feeling.
        Default policy: random writes. Over time, perceptrons improve.
        """
        # Default: random cell writes near the baby
        action = BabyAction()
        if self.energy > 10:
            # Write a few random cells nearby
            for _ in range(3):
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                dz = np.random.randint(-3, 4)
                gx = int(self.position[0]) + dx
                gy = int(self.position[1]) + dy
                gz = int(self.position[2]) + dz
                material = np.random.randint(0, NUM_MATERIALS)
                energy = np.random.uniform(0, 5)
                action.writes.append(CellWrite(gx, gy, gz, material, energy))
        self._last_action = action
        return action

    def apply_action(self, action: BabyAction, world: WorldGrid) -> int:
        """
        Apply the baby's action to the world — write cells.
        Returns number of cells written.
        """
        written = 0
        for w in action.writes:
            if world.write_cell(w.x, w.y, w.z, w.material, w.energy):
                written += 1
        return written

    def absorb_energy(self, world: WorldGrid, radius: int = 2) -> float:
        """
        Absorb energy from nearby cells (organic material).
        Returns amount absorbed.
        """
        gx, gy, gz = int(self.position[0]), int(self.position[1]), int(self.position[2])
        absorbed = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    i = world.idx(gx + dx, gy + dy, gz + dz)
                    if world.material[i] == MATERIAL_ORGANIC:
                        take = min(world.energy[i], 1.0)
                        world.energy[i] -= take
                        absorbed += take
        return absorbed

    def learn(self, energy_delta: float):
        """
        Simple learning: strengthen what increased energy, weaken what decreased.
        Updates perceptron weights.
        """
        if energy_delta > 0:
            # Good — reinforce
            lr = 0.01
        else:
            # Bad — weaken
            lr = -0.01

        # Simple body input
        body_input = np.array([
            self.energy / self.params.start_energy,
            self.position[0] / self.params.grid_size[0],
            self.position[1] / self.params.grid_size[1],
        ], dtype=np.float32)

        error = np.sign(energy_delta) * min(abs(energy_delta), 1.0)
        self.perceptron_body.update(body_input, np.array([error, error]), lr=lr)

    @property
    def alive(self) -> bool:
        return self.entity.alive and self.energy > 0

    @property
    def tick_count(self) -> int:
        return self._total_ticks

    def info(self) -> dict:
        return {
            "id": self.entity.id,
            "position": self.position.tolist(),
            "energy": self.energy,
            "ticks": self._total_ticks,
            "alive": self.alive,
        }


# ── SimScene ─────────────────────────────────────────────────────────────────

class SimScene:
    """The virtual world — grid, entities, babies."""

    def __init__(self, params: WorldParams | None = None):
        self.params = params or WorldParams()
        self.world = WorldGrid(self.params.grid_size)
        self.entities: list[Entity] = []
        self._next_entity_id = 1
        self._devices: list[SimBaby] = []
        self._tick = 0
        self._cell_update_fn: Callable | None = None

    def add_baby(self, baby: SimBaby):
        """Add a baby to the world."""
        self._devices.append(baby)
        self.entities.append(baby.entity)

    def spawn_babies(self, count: int | None = None):
        """Spawn babies at random positions."""
        n = count or self.params.start_agents
        for _ in range(n):
            baby = SimBaby(
                initial_energy=self.params.start_energy,
                params=self.params,
            )
            self.add_baby(baby)

    def place_material(self, x: int, y: int, z: int, material: int,
                       energy: float = 0.0, temperature: float = 20.0):
        """Place material in the world grid."""
        self.world.place_material(x, y, z, material, energy, temperature)

    def update_cells(self):
        """Run cell update function on the world grid."""
        fn = self._cell_update_fn or cell_update_default
        fn(self.world, self.params)

    def get_baby(self, baby_id: int) -> SimBaby | None:
        for d in self._devices:
            if d.entity.id == baby_id:
                return d
        return None

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def alive_babies(self) -> list[SimBaby]:
        return [d for d in self._devices if d.alive]

    def info(self) -> dict:
        return {
            "tick": self._tick,
            "grid_size": self.params.grid_size,
            "total_energy": self.world.total_energy,
            "total_signal": self.world.total_signal,
            "entities": len(self.entities),
            "alive_babies": len(self.alive_babies),
        }


# ── Simulation ───────────────────────────────────────────────────────────────

class Simulation:
    """
    Tick-based simulation loop.

    Each tick:
        1. World compute (diffusion, waves, conservation)
        2. For each alive baby:
           a. Perceive (read cells, entities, body)
           b. Feel (energy change)
           c. React (write cells)
           d. Apply action (write to world)
           e. Learn (update perceptron weights)
           f. Passive drain + perception cost
        3. Remove dead babies
    """

    def __init__(self, scene: SimScene, max_ticks: int = 100,
                 verbose: bool = False):
        self.scene = scene
        self.max_ticks = max_ticks
        self.verbose = verbose
        self._tick_log: list[dict] = []
        self._running = False

    def step(self) -> list[dict]:
        """Run one simulation tick. Returns per-baby results."""
        self.scene._tick += 1
        results = []

        # World compute first
        self.scene.update_cells()

        for baby in list(self.scene.alive_babies):
            if not baby.alive:
                continue

            t0 = time.time()
            prev_energy = baby.energy

            # 1. Perceive
            perception = baby.perceive(self.scene.world)
            baby.entity.energy -= self.scene.params.see_cost

            # 2. Feel
            energy_delta = baby.feel(prev_energy)

            # 3. React
            action = baby.react(perception, energy_delta)

            # 4. Apply action
            cells_written = baby.apply_action(action, self.scene.world)
            baby.entity.energy -= self.scene.params.write_cost * cells_written

            # 5. Learn
            baby.learn(energy_delta)

            # 6. Absorb energy from nearby organic material
            absorbed = baby.absorb_energy(self.scene.world)
            baby.entity.energy += absorbed

            # 7. Passive drain
            baby.entity.energy -= self.scene.params.passive_drain
            baby.entity.energy = max(0.0, baby.entity.energy)
            baby._total_ticks += 1

            elapsed = (time.time() - t0) * 1000

            result = {
                "baby_id": baby.entity.id,
                "tick": self.scene.tick,
                "energy": baby.energy,
                "energy_delta": energy_delta,
                "cells_written": cells_written,
                "absorbed": absorbed,
                "total_ms": elapsed,
                "alive": baby.alive,
            }
            results.append(result)

            if self.verbose:
                logger.info(
                    f"tick={self.scene.tick} baby={baby.entity.id} "
                    f"energy={baby.energy:.1f} "
                    f"delta={energy_delta:+.1f} "
                    f"wrote={cells_written}"
                )

        # Remove dead babies from entities
        self.scene.entities = [e for e in self.scene.entities if e.alive]
        self._tick_log.extend(results)
        return results

    def run(self) -> list[dict]:
        """Run the full simulation loop."""
        self._running = True
        all_results = []
        for i in range(self.max_ticks):
            if not self._running:
                break
            results = self.step()
            all_results.extend(results)
        self._running = False
        return all_results

    def stop(self):
        self._running = False

    @property
    def tick_log(self) -> list[dict]:
        return list(self._tick_log)

    def summary(self) -> dict:
        if not self._tick_log:
            return {"total_ticks": 0, "total_actions": 0}

        alive = [r for r in self._tick_log if r["alive"]]
        dead = [r for r in self._tick_log if not r["alive"]]

        return {
            "total_ticks": self.scene.tick,
            "total_baby_ticks": len(self._tick_log),
            "alive_at_end": len(alive) > 0,
            "deaths": len(dead),
            "avg_energy": float(np.mean([r["energy"] for r in self._tick_log])) if self._tick_log else 0,
            "total_cells_written": sum(r["cells_written"] for r in self._tick_log),
            "total_energy_absorbed": sum(r["absorbed"] for r in self._tick_log),
        }
