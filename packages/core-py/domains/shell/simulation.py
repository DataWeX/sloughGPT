"""
Simulation engine — baby agents in a 3D grid world.

The world is a 3D spatial grid where every cell has:
  material (unnamed, 0-7), energy, temperature.

Agents are babies:
  - born alone, no knowledge, random perceptron weights
  - perceive: read nearby cells, detect nearby entities, read own body
  - feel: energy going up (good) or down (bad)
  - act: cells perceptron gates what and how much to write (no action menu)
  - socialize: cooperate (share energy) or compete (contest energy)
  - learn: strengthen what increased energy, weaken what decreased

The world computes on its own (diffusion, waves, energy conservation).
Babies read results. The world ticks: perceive → feel → react → world compute.

Multi-agent: several babies share the world. Each perceives the others
within its see radius and runs a social step against the nearest neighbor.
Cooperation and competition are decided by the entity perceptron (two gates)
and reinforced through the same energy-delta learning rule as everything else.

No agent knows the rules. They discover them through experiment.
"""

from __future__ import annotations

import time
import logging
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np

from .memory import EpisodicMemory

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
    write_energy_scale: float = 1.0  # max energy deposited per written cell (out[2] × scale)
    passive_drain: float = 0.5
    energy_loss: float = 0.02
    diffusion_rate: float = 0.1
    wave_speed: float = 1.0
    signal_decay: float = 0.15

    start_energy: float = 100.0
    start_agents: int = 4
    max_entities: int = 128

    # Social interaction (multi-agent)
    social_enabled: bool = True
    social_radius: float = 3.0
    share_fraction: float = 0.1     # fraction of surplus energy shared per act
    contest_take: float = 2.0       # energy taken per contest against a weaker agent
    contest_threshold: float = 0.5  # perceptron competition gate
    cooperate_threshold: float = 0.5  # perceptron cooperation gate

    # Directed communication (Stage 6). Opt-in: off by default so the locked
    # single/group selection proofs keep their exact genome layout. When on,
    # each baby gains a ``perceptron_message`` (entity-input -> 1 gate) that
    # decides whether to signal a specific neighbor; the recipient perceives
    # the message one tick later as an extra entity feature at index 5, so it
    # is only visible to brains built with ``entity_input_dim >= 6``.
    message_enabled: bool = False
    message_cost: float = 0.5       # energy spent per unit of message amplitude
    message_range: float = 5.0      # max distance for direct delivery
    message_gate_threshold: float = 0.5  # perceptron gate must clear to emit

    # World generation (opt-in terrain, deterministic on (grid_size, world_seed))
    generate_world: bool = False
    world_seed: int = 0

    # Temperature & combustion
    ambient_temp: float = 20.0      # temperature the world relaxes toward
    ambient_cooling: float = 0.01   # fraction of the temp gap closed per tick
    ignition_temp: float = 100.0    # organic material ignites above this
    burn_temp: float = 150.0        # temperature a live ember sustains itself at

    # Material behaviors (world-computer physics)
    ember_heat_rate: float = 0.05     # fraction of ember fuel emitted per tick
    ember_energy_fraction: float = 0.5  # of the emitted fuel, share given as energy
    heat_to_temp: float = 1.0         # converts emitted heat into temperature units
    organic_metabolism: float = 0.001  # fraction of organic energy lost to rot/tick
    living_growth_rate: float = 0.05   # fraction of living energy spent growing/tick
    living_growth_cost: float = 2.0    # energy a living cell must spend per new cell
    growth_transfer_fraction: float = 0.8  # of that cost, share moved into the new cell
    metal_conduction_boost: float = 3.0   # extra diffusion carried by metal cells
    water_signal_dampen: float = 0.5      # fraction of signal removed at water cells
    water_cool_rate: float = 0.05         # water cells relax toward ambient per tick

    # Perception input dimensions (for perceptrons)
    cells_input_dim: int = 5  # material, energy, temperature, occupancy, signal
    body_input_dim: int = 3   # energy, position x, position y
    entity_input_dim: int = 5  # type, energy, distance, angle, kin signal (+ directed-message amplitude at index 5 when >= 6)

    # Episodic memory (ring buffer)
    memory_capacity: int = 64   # episodes remembered before the oldest is evicted
    memory_lookback: int = 5    # recent episodes averaged as the learning baseline
    memory_inherit: int = 8     # episodes a baby consolidates into its offspring's memory

    # Brain & movement (Stage 5 cognition)
    brain_hidden_units: int = 0     # hidden layer per decision perceptron (0 = single layer)
    move_cost: float = 0.2          # energy spent per grid step a baby takes
    move_threshold: float = 10.0    # babies below this energy stay still (like react)
    learning_enabled: bool = True   # in-life delta-rule weight updates (off = pure evolution)


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
        if material == MATERIAL_SIGNAL:
            # Writing a signal cell converts the deposited energy into
            # broadcast amplitude — the wave engine propagates it outward.
            self.signal[i] += energy

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
        if material == MATERIAL_SIGNAL:
            # Signal writes emit: the deposited energy becomes wave amplitude.
            self.signal[i] += energy
        return True

    def get_nearby_cells(self, cx: int, cy: int, cz: int,
                         radius: float) -> dict[str, np.ndarray]:
        """
        Read cells within radius of a point.
        Returns flattened arrays of cell properties.
        """
        dx, dy, dz, dist = _sphere_offsets(radius)
        if dx.size == 0:
            return {
                "material": np.array([], dtype=np.int32),
                "energy": np.array([], dtype=np.float32),
                "temperature": np.array([], dtype=np.float32),
                "signal": np.array([], dtype=np.float32),
                "distance": np.array([], dtype=np.float32),
                "count": 0,
            }
        flat_idx = (
            ((cx + dx) % self.nx) * self.ny * self.nz
            + ((cy + dy) % self.ny) * self.nz
            + ((cz + dz) % self.nz)
        )
        return {
            "material": self.material[flat_idx].copy(),
            "energy": self.energy[flat_idx].copy(),
            "temperature": self.temperature[flat_idx].copy(),
            "signal": self.signal[flat_idx].copy(),
            "distance": dist.astype(np.float32),
            "count": flat_idx.size,
        }

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.energy))

    @property
    def total_signal(self) -> float:
        return float(np.sum(self.signal))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the grid to a JSON-safe dict.

        Returns:
            Dict with size and the four parallel cell arrays (material,
            energy, temperature, signal) as flat lists.
        """
        return {
            "size": list(self.size),
            "material": self.material.tolist(),
            "energy": self.energy.tolist(),
            "temperature": self.temperature.tolist(),
            "signal": self.signal.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldGrid":
        """
        Rebuild a grid from :meth:`to_dict` output.

        Args:
            data: serialized grid dict.

        Returns:
            A new WorldGrid with the stored cell state.

        Raises:
            ValueError: if a stored array length does not match the stored size.
        """
        size = tuple(int(v) for v in data["size"])
        grid = cls(size)
        arrays = {
            "material": (np.asarray(data["material"], dtype=np.int32), grid.total),
            "energy": (np.asarray(data["energy"], dtype=np.float32), grid.total),
            "temperature": (np.asarray(data["temperature"], dtype=np.float32), grid.total),
            "signal": (np.asarray(data["signal"], dtype=np.float32), grid.total),
        }
        for name, (arr, expected) in arrays.items():
            if arr.size != expected:
                raise ValueError(
                    f"{name} array length {arr.size} does not match grid size {expected}"
                )
            setattr(grid, name, arr.reshape(-1))
        return grid


# ── Cell Update Functions (world-computer) ───────────────────────────────────

def cell_update_diffusion(grid: WorldGrid, params: WorldParams):
    """
    Energy diffusion between neighboring cells. High → low.

    Vectorized: the six wrap-around neighbor directions are processed with
    rolled array views, each direction performing a symmetric pairwise exchange
    that conserves the energy (and temperature) totals exactly.
    """
    rate = params.diffusion_rate
    energy = grid.energy.reshape(grid.size).copy()
    temp = grid.temperature.reshape(grid.size).copy()
    for axis in (0, 1, 2):
        for shift in (1, -1):
            diff = (np.roll(energy, shift, axis=axis) - energy) * rate
            energy += diff
            energy -= np.roll(diff, -shift, axis=axis)

            diff_t = (np.roll(temp, shift, axis=axis) - temp) * rate * 0.1
            temp += diff_t
            temp -= np.roll(diff_t, -shift, axis=axis)

    grid.energy = energy.reshape(-1)
    grid.temperature = temp.reshape(-1)


def cell_update_waves(grid: WorldGrid, params: WorldParams):
    """
    Wave propagation through the signal field.

    Each direction pass, every cell with signal emits a fixed fraction
    (``wave_speed`` × ``1 - signal_decay``) toward its neighbor on that side —
    the fraction is taken from the running field, so amplitude splits across
    the six directions exactly as a broadcast spreading outward. Never sends
    more than the cell currently holds, so the field stays non-negative.
    """
    speed = params.wave_speed * (1.0 - params.signal_decay)
    if speed <= 0:
        return
    signal = grid.signal.reshape(grid.size).copy()
    for axis in (0, 1, 2):
        for shift in (1, -1):
            # Signal is never negative (the invariant is maintained below), so
            # ``signal * speed`` equals the guarded ``where(signal > 0, ...)``
            # bit-for-bit while allocating one fewer full-size array per pass.
            transfer = signal * speed
            signal += np.roll(transfer, shift, axis=axis)
            signal -= transfer
    grid.signal = signal.reshape(-1).astype(np.float32)


# Offsets of the six orthogonal neighbors (axis-aligned 3D grid).
_NEIGHBOR_OFFSETS = (
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)

# The neighborhood a baby perceives is a pure function of radius, so the
# sphere offsets are computed once per radius and reused every tick.
_SPHERE_CACHE: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

# The cube a baby absorbs from is likewise a pure function of radius.
_CUBE_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _cube_offsets(radius: int):
    """Cached (dx, dy, dz) offset arrays for the full cube of the radius.

    Order matches the reference triple loop (dx slowest, dz fastest), so the
    visited cell order is identical to the original implementation.
    """
    cached = _CUBE_CACHE.get(radius)
    if cached is not None:
        return cached
    ar = np.arange(-radius, radius + 1)
    dx, dy, dz = np.meshgrid(ar, ar, ar, indexing="ij")
    offsets = (dx.ravel(), dy.ravel(), dz.ravel())
    _CUBE_CACHE[radius] = offsets
    return offsets


def _sphere_offsets(radius: float):
    """Cached (dx, dy, dz, distance) offset arrays for a sphere of the radius.

    Order matches the reference triple loop (dx slowest, dz fastest), so the
    returned cell order is identical to the original implementation.
    """
    cached = _SPHERE_CACHE.get(radius)
    if cached is not None:
        return cached
    r = int(np.ceil(radius))
    ar = np.arange(-r, r + 1)
    dx, dy, dz = np.meshgrid(ar, ar, ar, indexing="ij")
    d = np.sqrt(
        dx.astype(np.float64) ** 2
        + dy.astype(np.float64) ** 2
        + dz.astype(np.float64) ** 2
    )
    inside = d <= radius
    offsets = (dx[inside], dy[inside], dz[inside], d[inside])
    _SPHERE_CACHE[radius] = offsets
    return offsets


def cell_update_combustion(grid: WorldGrid, params: WorldParams):
    """Organic ignites into ember above ignition temperature. Energy is the fuel."""
    hot = (grid.material == MATERIAL_ORGANIC) & (grid.temperature > params.ignition_temp)
    if hot.any():
        grid.material[hot] = MATERIAL_EMBER
        grid.temperature[hot] = params.burn_temp


def cell_update_metabolism(grid: WorldGrid, params: WorldParams):
    """Organic material slowly rots — a small energy sink."""
    if params.organic_metabolism <= 0:
        return
    mask = grid.material == MATERIAL_ORGANIC
    if mask.any():
        grid.energy[mask] *= (1.0 - params.organic_metabolism)


def cell_update_ember(grid: WorldGrid, params: WorldParams):
    """
    Ember burns its stored energy: part is radiated to neighbors as energy,
    the rest becomes heat (temperature). Exhausted ember cools to inert stone.

    Vectorized: every ember radiates from the fuel it held at the start of the
    tick (a snapshot, so ember-ember feedback never re-burns the same fuel
    twice within a tick), emitting an equal share to each of its six
    neighbors. The energy portion is a transfer (conserved); the heat portion
    leaves the energy pool, matching the documented burn physics.
    """
    if params.ember_heat_rate <= 0:
        return
    material = grid.material
    energy = grid.energy
    temperature = grid.temperature
    ember_ids = np.flatnonzero(material == MATERIAL_EMBER)
    if ember_ids.size == 0:
        return
    n = float(len(_NEIGHBOR_OFFSETS))
    rate = params.ember_heat_rate
    fuel = energy[ember_ids].astype(np.float64)
    d = np.minimum(fuel * rate, fuel)

    alive = fuel > 0.0
    if not alive.any():
        material[ember_ids] = MATERIAL_STONE
        return
    dead_ids = ember_ids[~alive]
    if dead_ids.size:
        material[dead_ids] = MATERIAL_STONE

    energy[ember_ids] -= d
    e_per = d * params.ember_energy_fraction / n
    t_per = (
        d * (1.0 - params.ember_energy_fraction) * params.heat_to_temp / n
    )
    nx, ny, nz = grid.size
    ex = ember_ids // (ny * nz)
    rem = ember_ids % (ny * nz)
    ey = rem // nz
    ez = rem % nz
    # Scatter each ember's equal share to its six neighbors. Under a fixed
    # offset the neighbor map is a bijection over the ember set, so targets
    # within one pass are distinct — a direct fancy-indexed add is exact.
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        ni = (
            ((ex + dx) % nx) * ny * nz
            + ((ey + dy) % ny) * nz
            + ((ez + dz) % nz)
        )
        energy[ni] += e_per
        temperature[ni] += t_per

    burned_ids = ember_ids[energy[ember_ids] <= 0.0]
    if burned_ids.size:
        material[burned_ids] = MATERIAL_STONE
    keep_ids = ember_ids[energy[ember_ids] > 0.0]
    if keep_ids.size:
        temperature[keep_ids] = np.maximum(temperature[keep_ids], params.burn_temp)


def cell_update_living(grid: WorldGrid, params: WorldParams):
    """
    Living material grows: it spends energy to turn adjacent air into organic.
    Too poor to afford a new cell, it rests. Deterministic — the first free
    neighbor in scan order wins, so no RNG is consumed.

    Vectorized: living cells are processed in ascending flat-index order (the
    same order the loop used), and for each of the six neighbor directions the
    air targets are claimed in one batched pass, resolving collisions in favor
    of the earliest cell. Exactly replicates first-free-neighbor semantics.
    """
    if params.living_growth_rate <= 0 or params.living_growth_cost <= 0:
        return
    material = grid.material
    energy = grid.energy
    living = np.flatnonzero(material == MATERIAL_LIVING)
    if living.size == 0:
        return
    fuel = energy[living].astype(np.float64)
    spend = np.minimum(fuel * params.living_growth_rate, fuel)
    can_grow = spend >= params.living_growth_cost
    if not can_grow.any():
        return
    pending = np.flatnonzero(can_grow)
    nx, ny, nz = grid.size
    x = living[pending] // (ny * nz)
    rem = living[pending] % (ny * nz)
    y = rem // nz
    z = rem % nz
    claimed = np.zeros(grid.total, dtype=bool)
    targets = np.empty(pending.size, dtype=np.int64)
    assigned = np.zeros(pending.size, dtype=bool)
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        still = ~assigned
        if not still.any():
            break
        idx_still = np.flatnonzero(still)
        xn = (x[idx_still] + dx) % nx
        yn = (y[idx_still] + dy) % ny
        zn = (z[idx_still] + dz) % nz
        ni = xn * ny * nz + yn * nz + zn
        free = (material[ni] == MATERIAL_AIR) & (~claimed[ni])
        cand_idx = idx_still[free]
        cand_targets = ni[free]
        if cand_idx.size == 0:
            continue
        uniq_targets, first = np.unique(cand_targets, return_index=True)
        winners = cand_idx[first]
        claimed[uniq_targets] = True
        targets[winners] = uniq_targets
        assigned[winners] = True
    grew = np.flatnonzero(assigned)
    if grew.size:
        cost = params.living_growth_cost
        material[targets[grew]] = MATERIAL_ORGANIC
        energy[targets[grew]] += cost * params.growth_transfer_fraction
        energy[living[grew]] -= cost


def cell_update_water(grid: WorldGrid, params: WorldParams):
    """Water damps signal and relaxes temperature toward ambient faster than air."""
    water = grid.material == MATERIAL_WATER
    if not water.any():
        return
    if params.water_signal_dampen > 0:
        grid.signal[water] *= (1.0 - params.water_signal_dampen)
    if params.water_cool_rate > 0:
        grid.temperature[water] += (
            (params.ambient_temp - grid.temperature[water]) * params.water_cool_rate
        )


def cell_update_conduction(grid: WorldGrid, params: WorldParams):
    """Metal conducts energy rapidly — extra pairwise diffusion on metal cells."""
    boost = params.metal_conduction_boost
    if boost <= 0:
        return
    rate = params.diffusion_rate * boost
    metal = grid.material == MATERIAL_METAL
    if not metal.any():
        return
    energy = grid.energy.astype(np.float64)
    idx = np.flatnonzero(metal)
    nx, ny, nz = grid.size
    x = idx // (ny * nz)
    rem = idx % (ny * nz)
    y = rem // nz
    z = rem % nz
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        ni = ((x + dx) % nx) * ny * nz + ((y + dy) % ny) * nz + ((z + dz) % nz)
        de = (energy[ni] - energy[idx]) * rate
        energy[idx] += de
        energy[ni] -= de
    grid.energy = energy.astype(np.float32)


def cell_update_temperature(grid: WorldGrid, params: WorldParams):
    """The whole world relaxes toward ambient temperature."""
    if params.ambient_cooling > 0:
        grid.temperature += (
            (params.ambient_temp - grid.temperature) * params.ambient_cooling
        )


def cell_update_materials(grid: WorldGrid, params: WorldParams):
    """All material behaviors: ignition, rot, ember burn, growth, water, metal."""
    cell_update_combustion(grid, params)
    cell_update_metabolism(grid, params)
    cell_update_ember(grid, params)
    cell_update_living(grid, params)
    cell_update_water(grid, params)
    cell_update_conduction(grid, params)


def cell_update_energy_conservation(grid: WorldGrid, params: WorldParams):
    """Enforce energy conservation — total energy can only decrease."""
    if params.energy_loss > 0:
        grid.energy *= (1.0 - params.energy_loss)
    grid.energy = np.maximum(grid.energy, 0.0)
    grid.temperature = np.clip(grid.temperature, -273.15, 1000.0)


def cell_update_default(grid: WorldGrid, params: WorldParams):
    """Default cell update: diffusion → waves → materials → temperature → conservation."""
    cell_update_diffusion(grid, params)
    cell_update_waves(grid, params)
    cell_update_materials(grid, params)
    cell_update_temperature(grid, params)
    cell_update_energy_conservation(grid, params)


# ── World Generation (terrain) ───────────────────────────────────────────────

def generate_world(grid: WorldGrid, params: WorldParams,
                   seed: int | None = None) -> None:
    """
    Build terrain into an empty grid using a local seeded RNG.

    Deterministic on ``(grid_size, world_seed)`` — the generator draws from
    ``np.random.default_rng``, never the global stream, so scene snapshots and
    restore stay RNG-neutral. Layout: stone floor (y=0), water pools and
    organic patches on the surface (y=1), buried ember vents in the floor.

    Args:
        grid: the world grid to fill (must already be air/empty).
        params: world rules; uses ``world_seed`` when ``seed`` is not given.
        seed: optional explicit seed (overrides ``params.world_seed``).

    Side effects:
        - mutates ``grid.material``, ``grid.energy``, ``grid.temperature``.
    """
    rng = np.random.default_rng(seed if seed is not None else params.world_seed)
    nx, ny, nz = grid.size
    if ny < 1:
        return

    # Stone floor across the whole base layer.
    for x in range(nx):
        for z in range(nz):
            i = grid.idx(x, 0, z)
            grid.material[i] = MATERIAL_STONE
            grid.temperature[i] = params.ambient_temp + float(rng.uniform(0.0, 2.0))
    if ny < 2:
        return

    max_r = max(2, min(nx, nz) // 8)

    # Water pools on the surface.
    for _ in range(max(1, int(rng.integers(1, max(2, nx // 16))))):
        cx = int(rng.integers(0, nx))
        cz = int(rng.integers(0, nz))
        r = int(rng.integers(1, max_r + 1))
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dz * dz <= r * r:
                    grid.material[grid.idx(cx + dx, 1, cz + dz)] = MATERIAL_WATER

    # Organic patches on the surface — the food supply babies discover.
    for _ in range(max(1, int(rng.integers(1, max(2, nx // 16))))):
        cx = int(rng.integers(0, nx))
        cz = int(rng.integers(0, nz))
        r = int(rng.integers(1, max_r + 1))
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dz * dz <= r * r:
                    i = grid.idx(cx + dx, 1, cz + dz)
                    if grid.material[i] == MATERIAL_AIR:
                        grid.material[i] = MATERIAL_ORGANIC
                        grid.energy[i] = float(rng.uniform(50.0, 300.0))

    # Buried ember vents — heat sources in the stone floor.
    for _ in range(max(1, int(rng.integers(1, 3)))):
        x = int(rng.integers(0, nx))
        z = int(rng.integers(0, nz))
        i = grid.idx(x, 0, z)
        grid.material[i] = MATERIAL_EMBER
        grid.energy[i] = float(rng.uniform(200.0, 800.0))
        grid.temperature[i] = params.burn_temp


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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the entity to a JSON-safe dict.

        Returns:
            Dict with id, position, energy, entity_type, alive.
        """
        return {
            "id": int(self.id),
            "position": self.position.tolist(),
            "energy": float(self.energy),
            "entity_type": int(self.entity_type),
            "alive": bool(self.alive),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """
        Rebuild an entity from :meth:`to_dict` output.

        Args:
            data: serialized entity dict.

        Returns:
            A new Entity with the stored state.
        """
        entity = cls.__new__(cls)
        entity.id = int(data["id"])
        entity.position = np.asarray(data["position"], dtype=np.float64)
        entity.energy = float(data["energy"])
        entity.entity_type = EntityType(int(data["entity_type"]))
        entity.alive = bool(data["alive"])
        return entity


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

    Random weights at birth. No knowledge. Optionally a **deeper brain**:
    when ``hidden_units > 0`` the input is first projected through a fixed
    random sigmoid layer (never trained — no backpropagation), and the
    resulting nonlinear features (concatenated with the raw input) feed the
    readout that the delta rule and evolution shape. The hidden projection is
    genetic material: it is serialized and inherited so offspring keep their
    parents' feature space.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 0):
        self.hidden_units = int(hidden_units)
        if self.hidden_units > 0:
            self.H = np.random.randn(int(input_dim), self.hidden_units).astype(np.float32) * 0.5
            self.bh = np.random.randn(self.hidden_units).astype(np.float32) * 0.1
            readout_in = int(input_dim) + self.hidden_units
        else:
            self.H = None
            self.bh = None
            readout_in = int(input_dim)
        self.W = np.random.randn(readout_in, int(output_dim)).astype(np.float32) * 0.1
        self.b = np.zeros(int(output_dim), dtype=np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        """
        Representation fed to the readout.

        Without a hidden layer this is the raw input. With one, it is the raw
        input plus the nonlinear hidden features (a skip connection), so the
        deeper brain can always fall back to a plain perceptron on the raw
        signal while gaining the extra hidden capacity.
        """
        arr = np.asarray(x, dtype=np.float32)
        if self.H is None:
            return arr
        h = self._sigmoid(arr @ self.H + self.bh)
        return np.concatenate([arr, h])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: sigmoid(features(x) @ W + b)."""
        return self._sigmoid(self._features(x) @ self.W + self.b)

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """
        Simple weight update — reinforce what worked, weaken what didn't.

        ``error`` is positive (good) or negative (bad). Only the readout
        weights change; the hidden projection stays fixed (no backpropagation).

        Args:
            x: input features.
            error: scalar or vector reward signal.
            lr: learning rate (sign encodes reinforce vs weaken).
        """
        features = self._features(x)
        pred = self._sigmoid(features @ self.W + self.b)
        # Simple delta rule: weight update proportional to error * features
        grad = np.outer(features, error * pred * (1.0 - pred))
        self.W += grad * lr
        self.b += error * pred * (1.0 - pred) * lr

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the perceptron weights to a JSON-safe dict.

        Returns:
            Dict with the weight matrix W and bias vector b as lists; when a
            hidden projection is present, H and bh are included as well.
        """
        data: dict[str, Any] = {
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }
        if self.H is not None:
            data["H"] = self.H.tolist()
            data["bh"] = self.bh.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Perceptron":
        """
        Rebuild a perceptron from :meth:`to_dict` output without touching RNG.

        Args:
            data: serialized weight dict.

        Returns:
            A new Perceptron with the stored weights (hidden projection
            restored only when present in the payload).
        """
        p = object.__new__(cls)
        p.W = np.asarray(data["W"], dtype=np.float32).copy()
        p.b = np.asarray(data["b"], dtype=np.float32).copy()
        if "H" in data:
            p.H = np.asarray(data["H"], dtype=np.float32).copy()
            p.bh = np.asarray(data["bh"], dtype=np.float32).copy()
            p.hidden_units = int(p.H.shape[1])
        else:
            p.H = None
            p.bh = None
            p.hidden_units = 0
        return p


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
                 params: WorldParams | None = None,
                 group_id: int = 0):
        SimBaby._next_id += 1
        self.params = params or WorldParams()
        self.group_id: int = int(group_id)

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
        hidden = self.params.brain_hidden_units
        self.perceptron_cells = Perceptron(cells_input_dim, 3, hidden_units=hidden)
        self.perceptron_body = Perceptron(body_input_dim, 2)
        self.perceptron_entity = Perceptron(entity_input_dim, 2)
        self.perceptron_move = Perceptron(cells_input_dim, 3, hidden_units=hidden)
        self.perceptron_message: Perceptron | None = None
        if self.params.message_enabled:
            # Directed-signal brain: reads a target neighbor's entity features
            # and emits a single gate whose value is the message amplitude.
            self.perceptron_message = Perceptron(entity_input_dim, 1)

        # Inbox — messages delivered to this baby this tick (sender_id -> amplitude).
        # Filled by the scene's delivery pass at the start of each tick.
        self._inbox: dict[int, float] = {}
        self._last_message_input: np.ndarray | None = None
        self._last_message_out: float | None = None

        self._last_perception: Perception | None = None
        self._last_action: BabyAction | None = None
        self._last_features: np.ndarray | None = None
        self._last_gates: tuple[float, ...] | None = None
        self._last_move: np.ndarray | None = None
        self._total_ticks = 0
        self._previous_energy = initial_energy
        self._last_learning: dict | None = None

        # Episodic memory — a ring buffer of lived experiences
        self.memory = EpisodicMemory(capacity=self.params.memory_capacity)

    @property
    def position(self) -> np.ndarray:
        return self.entity.position

    @property
    def energy(self) -> float:
        return self.entity.energy

    def distance_to_point(self, point: np.ndarray) -> float:
        """Euclidean distance to a point in world coordinates."""
        return float(np.linalg.norm(self.position - point))

    def perceive(self, world: WorldGrid, babies: list[SimBaby] | None = None) -> Perception:
        """
        Read nearby cells, detect entities, read own body.
        Energy cost is deducted.

        Args:
            world: the world grid.
            babies: optional list of other agents to detect as entities.

        Returns:
            Perception with nearby_cells, nearby_entities, agent_body.
        """
        t0 = time.time()
        gx, gy, gz = int(self.position[0]), int(self.position[1]), int(self.position[2])

        cells = world.get_nearby_cells(gx, gy, gz, self.params.see_radius)

        # Add noise proportional to distance
        if cells["count"] > 0:
            noise_scale = cells["distance"] / (self.params.see_radius + 1e-8)
            cells["energy"] += np.random.normal(0, 0.05 * noise_scale).astype(np.float32)
            cells["temperature"] += np.random.normal(0, 0.5 * noise_scale).astype(np.float32)

        # Detect nearby agents (social perception)
        entities: list[dict[str, Any]] = []
        if babies is not None:
            for other in babies:
                if not other.alive or other.entity.id == self.entity.id:
                    continue
                d = other.distance_to_point(self.position)
                if d <= self.params.see_radius:
                    dx, dy, dz = other.position - self.position
                    angle = float(np.arctan2(dz, dx))
                    entities.append({
                        "id": other.entity.id,
                        "type": int(other.entity.entity_type),
                        "energy": other.energy,
                        "distance": d,
                        "angle": angle,
                        "group_id": other.group_id,
                        "message": self._inbox.get(other.entity.id, 0.0),
                    })

        # Simple body readout
        body = {
            "position": self.position.tolist(),
            "energy": self.energy,
        }

        elapsed = (time.time() - t0) * 1000

        p = Perception(
            nearby_cells=cells,
            nearby_entities=entities,
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

    def _perception_features(self, perception: Perception) -> np.ndarray:
        """
        Build the cells feature vector for the cells perceptron.

        Build the cells feature vector for the cells perceptron.

        Five normalized features drawn from the nearest-cell readout:
        material, energy, temperature, occupancy (cell count), and signal
        (broadcast strength). The vector is sliced to ``cells_input_dim`` so
        a smaller dim simply drops the trailing features.

        Args:
            perception: the perception to summarize.

        Returns:
            np.ndarray of shape (cells_input_dim,), values in [0, 1].
        """
        cells = perception.nearby_cells
        n = int(cells.get("count", 0))
        if n == 0:
            return np.zeros(self.params.cells_input_dim, dtype=np.float32)
        return np.clip(np.array([
            float(np.mean(cells["material"])) / max(float(NUM_MATERIALS), 1.0),
            float(np.mean(cells["energy"])) / max(self.params.start_energy, 1.0),
            float(np.mean(cells["temperature"])) / 100.0,
            min(float(n) / 16.0, 1.0),
            min(float(np.mean(cells["signal"])), 1.0),  # broadcast strength in radius
        ], dtype=np.float32), 0.0, 1.0)[:self.params.cells_input_dim]

    def react(self, perception: Perception, energy_delta: float) -> BabyAction:
        """
        Produce an action based on perception and feeling.

        The cells perceptron gates the write policy — three outputs:
          out[0] drives how many cells to write (0..6),
          out[1] selects which material,
          out[2] scales how much energy to deposit.
        With random weights the policy is effectively random; learning
        reshapes it from experience. A starving baby (energy <= 10) stays still.
        """
        action = BabyAction()
        if self.energy > 10:
            features = self._perception_features(perception)
            out = self.perceptron_cells.forward(features)
            n_writes = int(round(float(out[0]) * 6))
            material = int(float(out[1]) * NUM_MATERIALS) % NUM_MATERIALS
            energy = float(out[2]) * self.params.write_energy_scale
            for _ in range(n_writes):
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                dz = np.random.randint(-3, 4)
                gx = int(self.position[0]) + dx
                gy = int(self.position[1]) + dy
                gz = int(self.position[2]) + dz
                action.writes.append(CellWrite(gx, gy, gz, material, energy))
            self._last_features = features
            self._last_gates = (float(out[0]), float(out[1]), float(out[2]))
        else:
            self._last_features = None
            self._last_gates = None
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

    def decide_move(self, perception: Perception) -> np.ndarray | None:
        """
        Decide a movement step from perception (movement perceptron).

        The move perceptron reads the same cells features as the write policy
        and emits three sigmoid gates. Each gate maps to a grid step of -1 or
        +1 on its axis via ``sign(gate - 0.5)``; a gate held exactly at 0.5
        means "stay put on this axis". A starving baby (energy <=
        move_threshold) stays still, matching the write policy. The simulation
        loop applies the returned direction (one grid step, wrapped at world
        boundaries) and charges ``move_cost``.

        Movement is emergent: no pathfinding, no target, no menu. Evolution
        selects for policies that wander toward food and hold their gates at
        0.5 while absorbing it — "wander, then stop on food".
        """
        if self.energy <= self.params.move_threshold:
            self._last_move = None
            return None
        features = self._perception_features(perception)
        out = self.perceptron_move.forward(features)
        direction = np.sign(out - 0.5).astype(np.float32)
        self._last_move = direction
        return direction

    def absorb_energy(self, world: WorldGrid, radius: int = 2) -> float:
        """
        Absorb energy from nearby cells (organic material).
        Returns amount absorbed.
        """
        gx, gy, gz = int(self.position[0]), int(self.position[1]), int(self.position[2])
        dx, dy, dz = _cube_offsets(radius)
        idx = (
            ((gx + dx) % world.nx) * world.ny * world.nz
            + ((gy + dy) % world.ny) * world.nz
            + ((gz + dz) % world.nz)
        )
        organic = world.material[idx] == MATERIAL_ORGANIC
        if not organic.any():
            return 0.0
        idx = idx[organic]
        if idx.size != np.unique(idx).size:
            # Wrap-around aliasing: a cell is visited more than once, and each
            # visit reads the running (already reduced) value. Only the
            # sequential loop reproduces that exactly.
            absorbed = 0.0
            for i in idx:
                take = min(world.energy[i], 1.0)
                world.energy[i] -= take
                absorbed += take
            return absorbed
        take = np.minimum(world.energy[idx], np.float32(1.0))
        world.energy[idx] -= take
        absorbed = np.float32(0.0)
        for v in take:
            absorbed += v
        return float(absorbed)

    def learn(self, energy_delta: float):
        """
        Learning with an episodic-memory baseline (surprise-based).

        The baby expects to keep getting what it has gotten recently: the
        baseline is the mean reward of recent episodes. Outcomes that diverge
        from that baseline — a bigger gain than expected, a worse loss than
        expected — update the perceptrons harder (predictive coding), while
        unsurprising outcomes nudge weights less. The sign rule is unchanged:
        gains reinforce, losses weaken.

        Updates the body, cells, and entity perceptrons, then records the
        episode (features + action + reward) to episodic memory.
        """
        baseline = self.memory.mean_reward()  # 0.0 when the buffer is empty
        surprise = abs(energy_delta - baseline)
        scale = 0.5 + min(surprise, 1.0)      # learning rate multiplier, 0.5x..1.5x
        self._last_learning = {
            "baseline": baseline,
            "surprise": surprise,
            "scale": scale,
        }

        if energy_delta > 0:
            # Good — reinforce
            lr = 0.01 * scale
        else:
            # Bad — weaken
            lr = -0.01 * scale

        error = np.sign(energy_delta) * min(abs(energy_delta), 1.0)

        if self.params.learning_enabled:
            # Simple body input
            body_input = np.array([
                self.energy / self.params.start_energy,
                self.position[0] / self.params.grid_size[0],
                self.position[1] / self.params.grid_size[1],
            ], dtype=np.float32)

            self.perceptron_body.update(body_input, np.array([error, error]), lr=lr)

            # Learn cells behavior from the last perception
            last = self._last_perception
            if last is not None:
                cells_input = self._perception_features(last)
                self.perceptron_cells.update(
                    cells_input,
                    np.array([error, error, error], dtype=np.float32),
                    lr=lr,
                )

            # Learn movement from the last perception + move decision
            if last is not None and self._last_move is not None:
                move_input = self._perception_features(last)
                self.perceptron_move.update(
                    move_input,
                    np.array([error, error, error], dtype=np.float32),
                    lr=lr * 0.5,
                )

            # Learn social behavior from last perceived neighbor (if any)
            if last is not None and last.nearby_entities:
                nearest = min(last.nearby_entities, key=lambda e: e["distance"])
                entity_input = self._entity_features(nearest)
                self.perceptron_entity.update(
                    entity_input,
                    np.array([error, error]),
                    lr=lr * 0.5,
                )

            # Learn directed messaging from the last message decision (if any)
            if self._last_message_input is not None and self.perceptron_message is not None:
                self.perceptron_message.update(
                    self._last_message_input,
                    np.array([error]),
                    lr=lr * 0.5,
                )
                self._last_message_input = None
                self._last_message_out = None

        # Remember this episode
        if self._last_features is not None and self._last_gates is not None:
            self.memory.record(
                features=self._last_features,
                action=self._last_gates,
                reward=energy_delta,
                tick=self._total_ticks,
            )

    def recall_memories(self, k: int | None = None, by_reward: bool = False) -> list[dict]:
        """
        Retrieve remembered episodes as plain dicts.

        Args:
            k: number of episodes to return (defaults to the world's
                memory_lookback); capped by the buffer size.
            by_reward: when True return the k highest-reward episodes,
                otherwise the k most recent.

        Returns:
            List of dicts with features, action, reward, tick.
        """
        k = self.params.memory_lookback if k is None else k
        return [
            {
                "features": e.features.tolist(),
                "action": list(e.action),
                "reward": e.reward,
                "tick": e.tick,
            }
            for e in self.memory.recall(k, by_reward=by_reward)
        ]

    def share_energy(self, other: SimBaby) -> float:
        """
        Cooperative act — give a fraction of surplus energy to a neighbor.

        Args:
            other: the receiving agent.

        Returns:
            Energy transferred.
        """
        if not other.alive or self.energy <= 0:
            return 0.0
        transfer = min(self.energy * self.params.share_fraction, other.params.start_energy)
        if transfer <= 0:
            return 0.0
        self.entity.energy -= transfer
        other.entity.energy += transfer
        return transfer

    def contest_energy(self, other: SimBaby) -> float:
        """
        Competitive act — take energy from a weaker neighbor.

        Args:
            other: the contested agent.

        Returns:
            Energy taken (0 if the other agent is not weaker).
        """
        if not other.alive or other.energy >= self.energy:
            return 0.0
        take = min(self.params.contest_take, other.energy)
        other.entity.energy -= take
        self.entity.energy += take
        return take

    def _entity_features(self, entity: dict) -> np.ndarray:
        """
        Build the entity feature vector for a detected neighbor.

        Six features: entity type, energy, distance, angle, same-tribe kin
        signal, and the amplitude of any directed message received from that
        neighbor this tick. The vector is sliced to ``entity_input_dim``, so a
        smaller dim simply drops the trailing features — a brain built with
        dim 5 never sees the message channel, one built with dim 6 does.

        Args:
            entity: a perception entity dict (see ``perceive``).

        Returns:
            np.ndarray of shape (entity_input_dim,), values in [0, 1].
        """
        return np.array([
            float(entity["type"]) / max(int(EntityType.EFFECTOR) + 1, 1),
            entity["energy"] / max(self.params.start_energy, 1.0),
            entity["distance"] / max(self.params.see_radius, 1.0),
            (entity["angle"] + np.pi) / (2 * np.pi),
            1.0 if entity.get("group_id") == self.group_id else 0.0,  # kin signal
            min(float(entity.get("message", 0.0)), 1.0),  # directed message amplitude
        ], dtype=np.float32)[:self.params.entity_input_dim]

    def decide_message(self, other: SimBaby) -> float:
        """
        Decide whether to send a directed message to a specific neighbor.

        The message perceptron reads the target's entity features (same input
        the social brain sees, so signaling can be keyed to need, kin, or
        wealth) and emits one sigmoid gate. When the gate clears
        ``message_gate_threshold`` a message is emitted with the gate value as
        its amplitude — a louder signal carries more information and costs
        more energy. The simulation loop delivers the message to the target's
        inbox at the start of the next tick (one-tick latency).

        Args:
            other: the intended recipient.

        Returns:
            Amplitude in [0, 1]; 0.0 means no message is emitted.
        """
        if self.perceptron_message is None or not other.alive:
            self._last_message_input = None
            self._last_message_out = None
            return 0.0
        d = other.distance_to_point(self.position)
        entity = {
            "type": int(other.entity.entity_type),
            "energy": other.energy,
            "distance": d,
            "angle": float(np.arctan2(other.position[2] - self.position[2],
                                      other.position[0] - self.position[0])),
            "group_id": other.group_id,
            "id": other.entity.id,
            "message": self._inbox.get(other.entity.id, 0.0),
        }
        features = self._entity_features(entity)
        out = self.perceptron_message.forward(features)
        gate = float(out[0])
        self._last_message_input = features
        self._last_message_out = gate
        if gate >= self.params.message_gate_threshold:
            return gate
        return 0.0

    def social_step(self, other: SimBaby) -> dict[str, float]:
        """
        Perceptron-driven social decision against one neighbor.

        The entity perceptron outputs two gates: competition and cooperation.
        Cooperation only happens when the agent is well fed (has surplus);
        competition happens against weaker agents.

        Args:
            other: the neighbor agent.

        Returns:
            Dict with the chosen act and energy moved.
        """
        if not other.alive:
            return {"act": "none", "energy_moved": 0.0}

        d = other.distance_to_point(self.position)
        entity = {
            "type": int(other.entity.entity_type),
            "energy": other.energy,
            "distance": d,
            "angle": float(np.arctan2(other.position[2] - self.position[2],
                                      other.position[0] - self.position[0])),
            "group_id": other.group_id,
            "id": other.entity.id,
            "message": self._inbox.get(other.entity.id, 0.0),
        }
        entity_input = self._entity_features(entity)

        out = self.perceptron_entity.forward(entity_input)
        compete = float(out[0])
        cooperate = float(out[1])

        if cooperate >= self.params.cooperate_threshold and self.energy > self.params.start_energy:
            moved = self.share_energy(other)
            return {"act": "cooperate", "energy_moved": moved}
        if compete >= self.params.contest_threshold and other.energy < self.energy:
            moved = self.contest_energy(other)
            return {"act": "contest", "energy_moved": moved}
        return {"act": "none", "energy_moved": 0.0}

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
            "memory": self.memory.stats(),
            "learning": self._last_learning,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the baby to a JSON-safe dict.

        Captures every field that affects future ticks: the entity, all three
        perceptrons, episodic memory (exact ring-buffer state), and the total
        tick counter. Transient perception state (``_last_*``) is recomputed on
        the next tick and is not stored.

        Returns:
            Dict with entity, perceptron weights, memory, and total_ticks.
        """
        return {
            "entity": self.entity.to_dict(),
            "perceptron_cells": self.perceptron_cells.to_dict(),
            "perceptron_body": self.perceptron_body.to_dict(),
            "perceptron_entity": self.perceptron_entity.to_dict(),
            "perceptron_move": self.perceptron_move.to_dict(),
            "perceptron_message": self.perceptron_message.to_dict()
            if self.perceptron_message is not None else None,
            "memory": self.memory.to_dict(),
            "total_ticks": int(self._total_ticks),
            "group_id": int(self.group_id),
            "learning": {
                k: float(v) for k, v in (self._last_learning or {}).items()
            } or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any],
                  params: WorldParams | None = None,
                  entity: Entity | None = None) -> "SimBaby":
        """
        Rebuild a baby from :meth:`to_dict` output without touching RNG.

        Args:
            data: serialized baby dict.
            params: the world parameters the baby was born under (restored
                scenes pass the same params object for all babies).
            entity: the matching Entity instance; when omitted one is built
                from the stored entity dict.

        Returns:
            A new SimBaby with the stored weights, memory, and counters.
        """
        baby = object.__new__(cls)
        baby.params = params or WorldParams()
        baby.entity = entity or Entity.from_dict(data["entity"])
        baby.perceptron_cells = Perceptron.from_dict(data["perceptron_cells"])
        baby.perceptron_body = Perceptron.from_dict(data["perceptron_body"])
        baby.perceptron_entity = Perceptron.from_dict(data["perceptron_entity"])
        move_data = data.get("perceptron_move")
        if move_data and move_data.get("W"):
            baby.perceptron_move = Perceptron.from_dict(move_data)
        else:
            # Legacy payloads predate movement — mint a fresh move brain
            # (touches RNG only on the legacy path).
            baby.perceptron_move = Perceptron(
                baby.params.cells_input_dim, 3,
                hidden_units=baby.params.brain_hidden_units,
            )
        baby.memory = EpisodicMemory.from_dict(data["memory"])
        baby._total_ticks = int(data["total_ticks"])
        baby.group_id = int(data.get("group_id", 0))
        baby._last_learning = data.get("learning")
        message_data = data.get("perceptron_message")
        if message_data and message_data.get("W"):
            baby.perceptron_message = Perceptron.from_dict(message_data)
        else:
            baby.perceptron_message = None
        baby._inbox = {}
        baby._last_message_input = None
        baby._last_message_out = None
        baby._last_perception = None
        baby._last_action = None
        baby._last_features = None
        baby._last_gates = None
        baby._last_move = None
        baby._previous_energy = baby.entity.energy
        return baby


# ── SimScene ─────────────────────────────────────────────────────────────────

class SimScene:
    """The virtual world — grid, entities, babies."""

    def __init__(self, params: WorldParams | None = None):
        self.params = params or WorldParams()
        self.world = WorldGrid(self.params.grid_size)
        if self.params.generate_world:
            generate_world(self.world, self.params)
        self.entities: list[Entity] = []
        self._next_entity_id = 1
        self._devices: list[SimBaby] = []
        self._tick = 0
        self._cell_update_fn: Callable | None = None
        # Directed-message bus: (sender_id, target_id, amplitude) tuples posted
        # during the current tick, routed into targets' inboxes at the start of
        # the next tick so delivery is order-independent and one tick latent.
        self._pending_messages: list[tuple[int, int, float]] = []

    def add_baby(self, baby: SimBaby):
        """Add a baby to the world."""
        self._devices.append(baby)
        self.entities.append(baby.entity)

    def _surface_y(self, x: int, z: int) -> int:
        """Lowest y whose cell is not air — the ground a baby can stand on."""
        for y in range(self.world.ny):
            if self.world.material[self.world.idx(x, y, z)] != MATERIAL_AIR:
                return y
        return max(self.world.ny - 1, 0)

    def spawn_babies(self, count: int | None = None):
        """
        Spawn babies at random positions.

        In a generated world they land on the ground surface (first non-air
        cell) at a random x/z; in an empty world they drop at a random cell.
        """
        n = count or self.params.start_agents
        nx, ny, nz = self.params.grid_size
        for _ in range(n):
            if self.params.generate_world:
                x = int(np.random.randint(0, nx))
                z = int(np.random.randint(0, nz))
                y = self._surface_y(x, z)
                position = np.array([x + 0.5, y, z + 0.5], dtype=np.float64)
            else:
                position = np.array([
                    int(np.random.randint(0, max(nx, 1))),
                    int(np.random.randint(0, max(ny, 1))),
                    int(np.random.randint(0, max(nz, 1))),
                ], dtype=np.float64)
            baby = SimBaby(
                position=position,
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
    def babies(self) -> list[SimBaby]:
        """All babies in the world (alive or not)."""
        return list(self._devices)

    @property
    def alive_babies(self) -> list[SimBaby]:
        return [d for d in self._devices if d.alive]

    def nearby_babies(self, position: np.ndarray, radius: float,
                      exclude_id: int | None = None) -> list[SimBaby]:
        """
        Find alive babies within radius of a point.

        Args:
            position: query point.
            radius: max distance to include.
            exclude_id: optional baby id to skip (self).

        Returns:
            List of alive SimBaby instances, sorted by distance.
        """
        out = []
        for d in self._devices:
            if not d.alive:
                continue
            if exclude_id is not None and d.entity.id == exclude_id:
                continue
            if d.distance_to_point(position) <= radius:
                out.append(d)
        out.sort(key=lambda b: b.distance_to_point(position))
        return out

    def deliver_messages(self) -> None:
        """
        Route the pending-message bus into targets' inboxes.

        Called once per tick before perception. Every baby's inbox is cleared
        first so a message is perceived exactly during the tick after it was
        sent. Messages addressed to a dead baby are dropped; if a sender
        posted several messages to the same target in one tick only the
        strongest amplitude is kept.

        Side effects:
            - clears every baby's ``_inbox``
            - fills targets' ``_inbox`` from ``_pending_messages``
            - empties ``_pending_messages``
        """
        for b in self._devices:
            b._inbox.clear()
        pending = self._pending_messages
        self._pending_messages = []
        for sender_id, target_id, amplitude in pending:
            tgt = self.get_baby(target_id)
            if tgt is None or not tgt.alive:
                continue
            tgt._inbox[sender_id] = max(tgt._inbox.get(sender_id, 0.0), amplitude)

    def info(self) -> dict:
        return {
            "tick": self._tick,
            "grid_size": self.params.grid_size,
            "total_energy": self.world.total_energy,
            "total_signal": self.world.total_signal,
            "entities": len(self.entities),
            "alive_babies": len(self.alive_babies),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the entire scene to a JSON-safe dict — a snapshot.

        Captures the params, grid, all entities (baby and non-baby), every
        baby's weights/memory/counters, the tick counter, and the next entity
        id so restored scenes keep generating collision-free ids. The cell
        update function is not serializable (it is a callable) and is reset to
        the default on restore.

        Returns:
            Dict that can be passed to :meth:`from_dict` or JSON-dumped.
        """
        return {
            "version": 1,
            "params": asdict(self.params),
            "tick": self._tick,
            "next_entity_id": self._next_entity_id,
            "world": self.world.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "babies": [b.to_dict() for b in self._devices],
            "pending_messages": [list(m) for m in self._pending_messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimScene":
        """
        Rebuild a scene from :meth:`to_dict` output.

        Babies are reattached to their original Entity objects so the
        ``entities`` list and ``_devices`` list share identity, exactly as a
        live scene does. The global baby id counter is advanced past the
        largest restored id so future spawns do not collide.

        Args:
            data: serialized scene dict.

        Returns:
            A new SimScene equivalent to the saved one.
        """
        params = WorldParams(**data["params"])
        world = WorldGrid.from_dict(data["world"])
        entities = [Entity.from_dict(e) for e in data["entities"]]
        by_id = {e.id: e for e in entities}
        babies = [
            SimBaby.from_dict(b, params, entity=by_id.get(b["entity"]["id"]))
            for b in data["babies"]
        ]
        scene = cls(params)
        scene.world = world
        scene.entities = entities
        scene._devices = babies
        scene._tick = int(data["tick"])
        scene._next_entity_id = int(data["next_entity_id"])
        scene._pending_messages = [
            tuple(m) for m in data.get("pending_messages", [])
        ]
        max_id = max((e.id for e in entities), default=0)
        if SimBaby._next_id <= max_id:
            SimBaby._next_id = max_id + 1
        return scene


# ── Simulation ───────────────────────────────────────────────────────────────

class Simulation:
    """
    Tick-based simulation loop.

    Each tick:
        1. Deliver last tick's directed messages (inbox fill)
        2. World compute (diffusion, waves, conservation)
        3. For each alive baby:
           a. Perceive (read cells, entities, body)
           b. Feel (energy change)
           c. React (write cells)
           d. Apply action (write to world)
           e. Learn (update perceptron weights)
           f. Directed message (signal a specific neighbor, delivered next tick)
           g. Social step (cooperate or contest against nearest neighbor)
           h. Passive drain + perception cost
        4. Remove dead babies
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

        # Deliver last tick's directed messages before anyone perceives, so a
        # message is readable exactly one tick after it was sent.
        if self.scene.params.message_enabled:
            self.scene.deliver_messages()

        results = []

        # World compute first
        self.scene.update_cells()

        alive = list(self.scene.alive_babies)

        for baby in alive:
            if not baby.alive:
                continue

            t0 = time.time()
            prev_energy = baby.energy

            # 1. Perceive (cells + nearby agents)
            perception = baby.perceive(self.scene.world, babies=alive)
            baby.entity.energy -= self.scene.params.see_cost

            # 2. Feel
            energy_delta = baby.feel(prev_energy)

            # 3. React
            action = baby.react(perception, energy_delta)

            # 4. Apply action — the baby funds the cells it writes, so the
            #    world conserves energy (deposits are a transfer, not creation)
            cells_written = baby.apply_action(action, self.scene.world)
            deposited = float(sum(w.energy for w in action.writes))
            baby.entity.energy -= (
                self.scene.params.write_cost * cells_written + deposited
            )

            # 4b. Move — a born ability to relocate one grid step per tick.
            #     Charged move_cost; the cost shows up in the next tick's
            #     energy_delta, which shapes the movement perceptron.
            moved = False
            direction = baby.decide_move(perception)
            if direction is not None:
                step_v = np.round(direction).astype(np.int64)
                if np.any(step_v != 0):
                    grid = np.asarray(self.scene.params.grid_size, dtype=np.float64)
                    new_pos = (baby.position + step_v) % grid
                    baby.entity.position = new_pos.astype(np.float64)
                    baby.entity.energy -= self.scene.params.move_cost
                    moved = True

            # 4c. Directed message — signal a specific neighbor. The message
            #     perceptron decides whether to address the neediest nearby
            #     baby within range; the gate value becomes the amplitude, and
            #     the cost scales with it (a louder signal costs more). The
            #     message is delivered to the target's inbox at the start of
            #     the next tick, so the recipient perceives it as the extra
            #     entity feature at index 5 (visible to dim-6 brains).
            message_energy = 0.0
            message_amplitude = 0.0
            if self.scene.params.message_enabled:
                msg_neighbors = self.scene.nearby_babies(
                    baby.position, self.scene.params.message_range,
                    exclude_id=baby.entity.id,
                )
                baby._last_message_input = None
                baby._last_message_out = None
                if msg_neighbors:
                    msg_target = min(msg_neighbors, key=lambda n: n.energy)
                    message_amplitude = baby.decide_message(msg_target)
                    if message_amplitude > 0.0:
                        self.scene._pending_messages.append(
                            (baby.entity.id, msg_target.entity.id, float(message_amplitude))
                        )
                        message_energy = (
                            self.scene.params.message_cost * message_amplitude
                        )
                        baby.entity.energy -= message_energy

            # 5. Social step — cooperate or contest the neediest nearby baby.
            #    Cooperation targets the hungriest neighbor: a gift to a
            #    starving tribe-mate raises the tribe's geometric-mean fitness
            #    far more than one to a well-fed member. Contest likewise
            #    presses the weakest. The kin signal in the entity input lets
            #    the perceptron learn whom to help and whom to rob.
            #    ``social_enabled=False`` (used by the single-population
            #    emergence proof) isolates individual foraging from the
            #    inter-agent stealing that a contest-based selection would
            #    otherwise reward.
            social_act = "none"
            social_energy = 0.0
            if self.scene.params.social_enabled:
                neighbors = self.scene.nearby_babies(
                    baby.position, self.scene.params.social_radius,
                    exclude_id=baby.entity.id,
                )
                if neighbors:
                    target = min(neighbors, key=lambda n: n.energy)
                    social = baby.social_step(target)
                    social_act = social["act"]
                    social_energy = social["energy_moved"]

            # 6. Learn
            baby.learn(energy_delta)

            # 7. Absorb energy from nearby organic material
            absorbed = baby.absorb_energy(self.scene.world)
            baby.entity.energy += absorbed

            # 8. Passive drain
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
                "moved": moved,
                "message_amplitude": message_amplitude,
                "message_energy": message_energy,
                "absorbed": absorbed,
                "social_act": social_act,
                "social_energy": social_energy,
                "total_ms": elapsed,
                "alive": baby.alive,
            }
            results.append(result)

            if self.verbose:
                logger.info(
                    f"tick={self.scene.tick} baby={baby.entity.id} "
                    f"energy={baby.energy:.1f} "
                    f"delta={energy_delta:+.1f} "
                    f"wrote={cells_written} "
                    f"social={social_act}({social_energy:+.1f})"
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
            "cooperations": sum(1 for r in self._tick_log if r["social_act"] == "cooperate"),
            "contests": sum(1 for r in self._tick_log if r["social_act"] == "contest"),
            "social_energy_moved": sum(r["social_energy"] for r in self._tick_log),
        }
