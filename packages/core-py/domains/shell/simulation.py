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

from .memory import EpisodicMemory, WorldMemory

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

    # Durable structures (Stage 7). Opt-in: off by default so the locked
    # single/group selection proofs keep their exact energy flow and genome
    # layout. When on, cell-write deposits near a nest feed its bank (a
    # transfer, never creation), enough deposited energy seeds a new nest at
    # the written cell, and a baby under its start energy can draw from its
    # own tribe's nearest nest — a starvation buffer that makes territoriality
    # and resource pooling worth evolving.
    structure_enabled: bool = False
    nest_radius: float = 2.0        # a write within this distance of a nest feeds it
    nest_seed_energy: float = 3.0   # deposited energy needed to seed a new nest
    nest_draw_rate: float = 1.0     # max energy drawn per tick from one nest
    nest_use_radius: float = 2.0    # a baby must stand this close to draw
    nest_decay: float = 0.002       # fraction of stored energy lost per tick
    max_nests: int = 8              # world-wide nest cap (limited territory)

    # Cultural transmission (Stage 7). Opt-in: off by default so the locked
    # single/group selection proofs keep their exact genome layout and RNG
    # draw order. When on, each baby gains a ``perceptron_teach``
    # (entity-input -> 1 gate) that decides whether to teach the neediest
    # nearby baby: the teacher pays ``teach_cost * gate`` energy and blends
    # the student's behavior perceptrons toward its own learned weights (a
    # louder lesson has a stronger effect) — learned behavior moves laterally
    # between living agents, not just vertically at birth (memotype) or via
    # selection. Episode transfer is limited (``teach_memotype_cap=1``): bulk
    # episode copies raised the student's reward baseline and dampened its own
    # subsequent learning, making culture net-negative in the honest-reward
    # world (benchmark_culture). ``0`` disables episode transfer entirely.
    teaching_enabled: bool = False
    teach_cost: float = 0.5          # energy spent per unit of lesson amplitude
    teach_range: float = 5.0         # max distance for a lesson
    teach_gate_threshold: float = 0.5  # perceptron gate must clear to teach
    teach_weight_blend: float = 0.1  # fraction of the weight gap closed per lesson
    teach_memotype_cap: int = 1      # best episodes copied per lesson (0 = none)

    # World-level long-term memory (Stage 7). Opt-in: off by default so the
    # locked selection proofs keep their exact energy flow and genome layout.
    # When on, the scene carries a WorldMemory reservoir that never evicts:
    # a dying baby deposits its best episodes (``memory_deposit``), the
    # evolution engine deposits every survivor at generation boundaries, and
    # a newborn is seeded (``memory_seed``) from the reservoir's best episodes
    # — so lived experience survives death and crosses lineages, beyond the
    # capped parent->child memotype.
    memory_enabled: bool = False
    memory_deposit: int = 8    # episodes a baby deposits into the world reservoir
    memory_seed: int = 4       # episodes a newborn is seeded with from the reservoir

    # Predator-prey dynamics (Stage 8). Opt-in: off by default so the locked
    # selection proofs keep their exact genome layout and RNG draw order.
    # When on, each baby gains a ``perceptron_predation`` (entity-input -> 1
    # gate) that decides whether to hunt the weakest nearby baby within range.
    # A strike is lethal: the prey's full energy transfers to the predator (a
    # transfer, not creation — the world still conserves energy) and the prey
    # dies. The predator pays ``predation_cost`` for the strike, so the gate
    # is shaped by the honest same-tick net reward: hunting pays while prey
    # energy exceeds the strike cost, and self-limits as prey grows scarce
    # (a lone predator that eats its own population starves with it).
    predation_enabled: bool = False
    predation_cost: float = 0.5       # energy spent to execute a strike
    predation_range: float = 3.0      # max distance a predator can strike
    predation_gate_threshold: float = 0.5  # perceptron gate must clear to hunt

    # Territoriality (Stage 9). Opt-in: off by default so the locked
    # selection proofs keep their exact genome layout and RNG draw order.
    # Territory is CLAIMED by building nests (Stage 7): a tribe's region is
    # the ground within ``territory_radius`` of the nearest nest it owns
    # (world-wide cap ``max_nests``). When the channel is on, the region
    # becomes a two-sided resource: a hungry baby standing on foreign ground
    # (within ``territory_radius`` of a foreign tribe's nearest nest) can RAID
    # that bank — a one-way drain, ``nest_draw_rate`` per tick, that is the
    # value defending protects — and each baby gains a ``perceptron_territory``
    # (entity-input -> 1 gate) that decides whether to DEFEND: standing on its
    # own tribe's territory, a cleared gate evicts the nearest foreign baby
    # within ``defend_range``. The toll scales with the trespasser's own
    # energy — ``defend_take_fraction`` of it transfers to the defender
    # (capped so the eviction is never lethal), so evicting a rich trespasser
    # pays and the gate has an honest gradient to learn "this trespasser is
    # worth evicting". The trespasser is softly pushed ``defend_push`` cells
    # away from the defender (a pure relocation, never a kill — a small shove,
    # not a stranding), and the defender pays ``defend_cost`` (a transfer, not
    # creation — the world conserves energy). The gate is shaped by the honest
    # same-tick net reward: defending pays while a trespasser carries more
    # energy than the eviction costs and self-limits as trespassers grow
    # scarce.
    territoriality_enabled: bool = False
    territory_radius: float = 3.0       # a tribe's region = within this of its nearest nest
    defend_range: float = 3.0           # max distance a defender can evict a trespasser
    defend_cost: float = 0.5            # energy spent to execute an eviction
    defend_take_fraction: float = 0.5   # share of the trespasser's energy taken as toll
    defend_push: float = 1.0            # cells the evicted trespasser is shoved away
    defend_gate_threshold: float = 0.5  # perceptron gate must clear to defend

    # In-world life cycle (Stage 10). Opt-in: off by default so the locked
    # selection proofs keep their exact genome layout and RNG draw order.
    # When on, each baby gains a ``perceptron_reproduce`` (body-input -> 1
    # gate) that decides whether it is ready to breed. A baby whose gate
    # clears while it stands above ``reproduce_energy_threshold`` spawns an
    # offspring near itself: the child's starting energy is ``birth_cost`` —
    # up to ``birth_nest_fraction`` of it drawn from the tribe's nearest
    # nest bank (the tribe funds the child's start), the rest from the
    # parent — a pure transfer, never creation, so the world conserves
    # energy. The child inherits the parent's learned behavior weights and
    # best episodes (an asexual offspring of the living lineage), joins the
    # parent's tribe, and is seeded from the world reservoir like any
    # newborn. Starvation (energy <= 0) still removes babies every tick, so
    # births and deaths now happen INSIDE the tick loop and a scene's
    # population can self-sustain without the evolution engine re-seeding
    # it. Population is bounded by ``max_entities`` and, ultimately, by the
    # world's conserved energy budget — a birth is a transfer, never
    # creation, so energy is the carrying capacity.
    lifecycle_enabled: bool = False
    reproduce_gate_threshold: float = 0.5   # perceptron gate must clear to breed
    reproduce_energy_threshold: float = 150.0  # parent must exceed this energy to breed
    birth_cost: float = 50.0      # total energy transferred to the offspring at birth
    birth_nest_fraction: float = 0.5  # share of birth_cost drawn from the tribe's nest bank
    birth_range: float = 2.0      # max distance the offspring is placed from the parent

    # Division of labor (Stage 11). Opt-in: off by default so the locked
    # selection proofs keep their exact genome layout and RNG draw order.
    # When on, each baby gains a ``perceptron_role`` (body-input -> 1 gate)
    # whose value is a heritable POSTURE. Below ``role_gate_threshold`` the
    # baby is a BUILDER: while it carries genuine surplus (above
    # ``start_energy``) and stands within ``nest_use_radius`` of its tribe's
    # nearest nest, it banks ``role_deposit_fraction`` of that surplus into
    # the bank (a deliberate transfer, replacing the noisy cell-write
    # deposits — a builder lifts its tribe's famine floor). At or above the
    # threshold the baby is a WARRIOR: standing within ``territory_radius``
    # of a FOREIGN tribe's nearest nest it raids that bank even when not
    # hungry, capped at ``role_raid_fraction`` of ``nest_draw_rate`` and the
    # bank (a warrior lifts its tribe's mean). Both acts land in the same
    # tick's honest net reward (step 8b), so the posture is selected for only
    # where it pays, and geometric-mean group selection rewards tribes that
    # field both postures.
    specialization_enabled: bool = False
    role_gate_threshold: float = 0.5  # gate < threshold = Builder, >= = Warrior
    role_deposit_fraction: float = 0.1  # share of surplus a Builder banks per tick
    role_raid_fraction: float = 0.5     # share of nest_draw_rate a Warrior may raid

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
    cells_input_dim: int = 5  # material, energy, temperature, occupancy, signal (+ daylight at index 5 when >= 6)
    body_input_dim: int = 3   # energy, position x, position y
    entity_input_dim: int = 5  # type, energy, distance, angle, kin signal (+ directed-message amplitude at index 5 when >= 6)

    # Solar energy cycle (Stage 13) — energy enters the world ONLY from the
    # boundary (the sky), along a diurnal curve. When off the world stays a
    # closed system, so the locked selection proofs remain bit-identical.
    solar_enabled: bool = False
    solar_day_ticks: int = 24          # full day/night cycle length in ticks
    solar_phase: int = 0               # tick offset (0 = sunrise)
    solar_min_intensity: float = 0.0   # night light level (0 = dark)
    solar_max_intensity: float = 1.0   # noon light level
    solar_deposit_rate: float = 0.4    # energy per lit surface cell per tick at full sun

    # Seasonal year envelope (Stage 14) — the diurnal curve rides inside a
    # slower cosine year. When off (``solar_season_ticks == 0``) the world is
    # exactly the Stage 13 diurnal world, so the locked selection proofs stay
    # bit-identical. The envelope is a pure deterministic function of tick —
    # it consumes no RNG.
    solar_season_ticks: int = 0        # full year length in ticks (0 = no seasons)
    solar_seasonality: float = 1.0     # 0 = flat (always the diurnal mean), 1 = full swing

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
        self.light = 0.0  # global daylight intensity this tick (0 = night)

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


# ── Structures ───────────────────────────────────────────────────────────────

@dataclass
class Nest:
    """
    A durable structure — a bank of stored energy anchored to a grid cell.

    Seeded by a baby's cell-write that carries enough deposited energy; fed
    by later writes within ``nest_radius`` of it. Any tribe-mate that is
    under its start energy can draw from the nearest same-group nest, so a
    nest is a starvation buffer that makes territoriality and resource
    pooling worth evolving. Stored energy decays a little every tick, so an
    abandoned structure erodes to nothing.
    """
    id: int
    position: np.ndarray
    stored_energy: float
    owner_group_id: int
    alive: bool = True

    def distance_to_point(self, point: np.ndarray) -> float:
        """Euclidean distance to a point in world coordinates."""
        return float(np.linalg.norm(self.position - point))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the nest to a JSON-safe dict.

        Returns:
            Dict with id, position, stored_energy, owner_group_id, alive.
        """
        return {
            "id": int(self.id),
            "position": self.position.tolist(),
            "stored_energy": float(self.stored_energy),
            "owner_group_id": int(self.owner_group_id),
            "alive": bool(self.alive),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Nest":
        """
        Rebuild a nest from :meth:`to_dict` output.

        Args:
            data: serialized nest dict.

        Returns:
            A new Nest with the stored state.
        """
        nest = cls.__new__(cls)
        nest.id = int(data["id"])
        nest.position = np.asarray(data["position"], dtype=np.float64)
        nest.stored_energy = float(data["stored_energy"])
        nest.owner_group_id = int(data["owner_group_id"])
        nest.alive = bool(data["alive"])
        return nest


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
            error: scalar or vector reward signal (positive reinforces,
            negative weakens; the sign is the direction, not a learning-rate
            flag).
            lr: learning rate (always positive).
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
        self.perceptron_teach: Perceptron | None = None
        if self.params.teaching_enabled:
            # Cultural brain: reads a target's entity features and emits a
            # single gate whose value is the lesson amplitude. Constructed
            # from fixed zeros (no RNG draw) so the four behavior brains'
            # RNG draw order — and the whole perception-noise stream — is
            # unchanged whether or not teaching is enabled (locked proofs and
            # a perfectly controlled culture benchmark). The genome's
            # ``apply_to`` overwrites these weights at birth anyway.
            p = object.__new__(Perceptron)
            p.hidden_units = 0
            p.H = None
            p.bh = None
            p.W = np.zeros((entity_input_dim, 1), dtype=np.float32)
            p.b = np.zeros(1, dtype=np.float32)
            self.perceptron_teach = p

        self.perceptron_predation: Perceptron | None = None
        if self.params.predation_enabled:
            # Predator brain: reads a target's entity features and emits a
            # single gate whose value is the hunt decision strength.
            # Constructed from fixed zeros (no RNG draw) so the behavior
            # brains' RNG draw order — and the whole perception-noise
            # stream — is unchanged whether or not predation is enabled
            # (locked proofs and a controlled predator-prey benchmark). The
            # genome's ``apply_to`` overwrites these weights at birth anyway.
            p = object.__new__(Perceptron)
            p.hidden_units = 0
            p.H = None
            p.bh = None
            p.W = np.zeros((entity_input_dim, 1), dtype=np.float32)
            p.b = np.zeros(1, dtype=np.float32)
            self.perceptron_predation = p

        self.perceptron_territory: Perceptron | None = None
        if self.params.territoriality_enabled:
            # Territory brain: reads a trespasser's entity features and emits
            # a single gate whose value is the defense decision strength.
            # Constructed from fixed zeros (no RNG draw) so the behavior
            # brains' RNG draw order — and the whole perception-noise
            # stream — is unchanged whether or not territoriality is enabled
            # (locked proofs and a controlled territoriality benchmark). The
            # genome's ``apply_to`` overwrites these weights at birth anyway.
            p = object.__new__(Perceptron)
            p.hidden_units = 0
            p.H = None
            p.bh = None
            p.W = np.zeros((entity_input_dim, 1), dtype=np.float32)
            p.b = np.zeros(1, dtype=np.float32)
            self.perceptron_territory = p

        self.perceptron_reproduce: Perceptron | None = None
        if self.params.lifecycle_enabled:
            # Reproduction brain: reads the parent's own body state (energy,
            # position) and emits a single gate whose value is the readiness
            # to breed. Constructed from fixed zeros (no RNG draw) so the
            # behavior brains' RNG draw order — and the whole perception-noise
            # stream — is unchanged whether or not lifecycle is enabled
            # (locked proofs and a controlled lifecycle benchmark). The
            # genome's ``apply_to`` overwrites these weights at birth anyway.
            p = object.__new__(Perceptron)
            p.hidden_units = 0
            p.H = None
            p.bh = None
            p.W = np.zeros((body_input_dim, 1), dtype=np.float32)
            p.b = np.zeros(1, dtype=np.float32)
            self.perceptron_reproduce = p

        self.perceptron_role: Perceptron | None = None
        if self.params.specialization_enabled:
            # Role brain: reads the baby's own body state (energy, position)
            # and emits a single gate whose value is its posture — Builder
            # below ``role_gate_threshold``, Warrior at or above it.
            # Constructed from fixed zeros (no RNG draw) so the behavior
            # brains' RNG draw order — and the whole perception-noise stream
            # — is unchanged whether or not specialization is enabled (locked
            # proofs and a controlled division-of-labor benchmark). The
            # genome's ``apply_to`` overwrites these weights at birth anyway.
            p = object.__new__(Perceptron)
            p.hidden_units = 0
            p.H = None
            p.bh = None
            p.W = np.zeros((body_input_dim, 1), dtype=np.float32)
            p.b = np.zeros(1, dtype=np.float32)
            self.perceptron_role = p

        # Inbox — messages delivered to this baby this tick (sender_id -> amplitude).
        # Filled by the scene's delivery pass at the start of each tick.
        self._inbox: dict[int, float] = {}
        self._last_message_input: np.ndarray | None = None
        self._last_message_out: float | None = None
        self._last_teach_input: np.ndarray | None = None
        self._last_teach_out: float | None = None
        self._last_predation_input: np.ndarray | None = None
        self._last_predation_out: float | None = None
        self._last_defend_input: np.ndarray | None = None
        self._last_defend_out: float | None = None
        self._last_reproduce_input: np.ndarray | None = None
        self._last_reproduce_out: float | None = None
        self._last_role_input: np.ndarray | None = None
        self._last_role_out: float | None = None

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

    def perceive(self, world: WorldGrid, babies: list[SimBaby] | None = None,
                 nests: list[Nest] | None = None) -> Perception:
        """
        Read nearby cells, detect entities and structures, read own body.
        Energy cost is deducted.

        Args:
            world: the world grid.
            babies: optional list of other agents to detect as entities.
            nests: optional list of durable structures to detect (visible only
                when ``structure_enabled``).

        Returns:
            Perception with nearby_cells, nearby_entities, agent_body.
        """
        t0 = time.time()
        gx, gy, gz = int(self.position[0]), int(self.position[1]), int(self.position[2])

        cells = world.get_nearby_cells(gx, gy, gz, self.params.see_radius)

        # Stage 13: when the sun is on, the world carries a global daylight
        # level that dim-6+ brains can read as the extra cells feature.
        if self.params.solar_enabled:
            cells["light"] = np.array([world.light], dtype=np.float32)

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

        # Detect durable structures (Stage 7). Nests appear as objects keyed
        # by their stored energy, so the entity brain can learn to approach a
        # full bank, linger near its own tribe's, and avoid hostile territory.
        if nests is not None and self.params.structure_enabled:
            for nest in nests:
                if not nest.alive:
                    continue
                d = nest.distance_to_point(self.position)
                if d <= self.params.see_radius:
                    entities.append({
                        "id": nest.id,
                        "type": int(EntityType.OBJECT),
                        "energy": nest.stored_energy,
                        "distance": d,
                        "angle": float(np.arctan2(
                            nest.position[2] - self.position[2],
                            nest.position[0] - self.position[0])),
                        "group_id": nest.owner_group_id,
                        "message": 0.0,
                        "is_nest": True,
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
            min(float(np.mean(cells.get("light", [0.0]))), 1.0),  # daylight
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

        # lr is always positive: the error sign (from energy_delta) encodes
        # direction — gains push outputs up, losses push them down. A negative
        # lr on losses used to double-flip the delta-rule sign (error * lr
        # becomes positive), so "weaken" actually reinforced the bad outcome
        # and every loss drove the perceptrons toward saturation.
        lr = 0.01 * scale

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

            # Learn teaching from the last lesson decision (if any)
            if self._last_teach_input is not None and self.perceptron_teach is not None:
                self.perceptron_teach.update(
                    self._last_teach_input,
                    np.array([error]),
                    lr=lr * 0.5,
                )
                self._last_teach_input = None
                self._last_teach_out = None

            # Learn predation from the last hunt decision (if any)
            if (self._last_predation_input is not None
                    and self.perceptron_predation is not None):
                self.perceptron_predation.update(
                    self._last_predation_input,
                    np.array([error]),
                    lr=lr * 0.5,
                )
                self._last_predation_input = None
                self._last_predation_out = None

            # Learn territoriality from the last defense decision (if any)
            if (self._last_defend_input is not None
                    and self.perceptron_territory is not None):
                self.perceptron_territory.update(
                    self._last_defend_input,
                    np.array([error]),
                    lr=lr * 0.5,
                )
                self._last_defend_input = None
                self._last_defend_out = None

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

    def decide_teach(self, other: SimBaby) -> float:
        """
        Decide whether to teach a specific neighbor (cultural act).

        Mirrors ``decide_message``: the teach perceptron reads the target's
        entity features (so a lesson can be keyed to need, kin, or wealth)
        and emits one sigmoid gate. When the gate clears
        ``teach_gate_threshold`` a lesson is given with the gate value as its
        amplitude — a louder lesson transfers more behavior and costs more
        energy. The simulation loop picks the neediest nearby tribe-mate as
        the target (cultural transmission is in-group) and charges the cost
        immediately. Teaching cost is a real energy outlay and lands in the
        teacher's same-tick net reward (step 8b), so the teach perceptron is
        shaped by the full honest outcome of the tick.

        Args:
            other: the intended student.

        Returns:
            Lesson amplitude in [0, 1]; 0.0 means no lesson is given.
        """
        if (self.perceptron_teach is None or not other.alive
                or self.energy <= self.params.start_energy):
            # Teaching is only worth it when the teacher has surplus energy
            # (mirrors cooperation's surplus condition) — a starving teacher
            # that spends energy on a lesson is pure waste.
            self._last_teach_input = None
            self._last_teach_out = None
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
        out = self.perceptron_teach.forward(features)
        gate = float(out[0])
        self._last_teach_input = features
        self._last_teach_out = gate
        if gate >= self.params.teach_gate_threshold:
            return gate
        return 0.0

    def teach(self, other: SimBaby, amplitude: float) -> int:
        """
        Transfer learned behavior to a neighbor — cultural transmission.

        The student's behavior perceptrons (cells, body, entity, move) blend
        toward the teacher's learned weights by ``teach_weight_blend *
        amplitude`` — a louder lesson closes a larger fraction of the weight
        gap. The teacher's highest-reward episodes are then copied into the
        student's episodic memory, so lived experience moves laterally between
        living agents (the memotype still only moves vertically at birth).

        Args:
            other: the student agent (weights and memory modified in place).
            amplitude: lesson amplitude in (0, 1] from the teach gate.

        Returns:
            Number of episodes copied into the student's memory.

        Side effects:
            - modifies ``other``'s behavior perceptron weights and memory.
        """
        if not other.alive or amplitude <= 0.0:
            return 0
        blend = self.params.teach_weight_blend * min(float(amplitude), 1.0)
        for name in ("cells", "body", "entity", "move"):
            t = getattr(self, f"perceptron_{name}")
            s = getattr(other, f"perceptron_{name}")
            s.W[:] = (s.W + blend * (t.W - s.W)).astype(np.float32)
            s.b[:] = (s.b + blend * (t.b - s.b)).astype(np.float32)
            if t.H is not None and s.H is not None:
                s.H[:] = (s.H + blend * (t.H - s.H)).astype(np.float32)
                s.bh[:] = (s.bh + blend * (t.bh - s.bh)).astype(np.float32)
        cap = int(self.params.teach_memotype_cap)
        copied = 0
        for e in self.memory.recall(cap, by_reward=True):
            other.memory.record(
                features=np.asarray(e.features, dtype=np.float32),
                action=tuple(e.action),
                reward=float(e.reward),
                tick=int(e.tick),
            )
            copied += 1
        return copied

    def decide_predation(self, other: SimBaby) -> float:
        """
        Decide whether to hunt a specific neighbor (predator-prey act).

        Mirrors ``decide_message``: the predation perceptron reads the
        target's entity features (so a hunt can be keyed to need, kin, or
        wealth — the kin signal lets selection learn not to eat its own
        tribe) and emits one sigmoid gate. When the gate clears
        ``predation_gate_threshold`` a lethal strike is executed by the
        simulation loop: the prey's full energy transfers to the predator
        and the prey dies. The gate value itself is not scaled into energy —
        the strike is all-or-nothing — but it is recorded so the delta-rule
        learner can shape the gate from the honest same-tick net reward
        (strike gain minus ``predation_cost``).

        Args:
            other: the intended prey.

        Returns:
            Hunt decision strength in [0, 1]; 0.0 means no strike.
        """
        if self.perceptron_predation is None or not other.alive:
            self._last_predation_input = None
            self._last_predation_out = None
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
        out = self.perceptron_predation.forward(features)
        gate = float(out[0])
        self._last_predation_input = features
        self._last_predation_out = gate
        if gate >= self.params.predation_gate_threshold:
            return gate
        return 0.0

    def hunt(self, other: SimBaby) -> float:
        """
        Execute a lethal strike: consume a weaker neighbor for energy.

        The prey's full energy transfers to the predator and the prey dies —
        a transfer, not creation, so the world still conserves energy. The
        predator's strike cost (``predation_cost``) is charged by the
        simulation loop after the strike, so it lands in the same tick's
        honest net reward and shapes the predation gate.

        Args:
            other: the prey agent (killed in place).

        Returns:
            Energy gained by the predator (the prey's full energy).

        Side effects:
            - sets ``other``'s energy to 0 and ``other``'s alive flag to False
            - adds the prey's energy to ``self``'s energy
        """
        if not other.alive or self.perceptron_predation is None:
            return 0.0
        gained = float(other.entity.energy)
        other.entity.energy = 0.0
        other.entity.alive = False
        self.entity.energy += gained
        return gained

    def decide_defend(self, other: SimBaby) -> float:
        """
        Decide whether to evict a trespasser from the tribe's territory.

        The territory perceptron reads the trespasser's entity features and
        emits a single gate. When the gate clears ``defend_gate_threshold``
        an eviction is executed by the simulation loop: the trespasser is
        shoved ``defend_push`` cells away and ``defend_take_fraction`` of its
        energy transfers to the defender, who pays ``defend_cost``. The
        eviction also keeps a rival raider off the tribe's nest bank (step
        7b), so a cleared gate protects real shared energy. The gate value
        itself is not scaled into energy — the eviction is all-or-nothing —
        but it is recorded so the delta-rule learner can shape the gate from
        the honest same-tick net reward (toll gained minus ``defend_cost``).
        Because the toll scales with the trespasser's own energy, the gate
        reads that feature and learns to evict rich trespassers and leave lean
        ones alone. The simulation loop only offers decisions when the
        defender stands on its own tribe's territory, so the gate learns "this
        trespasser is worth evicting", not "where is my territory".

        Args:
            other: the intended trespasser (a baby of another tribe).

        Returns:
            Defense decision strength in [0, 1]; 0.0 means no eviction.
        """
        if self.perceptron_territory is None or not other.alive:
            self._last_defend_input = None
            self._last_defend_out = None
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
        out = self.perceptron_territory.forward(features)
        gate = float(out[0])
        self._last_defend_input = features
        self._last_defend_out = gate
        if gate >= self.params.defend_gate_threshold:
            return gate
        return 0.0

    def decide_reproduce(self) -> float:
        """
        Decide whether this baby is ready to breed.

        The reproduction perceptron reads the baby's own body state — its
        energy relative to start and its position — and emits one sigmoid
        gate. When the gate clears ``reproduce_gate_threshold`` (and the
        simulation loop has confirmed the baby stands above
        ``reproduce_energy_threshold`` with room in the world) an offspring
        is attempted. Reproduction is a heritable strategy trait, not an
        in-life reflex: like the predation/territory gates it is shaped by
        selection (the genome's dedicated RNG stream), and the offspring's
        survival is its payoff. The immediate energy outlay still lands in
        the parent's honest same-tick net reward (step 8b), so the gate is
        selected for only when breeding is affordable.

        Args:
            None.

        Returns:
            Gate value in [0, 1]; 0.0 means no offspring is attempted.
        """
        if self.perceptron_reproduce is None:
            self._last_reproduce_input = None
            self._last_reproduce_out = None
            return 0.0
        body = np.array([
            self.energy / max(self.params.start_energy, 1.0),
            self.position[0] / max(self.params.grid_size[0], 1.0),
            self.position[1] / max(self.params.grid_size[1], 1.0),
        ], dtype=np.float32)[:self.params.body_input_dim]
        out = self.perceptron_reproduce.forward(body)
        gate = float(out[0])
        self._last_reproduce_input = body
        self._last_reproduce_out = gate
        if gate >= self.params.reproduce_gate_threshold:
            return gate
        return 0.0

    def decide_role(self) -> float:
        """
        Decide this baby's role posture: Builder or Warrior.

        The role perceptron reads the baby's own body state (energy,
        position) and emits one sigmoid gate. Below ``role_gate_threshold``
        the baby is a BUILDER — while it carries genuine surplus and stands
        near its tribe's nearest nest it banks a share of that surplus
        (lifting the tribe's famine floor). At or above the threshold it is
        a WARRIOR — standing on a FOREIGN tribe's ground it raids that bank
        even when it is not hungry (lifting the tribe's mean). Both acts
        land in the same tick's honest net reward (step 8b). The posture is
        a heritable strategy trait, not an in-life reflex: like the
        reproduction gate it is shaped by selection (the genome's dedicated
        RNG stream), and the survival of the baby's own tribe is its payoff
        — geometric-mean group selection rewards tribes that field both
        postures.

        Args:
            None.

        Returns:
            Gate value in [0, 1]; the simulation loop interprets values
            below ``role_gate_threshold`` as Builder and values at or above
            it as Warrior.
        """
        if self.perceptron_role is None:
            self._last_role_input = None
            self._last_role_out = None
            return 0.0
        body = np.array([
            self.energy / max(self.params.start_energy, 1.0),
            self.position[0] / max(self.params.grid_size[0], 1.0),
            self.position[1] / max(self.params.grid_size[1], 1.0),
        ], dtype=np.float32)[:self.params.body_input_dim]
        out = self.perceptron_role.forward(body)
        gate = float(out[0])
        self._last_role_input = body
        self._last_role_out = gate
        return gate

    def spawn_child(self, position: np.ndarray) -> "SimBaby":
        """
        Create an asexual offspring of this living lineage.

        The child copies its parent's behavior perceptron weights (all
        channels the parent carries — the four behavior brains plus any
        opt-in gates) and consolidates the parent's best episodes as its
        memotype, exactly as ``Genome.from_baby`` would at a generation
        boundary. The child joins the parent's tribe and is born carrying
        ``birth_cost`` of energy (the simulation loop deducts that amount
        from the parent and the tribe's nest as a conservation-safe
        transfer).

        Args:
            position: the offspring's spawn point.

        Returns:
            A new SimBaby with the parent's weights, memory, and tribe.
        """
        child = SimBaby(
            position=position,
            initial_energy=self.params.birth_cost,
            params=self.params,
            group_id=self.group_id,
        )
        names = ("cells", "body", "entity", "move", "message", "teach",
                 "predation", "territory", "reproduce", "role")
        for name in names:
            src = getattr(self, f"perceptron_{name}", None)
            dst = getattr(child, f"perceptron_{name}", None)
            if src is None or dst is None:
                continue
            dst.W[:] = src.W.copy()
            dst.b[:] = src.b.copy()
            if src.H is not None:
                dst.H = np.array(src.H, dtype=np.float32).copy()
                dst.bh = np.array(src.bh, dtype=np.float32).copy()
        cap = max(1, int(self.params.memory_inherit))
        for e in self.memory.recall(cap, by_reward=True):
            child.memory.record(e.features, e.action, e.reward, e.tick)
        return child

    def defend(self, other: SimBaby, anchor: np.ndarray) -> float:
        """
        Execute an eviction: shove a trespasser away and take a toll.

        ``defend_take_fraction`` of the trespasser's energy transfers to the
        defender, capped so the eviction is never lethal — it is a toll, not
        a strike, and it scales with how much the trespasser is carrying, so
        evicting a rich foreigner pays. The trespasser is shoved one small
        step (``defend_push`` cells) away from the defender in the ground
        plane (a pure relocation, never a kill, and never a stranding). The
        defender's own ``defend_cost`` is charged by the simulation loop
        after the eviction, so it lands in the same tick's honest net reward
        and shapes the territory gate.

        Args:
            other: the trespassing agent (drained and shoved in place).
            anchor: the defended nest's position (retained for signature
                compatibility; the push no longer targets the territory edge).

        Returns:
            Energy gained by the defender (the toll taken from the trespasser).

        Side effects:
            - reduces ``other``'s energy and relocates ``other`` one cell
            - adds the toll to ``self``'s energy
        """
        if not other.alive or self.perceptron_territory is None:
            return 0.0
        take = min(self.params.defend_take_fraction
                   * float(other.entity.energy),
                   max(float(other.entity.energy) - 1e-6, 0.0))
        if take > 0.0:
            other.entity.energy -= take
            self.entity.energy += take
        other.entity.position = self._push_away(other)
        return take

    def _push_away(self, other: SimBaby) -> np.ndarray:
        """
        Compute the soft shove position for an evicted trespasser.

        The evicted baby is pushed ``defend_push`` cells away from the
        defender along the defender->baby ground-plane direction, keeping its
        height. The world is a torus, so the new point wraps at the grid
        edges. This is a small relocation, never a stranding: the trespasser
        stays close to where it was, just outside arm's reach of the
        defender.

        Args:
            other: the evicted trespasser.

        Returns:
            A new (x, y, z) position one shove away from the defender.
        """
        dx = float(other.position[0] - self.position[0])
        dz = float(other.position[2] - self.position[2])
        d = float(np.hypot(dx, dz))
        if d < 1e-9:
            dx, dz, d = 1.0, 0.0, 1.0
        gx, gy, gz = self.params.grid_size
        sx = (other.position[0] + dx / d * self.params.defend_push) % gx
        sz = (other.position[2] + dz / d * self.params.defend_push) % gz
        return np.array([sx, float(other.position[1]), sz], dtype=np.float64)

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
            "perceptron_teach": self.perceptron_teach.to_dict()
            if self.perceptron_teach is not None else None,
            "perceptron_predation": self.perceptron_predation.to_dict()
            if self.perceptron_predation is not None else None,
            "perceptron_territory": self.perceptron_territory.to_dict()
            if self.perceptron_territory is not None else None,
            "perceptron_reproduce": self.perceptron_reproduce.to_dict()
            if self.perceptron_reproduce is not None else None,
            "perceptron_role": self.perceptron_role.to_dict()
            if self.perceptron_role is not None else None,
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
        teach_data = data.get("perceptron_teach")
        if teach_data and teach_data.get("W"):
            baby.perceptron_teach = Perceptron.from_dict(teach_data)
        else:
            baby.perceptron_teach = None
        pred_data = data.get("perceptron_predation")
        if pred_data and pred_data.get("W"):
            baby.perceptron_predation = Perceptron.from_dict(pred_data)
        else:
            baby.perceptron_predation = None
        terr_data = data.get("perceptron_territory")
        if terr_data and terr_data.get("W"):
            baby.perceptron_territory = Perceptron.from_dict(terr_data)
        else:
            baby.perceptron_territory = None
        repro_data = data.get("perceptron_reproduce")
        if repro_data and repro_data.get("W"):
            baby.perceptron_reproduce = Perceptron.from_dict(repro_data)
        else:
            baby.perceptron_reproduce = None
        role_data = data.get("perceptron_role")
        if role_data and role_data.get("W"):
            baby.perceptron_role = Perceptron.from_dict(role_data)
        else:
            baby.perceptron_role = None
        baby._inbox = {}
        baby._last_message_input = None
        baby._last_message_out = None
        baby._last_teach_input = None
        baby._last_teach_out = None
        baby._last_predation_input = None
        baby._last_predation_out = None
        baby._last_defend_input = None
        baby._last_defend_out = None
        baby._last_reproduce_input = None
        baby._last_reproduce_out = None
        baby._last_role_input = None
        baby._last_role_out = None
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

    def __init__(self, params: WorldParams | None = None,
                 world_memory: WorldMemory | None = None):
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
        # Durable structures (Stage 7) — banked-energy nests rooted at cells.
        self.nests: list[Nest] = []
        self._next_nest_id = 1
        # World-level long-term memory (Stage 7) — the collective reservoir.
        # When the channel is off the scene simply carries no reservoir; an
        # injected one (from the evolution engine) persists across generations.
        self.world_memory: WorldMemory | None = (
            world_memory if world_memory is not None
            else (WorldMemory() if self.params.memory_enabled else None)
        )
        self.memory_seeds_given = 0
        # In-world life cycle (Stage 10) — births granted inside the tick loop.
        self.births = 0
        # Solar boundary source (Stage 13) — cumulative energy the sky has
        # deposited onto the surface, and how many ticks were lit. The
        # conservation tripwire subtracts the boundary deposit so the internal
        # channels must still never create energy.
        self.solar_energy_deposited = 0.0
        self.solar_lit_ticks = 0
        # Stage 14: the current year-envelope state (pure function of tick).
        self.solar_season_index = 0   # which quadrant of the year this tick is in
        self.solar_season_factor = 1.0  # multiplicative daylight envelope (0..1)
        self.solar_year = 0           # how many full years have elapsed

    def add_baby(self, baby: SimBaby):
        """
        Add a baby to the world.

        When the world memory channel is on and the reservoir holds episodes,
        the newborn is seeded with the ``memory_seed`` best episodes — lived
        experience crosses lineages at birth, on top of the memotype that
        ``Genome.apply_to`` records into the same ring buffer beforehand.

        Side effects:
            - appends the baby to ``_devices`` and ``entities``
            - records world-memory episodes into the baby's episodic memory
            - raises ``memory_seeds_given`` by the number of episodes seeded
        """
        if self.world_memory is not None and self.params.memory_seed > 0:
            for e in self.world_memory.recall(self.params.memory_seed,
                                              by_reward=True):
                baby.memory.record(e.features, e.action, e.reward, e.tick)
                self.memory_seeds_given += 1
        self._devices.append(baby)
        self.entities.append(baby.entity)

    def deposit_memory(self, baby: SimBaby) -> int:
        """
        Deposit a baby's best episodes into the world memory reservoir.

        Called on death (in-sim) and on every survivor at a generation
        boundary (evolution engine). Each deposit keeps the ``memory_deposit``
        highest-reward episodes from the baby's ring buffer, stamped with its
        tribe and id, and the reservoir never evicts.

        Args:
            baby: the depositing baby.

        Returns:
            Number of episodes deposited.

        Side effects:
            - appends episodes to ``self.world_memory``
        """
        if self.world_memory is None:
            return 0
        return self.world_memory.consolidate(
            baby.memory, self.params.memory_deposit,
            group_id=baby.group_id, donor_id=baby.entity.id,
        )

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

    def apply_solar(self):
        """
        Deposit energy from the sky along the diurnal curve (Stage 13) inside
        a slower seasonal year envelope (Stage 14).

        The sun is a boundary source: it adds energy ONLY onto the topmost
        non-air cell of each column — the ground the sky can see — scaled by
        the current daylight intensity. A half-wave sinusoid over
        ``solar_day_ticks`` gives full darkness at night, sunrise/noon/sunset
        through the day, and wraps at the horizon. When ``solar_season_ticks``
        is non-zero the daily peak rides a cosine ``solar_seasonality`` year,
        so midsummer noon outshines midwinter noon. Nothing inside the world
        creates energy: grid, babies, and nests still only move it around, so
        the internal conservation invariant holds exactly once each tick's
        boundary deposit is accounted for (the ``_conservation_sweep``
        subtracts it).

        The year envelope is a pure, deterministic function of tick — it
        consumes no RNG, so the same-seed locked selection proofs are
        bit-identical whether or not seasons are on.

        Side effects:
            - raises ``world.light`` to the tick's daylight intensity (0..1)
              modulated by the seasonal envelope
            - records ``solar_season_index`` (0..3 year quadrant),
              ``solar_season_factor`` (0..1 daylight envelope) and
              ``solar_year`` (tick // season_ticks) on the scene
            - adds energy to the exposed surface cells, raising
              ``solar_energy_deposited`` and ``solar_lit_ticks``
        """
        p = self.params
        day = max(int(p.solar_day_ticks), 1)
        phase = (self._tick + int(p.solar_phase)) % day
        angle = 2.0 * np.pi * phase / day
        intensity = (p.solar_min_intensity
                     + (p.solar_max_intensity - p.solar_min_intensity)
                     * max(0.0, float(np.sin(angle))))

        season_ticks = int(p.solar_season_ticks)
        if season_ticks <= 0:
            season_factor = 1.0
            season_index = 0
            year = 0
        else:
            season_ticks = max(season_ticks, 1)
            season_index = (self._tick % season_ticks) // max(day, 1)
            season_factor = (
                (1.0 - float(p.solar_seasonality))
                + float(p.solar_seasonality)
                * (0.5 + 0.5 * np.cos(2.0 * np.pi * self._tick / season_ticks))
            )
            year = self._tick // season_ticks
        self.solar_season_index = season_index
        self.solar_season_factor = float(season_factor)
        self.solar_year = year

        self.world.light = float(intensity) * float(season_factor)
        if intensity <= 0.0:
            return

        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz
        exposed = self.world.material.reshape(nx, ny, nz) != MATERIAL_AIR
        has_ground = exposed.any(axis=1)
        top_y = ny - 1 - np.argmax(exposed[:, ::-1], axis=1)
        X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        cols = has_ground
        idx = X[cols] * (ny * nz) + top_y[cols] * nz + Z[cols]
        if idx.size == 0:
            return
        deposit = intensity * float(p.solar_deposit_rate) * float(season_factor)
        self.world.energy[idx] += np.float32(deposit)
        self.solar_energy_deposited += float(idx.size) * deposit
        self.solar_lit_ticks += 1

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

    def nearest_nest(self, point: np.ndarray, radius: float,
                     group_id: int | None = None) -> Nest | None:
        """
        Find the closest alive nest within radius of a point.

        Args:
            point: query point.
            radius: max distance to include.
            group_id: when given, only nests owned by this tribe are considered.

        Returns:
            The nearest qualifying nest, or None.
        """
        best = None
        best_d = float("inf")
        for n in self.nests:
            if not n.alive:
                continue
            if group_id is not None and n.owner_group_id != group_id:
                continue
            d = n.distance_to_point(point)
            if d <= radius and d < best_d:
                best, best_d = n, d
        return best

    def route_build(self, action: BabyAction, baby: SimBaby) -> tuple[float, int]:
        """
        Route a baby's cell-write deposits into durable structures.

        When structures are enabled, each applied write that carries energy is
        rerouted: a write near an existing nest feeds its bank; a write far
        from any nest seeds a new nest if it carries enough energy and the
        world-wide nest cap has not been reached; otherwise the cell keeps its
        energy as usual. Fed and seed writes leave the cell as material with
        zero energy, so the deposit is a transfer from the baby into the
        structure bank — never creation.

        Args:
            action: the baby's action (its cell writes).
            baby: the writing baby (owner of any newly seeded nest).

        Returns:
            Tuple (nested, seeded) — energy banked this call, nests seeded.

        Side effects:
            - grows the bank of an existing nest or appends a new Nest
            - zeroes the energy of cells that fed or seeded a nest
        """
        if not self.params.structure_enabled:
            return 0.0, 0
        nested = 0.0
        seeded = 0
        for w in action.writes:
            if w.energy <= 0:
                continue
            if not (0 <= w.x < self.world.nx
                    and 0 <= w.y < self.world.ny
                    and 0 <= w.z < self.world.nz):
                continue
            anchor = np.array([w.x + 0.5, w.y, w.z + 0.5], dtype=np.float64)
            nest = self.nearest_nest(anchor, self.params.nest_radius)
            if nest is not None:
                nest.stored_energy += w.energy
                nested += w.energy
                self.world.energy[self.world.idx(w.x, w.y, w.z)] = 0.0
            elif (len(self.nests) < self.params.max_nests
                    and w.energy >= self.params.nest_seed_energy):
                self.nests.append(Nest(
                    id=self._next_nest_id,
                    position=anchor,
                    stored_energy=w.energy,
                    owner_group_id=baby.group_id,
                ))
                self._next_nest_id += 1
                seeded += 1
                self.world.energy[self.world.idx(w.x, w.y, w.z)] = 0.0
        return nested, seeded

    def update_nests(self) -> None:
        """
        Apply structure upkeep — stored energy decays, empty nests erode away.

        Each tick a fraction of every nest's bank is lost to entropy. A nest
        whose bank is empty is pruned, leaving its anchor cell as ordinary
        material rubble.

        Side effects:
            - reduces each nest's stored_energy
            - removes empty nests from ``self.nests``
        """
        if not self.params.structure_enabled:
            return
        alive = []
        for n in self.nests:
            n.stored_energy -= self.params.nest_decay * n.stored_energy
            if n.stored_energy > 1e-9:
                alive.append(n)
        self.nests = alive

    def draw_nest(self, baby: SimBaby) -> float:
        """
        Draw stored energy from the baby's own tribe's nearest nest.

        A baby only draws when it is under its start energy — the nest is a
        starvation buffer, never a hoarding mechanism. The draw is limited by
        the draw rate, the gap back to start energy, and the nest's remaining
        bank, and only a nest owned by the baby's tribe can be tapped — other
        tribes' structures are off-limits territory.

        Args:
            baby: the drawing baby.

        Returns:
            Energy transferred from the nest to the baby.

        Side effects:
            - reduces the nest's stored_energy
            - raises the baby's energy
        """
        if not self.params.structure_enabled:
            return 0.0
        if baby.energy >= self.params.start_energy:
            return 0.0
        nest = self.nearest_nest(baby.position, self.params.nest_use_radius,
                                 group_id=baby.group_id)
        if nest is None:
            return 0.0
        gap = self.params.start_energy - baby.energy
        draw = min(self.params.nest_draw_rate, gap, nest.stored_energy)
        if draw <= 0:
            return 0.0
        nest.stored_energy -= draw
        baby.entity.energy += draw
        return float(draw)

    def raid_nest(self, baby: SimBaby) -> float:
        """
        Foreign theft: drain stored energy from a rival tribe's nest.

        When territoriality is on, a hungry baby standing on foreign ground —
        within ``territory_radius`` of a FOREIGN tribe's nearest nest — can
        draw from that bank at the same rate as an owner (``nest_draw_rate``,
        capped by the gap back to start energy and the bank). The draw is a
        transfer, never creation: the rival tribe's shared bank is the value
        that defending (evicting the raider) protects. This is the one-way
        drain that makes territoriality worth evolving — an unguarded bank is
        siphoned, a defended one is kept.

        Args:
            baby: the raiding agent.

        Returns:
            Energy stolen from the foreign nest into the baby.

        Side effects:
            - reduces the foreign nest's stored_energy
            - raises the baby's energy
        """
        if not self.params.territoriality_enabled:
            return 0.0
        if baby.energy >= self.params.start_energy:
            return 0.0
        best = None
        best_d = float("inf")
        for n in self.nests:
            if not n.alive or n.owner_group_id == baby.group_id:
                continue
            d = n.distance_to_point(baby.position)
            if d <= self.params.territory_radius and d < best_d:
                best, best_d = n, d
        if best is None:
            return 0.0
        gap = self.params.start_energy - baby.energy
        steal = min(self.params.nest_draw_rate, gap, best.stored_energy)
        if steal <= 0:
            return 0.0
        best.stored_energy -= steal
        baby.entity.energy += steal
        return float(steal)

    def deposit_nest(self, baby: SimBaby) -> float:
        """
        Builder act: bank genuine surplus into the tribe's nearest nest.

        A Builder carries a role posture (see ``decide_role``). While it
        stands within ``nest_use_radius`` of its OWN tribe's nearest nest
        and holds energy above its start, it deliberately banks
        ``role_deposit_fraction`` of that surplus into the bank — a
        deliberate transfer that replaces the noisy cell-write deposits and
        lifts the tribe's famine floor. The transfer is conservation-safe:
        it is a pure movement of energy from the baby into the nest, never
        creation. The banking is capped so a Builder never banks more than
        the surplus available (it keeps its start energy as a working
        buffer). Because the deposit lands in the same tick's honest net
        reward, the posture is selected for only where banking pays.

        Args:
            baby: the Builder depositing its surplus.

        Returns:
            Energy transferred from the baby into the nest bank.

        Side effects:
            - reduces the baby's energy
            - raises the nest's stored_energy
        """
        if not self.params.specialization_enabled:
            return 0.0
        surplus = baby.energy - self.params.start_energy
        if surplus <= 0.0:
            return 0.0
        nest = self.nearest_nest(baby.position, self.params.nest_use_radius,
                                 group_id=baby.group_id)
        if nest is None:
            return 0.0
        amount = min(surplus, surplus * self.params.role_deposit_fraction)
        if amount <= 0.0:
            return 0.0
        baby.entity.energy -= amount
        nest.stored_energy += amount
        return float(amount)

    def role_raid(self, baby: SimBaby) -> float:
        """
        Warrior act: raid a foreign tribe's nest even when not hungry.

        A Warrior carries a role posture (see ``decide_role``). Standing
        within ``territory_radius`` of a FOREIGN tribe's nearest nest it
        raids that bank at ``role_raid_fraction`` of the owner's draw rate,
        capped by the bank — the warrior acts even above start energy, so a
        defended bank's value is what the defender keeps by evicting the
        raider. The raid is a transfer, never creation: the rival tribe's
        shared bank is moved into the raider and, as with any same-tick
        energy movement, lands in the raider's honest net reward. A warrior
        that finds no foreign nest (or no bank) takes nothing.

        Args:
            baby: the Warrior raiding the foreign bank.

        Returns:
            Energy stolen from the foreign nest into the baby.

        Side effects:
            - reduces the foreign nest's stored_energy
            - raises the baby's energy
        """
        if not self.params.specialization_enabled:
            return 0.0
        best = None
        best_d = float("inf")
        for n in self.nests:
            if not n.alive or n.owner_group_id == baby.group_id:
                continue
            d = n.distance_to_point(baby.position)
            if d <= self.params.territory_radius and d < best_d:
                best, best_d = n, d
        if best is None:
            return 0.0
        cap = self.params.nest_draw_rate * self.params.role_raid_fraction
        steal = min(cap, best.stored_energy)
        if steal <= 0.0:
            return 0.0
        best.stored_energy -= steal
        baby.entity.energy += steal
        return float(steal)

    def birth(self, parent: SimBaby) -> tuple[int | None, float]:
        """
        Birth an offspring near a parent — a transfer, never creation.

        The offspring's starting energy is ``birth_cost``: up to
        ``birth_nest_fraction`` of it is drawn from the tribe's nearest
        nest bank (the tribe funds the child's start), the remainder comes
        from the parent's own surplus. Total world energy is conserved —
        the child's energy is exactly what the parent and nest lose. If the
        parent cannot fund its share the birth aborts and the nest draw is
        refunded (breeding requires genuine surplus). The child is placed
        within ``birth_range`` of the parent (torus-wrapped), inherits the
        parent's learned behavior and tribe via :meth:`SimBaby.spawn_child`,
        and is seeded from the world reservoir like any newborn. The
        world's ``max_entities`` cap bounds population even where energy is
        plentiful.

        Args:
            parent: the breeding baby (its energy is reduced in place).

        Returns:
            ``(child_id, energy_moved)`` or ``(None, 0.0)`` when no birth
            happened.

        Side effects:
            - reduces the parent's energy and the tribe nest's bank
            - appends a new baby and entity to the scene
        """
        if not self.params.lifecycle_enabled:
            return None, 0.0
        if len(self.alive_babies) >= self.params.max_entities:
            return None, 0.0
        nest_share = 0.0
        nest = None
        if self.params.structure_enabled:
            nest = self.nearest_nest(parent.position,
                                     self.params.nest_use_radius,
                                     group_id=parent.group_id)
            if nest is not None:
                nest_share = min(
                    self.params.birth_cost * self.params.birth_nest_fraction,
                    nest.stored_energy,
                )
                nest.stored_energy -= nest_share
        parent_share = self.params.birth_cost - nest_share
        if parent_share > parent.energy:
            if nest is not None and nest_share > 0.0:
                nest.stored_energy += nest_share
            return None, 0.0
        parent.entity.energy -= parent_share
        gx, gy, gz = self.params.grid_size
        dx = float(np.random.uniform(-self.params.birth_range,
                                     self.params.birth_range))
        dz = float(np.random.uniform(-self.params.birth_range,
                                     self.params.birth_range))
        child_pos = np.array([
            (parent.position[0] + dx) % gx,
            float(parent.position[1]),
            (parent.position[2] + dz) % gz,
        ], dtype=np.float64)
        child = parent.spawn_child(child_pos)
        self.add_baby(child)
        self.births += 1
        return child.entity.id, float(self.params.birth_cost)

    def info(self) -> dict:
        return {
            "tick": self._tick,
            "grid_size": self.params.grid_size,
            "total_energy": self.world.total_energy,
            "total_signal": self.world.total_signal,
            "entities": len(self.entities),
            "alive_babies": len(self.alive_babies),
            "nests": len(self.nests),
            "nest_energy": float(sum(n.stored_energy for n in self.nests)),
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
            "nests": [n.to_dict() for n in self.nests],
            "next_nest_id": self._next_nest_id,
            "world_memory": self.world_memory.to_dict()
            if self.world_memory is not None else None,
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
            SimBaby.from_dict(b, params, entity=by_id.get((b.get("entity") or {}).get("id")))
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
        scene.nests = [Nest.from_dict(n) for n in data.get("nests", [])]
        scene._next_nest_id = int(data.get("next_nest_id", 1))
        wm = data.get("world_memory")
        scene.world_memory = WorldMemory.from_dict(wm) if wm else None
        max_id = max((e.id for e in entities), default=0)
        if SimBaby._next_id <= max_id:
            SimBaby._next_id = max_id + 1
        max_nest = max((n.id for n in scene.nests), default=0)
        if scene._next_nest_id <= max_nest:
            scene._next_nest_id = max_nest + 1
        return scene


# ── Simulation ───────────────────────────────────────────────────────────────

class Simulation:
    """
    Tick-based simulation loop.

    Each tick:
        1. Deliver last tick's directed messages (inbox fill)
        1b. Structure upkeep (nest decay)
        2. World compute (diffusion, waves, conservation)
        3. For each alive baby:
           a. Perceive (read cells, entities, nests, body)
           b. Feel (energy change)
           c. React (write cells)
           d. Apply action (write to world)
           e. Build (route write deposits into nests)
           f. Move (born ability — one grid step per tick)
           g. Directed message (signal a specific neighbor, delivered next tick)
           h. Teach (cultural transfer to the neediest neighbor)
           i. Social step (cooperate or contest against nearest neighbor)
           i2. Defend (territoriality: evict trespassers from the tribe's region)
           j. Learn (update perceptron weights)
           k. Absorb (energy from nearby organic material)
           l. Draw (tap own tribe's nest when under start energy)
           m. Passive drain + perception cost
        4. Remove dead babies (their best episodes are deposited into the
           world memory reservoir first, when the channel is on)
    """

    def __init__(self, scene: SimScene, max_ticks: int = 100,
                 verbose: bool = False, render_bridge=None):
        self.scene = scene
        self.max_ticks = max_ticks
        self.verbose = verbose
        self._tick_log: list[dict] = []
        self._running = False
        self._render_bridge = render_bridge

    def step(self) -> list[dict]:
        """Run one simulation tick. Returns per-baby results."""
        self.scene._tick += 1

        # Deliver last tick's directed messages before anyone perceives, so a
        # message is readable exactly one tick after it was sent.
        if self.scene.params.message_enabled:
            self.scene.deliver_messages()

        # Structure upkeep — nest banks decay a little every tick.
        if self.scene.params.structure_enabled:
            self.scene.update_nests()

        results = []

        # World compute first
        self.scene.update_cells()

        # Solar boundary source — the sky deposits energy onto the surface
        # after diffusion, so it never competes with a cell's internal math.
        if self.scene.params.solar_enabled:
            self.scene.apply_solar()

        # Render the world state after all compute, before baby actions.
        # The bridge converts the WorldGrid into a Cycles Scene and renders it.
        rendered_image = None
        if self._render_bridge is not None:
            rendered_image = self._render_bridge.render_tick(
                self.scene.world,
                babies=list(self.scene.alive_babies),
                nests=self.scene.nests if self.scene.params.structure_enabled else None,
            )

        alive = list(self.scene.alive_babies)

        for baby in alive:
            if not baby.alive:
                continue

            t0 = time.time()
            prev_energy = baby.energy

            # 1. Perceive (cells + nearby agents + structures)
            perception = baby.perceive(self.scene.world, babies=alive,
                                       nests=self.scene.nests)
            baby.entity.energy -= self.scene.params.see_cost

            # 2. Feel — the immediate sensation (only see_cost has been paid
            #    so far this tick). This feeds the action decision, it is NOT
            #    the reward: the honest reward is the net tick delta measured
            #    after every gain and drain (step 8b).
            sensation = baby.feel(prev_energy)

            # 3. React
            action = baby.react(perception, sensation)

            # 4. Apply action — the baby funds the cells it writes, so the
            #    world conserves energy (deposits are a transfer, not creation)
            cells_written = baby.apply_action(action, self.scene.world)

            # 4a. Route write deposits into durable structures. A write near a
            #     nest feeds its bank; a substantial write far from any nest
            #     seeds a new one (the deposit becomes banked savings instead
            #     of a free-standing cell, so the world still conserves energy).
            nested = 0.0
            seeded = 0
            if self.scene.params.structure_enabled:
                nested, seeded = self.scene.route_build(action, baby)

            deposited = float(sum(w.energy for w in action.writes))
            baby.entity.energy -= (
                self.scene.params.write_cost * cells_written + deposited
            )

            # 4b. Move — a born ability to relocate one grid step per tick.
            #     Charged move_cost; the cost lands in the same tick's net
            #     reward (step 8b), giving the movement perceptron immediate
            #     feedback on where it went.
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

            # 4d. Cultural act — teach the neediest nearby baby. The teach
            #     perceptron decides whom to teach (the hungriest neighbor in
            #     range); the gate value becomes the lesson amplitude, and the
            #     cost scales with it (a louder lesson transfers more behavior
            #     and costs more). The student's behavior weights blend toward
            #     the teacher's learned weights and its best episodes are
            #     copied into the student's memory — learned behavior moves
            #     laterally between living agents.
            teaching_energy = 0.0
            teaching_amplitude = 0.0
            taught_episodes = 0
            if self.scene.params.teaching_enabled:
                teach_neighbors = self.scene.nearby_babies(
                    baby.position, self.scene.params.teach_range,
                    exclude_id=baby.entity.id,
                )
                # Cultural transmission is IN-GROUP (mirrors the tribe-scoped
                # nest draw): knowledge given to a rival tribe is a net tribe
                # loss, so group selection would extinguish cross-tribe
                # teaching before it can ever be selected for. The teach
                # perceptron still decides whether and how loud to teach a
                # tribe-mate, and pays the cost.
                teach_neighbors = [n for n in teach_neighbors
                                   if n.group_id == baby.group_id]
                baby._last_teach_input = None
                baby._last_teach_out = None
                if teach_neighbors:
                    teach_target = min(teach_neighbors, key=lambda n: n.energy)
                    teaching_amplitude = baby.decide_teach(teach_target)
                    if teaching_amplitude > 0.0:
                        taught_episodes = baby.teach(teach_target, teaching_amplitude)
                        teaching_energy = (
                            self.scene.params.teach_cost * teaching_amplitude
                        )
                        baby.entity.energy -= teaching_energy

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

            # 6. Predation — consume a weaker neighbor for energy. The
            #     predation perceptron decides whether to hunt the weakest
            #     nearby baby within range; a strike is lethal, transfers the
            #     prey's full energy to the predator, and costs
            #     ``predation_cost``. Predation is an opt-in channel (off by
            #     default): when off no perceptron exists and no RNG is drawn,
            #     so the locked selection proofs keep their exact genome
            #     layout and energy flow. When on, the strike lands in the
            #     same tick's honest net reward (step 8b), so the gate is
            #     shaped by the true outcome — hunting pays while prey energy
            #     exceeds the strike cost and self-limits as prey runs scarce.
            predation_amplitude = 0.0
            predation_energy = 0.0
            prey_id = None
            if self.scene.params.predation_enabled:
                prey_neighbors = self.scene.nearby_babies(
                    baby.position, self.scene.params.predation_range,
                    exclude_id=baby.entity.id,
                )
                baby._last_predation_input = None
                baby._last_predation_out = None
                if prey_neighbors:
                    weaker = [n for n in prey_neighbors if n.energy < baby.energy]
                    if weaker:
                        prey = min(weaker, key=lambda n: n.energy)
                        predation_amplitude = baby.decide_predation(prey)
                        if predation_amplitude > 0.0:
                            predation_energy = baby.hunt(prey)
                            prey_id = prey.entity.id
                            baby.entity.energy -= self.scene.params.predation_cost

            # 6b. Territoriality — defend the tribe's region. When a baby
            #     stands on its own tribe's territory (within
            #     ``territory_radius`` of its tribe's nearest nest) it may
            #     evict the nearest foreign baby within ``defend_range``: the
            #     territory perceptron decides whether the eviction pays, a
            #     cleared gate shoves the trespasser ``defend_push`` cells
            #     away and transfers ``defend_take_fraction`` of its energy to
            #     the defender (a toll that scales with what the trespasser
            #     carries, never lethal), and the defender pays
            #     ``defend_cost``. Eviction is what keeps a rival raider from
            #     draining the tribe's nest bank (step 7b), so defense
            #     protects a real shared resource. Territoriality is an opt-in
            #     channel (off by default): when off no perceptron exists and
            #     no RNG is drawn, so the locked selection proofs keep their
            #     exact genome layout and energy flow. When on, the eviction
            #     lands in the same tick's honest net reward (step 8b), so
            #     the gate is shaped by the true outcome — defending pays
            #     while a trespasser carries more energy than the eviction
            #     costs and self-limits as trespassers run scarce.
            defended = False
            defend_amplitude = 0.0
            defend_energy = 0.0
            defended_id = None
            if self.scene.params.territoriality_enabled:
                home_nest = self.scene.nearest_nest(
                    baby.position, self.scene.params.territory_radius,
                    group_id=baby.group_id,
                )
                baby._last_defend_input = None
                baby._last_defend_out = None
                if home_nest is not None:
                    strangers = [
                        n for n in self.scene.nearby_babies(
                            baby.position, self.scene.params.defend_range,
                            exclude_id=baby.entity.id,
                        ) if n.group_id != baby.group_id
                    ]
                    if strangers:
                        target = min(strangers,
                                     key=lambda n: n.distance_to_point(
                                         baby.position))
                        defend_amplitude = baby.decide_defend(target)
                        if defend_amplitude > 0.0:
                            defend_energy = baby.defend(
                                target, home_nest.position)
                            defended_id = target.entity.id
                            defended = True
                            baby.entity.energy -= self.scene.params.defend_cost

            # 6c. Reproduction — breed an offspring when the gate clears.
            #     A baby must clear its reproduce gate AND stand above
            #     ``reproduce_energy_threshold`` (genuine surplus). The
            #     offspring's start energy is ``birth_cost`` — up to
            #     ``birth_nest_fraction`` of it drawn from the tribe's nest
            #     bank, the rest from the parent (a transfer, never
            #     creation, so the world conserves energy). The outlay lands
            #     in the parent's same-tick honest net reward, so the gate
            #     is selected for only when breeding is affordable. The
            #     child does not act until the next tick (the alive snapshot
            #     was taken before the loop), and it is removed with any
            #     other dead baby at the end of the step. Starvation already
            #     removes babies every tick, so births and deaths now happen
            #     INSIDE the tick loop and a scene's population can
            #     self-sustain without the engine re-seeding it.
            reproduced = False
            birth_energy = 0.0
            child_id = None
            if self.scene.params.lifecycle_enabled:
                baby._last_reproduce_input = None
                baby._last_reproduce_out = None
                if (baby.energy > self.scene.params.reproduce_energy_threshold
                        and len(self.scene.alive_babies)
                        < self.scene.params.max_entities):
                    if baby.decide_reproduce() > 0.0:
                        child_id, birth_energy = self.scene.birth(baby)
                        reproduced = child_id is not None

            # 7. Absorb energy from nearby organic material
            absorbed = baby.absorb_energy(self.scene.world)
            baby.entity.energy += absorbed

            # 7b. Draw from own tribe's nearest nest when under start energy —
            #     a starvation buffer that rewards territoriality. When the
            #     territoriality channel is on, a hungry baby on foreign
            #     ground can instead raid a rival tribe's nearest nest (the
            #     theft that defense protects).
            drawn = 0.0
            raided = 0.0
            if self.scene.params.structure_enabled:
                drawn = self.scene.draw_nest(baby)
                raided = self.scene.raid_nest(baby)

            # 7c. Division of labor — the role act. When specialization is
            #     on, the baby's role posture decides what it does this tick:
            #     a Builder standing near its own tribe's nest banks a share
            #     of its surplus (lifting the tribe's famine floor), a
            #     Warrior standing on a foreign tribe's ground raids that
            #     bank even when it is not hungry (lifting the tribe's
            #     mean). Both acts are deliberate transfers that land in the
            #     same tick's honest net reward (step 8b), so the posture —
            #     a heritable strategy trait shaped by the genome's dedicated
            #     RNG stream — is selected for only where it pays.
            role_deposited = 0.0
            role_raided = 0.0
            if self.scene.params.specialization_enabled:
                baby._last_role_input = None
                baby._last_role_out = None
                posture = baby.decide_role()
                if posture < self.scene.params.role_gate_threshold:
                    role_deposited = self.scene.deposit_nest(baby)
                else:
                    role_raided = self.scene.role_raid(baby)

            # 8. Passive drain
            baby.entity.energy -= self.scene.params.passive_drain
            baby.entity.energy = max(0.0, baby.entity.energy)
            baby._total_ticks += 1

            # 8b. Learn from the tick's honest net outcome. The reward is the
            #     full energy delta across this tick — perception, movement,
            #     writes, social transfers, teaching cost, absorption, nest
            #     draw, and passive drain — so a baby that reaches food
            #     genuinely feels the gain and one that wastes energy feels the
            #     loss. Two defects had to be fixed for honesty to work: the
            #     reward used to collapse to the uniform -0.5 see_cost (food
            #     absorption was measured after the delta was captured), and
            #     the delta rule's "weaken" branch double-flipped its sign
            #     (error * lr came out positive), so losses actually reinforced
            #     bad behavior and drove the perceptrons to saturation. With
            #     both fixed, episodes carry real outcomes and memotype
            #     inheritance plus lateral teaching copy genuinely good
            #     experience (benchmark_culture).
            net_delta = baby.energy - prev_energy
            baby.learn(net_delta)

            elapsed = (time.time() - t0) * 1000

            result = {
                "baby_id": baby.entity.id,
                "tick": self.scene.tick,
                "energy": baby.energy,
                "energy_delta": net_delta,
                "cells_written": cells_written,
                "moved": moved,
                "message_amplitude": message_amplitude,
                "message_energy": message_energy,
                "teaching_amplitude": teaching_amplitude,
                "teaching_energy": teaching_energy,
                "taught_episodes": taught_episodes,
                "predation_amplitude": predation_amplitude,
                "predation_energy": predation_energy,
                "prey_id": prey_id,
                "defended": defended,
                "defend_amplitude": defend_amplitude,
                "defend_energy": defend_energy,
                "defended_id": defended_id,
                "reproduced": reproduced,
                "birth_energy": birth_energy,
                "child_id": child_id,
                "absorbed": absorbed,
                "nested": nested,
                "seeded": seeded,
                "drawn": drawn,
                "raided": raided,
                "role_deposited": role_deposited,
                "role_raided": role_raided,
                "social_act": social_act,
                "social_energy": social_energy,
                "total_ms": elapsed,
                "alive": baby.alive,
            }
            results.append(result)

            if self.verbose:
                logger.debug(
                    f"tick={self.scene.tick} baby={baby.entity.id} "
                    f"energy={baby.energy:.1f} "
                    f"delta={net_delta:+.1f} "
                    f"wrote={cells_written} "
                    f"social={social_act}({social_energy:+.1f})"
                    f" prey={prey_id if predation_energy > 0.0 else '-'}"
                    f" defend={defended_id if defend_energy > 0.0 else '-'}"
                    f" child={child_id if reproduced else '-'}"
                    f" role_dep={role_deposited:.1f}"
                    f" role_raid={role_raided:.1f}"
                )

        # 4. Remove dead babies from entities. Before a baby leaves the world
        #    its best episodes are deposited into the world memory reservoir
        #    (when the channel is on) — lived experience survives the body.
        if self.scene.world_memory is not None:
            for d in self.scene._devices:
                if not d.alive:
                    self.scene.deposit_memory(d)
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
            "avg_energy": float(np.mean([r["energy"] for r in self._tick_log])) if self._tick_log else 0,
            "total_cells_written": sum(r["cells_written"] for r in self._tick_log),
            "total_energy_absorbed": sum(r["absorbed"] for r in self._tick_log),
            "cooperations": sum(1 for r in self._tick_log if r["social_act"] == "cooperate"),
            "contests": sum(1 for r in self._tick_log if r["social_act"] == "contest"),
            "social_energy_moved": sum(r["social_energy"] for r in self._tick_log),
            "total_nested": sum(r.get("nested", 0.0) for r in self._tick_log),
            "total_drawn": sum(r.get("drawn", 0.0) for r in self._tick_log),
            "nests_built": sum(r.get("seeded", 0) for r in self._tick_log),
            "lessons": sum(1 for r in self._tick_log if r.get("taught_episodes", 0) > 0),
            "episodes_taught": sum(r.get("taught_episodes", 0) for r in self._tick_log),
            "predations": sum(1 for r in self._tick_log
                              if r.get("predation_energy", 0.0) > 0.0),
            "predation_energy_moved": sum(r.get("predation_energy", 0.0)
                                          for r in self._tick_log),
            "defenses": sum(1 for r in self._tick_log if r.get("defended", False)),
            "defend_energy_moved": sum(r.get("defend_energy", 0.0)
                                       for r in self._tick_log),
            "raids": sum(1 for r in self._tick_log if r.get("raided", 0.0) > 0.0),
            "raid_energy_moved": sum(r.get("raided", 0.0)
                                     for r in self._tick_log),
            "births": sum(1 for r in self._tick_log if r.get("reproduced", False)),
            "birth_energy_moved": sum(r.get("birth_energy", 0.0)
                                      for r in self._tick_log),
            "role_deposits": sum(1 for r in self._tick_log
                                 if r.get("role_deposited", 0.0) > 0.0),
            "role_deposit_energy": sum(r.get("role_deposited", 0.0)
                                       for r in self._tick_log),
            "role_raids": sum(1 for r in self._tick_log
                              if r.get("role_raided", 0.0) > 0.0),
            "role_raid_energy": sum(r.get("role_raided", 0.0)
                                    for r in self._tick_log),
            "deaths": len(dead),
            "alive_count": len([b for b in self.scene.babies if b.alive]),
            "solar_energy_deposited": float(self.scene.solar_energy_deposited),
            "sunshine": float(self.scene.solar_lit_ticks)
            / max(float(self.scene.tick), 1.0),
            "light_final": float(self.scene.world.light),
        }
