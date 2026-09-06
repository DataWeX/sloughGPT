"""
Live realm view — watch the Programmable World Realm run, tick by tick.

Renders a side-on ASCII cross-section of the living grid: material glyphs,
a grayscale energy heat-map (brighter = more energy), babies overlaid with a
green-to-red energy read, and a skyline above the grid showing the sun rise
and set as the diurnal cycle (Stage 13) turns. When the seasonal year
envelope (Stage 14) is enabled the skyline also names the current season and
a second bar traces the daylight envelope across the whole year, so you watch
midsummer noon outshine midwinter noon. Every tick clears and redraws, so you
watch energy flood the surface at noon, drain into the dark at night, and
diffuse through the ground while babies hunt, forage and breed.

This is an observation surface for the realm — the counterpart to the
benchmarks. The benchmarks prove the physics with verdicts; this lets you
SEE the physics.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .evolution import EvolutionEngine, Genome
from .simulation import (
    MATERIAL_AIR,
    MATERIAL_EMBER,
    MATERIAL_LIVING,
    MATERIAL_METAL,
    MATERIAL_ORGANIC,
    MATERIAL_SIGNAL,
    MATERIAL_STONE,
    MATERIAL_WATER,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)

# Material glyphs for the side view. AIR renders as space so the sky and
# underground read naturally; the rest are single distinct symbols.
_SYMBOLS: dict[int, str] = {
    MATERIAL_AIR: " ",
    MATERIAL_WATER: "~",
    MATERIAL_STONE: "#",
    MATERIAL_ORGANIC: "o",
    MATERIAL_METAL: "M",
    MATERIAL_EMBER: "e",
    MATERIAL_LIVING: "@",
    MATERIAL_SIGNAL: "*",
}

# Material foreground colors (ANSI 256), keyed by the same constants.
_COLORS: dict[int, str] = {
    MATERIAL_AIR: "0",        # default — invisible, no background heat
    MATERIAL_WATER: "81",     # sky blue
    MATERIAL_STONE: "245",    # grey
    MATERIAL_ORGANIC: "114",  # moss green
    MATERIAL_METAL: "226",    # gold
    MATERIAL_EMBER: "203",    # ember red
    MATERIAL_LIVING: "120",   # bright green
    MATERIAL_SIGNAL: "213",   # magenta
}

_CLEAR = "\033[2J\033[H"
_RESET = "\033[0m"
_SKY = 24  # width of the sun skyline drawn above the grid

# Year quadrant names, in envelope order. The envelope peaks at tick 0, so
# quadrant 0 is midsummer and quadrant 2 (the midpoint) is midwinter.
_SEASONS = ("SUMMER", "AUTUMN", "WINTER", "SPRING")

# Sun glyph colours by season — midsummer burns hot yellow, midwinter a pale
# chill blue, autumn orange and spring green. Seasons off keeps the warm sun.
_SUN_COLORS = {
    0: "\033[93m",   # summer — hot yellow
    1: "\033[38;5;214m",  # autumn — orange
    2: "\033[96m",   # winter — pale cyan
    3: "\033[92m",   # spring — green
}
_SUMMER_SUN = "\033[93m"

# Year-envelope bar ramp (darkest to brightest) and its width.
_RAMP = " ▁▂▃▄▅▆▇█"


def _season_of(scene: SimScene) -> tuple[int, float]:
    """Year quadrant (0..3) and daylight envelope factor for this tick."""
    p = scene.params
    if int(p.solar_season_ticks) <= 0:
        return 0, 1.0
    return int(scene.solar_season_index) % 4, float(scene.solar_season_factor)


def _envelope_at(tick: int, season_ticks: int, seasonality: float) -> float:
    """Daylight envelope (0..1) at an arbitrary tick inside the year."""
    st = max(int(season_ticks), 1)
    return (1.0 - float(seasonality)
            + float(seasonality)
            * (0.5 + 0.5 * np.cos(2.0 * np.pi * tick / st)))


def _year_envelope_bar(scene: SimScene) -> str:
    """
    One-row bar tracing the daylight envelope across the whole year.

    Draws ``_SKY`` cells, each the envelope at that fraction of the year, with
    the current tick highlighted in yellow and the current envelope printed as
    a percentage. Midsummer reads as a bright ridge at the left, midwinter as
    a dark trough at the middle, spring/summer closing the loop.

    Args:
        scene: the live scene (seasonal year envelope must be on).

    Returns:
        The bar line, or an empty string when seasons are off.
    """
    p = scene.params
    st = int(p.solar_season_ticks)
    if st <= 0:
        return ""
    pos = int(scene.tick) % st
    cur = min(int(pos / st * _SKY), _SKY - 1)
    cells: list[str] = []
    for i in range(_SKY):
        frac = _envelope_at(int(i / _SKY * st), st, float(p.solar_seasonality))
        level = int(min(max(frac, 0.0), 1.0) * (len(_RAMP) - 1))
        ch = _RAMP[level]
        if i == cur:
            ch = f"\033[93m{ch}\033[0m"
        cells.append(ch)
    season_idx, factor = _season_of(scene)
    return (f"year {int(scene.solar_year)}  {_SEASONS[season_idx]:<6} "
            f"env {factor * 100.0:>3.0f}%  [{''.join(cells)}]")


def _skyline(scene: SimScene) -> str:
    """One-row skyline showing the sun's position from the day phase."""
    p = scene.params
    day = max(int(p.solar_day_ticks), 1)
    phase = (scene.tick + int(p.solar_phase)) % day
    light = float(scene.world.light)
    cells = [" "] * _SKY
    season_idx, _ = _season_of(scene)
    sun = _SUN_COLORS.get(season_idx, _SUMMER_SUN)
    if light > 1e-6:
        pos = min(int(light * (_SKY - 1)), _SKY - 1)
        cells[pos] = f"{sun}☀\033[0m"
    noon = int(abs(np.sin(2.0 * np.pi * phase / day)) * 100)
    prefix = ""
    if int(p.solar_season_ticks) > 0:
        prefix = (f"year {int(scene.solar_year)} "
                  f"{_SEASONS[season_idx]:<6} ")
    return (f"{prefix}day {phase:>2}/{day}  light {light:.2f}  "
            f"noon {noon:>3}%  "
            + "".join(cells))


def _heat_bg(energy: float, emax: float) -> str:
    """ANSI 256 grayscale background for a cell's energy level."""
    if emax <= 0.0:
        return "232"
    b = energy / emax
    return "233" if b <= 0.001 else f"{232 + int(23 * min(b, 1.0))}"


def render_frame(scene: SimScene, tick: int) -> list[str]:
    """
    Render one side-on frame of the living grid.

    Picks the middle z-slice, renders y upward (bottom row = ground floor),
    colours material glyphs by type and paints a grayscale heat-map behind
    each cell from its energy, then overlays every alive baby with a
    green-to-red energy glyph. The first lines carry the skyline (sun
    position, plus the seasonal year bar when the Stage 14 envelope is on),
    the tick/phase/energy summary, and a material legend.

    Args:
        scene: the live scene to draw.
        tick: current simulation tick (0-based).

    Returns:
        List of lines (terminal width = nz columns + padding).
    """
    w = scene.world
    nx, ny, nz = w.nx, w.ny, w.nz
    alive = [b for b in scene.alive_babies]

    # Pick the z-slice that holds the most babies so the view follows the
    # action; fall back to the middle slice when the world is empty.
    if alive:
        zcounts: dict[int, int] = {}
        for b in alive:
            bz = int(round(float(b.position[2]))) % nz
            zcounts[bz] = zcounts.get(bz, 0) + 1
        z = max(zcounts, key=lambda k: (zcounts[k], -abs(k - nz // 2)))
    else:
        z = nz // 2
    mat = w.material.reshape(nx, ny, nz)[:, :, z]
    energy = w.energy.reshape(nx, ny, nz)[:, :, z]
    emax = float(w.energy.max()) if w.energy.size else 0.0

    lines: list[str] = []
    lines.append(_skyline(scene))
    year_bar = _year_envelope_bar(scene)
    if year_bar:
        lines.append(year_bar)

    energy_total = float(w.energy.sum()) + sum(b.energy for b in scene.babies)
    lines.append(
        f"tick {tick:>3}  energy_total {energy_total:8.1f}  "
        f"solar_in {scene.solar_energy_deposited:8.1f}  "
        f"alive {len(alive):>2}/{len(scene.babies):>2}"
    )

    # Overlay EVERY alive baby by its torus-wrapped (x, y), whatever its z,
    # so the whole population is visible at once.
    babies: dict[tuple[int, int], SimBaby] = {}
    for b in alive:
        x, y, bz = (int(round(float(c))) for c in b.position)
        babies[(x % nx, y % ny)] = b

    for y in range(ny - 1, -1, -1):
        row = []
        for x in range(nx):
            m = int(mat[x, y])
            sym = _SYMBOLS.get(m, "?")
            baby = babies.get((x, y))
            if baby is not None:
                frac = min(max(baby.energy / max(scene.params.start_energy, 1.0), 0.0), 1.0)
                color = "46" if frac >= 0.6 else ("226" if frac >= 0.3 else "196")
                row.append(f"\033[{color}mB\033[0m")
            elif m == MATERIAL_AIR:
                row.append(" ")
            else:
                fg = _COLORS.get(m, "255")
                bg = _heat_bg(float(energy[x, y]), emax)
                row.append(f"\033[38;5;{fg};48;5;{bg}m{sym}\033[0m")
        lines.append("".join(row))

    legend = "  ".join(
        f"\033[38;5;{c}m{s}\033[0m {name}"
        for s, name, c in (
            (_SYMBOLS[MATERIAL_STONE], "stone", _COLORS[MATERIAL_STONE]),
            (_SYMBOLS[MATERIAL_ORGANIC], "organic", _COLORS[MATERIAL_ORGANIC]),
            (_SYMBOLS[MATERIAL_WATER], "water", _COLORS[MATERIAL_WATER]),
            (_SYMBOLS[MATERIAL_METAL], "metal", _COLORS[MATERIAL_METAL]),
            (_SYMBOLS[MATERIAL_EMBER], "ember", _COLORS[MATERIAL_EMBER]),
            (_SYMBOLS[MATERIAL_SIGNAL], "signal", _COLORS[MATERIAL_SIGNAL]),
            ("B", "baby", "226"),
        )
    )
    lines.append(legend)
    return lines


def live_view(scene: SimScene, ticks: int, fps: float = 8.0,
              out=None) -> dict:
    """
    Step a live scene and redraw the frame after every tick.

    Runs the real ``Simulation.step`` loop so the view shows the genuine tick
    pipeline (cells update → solar deposit → perceive → act → diffuse).
    Clears the terminal and prints the fresh frame each tick at ``fps``.

    Args:
        scene: the scene to animate (solar_enabled recommended).
        ticks: number of ticks to run.
        fps: frames per second (0 disables the delay for fast runs).
        out: output stream (defaults to sys.stdout).

    Returns:
        Final scene stats dict (energy_total, solar_in, alive, births).
    """
    out = out or sys.stdout
    delay = 1.0 / fps if fps > 0 else 0.0
    if scene.params.world_seed is not None:
        np.random.seed(int(scene.params.world_seed))
    sim = Simulation(scene, max_ticks=ticks)
    prev_alive = {b.entity.id for b in scene.alive_babies}
    births = 0
    deaths = 0
    try:
        for t in range(1, ticks + 1):
            sim.step()
            now = {b.entity.id for b in scene.alive_babies}
            births = scene.births
            deaths += len(prev_alive - now)
            prev_alive = now
            out.write(_CLEAR)
            out.write("\n".join(render_frame(scene, t)))
            out.write("\n")
            out.flush()
            if delay:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        out.write(_RESET)
    energy_total = float(scene.world.energy.sum()) + sum(
        b.energy for b in scene.babies)
    return {
        "ticks": ticks,
        "energy_total": energy_total,
        "solar_in": float(scene.solar_energy_deposited),
        "alive": sum(1 for b in scene.babies if b.alive),
        "population": len(scene.babies),
        "births": births,
        "deaths": deaths,
    }


def make_live_scene(*, grid: tuple[int, int, int] = (24, 12, 24),
                    population: int = 8,
                    organic_pools: int = 3,
                    day_ticks: int = 24,
                    solar_deposit_rate: float = 0.4,
                    seasons_per_year: int = 0,
                    seasonality: float = 1.0,
                    cells_input_dim: int = 6,
                    seed: int = 7) -> SimScene:
    """
    Build a solar-lit, populated scene ready for ``live_view``.

    Uses a generated world, per-column surface spawns, deterministic food
    pools and random genomes, with the sun on and ``cells_input_dim = 6`` so
    the babies can actually perceive daylight.

    Args:
        grid: world dimensions (x, y, z).
        population: babies to spawn on the surface.
        organic_pools: food pools to scatter.
        day_ticks: length of one full day/night cycle.
        solar_deposit_rate: noon energy per lit surface cell.
        seasons_per_year: days per year (0 disables the Stage 14 envelope).
        seasonality: year envelope swing 0..1 (1 = full summer/winter swing).
        cells_input_dim: cells perceptron width (>= 6 exposes the light).
        seed: world + RNG seed.

    Returns:
        The populated ``SimScene`` (not yet stepped).
    """
    np.random.seed(seed)
    params = WorldParams(
        grid_size=grid,
        generate_world=True,
        world_seed=seed,
        solar_enabled=True,
        solar_day_ticks=day_ticks,
        solar_deposit_rate=solar_deposit_rate,
        solar_max_intensity=1.0,
        solar_season_ticks=day_ticks * max(int(seasons_per_year), 0),
        solar_seasonality=seasonality,
        cells_input_dim=cells_input_dim,
        learning_enabled=True,
    )
    scene = SimScene(params=params)
    engine = EvolutionEngine(params=params, population_size=population,
                             generations=1, ticks_per_generation=day_ticks,
                             seed=seed)
    engine._place_food(scene, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)

    # Surface spawn: topmost non-air cell per column, baby stands just above.
    w = scene.world
    nx, ny, nz = w.nx, w.ny, w.nz
    exposed = w.material.reshape(nx, ny, nz) != MATERIAL_AIR
    top_y = ny - 1 - np.argmax(exposed[:, ::-1], axis=1)
    xs = np.random.randint(0, nx, population)
    zs = np.random.randint(0, nz, population)
    for x, z in zip(xs, zs):
        y = min(int(top_y[x, z]) + 1, ny - 1)
        b = SimBaby(
            position=np.array([x, y, z], dtype=np.float64),
            initial_energy=params.start_energy,
            params=params,
            group_id=0,
        )
        genome = Genome.random(params, rng, group_id=0)
        genome.apply_to(b)
        scene.add_baby(b)
    return scene
