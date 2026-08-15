"""
Live realm viewer — ``python3 -m domains.shell.realm_live``.

Watches the Programmable World Realm run: a solar-lit generated world,
babies spawned on the surface, and the sun rising and setting over the grid
while energy floods the ground and the babies react. Pass ``--seasons`` to
ride the diurnal curve inside the seasonal year envelope (Stage 14) — the
skyline then names the current season and a year bar traces the daylight
envelope, so midsummer noon visibly outshines midwinter noon. Ctrl+C stops
the run and prints the final conservation summary.

Flags are optional; the defaults give a good first look.

    python3 -m domains.shell.realm_live                # 240 ticks, 8 fps
    python3 -m domains.shell.realm_live --ticks 96 --day 16 --fps 12
    python3 -m domains.shell.realm_live --seed 3 --population 12
    python3 -m domains.shell.realm_live --seasons      # two years of seasons
    python3 -m domains.shell.realm_live --seasons --seasons-per-year 6
"""

from __future__ import annotations

import argparse

from .realm_view import live_view, make_live_scene


def _parse_grid(value: str) -> tuple[int, int, int]:
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 3 or any(p <= 0 for p in parts):
        raise argparse.ArgumentTypeError("expected 'x,y,z' of positive ints")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="realm_live",
        description="Watch the Programmable World Realm run, tick by tick.")
    parser.add_argument("--grid", type=_parse_grid, default=(24, 12, 24),
                        help="world dimensions x,y,z (default 24,12,24)")
    parser.add_argument("--population", type=int, default=8,
                        help="babies spawned on the surface")
    parser.add_argument("--pools", type=int, default=3,
                        help="organic food pools")
    parser.add_argument("--day", type=int, default=24,
                        help="ticks per day/night cycle")
    parser.add_argument("--rate", type=float, default=0.4,
                        help="noon energy per lit surface cell")
    parser.add_argument("--seasons", action="store_true",
                        help="ride the sun on the seasonal year envelope (Stage 14)")
    parser.add_argument("--seasons-per-year", type=int, default=4,
                        help="days per year (only with --seasons)")
    parser.add_argument("--seasonality", type=float, default=1.0,
                        help="year envelope swing 0..1 (only with --seasons)")
    parser.add_argument("--ticks", type=int, default=240,
                        help="total ticks to watch (default 240 = 10 days)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="frames per second (0 = as fast as possible)")
    parser.add_argument("--seed", type=int, default=7,
                        help="world + RNG seed")
    args = parser.parse_args(argv)

    scene = make_live_scene(
        grid=args.grid,
        population=args.population,
        organic_pools=args.pools,
        day_ticks=args.day,
        solar_deposit_rate=args.rate,
        seasons_per_year=args.seasons_per_year if args.seasons else 0,
        seasonality=args.seasonality,
        seed=args.seed,
    )
    stats = live_view(scene, ticks=args.ticks, fps=args.fps)
    print(
        f"\nrealm run complete  ticks={stats['ticks']}  "
        f"energy_total={stats['energy_total']:.1f}  "
        f"solar_in={stats['solar_in']:.1f}  "
        f"alive={stats['alive']}/{stats['population']}  "
        f"births={stats['births']}  deaths={stats['deaths']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
