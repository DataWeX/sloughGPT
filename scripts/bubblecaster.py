#!/usr/bin/env python3
"""
Bubble Caster — cast wishes up to the stars.

A field of stars drifts high above. You cast bubbles from the ground; each
bubble carries a sigil and follows a short ballistic arc. When a bubble
touches a star, the star drinks it and brightens one step. A star that reaches
full brightness lights up and lets one of its wishes be heard.

Every cast is appended to datasets/bubblecaster/casts.jsonl as a labeled
sample (sky layout + chosen angle/power -> hit outcome). That growing record
is what a model could later learn from: given a target star, predict the
angle and power that will reach it.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Tunables ---------------------------------------------------------------

SKY_W = 60
SKY_H = 16
CASTER_X = SKY_W // 2
CASTER_Y = SKY_H - 1
GRAVITY = 0.09          # downward acceleration per simulation step
STAR_RADIUS = 1.5
MIN_POWER, MAX_POWER = 2, 12
MIN_ANGLE, MAX_ANGLE = 12, 82
MAX_BRIGHTNESS = 3      # casts to fully light a star
MAX_STEPS = 320

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "bubblecaster"
DATA_PATH = DATA_DIR / "casts.jsonl"

WISHES = [
    "The sea keeps every galaxy it has ever swallowed.",
    "A name spoken twice is a promise that cannot be undone.",
    "Fire remembers the shape of everything it has burned.",
    "Silence, at the right hour, is a kind of song.",
    "Every dream is a seed that was never planted on purpose.",
    "The wind carries the letters we were too late to post.",
]


# --- Core data --------------------------------------------------------------

@dataclass
class Star:
    x: float
    y: float
    radius: float = STAR_RADIUS
    brightness: int = 0
    wish_id: int = 0
    lit: bool = False

    @property
    def lit_char(self) -> str:
        return "\033[1;36m*\033[0m" if self.lit else "\033[90m*\033[0m"


@dataclass
class Cast:
    angle: float          # degrees from horizontal, launcher looks up away from ground
    power: float          # launch speed, scaled by / MAX_POWER
    sigil: str            # visual / narrative token carried by the bubble
    target_index: int     # which star the caster was aiming at


@dataclass
class GameState:
    stars: list[Star] = field(default_factory=list)
    score: int = 0
    casts: int = 0
    echoes: list[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


def new_sky(count: int = 6) -> list[Star]:
    """Spawn `count` stars across the upper half of the sky, non-overlapping."""
    stars: list[Star] = []
    wish_ids = list(range(len(WISHES)))
    random.shuffle(wish_ids)
    attempts = 0
    while len(stars) < count and attempts < 200:
        attempts += 1
        s = Star(
            x=random.uniform(4, SKY_W - 4),
            y=random.uniform(1, SKY_H / 2),
            radius=STAR_RADIUS,
            brightness=0,
            wish_id=wish_ids[len(stars) % len(WISHES)],
        )
        if all(math.hypot(s.x - o.x, s.y - o.y) > 4 for o in stars):
            stars.append(s)
    return stars


def simulate_cast(stars: list[Star], cast: Cast) -> dict:
    """
    Ballistic trajectory of a bubble from the caster's position.

    Returns a dict with the bubble's successive positions and the hit outcome:
      hit         - bool, any star touched
      hit_index   - index into `stars`, or None
      steps       - number of simulation steps before the bubble popped
      coords      - list of (x, y) positions crossed, for rendering
    """
    rad = math.radians(cast.angle)
    # Bubble launch vector points up and away; vertical is negative (sky is y=0).
    vx = math.cos(rad) * (cast.power / MAX_POWER) * 0.9
    vy = -math.sin(rad) * (cast.power / MAX_POWER) * 1.1
    x, y = float(CASTER_X), float(CASTER_Y)

    coords = [(x, y)]
    hit = False
    hit_index: Optional[int] = None
    steps = 0

    for _ in range(MAX_STEPS):
        x += vx
        y += vy
        vy += GRAVITY
        steps += 1
        coords.append((x, y))

        if y >= CASTER_Y:                # landed back on the ground
            break
        if x < 0 or x > SKY_W:           # drifted out of the sky
            break

        for i, s in enumerate(stars):
            if math.hypot(x - s.x, y - s.y) <= s.radius:
                hit = True
                hit_index = i
                break
        if hit:
            break

    return {"hit": hit, "hit_index": hit_index, "steps": steps, "coords": coords}


def append_record(state: GameState, cast: Cast, result: dict, score_delta: int) -> int:
    """Append one labeled cast sample to the JSONL dataset, returning total rows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "cast",
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": state.session_id,
        "sky": [asdict(s) for s in state.stars],
        "cast": asdict(cast),
        "outcome": {
            "hit": result["hit"],
            "target_hit": result["hit_index"] == cast.target_index,
            "hit_index": result["hit_index"],
            "steps": result["steps"],
            "score_delta": score_delta,
        },
    }
    with DATA_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return stream_count()


def stream_count() -> int:
    """Number of cast samples already gathered in the dataset."""
    if not DATA_PATH.exists():
        return 0
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def fit_aim_model(records=None):
    """
    Least-squares power predictor: power ~ a * distance + b.

    Reads the recorded casts, where `distance` is the horizontal gap between
    the caster and the aimed star. With enough samples the fitted line gives an
    assisted-aim suggestion. Returns (a, b) or None when data is too thin.
    """
    if records is None:
        records = list(_iter_records())
    xs, ys = [], []
    for r in records:
        cast = r.get("cast", {})
        idx = cast.get("target_index")
        sky = r.get("sky", [])
        if idx is None or not (0 <= idx < len(sky)):
            continue
        dx = (sky[idx]["x"] if isinstance(sky[idx], dict) else sky[idx].x) - CASTER_X
        if abs(dx) > 0.01:
            xs.append(abs(dx))
            ys.append(float(cast.get("power", 0)))
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    b = my - a * mx
    return a, b


def _iter_records():
    if not DATA_PATH.exists():
        return
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- Rendering --------------------------------------------------------------

def render_sky(stars: list[Star], bubble=None):
    grid = [[" " for _ in range(SKY_W)] for _ in range(SKY_H)]
    for s in stars:
        if 0 <= int(s.y) < SKY_H and 0 <= int(s.x) < SKY_W:
            grid[int(s.y)][int(s.x)] = s.lit_char
    if bubble is not None:
        x, y = bubble
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < SKY_W and 0 <= yi < SKY_H:
            grid[yi][xi] = "o"
    grid[CASTER_Y][CASTER_X] = "\033[1;35m@\033[0m"
    return "\n".join("".join(row) for row in grid)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def divider(char: str = "─", length: int = 50):
    print(f"\033[90m{char * length}\033[0m")


def header(text: str):
    print(f"\n\033[1;36m{'=' * 50}\033[0m")
    print(f"\033[1;36m{text:^50}\033[0m")
    print(f"\033[1;36m{'=' * 50}\033[0m\n")


def get_number(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            n = int(input(f"  \033[32m{prompt} \033[0m").strip())
            if lo <= n <= hi:
                return n
            print(f"  \033[31mEnter {lo}-{hi}\033[0m")
        except ValueError:
            print(f"  \033[31mEnter a number\033[0m")


def choice_menu(options: list[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  \033[33m[{i}]\033[0m {opt}")
    print()
    return get_number(">", 1, len(options))


def status_line(state: GameState):
    lit = sum(1 for s in state.stars if s.lit)
    total = sum(1 for s in state.stars if s.lit) + sum(
        1 for s in state.stars if not s.lit
    )
    print(f"  \033[90mcasts:\033[0m {state.casts}   \033[90mscore:\033[0m {state.score:>4}   "
          f"\033[90mstars lit:\033[0m {lit}/{total}")


def animate_cast(stars: list[Star], cast: Cast):
    result = simulate_cast(stars, cast)
    for x, y in result["coords"][::2]:
        clear()
        divider()
        print(render_sky(stars, (x, y)))
        divider()
        time.sleep(0.03)
    return result


# --- Game flow --------------------------------------------------------------

def reveal_echo(state: GameState, star: Star):
    wish = WISHES[star.wish_id % len(WISHES)]
    state.echoes.append(wish)
    if len(state.echoes) > 6:
        state.echoes = state.echoes[-6:]
    print(f"\n  \033[1;36m*\033[0m A star lights up and lets a wish be heard:")
    print(f"    \033[90m\"{wish}\"\033[0m")


def cast_turn(state: GameState) -> None:
    clear()
    header("CAST A WISH")
    divider()
    print(render_sky(state.stars))
    divider()
    for i, s in enumerate(state.stars):
        lit = "\033[1;36mlit\033[0m" if s.lit else f"{s.brightness}/{MAX_BRIGHTNESS}"
        print(f"  \033[33m[{i + 1}]\033[0m star at ({s.x:.0f},{s.y:.0f})  brightness {lit}")
    divider()

    target = get_number("Aim at star #", 1, len(state.stars)) - 1
    angle = get_number("Angle (deg, 12-82)", MIN_ANGLE, MAX_ANGLE)
    power = get_number("Power (2-12)", MIN_POWER, MAX_POWER)
    sigil = random.choice(["◆", "❖", "◈", "✦", "◇"])

    cast = Cast(angle=float(angle), power=float(power), sigil=sigil, target_index=target)

    result = animate_cast(state.stars, cast)
    state.casts += 1

    star = state.stars[result["hit_index"]] if result["hit_index"] is not None else None
    if star is None:
        print("  \033[90mThe bubble pops against the empty sky.\033[0m")
        delta = 0
    else:
        star.brightness += 1
        delta = 2
        if star.brightness >= MAX_BRIGHTNESS:
            star.brightness = MAX_BRIGHTNESS
            if not star.lit:
                star.lit = True
                delta = 10
                reveal_echo(state, star)
        else:
            print(f"  \033[33mThe star drinks the bubble and brightens.\033[0m +{delta}")
    state.score += delta

    append_record(state, cast, result, delta)
    time.sleep(1)


def main_loop():
    clear()
    header("BUBBLE CASTER")
    divider()
    print("  Cast wishes up to the stars.\n"
          "  A star that drinks {n} bubbles lights up and\n"
          "  lets one of its wishes be heard.\n".format(n=MAX_BRIGHTNESS))
    print("  Every cast is written to")
    print(f"  \033[90m  {DATA_PATH}\033[0m")
    print(f"  \033[90m  ({stream_count()} casts already gathered)\033[0m")
    divider()
    input("\n  \033[32mPress Enter to begin\033[0m")

    state = GameState(stars=new_sky())
    while not all(s.lit for s in state.stars):
        cast_turn(state)
        clear()
        render_sky(state.stars)
        divider()
        status_line(state)
        if len(state.echoes):
            print("\n  \033[90mEchoes the stars have shared:\033[0m")
            for e in state.echoes:
                print(f"    \033[90m· {e}\033[0m")
        divider()
        if all(s.lit for s in state.stars):
            break
        print()
        options = ["Cast another wish", "Let the sky suggest the power", "Rest and quit"]
        choice = choice_menu(options)
        if choice == 2:
            suggest(state)
        elif choice == 3:
            print("\n  \033[90mThe stars keep your wishes until you return.\033[0m")
            return

    header("NIGHTFALL")
    print(f"\n  \033[1;36mAll {len(state.stars)} stars are lit.\033[0m")
    print(f"  \033[90m{state.casts} casts, score {state.score}, "
          f"{stream_count()} samples in the dataset.\033[0m")
    print("\n  \033[90mOne day the model will read those samples and learn to aim.\033[0m")


def suggest(state: GameState):
    model = fit_aim_model()
    if model is None:
        print("\n  \033[90mThe sky has not watched enough casts yet — keep casting.\033[0m")
        return
    a, b = model
    stars = [s for s in state.stars if not s.lit]
    if not stars:
        return
    s = stars[0]
    power = a * abs(s.x - CASTER_X) + b
    power = max(MIN_POWER, min(MAX_POWER, round(power)))
    print(f"\n  \033[90mThe sky whispers:\033[0m \"aim at ({s.x:.0f},{s.y:.0f}) "
          f"with power \033[1;33m{power}\033[0m\".\n"
          "  (it learned this from your own casts)\033[0m")
    input("\n  \033[32mPress Enter\033[0m")


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n  \033[90mThe sky holds your unfinished wishes.\033[0m\n")
        sys.exit(0)
