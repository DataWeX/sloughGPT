#!/usr/bin/env python3
"""
Bunker — a doomsday shelter management game.
Survive as long as you can. Make choices. Keep your people alive.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


SAVE_DIR = Path.home() / ".bunker_game"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = SAVE_DIR / "save.json"


class Season(Enum):
    SPRING = "Spring"
    SUMMER = "Summer"
    AUTUMN = "Autumn"
    WINTER = "Winter"


@dataclass
class Resources:
    food: int = 100
    water: int = 100
    power: int = 80
    morale: int = 70
    meds: int = 40
    scrap: int = 20

    def __str__(self):
        return (
            f"  \033[33mFood\033[0m: {self.food:>3}  "
            f"\033[36mWater\033[0m: {self.water:>3}  "
            f"\033[93mPower\033[0m: {self.power:>3}\n"
            f"  \033[35mMorale\033[0m: {self.morale:>3}  "
            f"\033[31mMeds\033[0m: {self.meds:>3}  "
            f"\033[37mScrap\033[0m: {self.scrap:>3}"
        )


@dataclass
class Upgrades:
    hydroponics: bool = False
    water_filter: bool = False
    solar_panels: bool = False
    infirmary: bool = False
    workshop: bool = False
    radio: bool = False
    reinforced_door: bool = False


@dataclass
class GameState:
    day: int = 1
    season: Season = Season.SPRING
    population: int = 12
    resources: Resources = field(default_factory=Resources)
    upgrades: Upgrades = field(default_factory=Upgrades)
    days_surface_habitable: int = 0
    scavenge_cooldown: int = 0
    game_over: bool = False
    win: bool = False
    log: list[str] = field(default_factory=list)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def slow_type(text: str, delay: float = 0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def divider(char: str = "─", length: int = 50):
    print(f"\033[90m{char * length}\033[0m")


def header(text: str):
    print(f"\n\033[1;36m{'=' * 50}\033[0m")
    print(f"\033[1;36m{text:^50}\033[0m")
    print(f"\033[1;36m{'=' * 50}\033[0m\n")


def choice_menu(options: list[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  \033[33m[{i}]\033[0m {opt}")
    print()
    while True:
        try:
            choice = input("  \033[32m> \033[0m").strip()
            n = int(choice)
            if 1 <= n <= len(options):
                return n
            print(f"  \033[31mEnter 1-{len(options)}\033[0m")
        except ValueError:
            print("  \033[31mEnter a number\033[0m")


def morning_report(state: GameState):
    clear()
    header(f"Day {state.day} — {state.season.value}")
    season_colors = {
        Season.SPRING: "\033[32m", Season.SUMMER: "\033[33m",
        Season.AUTUMN: "\033[31m", Season.WINTER: "\033[34m",
    }
    print(f"  {season_colors[state.season]}Population\033[0m: {state.population}")
    print(f"  Resources:")
    print(f"  {state.resources}")
    print(f"  Upgrades: ", end="")
    upgrades_list = []
    for name, val in asdict(state.upgrades).items():
        if val:
            label = name.replace("_", " ").title()
            upgrades_list.append(f"\033[36m{label}\033[0m")
    print(", ".join(upgrades_list) if upgrades_list else "\033[90mNone\033[0m")
    print(f"  \033[90mDays until surface habitable: {max(0, 365 - state.days_surface_habitable)}\033[0m")
    divider()


def daily_consumption(state: GameState) -> list[str]:
    msgs = []
    per_person = state.population

    food_cost = per_person
    water_cost = per_person
    power_cost = max(5, per_person // 2)

    if state.upgrades.hydroponics:
        food_cost = max(0, food_cost - 15)
    if state.upgrades.water_filter:
        water_cost = max(0, water_cost - 10)
    if state.upgrades.solar_panels:
        power_cost = max(0, power_cost - 8)

    state.resources.food -= food_cost
    state.resources.water -= water_cost
    state.resources.power -= power_cost

    if food_cost > 0:
        msgs.append(f"Consumed \033[33m{food_cost} food\033[0m")
    if water_cost > 0:
        msgs.append(f"Consumed \033[36m{water_cost} water\033[0m")
    if power_cost > 0:
        msgs.append(f"Consumed \033[93m{power_cost} power\033[0m")

    return msgs


def check_shortages(state: GameState) -> list[str]:
    msgs = []
    if state.resources.food <= 0:
        state.resources.food = 0
        deaths = random.randint(1, 3)
        state.population = max(1, state.population - deaths)
        state.resources.morale -= 15
        msgs.append(f"\033[31m{deaths} died from starvation!\033[0m")
    if state.resources.water <= 0:
        state.resources.water = 0
        deaths = random.randint(1, 3)
        state.population = max(1, state.population - deaths)
        state.resources.morale -= 15
        msgs.append(f"\033[31m{deaths} died from dehydration!\033[0m")
    if state.resources.power <= 0:
        state.resources.power = 0
        state.resources.morale -= 10
        msgs.append(f"\033[31mPower failure — morale drops\033[0m")
    if state.resources.morale <= 0:
        state.resources.morale = 0
        deaths = random.randint(1, 2)
        state.population = max(1, state.population - deaths)
        msgs.append(f"\033[31m{deaths} lost hope and left the bunker!\033[0m")
    if state.resources.meds <= 0:
        state.resources.meds = 0
        msgs.append(f"\033[31mNo medicine left!\033[0m")
    return msgs


EVENTS = [
    {
        "title": "Stranger at the Door",
        "desc": "A survivor is pounding on the airlock. They look injured but could be a threat.",
        "choices": [
            ("Let them in", "stranger_let_in"),
            ("Turn them away", "stranger_turn_away"),
        ],
    },
    {
        "title": "Geiger Counter Spikes",
        "desc": "Radiation levels on the surface are spiking. The ventilation system is straining.",
        "choices": [
            ("Seal ventilation (uses scrap)", "geiger_seal"),
            ("Risk it", "geiger_risk"),
        ],
    },
    {
        "title": "Underground Spring",
        "desc": "While digging an expansion tunnel, your team found an underground water spring!",
        "choices": [
            ("Drill and collect", "spring_drill"),
            ("Save the discovery for later", "spring_save"),
        ],
    },
    {
        "title": "Mutiny Brewing",
        "desc": "A faction within the bunker is unhappy with your leadership and planning a takeover.",
        "choices": [
            ("Address concerns openly", "mutiny_speak"),
            ("Exile the ringleader", "mutiny_exile"),
        ],
    },
    {
        "title": "Scavenge Team Returns",
        "desc": "Your surface scavenge team came back with a haul!",
        "choices": [
            ("Sort through supplies", "scavenge_sort"),
            ("Send them back out immediately", "scavenge_send"),
        ],
    },
    {
        "title": "Equipment Failure",
        "desc": "The water reclamation system is broken! You need scrap to fix it.",
        "choices": [
            ("Repair with scrap", "equip_repair"),
            ("Rig a temporary fix", "equip_rig"),
        ],
    },
    {
        "title": "Radio Signal",
        "desc": "You picked up a faint radio signal. Another bunker? A government broadcast?",
        "choices": [
            ("Try to respond", "radio_respond"),
            ("Stay silent — conserve power", "radio_silent"),
        ],
    },
    {
        "title": "Birth in the Bunker",
        "desc": "A baby is being born! This could lift everyone's spirits — or strain resources.",
        "choices": [
            ("Celebrate — boost morale", "birth_celebrate"),
            ("Focus on practicality", "birth_practical"),
        ],
    },
    {
        "title": "Raiders Approach",
        "desc": "Armed raiders have been spotted near the bunker entrance.",
        "choices": [
            ("Barricade and hide", "raiders_barricade"),
            ("Fight them off", "raiders_fight"),
        ],
    },
]


def resolve_event(state: GameState, event_id: str) -> str:
    """Resolve a named event choice and return a result string."""
    if event_id == "stranger_let_in":
        state.population += 1
        state.resources.food = max(0, state.resources.food - 5)
        state.resources.morale = min(100, state.resources.morale + 5)
        return "\033[32mA new survivor joins! Food strain increases slightly.\033[0m"
    elif event_id == "stranger_turn_away":
        state.resources.morale = max(0, state.resources.morale - 5)
        return "\033[31mYour people are uneasy about your decision.\033[0m"
    elif event_id == "geiger_seal":
        state.resources.scrap = max(0, state.resources.scrap - 5)
        return "\033[36mVentilation sealed. Everyone is safe.\033[0m"
    elif event_id == "geiger_risk":
        state.resources.morale = max(0, state.resources.morale - 8)
        state.resources.meds = max(0, state.resources.meds - 5)
        return "\033[31mSome people got sick from radiation exposure.\033[0m"
    elif event_id == "spring_drill":
        state.resources.water = min(200, state.resources.water + 40)
        state.resources.power = max(0, state.resources.power - 5)
        return "\033[36mFresh water collected! +40 water.\033[0m"
    elif event_id == "spring_save":
        return "\033[90mYou mark the location on the map.\033[0m"
    elif event_id == "mutiny_speak":
        state.resources.morale = min(100, state.resources.morale + 10)
        if random.random() < 0.7:
            return "\033[32mYou calmed the dissenters.\033[0m"
        else:
            return "\033[31mThey weren't convinced. Tensions remain.\033[0m"
    elif event_id == "mutiny_exile":
        state.population = max(1, state.population - 1)
        state.resources.morale = min(100, state.resources.morale + 5)
        return "\033[31mOne expelled. Order restored for now.\033[0m"
    elif event_id == "scavenge_sort":
        state.resources.food = min(200, state.resources.food + random.randint(5, 20))
        state.resources.scrap = min(100, state.resources.scrap + random.randint(3, 12))
        state.resources.meds = min(100, state.resources.meds + random.randint(2, 8))
        return "\033[32mSupplies sorted and stored.\033[0m"
    elif event_id == "scavenge_send":
        state.resources.morale = max(0, state.resources.morale - 5)
        return "\033[31mThe team is exhausted but complying.\033[0m"
    elif event_id == "equip_repair":
        state.resources.scrap = max(0, state.resources.scrap - 8)
        return "\033[36mSystem repaired. Water production restored.\033[0m"
    elif event_id == "equip_rig":
        state.resources.water = max(0, state.resources.water - 15)
        state.resources.power = max(0, state.resources.power - 10)
        return "\033[31mThe rig works but drains power and wastes water.\033[0m"
    elif event_id == "radio_respond":
        if random.random() < 0.4:
            state.upgrades.radio = True
            state.resources.morale = min(100, state.resources.morale + 15)
            return "\033[32mYou made contact! Other survivors are out there. +morale\033[0m"
        else:
            return "\033[31mNo response. The signal was a loop.\033[0m"
    elif event_id == "radio_silent":
        state.resources.power = min(200, state.resources.power + 5)
        return "\033[90mPower conserved.\033[0m"
    elif event_id == "birth_celebrate":
        state.population += 1
        state.resources.morale = min(100, state.resources.morale + 20)
        return "\033[32mA new life in the bunker! Everyone celebrates.\033[0m"
    elif event_id == "birth_practical":
        state.population += 1
        return "\033[90mA new addition. Quietly welcomed.\033[0m"
    elif event_id == "raiders_barricade":
        if state.upgrades.reinforced_door:
            return "\033[32mThe reinforced door holds. Raiders move on.\033[0m"
        else:
            state.resources.scrap = max(0, state.resources.scrap - 10)
            return "\033[31mYou hurriedly reinforce the door. They move on but took some scrap.\033[0m"
    elif event_id == "raiders_fight":
        casualties = random.randint(0, 2)
        state.population = max(1, state.population - casualties)
        state.resources.morale = max(0, state.resources.morale + 5)
        state.resources.scrap = min(100, state.resources.scrap + random.randint(3, 10))
        return "\033[33mYou fought them off. Casualties but gained some scrap.\033[0m"
    return "\033[90mNothing happens.\033[0m"

UPGRADE_EVENTS = [
    {
        "upgrade": "hydroponics",
        "label": "Hydroponics Bay",
        "cost": 25,
        "desc": "Grow food underground. Reduces daily food consumption.",
        "effect": "Reduces food consumption by 15/day.",
    },
    {
        "upgrade": "water_filter",
        "label": "Advanced Water Filter",
        "cost": 20,
        "desc": "Purifies more water from the aquifer.",
        "effect": "Reduces water consumption by 10/day.",
    },
    {
        "upgrade": "solar_panels",
        "label": "Solar Panel Array",
        "cost": 30,
        "desc": "Harvest surface sunlight for power.",
        "effect": "Reduces power consumption by 8/day.",
    },
    {
        "upgrade": "infirmary",
        "label": "Infirmary",
        "cost": 15,
        "desc": "Treat injuries and illness more effectively.",
        "effect": "Bonus morale recovery and fewer deaths from sickness.",
    },
    {
        "upgrade": "workshop",
        "label": "Workshop",
        "cost": 20,
        "desc": "Craft tools and repair equipment.",
        "effect": "Scrap is more effective for repairs.",
    },
    {
        "upgrade": "radio",
        "label": "Radio Tower",
        "cost": 10,
        "desc": "Communicate with the outside world.",
        "effect": "Unlocks new events and potential contact.",
    },
]


def workshop_action(state: GameState):
    """Open the workshop for crafting upgrades."""
    while True:
        clear()
        header(f"Workshop — Day {state.day}")
        print(f"  \033[37mScrap available: {state.resources.scrap}\033[0m\n")
        options = []
        upgrade_map = []

        for u in UPGRADE_EVENTS:
            already_have = getattr(state.upgrades, u["upgrade"])
            label = f"{u['label']}  (\033[33m{u['cost']} scrap\033[0m)"
            if already_have:
                label += "  \033[32m✓ Built\033[0m"
            else:
                label += f"  — {u['effect']}"
            options.append(label)
            upgrade_map.append(u)

        options.append("\033[90mBack to main menu\033[0m")

        choice = choice_menu(options)
        if choice == len(options):
            return

        u = upgrade_map[choice - 1]
        already_have = getattr(state.upgrades, u["upgrade"])
        if already_have:
            slow_type("\n  \033[33mAlready built!\033[0m")
            time.sleep(1)
            continue
        if state.resources.scrap < u["cost"]:
            slow_type(f"\n  \033[31mNeed {u['cost']} scrap — you have {state.resources.scrap}\033[0m")
            time.sleep(1.5)
            continue

        state.resources.scrap -= u["cost"]
        setattr(state.upgrades, u["upgrade"], True)
        slow_type(f"\n  \033[32m{u['label']} built!\033[0m")
        time.sleep(1.5)


def scavenge(state: GameState):
    """Go on a scavenging mission to the surface."""
    if state.scavenge_cooldown > 0:
        slow_type(f"\n  \033[31mScavenge team is resting ({state.scavenge_cooldown} days)\033[0m")
        time.sleep(1.5)
        return

    clear()
    header(f"Surface Scavenge — Day {state.day}")
    slow_type("You send a team to the surface to look for supplies...\n")
    time.sleep(0.5)

    risk = random.random()
    found_food = random.randint(5, 25)
    found_scrap = random.randint(2, 10)
    found_meds = random.randint(0, 5)

    if state.upgrades.workshop:
        found_scrap += 5

    if state.upgrades.radio:
        found_food += 5
        found_scrap += 3

    state.resources.food = min(200, state.resources.food + found_food)
    state.resources.scrap = min(100, state.resources.scrap + found_scrap)
    state.resources.meds = min(100, state.resources.meds + found_meds)

    print(f"  Found: \033[33m+{found_food} food\033[0m, \033[37m+{found_scrap} scrap\033[0m", end="")
    if found_meds:
        print(f", \033[31m+{found_meds} meds\033[0m", end="")
    print()

    if risk < 0.2:
        casualty = random.randint(0, 1)
        if casualty:
            state.population = max(1, state.population - 1)
            slow_type(f"\n  \033[31mOne of the scavengers didn't make it back.\033[0m")
        else:
            slow_type(f"\n  \033[33mThe team encountered danger but escaped unharmed.\033[0m")
        state.resources.morale = max(0, state.resources.morale - 5)
    elif risk < 0.05:
        state.resources.power = max(0, state.resources.power + random.randint(5, 15))
        slow_type(f"\n  \033[36mThey found spare batteries! +power\033[0m")

    state.scavenge_cooldown = 3
    time.sleep(2)


def trigger_random_event(state: GameState) -> Optional[str]:
    """Trigger a random event with player choices."""
    event = random.choice(EVENTS)
    clear()
    header(f"⚡ Event — Day {state.day}")
    print(f"  \033[1;37m{event['title']}\033[0m\n")
    slow_type(f"  {event['desc']}\n", delay=0.03)
    divider()
    print()

    options = [c[0] for c in event["choices"]]
    choice = choice_menu(options)

    _, event_id = event["choices"][choice - 1]
    result = resolve_event(state, event_id)

    print()
    slow_type(f"  {result}", delay=0.03)
    time.sleep(2)
    return result


def season_advance(state: GameState):
    """Advance season every 30 days."""
    seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
    idx = (state.day // 30) % 4
    state.season = seasons[idx]

    if state.day > 0 and state.day % 30 == 0:
        clear()
        header(f"Season Change — {state.season.value}")
        if state.season == Season.WINTER:
            slow_type("  Winter sets in. Surface temperatures plummet.\n  Power consumption increases.\n")
            state.resources.power -= 10
        elif state.season == Season.SUMMER:
            slow_type("  Summer heat. Water evaporates faster.\n")
            state.resources.water -= 5
        elif state.season == Season.SPRING:
            slow_type("  Spring thaws. The surface shows signs of recovery.\n")
            state.days_surface_habitable += 30
        time.sleep(2)


def check_win(state: GameState) -> bool:
    """Check if surface is habitable."""
    if state.days_surface_habitable >= 365:
        state.game_over = True
        state.win = True
        return True
    return False


def save_game(state: GameState):
    data = asdict(state)
    SAVE_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n  \033[90mGame saved.\033[0m")


def load_game() -> Optional[GameState]:
    if SAVE_PATH.exists():
        try:
            data = json.loads(SAVE_PATH.read_text())
            return GameState(**data)
        except Exception:
            pass
    return None


def main_loop():
    clear()
    saved = load_game()

    title_art = r"""
    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║     ██████  ██    ██ ███    ██       ║
    ║     ██   ██ ██    ██ ████   ██       ║
    ║     ██████  ██    ██ ██ ██  ██       ║
    ║     ██   ██ ██    ██ ██  ██ ██       ║
    ║     ██████   ██████  ██   ████       ║
    ║                                       ║
    ║        B U N K E R   S U R V I V A L ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """

    if saved:
        print(title_art)
        print(f"\n  \033[36mSave found — Day {saved.day}\033[0m")
        print()
        options = ["Continue", "New game", "Quit"]
        choice = choice_menu(options)
        if choice == 1:
            state = saved
            slow_type("\n  \033[32mWelcome back, Commander.\033[0m")
            time.sleep(1)
        elif choice == 2:
            state = GameState()
            if SAVE_PATH.exists():
                SAVE_PATH.unlink()
            slow_type("\n  \033[33mA new world awaits.\033[0m")
            time.sleep(1)
        else:
            print("\n  Goodbye, Commander.")
            sys.exit(0)
    else:
        print(title_art)
        print()
        slow_type("  The sirens faded hours ago.\n  The last news broadcast mentioned\n  something about nuclear winter\n  lasting for years.\n")
        time.sleep(1)
        slow_type("  \033[36mYou are the bunker commander now.\n  12 people. Limited supplies.\n  Keep everyone alive.\033[0m\n")
        time.sleep(1.5)
        state = GameState()

    while not state.game_over:
        morning_report(state)

        consume_msgs = daily_consumption(state)
        for msg in consume_msgs:
            print(f"  {msg}")

        shortage_msgs = check_shortages(state)
        for msg in shortage_msgs:
            print(f"  {msg}")

        if state.population <= 0:
            state.game_over = True
            break

        season_advance(state)
        if check_win(state):
            break

        if state.scavenge_cooldown > 0:
            state.scavenge_cooldown -= 1

        divider()
        print()

        if random.random() < 0.4:
            trigger_random_event(state)

        # Main action menu
        print()
        divider("─", 30)
        print("  \033[1;37mWhat do you do?\033[0m")
        print()
        options = [
            "Scavenge the surface",
            "Workshop — build upgrades",
            "Wait — conserve resources",
        ]
        choice = choice_menu(options)

        if choice == 1:
            scavenge(state)
        elif choice == 2:
            workshop_action(state)
        else:
            slow_type("  \033[90mAnother day in the bunker. Rest and conserve.\033[0m")
            state.resources.morale = min(100, state.resources.morale + 2)
            state.resources.power = min(200, state.resources.power + 3)
            time.sleep(1)

        state.day += 1
        save_game(state)

    # Game over
    clear()
    if state.win:
        header("🎉 VICTORY")
        slow_type(f"  After {state.day} days underground,\n  the surface is finally habitable.\n")
        time.sleep(0.5)
        slow_type(f"  \033[32m{state.population} survivors emerge into the sunlight.\033[0m\n")
        time.sleep(0.5)
        slow_type("  \033[36mYou led them through the darkest days.\n  The world can begin again.\033[0m\n")
    else:
        header("💀 GAME OVER")
        if state.population <= 0:
            slow_type("  Everyone is gone.\n  The bunker is silent.\n")
        else:
            slow_type(f"  After {state.day} days, the bunker fell.\n  {state.population} survivors remain.\n")

    print(f"  \033[90mDays survived: {state.day}\033[0m")
    print(f"  \033[90mFinal population: {state.population}\033[0m")
    print()

    if SAVE_PATH.exists():
        SAVE_PATH.unlink()

    options = ["Play again", "Quit"]
    choice = choice_menu(options)
    if choice == 1:
        main_loop()


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n  Goodbye, Commander.\n")
        sys.exit(0)
