"""
Phoneme Encoder — rule-based grapheme-to-phoneme for English TTS.

Maps text to a compact phoneme vocabulary. Not perfect, but dramatically
better than raw byte values (ord(c) % 256).

Phoneme set based on ARPAbet (simplified).
"""

from __future__ import annotations

import re
import numpy as np


# ── Phoneme inventory ──────────────────────────────────────────────────────

# Vowels (monophthongs + diphthongs)
VOWELS = {
    "AA": 0,   # father
    "AE": 1,   # cat
    "AH": 2,   # hut
    "AO": 3,   # thought
    "AW": 4,   # cow
    "AY": 5,   # bike
    "EH": 6,   # bet
    "ER": 7,   # bird
    "EY": 8,   # bake
    "IH": 9,   # bit
    "IY": 10,  # beat
    "OW": 11,  # boat
    "OY": 12,  # boy
    "UH": 13,  # book
    "UW": 14,  # boot
}

# Consonants
CONSONANTS = {
    "B": 15,   # bay
    "CH": 16,  # chin
    "D": 17,   # day
    "DH": 18,  # then
    "F": 19,   # fat
    "G": 20,   # go
    "HH": 21,  # hat
    "JH": 22,  # joy
    "K": 23,   # key
    "L": 24,   # law
    "M": 25,   # met
    "N": 26,   # net
    "NG": 27,  # sing
    "P": 28,   # pay
    "R": 29,   # ray
    "S": 30,   # sea
    "SH": 31,  # she
    "T": 32,   # tea
    "TH": 33,  # think
    "V": 34,   # vat
    "W": 35,   # way
    "Y": 36,   # yes
    "Z": 37,   # zoo
    "ZH": 38,  # vision
}

# Special tokens
PAD = 39
BOS = 40
EOS = 41
SPACE = 42
SILENCE = 43

NUM_PHONEMES = 44

# Reverse lookup
ID_TO_PHONEME = {v: k for k, v in {**VOWELS, **CONSONANTS}.items()}
ID_TO_PHONEME[PAD] = "_"
ID_TO_PHONEME[BOS] = "<"
ID_TO_PHONEME[EOS] = ">"
ID_TO_PHONEME[SPACE] = " "
ID_TO_PHONEME[SILENCE] = "-"


# ── Grapheme-to-Phoneme rules ─────────────────────────────────────────────

# Common English words with known pronunciations (top ~200 high-frequency)
_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "the": ["DH", "AH"],
    "a": ["AH"],
    "an": ["AE", "N"],
    "is": ["IH", "Z"],
    "it": ["IH", "T"],
    "to": ["T", "UW"],
    "of": ["AH", "V"],
    "and": ["AE", "N", "D"],
    "in": ["IH", "N"],
    "that": ["DH", "AE", "T"],
    "have": ["HH", "AE", "V"],
    "i": ["AY"],
    "for": ["F", "AO", "R"],
    "not": ["N", "AA", "T"],
    "on": ["AA", "N"],
    "with": ["W", "IH", "DH"],
    "he": ["HH", "IY"],
    "as": ["AE", "Z"],
    "you": ["Y", "UW"],
    "do": ["D", "UW"],
    "at": ["AE", "T"],
    "this": ["DH", "IH", "S"],
    "but": ["B", "AH", "T"],
    "his": ["HH", "IH", "Z"],
    "by": ["B", "AY"],
    "from": ["F", "R", "AH", "M"],
    "or": ["AO", "R"],
    "one": ["W", "AH", "N"],
    "had": ["HH", "AE", "D"],
    "what": ["W", "AH", "T"],
    "were": ["W", "ER"],
    "all": ["AO", "L"],
    "their": ["DH", "EH", "R"],
    "we": ["W", "IY"],
    "when": ["W", "EH", "N"],
    "your": ["Y", "AO", "R"],
    "can": ["K", "AE", "N"],
    "said": ["S", "EH", "D"],
    "there": ["DH", "EH", "R"],
    "use": ["Y", "UW", "Z"],
    "each": ["IY", "CH"],
    "which": ["W", "IH", "CH"],
    "she": ["SH", "IY"],
    "do": ["D", "UW"],
    "how": ["HH", "AW"],
    "their": ["DH", "EH", "R"],
    "if": ["IH", "F"],
    "will": ["W", "IH", "L"],
    "up": ["AH", "P"],
    "about": ["AH", "B", "AW", "T"],
    "out": ["AW", "T"],
    "many": ["M", "EH", "N", "IY"],
    "then": ["DH", "EH", "N"],
    "these": ["DH", "IY", "Z"],
    "so": ["S", "OW"],
    "some": ["S", "AH", "M"],
    "would": ["W", "UH", "D"],
    "make": ["M", "EY", "K"],
    "like": ["L", "AY", "K"],
    "him": ["HH", "IH", "M"],
    "into": ["IH", "N", "T", "UW"],
    "time": ["T", "AY", "M"],
    "has": ["HH", "AE", "Z"],
    "look": ["L", "UH", "K"],
    "two": ["T", "UW"],
    "more": ["M", "AO", "R"],
    "go": ["G", "OW"],
    "see": ["S", "IY"],
    "no": ["N", "OW"],
    "way": ["W", "EY"],
    "could": ["K", "UH", "D"],
    "my": ["M", "AY"],
    "than": ["DH", "AE", "N"],
    "first": ["F", "ER", "S", "T"],
    "been": ["B", "IH", "N"],
    "call": ["K", "AO", "L"],
    "who": ["HH", "UW"],
    "its": ["IH", "T", "S"],
    "now": ["N", "AW"],
    "find": ["F", "AY", "N", "D"],
    "long": ["L", "AO", "NG"],
    "down": ["D", "AW", "N"],
    "day": ["D", "EY"],
    "did": ["D", "IH", "D"],
    "get": ["G", "EH", "T"],
    "come": ["K", "AH", "M"],
    "made": ["M", "EY", "D"],
    "may": ["M", "EY"],
    "hello": ["HH", "EH", "L", "OW"],
    "yes": ["Y", "EH", "S"],
    "no": ["N", "OW"],
    "please": ["P", "L", "IY", "Z"],
    "thank": ["TH", "AE", "NG", "K"],
    "thanks": ["TH", "AE", "NG", "K", "S"],
    "sorry": ["S", "AA", "R", "IY"],
    "okay": ["OW", "K", "EY"],
    "right": ["R", "AY", "T"],
    "know": ["N", "OW"],
    "think": ["TH", "IH", "NG", "K"],
    "want": ["W", "AA", "N", "T"],
    "need": ["N", "IY", "D"],
    "good": ["G", "UH", "D"],
    "bad": ["B", "AE", "D"],
    "big": ["B", "IH", "G"],
    "small": ["S", "M", "AO", "L"],
    "new": ["N", "UW"],
    "old": ["OW", "L", "D"],
    "here": ["HH", "IH", "R"],
    "where": ["W", "EH", "R"],
    "when": ["W", "EH", "N"],
    "why": ["W", "AY"],
    "how": ["HH", "AW"],
    "what": ["W", "AH", "T"],
    "who": ["HH", "UW"],
    "name": ["N", "EY", "M"],
    "people": ["P", "IY", "P", "AH", "L"],
    "quick": ["K", "W", "IH", "K"],
    "brown": ["B", "R", "AW", "N"],
    "fox": ["F", "AO", "K", "S"],
    "they": ["DH", "EH"],
    "am": ["AE", "M"],
    "are": ["AA", "R"],
    "was": ["W", "AA", "Z"],
    "be": ["B", "IY"],
    "does": ["D", "AH", "Z"],
    "should": ["SH", "UH", "D"],
    "might": ["M", "AY", "T"],
    "shall": ["SH", "AE", "L"],
    "must": ["M", "AH", "S", "T"],
    "take": ["T", "EY", "K"],
    "give": ["G", "IH", "V"],
    "say": ["S", "EY"],
    "tell": ["T", "EH", "L"],
    "ask": ["AE", "S", "K"],
    "love": ["L", "AH", "V"],
    "hate": ["HH", "EY", "T"],
    "hear": ["HH", "IH", "R"],
    "eat": ["IY", "T"],
    "drink": ["D", "R", "IH", "NG", "K"],
    "sleep": ["S", "L", "IY", "P"],
    "walk": ["W", "AO", "K"],
    "run": ["R", "AH", "N"],
    "talk": ["T", "AO", "K"],
    "play": ["P", "L", "EY"],
    "help": ["HH", "EH", "L", "P"],
    "try": ["T", "R", "AY"],
    "start": ["S", "T", "AA", "R", "T"],
    "stop": ["S", "T", "AA", "P"],
    "goodbye": ["G", "UH", "D", "B", "AY"],
    "ok": ["OW", "K", "EY"],
    "those": ["DH", "OW", "Z"],
    "always": ["AO", "L", "W", "AY", "Z"],
    "never": ["N", "EH", "V", "ER"],
    "today": ["T", "AH", "D", "EY"],
    "tomorrow": ["T", "AH", "M", "AA", "R", "OW"],
    "yesterday": ["Y", "EH", "S", "T", "ER", "D", "EY"],
    "morning": ["M", "AO", "R", "N", "IH", "NG"],
    "afternoon": ["AE", "F", "T", "ER", "N", "UW"],
    "evening": ["IY", "V", "N", "IH", "NG"],
    "hot": ["HH", "AA", "T"],
    "cold": ["K", "OW", "L", "D"],
    "fast": ["F", "AE", "S", "T"],
    "slow": ["S", "L", "OW"],
    "easy": ["IY", "Z", "IY"],
    "hard": ["HH", "AA", "R", "D"],
    "time": ["T", "AY", "M"],
    "year": ["Y", "IH", "R"],
    "day": ["D", "EY"],
    "world": ["W", "ER", "L", "D"],
    "life": ["L", "AY", "F"],
    "hand": ["HH", "AE", "N", "D"],
    "part": ["P", "AA", "R", "T"],
    "place": ["P", "L", "EY", "S"],
    "case": ["K", "EY", "S"],
    "week": ["W", "IY", "K"],
    "company": ["K", "AH", "M", "P", "AH", "N", "IY"],
    "system": ["S", "IH", "S", "T", "AH", "M"],
    "program": ["P", "R", "OW", "G", "R", "AE", "M"],
    "question": ["K", "W", "EH", "S", "CH", "AH", "N"],
    "work": ["W", "ER", "K"],
    "government": ["G", "AH", "V", "ER", "N", "M", "AH", "N", "T"],
    "number": ["N", "AH", "M", "B", "ER"],
    "night": ["N", "AY", "T"],
    "point": ["P", "OY", "N", "T"],
    "home": ["HH", "OW", "M"],
    "water": ["W", "AO", "T", "ER"],
    "room": ["R", "UW", "M"],
    "mother": ["M", "AH", "DH", "ER"],
    "area": ["EH", "R", "IY", "AH"],
    "money": ["M", "AH", "N", "IY"],
    "story": ["S", "T", "AO", "R", "IY"],
    "fact": ["F", "AE", "K", "T"],
    "month": ["M", "AH", "N", "TH"],
    "lot": ["L", "AA", "T"],
    "right": ["R", "AY", "T"],
    "study": ["S", "T", "AH", "D", "IY"],
    "book": ["B", "UH", "K"],
    "eye": ["AY"],
    "job": ["JH", "AA", "B"],
    "word": ["W", "ER", "D"],
    "business": ["B", "IH", "Z", "N", "AH", "S"],
    "issue": ["IH", "SH", "UW"],
    "side": ["S", "AY", "D"],
    "kind": ["K", "AY", "N", "D"],
    "head": ["HH", "EH", "D"],
    "house": ["HH", "AW", "S"],
    "service": ["S", "ER", "V", "IH", "S"],
    "friend": ["F", "R", "EH", "N", "D"],
    "father": ["F", "AA", "DH", "ER"],
    "power": ["P", "AW", "ER"],
    "hour": ["AW", "ER"],
    "game": ["G", "EY", "M"],
    "line": ["L", "AY", "N"],
    "end": ["EH", "N", "D"],
    "member": ["M", "EH", "M", "B", "ER"],
    "law": ["L", "AO"],
    "car": ["K", "AA", "R"],
    "city": ["S", "IH", "T", "IY"],
    "community": ["K", "AH", "M", "Y", "UW", "N", "IH", "T", "IY"],
    "name": ["N", "EY", "M"],
    "president": ["P", "R", "EH", "Z", "AH", "N", "T"],
    "team": ["T", "IY", "M"],
    "minute": ["M", "IH", "N", "AH", "T"],
    "idea": ["AY", "D", "IY", "AH"],
    "body": ["B", "AA", "D", "IY"],
    "information": ["IH", "N", "F", "ER", "M", "EY", "SH", "AH", "N"],
    "river": ["R", "IH", "V", "ER"],
    "land": ["L", "AE", "N", "D"],
    "art": ["AA", "R", "T"],
    "food": ["F", "UW", "D"],
    "moment": ["M", "OW", "M", "AH", "N", "T"],
    "air": ["EH", "R"],
    "teacher": ["T", "IY", "CH", "ER"],
    "force": ["F", "AO", "R", "S"],
    "education": ["EH", "JH", "UH", "K", "EY", "SH", "AH", "N"],
    "computer": ["K", "AH", "M", "P", "Y", "UW", "T", "ER"],
    "internet": ["IH", "N", "T", "ER", "N", "EH", "T"],
    "phone": ["F", "OW", "N"],
    "email": ["IY", "M", "EY", "L"],
    "message": ["M", "EH", "S", "IH", "JH"],
    "music": ["M", "Y", "UW", "Z", "IH", "K"],
    "movie": ["M", "UW", "V", "IY"],
    "read": ["R", "IY", "D"],
    "write": ["R", "AY", "T"],
    "learn": ["L", "ER", "N"],
    "teach": ["T", "IY", "CH"],
    "school": ["S", "K", "UW", "L"],
    "student": ["S", "T", "UW", "D", "AH", "N", "T"],
    "class": ["K", "L", "AE", "S"],
    "doctor": ["D", "AA", "K", "T", "ER"],
    "hospital": ["HH", "AA", "S", "P", "IH", "T", "AH", "L"],
    "police": ["P", "AH", "L", "IY", "S"],
    "fire": ["F", "AY", "ER"],
    "emergency": ["IH", "M", "ER", "JH", "AH", "N", "S", "IY"],
    "weather": ["W", "EH", "DH", "ER"],
    "rain": ["R", "EY", "N"],
    "snow": ["S", "N", "OW"],
    "wind": ["W", "IH", "N", "D"],
    "sun": ["S", "AH", "N"],
    "moon": ["M", "UW", "N"],
    "star": ["S", "T", "AA", "R"],
    "sky": ["S", "K", "AY"],
    "tree": ["T", "R", "IY"],
    "flower": ["F", "L", "AW", "ER"],
    "animal": ["AE", "N", "IH", "M", "AH", "L"],
    "dog": ["D", "AA", "G"],
    "cat": ["K", "AE", "T"],
    "bird": ["B", "ER", "D"],
    "fish": ["F", "IH", "SH"],
    "color": ["K", "AH", "L", "ER"],
    "red": ["R", "EH", "D"],
    "blue": ["B", "L", "UW"],
    "green": ["G", "R", "IY", "N"],
    "yellow": ["Y", "EH", "L", "OW"],
    "black": ["B", "L", "AE", "K"],
    "white": ["W", "AY", "T"],
    "one": ["W", "AH", "N"],
    "two": ["T", "UW"],
    "three": ["TH", "R", "IY"],
    "four": ["F", "AO", "R"],
    "five": ["F", "AY", "V"],
    "six": ["S", "IH", "K", "S"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"],
    "nine": ["N", "AY", "N"],
    "ten": ["T", "EH", "N"],
    "hundred": ["HH", "AH", "N", "D", "R", "AH", "D"],
    "thousand": ["TH", "AW", "Z", "AH", "N", "D"],
    "million": ["M", "IH", "L", "Y", "AH", "N"],
    "billion": ["B", "IH", "L", "Y", "AH", "N"],
    "zero": ["Z", "IH", "R", "OW"],
    # Additional common words
    "apple": ["AE", "P", "AH", "L"],
    "banana": ["B", "AH", "N", "AE", "N", "AH"],
    "orange": ["AO", "R", "IH", "N", "JH"],
    "grape": ["G", "R", "EY", "P"],
    "bread": ["B", "R", "EH", "D"],
    "milk": ["M", "IH", "L", "K"],
    "coffee": ["K", "AH", "F", "IY"],
    "tea": ["T", "IY"],
    "juice": ["JH", "UW", "S"],
    "chair": ["CH", "EH", "R"],
    "table": ["T", "EY", "B", "AH", "L"],
    "door": ["D", "AO", "R"],
    "window": ["W", "IH", "N", "D", "OW"],
    "bed": ["B", "EH", "D"],
    "bath": ["B", "AE", "TH"],
    "car": ["K", "AA", "R"],
    "bus": ["B", "AH", "S"],
    "train": ["T", "R", "EY", "N"],
    "plane": ["P", "L", "EY", "N"],
    "bike": ["B", "AY", "K"],
    "book": ["B", "UH", "K"],
    "pen": ["P", "EH", "N"],
    "paper": ["P", "EY", "P", "ER"],
    "phone": ["F", "OW", "N"],
    "computer": ["K", "AH", "M", "P", "Y", "UW", "T", "ER"],
    "internet": ["IH", "N", "T", "ER", "N", "EH", "T"],
    "email": ["IY", "M", "EY", "L"],
    "website": ["W", "EH", "B", "S", "AY", "T"],
    "video": ["V", "IH", "D", "IY", "OW"],
    "music": ["M", "Y", "UW", "Z", "IH", "K"],
    "movie": ["M", "UW", "V", "IY"],
    "game": ["G", "EY", "M"],
    "sport": ["S", "P", "AO", "R", "T"],
    "team": ["T", "IY", "M"],
    "player": ["P", "L", "EY", "ER"],
    "coach": ["K", "OW", "CH"],
    "win": ["W", "IH", "N"],
    "lose": ["L", "UW", "Z"],
    "draw": ["D", "R", "AO"],
    "score": ["S", "K", "AO", "R"],
    "goal": ["G", "OW", "L"],
    "ball": ["B", "AO", "L"],
    "race": ["R", "EY", "S"],
    "jump": ["JH", "AH", "M", "P"],
    "swim": ["S", "W", "IH", "M"],
    "dance": ["D", "AE", "N", "S"],
    "sing": ["S", "IH", "NG"],
    "song": ["S", "AO", "NG"],
    "picture": ["P", "IH", "K", "CH", "ER"],
    "photo": ["F", "OW", "T", "OW"],
    "map": ["M", "AE", "P"],
    "box": ["B", "AA", "K", "S"],
    "bag": ["B", "AE", "G"],
    "hat": ["HH", "AE", "T"],
    "shoe": ["SH", "UW"],
    "shirt": ["SH", "ER", "T"],
    "coat": ["K", "OW", "T"],
    "dress": ["D", "R", "EH", "S"],
    "pants": ["P", "AE", "N", "T", "S"],
    "sock": ["S", "AA", "K"],
    "ring": ["R", "IH", "NG"],
    "watch": ["W", "AA", "CH"],
    "key": ["K", "IY"],
    "lock": ["L", "AA", "K"],
    "knife": ["N", "AY", "F"],
    "fork": ["F", "AO", "R", "K"],
    "spoon": ["S", "P", "UW", "N"],
    "plate": ["P", "L", "EY", "T"],
    "cup": ["K", "AH", "P"],
    "glass": ["G", "L", "AE", "S"],
    "bottle": ["B", "AA", "T", "AH", "L"],
    "can": ["K", "AE", "N"],
    "jar": ["JH", "AA", "R"],
    "bowl": ["B", "OW", "L"],
    "pan": ["P", "AE", "N"],
    "pot": ["P", "AA", "T"],
    "oven": ["AH", "V", "AH", "N"],
    "stove": ["S", "T", "OW", "V"],
    "fridge": ["F", "R", "IH", "JH"],
    "freezer": ["F", "R", "IY", "Z", "ER"],
    "sink": ["S", "IH", "NG", "K"],
    "toilet": ["T", "OY", "L", "AH", "T"],
    "shower": ["SH", "AW", "ER"],
    "towel": ["T", "AW", "AH", "L"],
    "soap": ["S", "OW", "P"],
    "brush": ["B", "R", "AH", "SH"],
    "comb": ["K", "OW", "M"],
    "mirror": ["M", "IH", "R", "ER"],
    "clock": ["K", "L", "AA", "K"],
    "lamp": ["L", "AE", "M", "P"],
    "candle": ["K", "AE", "N", "D", "AH", "L"],
    "fan": ["F", "AE", "N"],
    "heater": ["HH", "IY", "T", "ER"],
    "cooler": ["K", "UW", "L", "ER"],
    "vacuum": ["V", "AE", "K", "Y", "UW"],
    "broom": ["B", "R", "UW", "M"],
    "mop": ["M", "AA", "P"],
    "bucket": ["B", "AH", "K", "IH", "T"],
    "towel": ["T", "AW", "AH", "L"],
    "closet": ["K", "L", "OW", "Z", "IH", "T"],
    "drawer": ["D", "R", "AO", "ER"],
    "shelf": ["SH", "EH", "L", "F"],
    "floor": ["F", "L", "AO", "R"],
    "wall": ["W", "AO", "L"],
    "ceiling": ["S", "IY", "L", "IH", "NG"],
    "roof": ["R", "UW", "F"],
    "garden": ["G", "AA", "R", "D", "AH", "N"],
    "yard": ["Y", "AA", "R", "D"],
    "fence": ["F", "EH", "N", "S"],
    "gate": ["G", "EY", "T"],
    "path": ["P", "AE", "TH"],
    "road": ["R", "OW", "D"],
    "street": ["S", "T", "R", "IY", "T"],
    "bridge": ["B", "R", "IH", "JH"],
    "river": ["R", "IH", "V", "ER"],
    "lake": ["L", "EY", "K"],
    "mountain": ["M", "AW", "N", "T", "AH", "N"],
    "hill": ["HH", "IH", "L"],
    "valley": ["V", "AE", "L", "IY"],
    "forest": ["F", "AO", "R", "AH", "S", "T"],
    "desert": ["D", "EH", "Z", "ER", "T"],
    "island": ["AY", "L", "AH", "N", "D"],
    "beach": ["B", "IY", "CH"],
    "ocean": ["OW", "SH", "AH", "N"],
    "wave": ["W", "EY", "V"],
    "sand": ["S", "AE", "N", "D"],
    "rock": ["R", "AA", "K"],
    "stone": ["S", "T", "OW", "N"],
    "metal": ["M", "EH", "T", "AH", "L"],
    "wood": ["W", "UH", "D"],
    "plastic": ["P", "L", "AE", "S", "T", "IH", "K"],
    "glass": ["G", "L", "AE", "S"],
    "paper": ["P", "EY", "P", "ER"],
    "cotton": ["K", "AA", "T", "AH", "N"],
    "silk": ["S", "IH", "L", "K"],
    "wool": ["W", "UH", "L"],
    "leather": ["L", "EH", "DH", "ER"],
    "rubber": ["R", "AH", "B", "ER"],
    "cotton": ["K", "AA", "T", "AH", "N"],
    "sugar": ["SH", "UH", "G", "ER"],
    "salt": ["S", "AO", "L", "T"],
    "pepper": ["P", "EH", "P", "ER"],
    "butter": ["B", "AH", "T", "ER"],
    "honey": ["HH", "AH", "N", "IY"],
    "rice": ["R", "AY", "S"],
    "egg": ["EH", "G"],
    "cheese": ["CH", "IY", "Z"],
    "meat": ["M", "IY", "T"],
    "chicken": ["CH", "IH", "K", "AH", "N"],
    "fish": ["F", "IH", "SH"],
    "pork": ["P", "AO", "R", "K"],
    "beef": ["B", "IY", "F"],
    "lamb": ["L", "AE", "M"],
    "steak": ["S", "T", "EY", "K"],
    "soup": ["S", "UW", "P"],
    "salad": ["S", "AE", "L", "AH", "D"],
    "sandwich": ["S", "AE", "N", "D", "W", "IH", "CH"],
    "pizza": ["P", "IY", "T", "S", "AH"],
    "pasta": ["P", "AE", "S", "T", "AH"],
    "cake": ["K", "EY", "K"],
    "cookie": ["K", "UH", "K", "IY"],
    "pie": ["P", "AY"],
    "ice cream": ["AY", "S", "K", "R", "IY", "M"],
    "chocolate": ["CH", "AA", "K", "L", "AH", "T"],
    "candy": ["K", "AE", "N", "D", "IY"],
    "gum": ["G", "AH", "M"],
    "popcorn": ["P", "AA", "P", "K", "AO", "R", "N"],
    "nut": ["N", "AH", "T"],
    "fruit": ["F", "R", "UW", "T"],
    "vegetable": ["V", "EH", "JH", "T", "AH", "B", "AH", "L"],
    "corn": ["K", "AO", "R", "N"],
    "pea": ["P", "IY"],
    "bean": ["B", "IY", "N"],
    "potato": ["P", "AH", "T", "EY", "T", "OW"],
    "tomato": ["T", "AH", "M", "EY", "T", "OW"],
    "carrot": ["K", "AE", "R", "AH", "T"],
    "onion": ["AH", "N", "Y", "AH", "N"],
    "garlic": ["G", "AA", "R", "L", "IH", "K"],
    "lettuce": ["L", "EH", "T", "IH", "S"],
    "cucumber": ["K", "Y", "UW", "K", "AH", "M", "B", "ER"],
    "pepper": ["P", "EH", "P", "ER"],
    "mushroom": ["M", "AH", "SH", "R", "UW", "M"],
    "mango": ["M", "AE", "NG", "G", "OW"],
    "peach": ["P", "IY", "CH"],
    "plum": ["P", "L", "AH", "M"],
    "lemon": ["L", "EH", "M", "AH", "N"],
    "lime": ["L", "AY", "M"],
    "cherry": ["CH", "EH", "R", "IY"],
    "strawberry": ["S", "T", "R", "AO", "B", "ER", "IY"],
    "blueberry": ["B", "L", "UW", "B", "EH", "R", "IY"],
    "watermelon": ["W", "AO", "T", "ER", "M", "EH", "L", "AH", "N"],
    "pineapple": ["P", "AY", "N", "AE", "P", "AH", "L"],
    "coconut": ["K", "OW", "K", "AH", "N", "AH", "T"],
    "avocado": ["AE", "V", "AH", "K", "AA", "D", "OW"],
    "broccoli": ["B", "R", "AA", "K", "AH", "L", "IY"],
    "spinach": ["S", "P", "IH", "N", "IH", "CH"],
    "celery": ["S", "EH", "L", "ER", "IY"],
    "cabbage": ["K", "AE", "B", "IH", "JH"],
    "pumpkin": ["P", "AH", "M", "P", "K", "IH", "N"],
    "eggplant": ["EH", "G", "P", "L", "AE", "N", "T"],
    "asparagus": ["AH", "S", "P", "EH", "R", "AH", "G", "AH", "S"],
    "zucchini": ["Z", "UW", "K", "IY", "N", "IY"],
    "turnip": ["T", "ER", "N", "IH", "P"],
    "radish": ["R", "AE", "D", "IH", "SH"],
    "beet": ["B", "IY", "T"],
    "parsnip": ["P", "AA", "R", "S", "N", "IH", "P"],
    "kale": ["K", "EY", "L"],
    "cereal": ["S", "IH", "R", "IY", "AH", "L"],
    "oatmeal": ["OW", "T", "M", "IY", "L"],
    "pancake": ["P", "AE", "N", "K", "EY", "K"],
    "waffle": ["W", "AA", "F", "AH", "L"],
    "donut": ["D", "OW", "N", "AH", "T"],
    "muffin": ["M", "AH", "F", "IH", "N"],
    "croissant": ["K", "R", "W", "AA", "S", "AA", "N"],
    "baguette": ["B", "AE", "G", "EH", "T"],
    "pretzel": ["P", "R", "EH", "T", "S", "AH", "L"],
    "cracker": ["K", "R", "AE", "K", "ER"],
    "popcorn": ["P", "AA", "P", "K", "AO", "R", "N"],
    "peanut": ["P", "IY", "N", "AH", "T"],
    "almond": ["AA", "M", "AH", "N", "D"],
    "walnut": ["W", "AO", "L", "N", "AH", "T"],
    "cashew": ["K", "AE", "SH", "UW"],
    "pecan": ["P", "IH", "K", "AE", "N"],
    "hazelnut": ["HH", "EY", "Z", "AH", "L", "N", "AH", "T"],
    "macadamia": ["M", "AE", "K", "AH", "D", "EY", "M", "IY", "AH"],
    "pistachio": ["P", "IH", "S", "T", "AE", "SH", "IY", "OW"],
    "walnut": ["W", "AO", "L", "N", "AH", "T"],
    "pecan": ["P", "IH", "K", "AE", "N"],
    "hazelnut": ["HH", "EY", "Z", "AH", "L", "N", "AH", "T"],
    "macadamia": ["M", "AE", "K", "AH", "D", "EY", "M", "IY", "AH"],
    "pistachio": ["P", "IH", "S", "T", "AE", "SH", "IY", "OW"],
}

# Letter-to-sound rules (fallback when not in dictionary)
_VOWEL_GROUPS = set("aeiouy")
_CONSONANT_GROUPS = set("bcdfghjklmnpqrstvwxyz")

# Digraphs/trigraphs (order matters — check longest first)
DIGRAPHS = [
    ("tion", ["SH", "AH", "N"]),
    ("sion", ["ZH", "AH", "N"]),
    ("ough", ["AH", "F"]),  # "rough" pattern
    ("ight", ["AY", "T"]),
    ("ould", ["UH", "D"]),
    ("ound", ["AW", "N", "D"]),
    ("ount", ["AW", "N", "T"]),
    ("ouse", ["AW", "S"]),
    ("outh", ["AW", "TH"]),
    ("ing", ["IH", "NG"]),
    ("tion", ["SH", "AH", "N"]),
    ("ous", ["AH", "S"]),
    ("ious", ["IY", "AH", "S"]),
    ("eous", ["IY", "AH", "S"]),
    ("ble", ["B", "AH", "L"]),
    ("ple", ["P", "AH", "L"]),
    ("tle", ["T", "AH", "L"]),
    ("dle", ["D", "AH", "L"]),
    ("gle", ["G", "AH", "L"]),
    ("ck", ["K"]),
    ("ph", ["F"]),
    ("sh", ["SH"]),
    ("ch", ["CH"]),
    ("th", ["TH"]),
    ("wh", ["W"]),
    ("gh", []),  # silent in most positions
    ("kn", ["N"]),
    ("wr", ["R"]),
    ("gn", ["N"]),
    ("mb", ["M"]),
]

# Single letter rules (fallback)
_LETTER_RULES: dict[str, list[str]] = {
    "a": ["AE"],
    "b": ["B"],
    "c": ["K"],
    "d": ["D"],
    "e": ["EH"],
    "f": ["F"],
    "g": ["G"],
    "h": ["HH"],
    "i": ["IH"],
    "j": ["JH"],
    "k": ["K"],
    "l": ["L"],
    "m": ["M"],
    "n": ["N"],
    "o": ["AO"],
    "p": ["P"],
    "q": ["K"],
    "r": ["R"],
    "s": ["S"],
    "t": ["T"],
    "u": ["AH"],
    "v": ["V"],
    "w": ["W"],
    "x": ["K", "S"],
    "y": ["Y"],
    "z": ["Z"],
}


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWEL_GROUPS


def _apply_rules(word: str) -> list[str]:
    """Apply letter-to-sound rules to a word not in the dictionary."""
    result: list[str] = []
    i = 0
    w = word.lower()

    while i < len(w):
        matched = False

        # Try longest match first (4, 3, 2 chars)
        for length in (4, 3, 2):
            chunk = w[i : i + length]
            for pattern, sounds in DIGRAPHS:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            if ch in _LETTER_RULES:
                result.extend(_LETTER_RULES[ch])
            i += 1

    return result


def text_to_phonemes(text: str) -> list[str]:
    """Convert English text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-Z']+|[^a-zA-Z']+", text)

    for token in tokens:
        if token.isspace():
            result.append(ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _PRONUNCIATION_DICT:
                result.extend(_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_apply_rules(token))
            result.append(ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(ID_TO_PHONEME[SILENCE])

    result.append(ID_TO_PHONEME[EOS])
    return result


# Phoneme-to-grapheme mapping for decode
PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Vowels
    "AA": "a", "AE": "a", "AH": "u", "AO": "o",
    "AW": "ow", "AY": "i", "EH": "e", "ER": "er",
    "EY": "a", "IH": "i", "IY": "ee", "OW": "o",
    "OY": "oy", "UH": "u", "UW": "oo",
    # Consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "th",
    "F": "f", "G": "g", "HH": "h", "JH": "j",
    "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ng", "P": "p", "R": "r", "S": "s",
    "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}


class PhonemeEncoder:
    """Encodes text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **VOWELS, **CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def encode_batch(self, texts: list[str], pad_to_max: bool = True) -> np.ndarray:
        """Convert multiple texts to padded phoneme ID array.

        Args:
            texts: List of text strings
            pad_to_max: If True, pad all sequences to the length of the longest.
                       If False, return list of variable-length arrays.
        Returns:
            (batch_size, max_seq_len) int32 array of padded phoneme IDs
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.int32)

        all_ids = [self.encode(t).flatten() for t in texts]

        if not pad_to_max:
            # Return list of arrays with different lengths
            result = np.zeros((len(all_ids), max(len(ids) for ids in all_ids)), dtype=np.int32)
            for i, ids in enumerate(all_ids):
                result[i, :len(ids)] = ids
            return result

        max_len = max(len(ids) for ids in all_ids)
        result = np.zeros((len(all_ids), max_len), dtype=np.int32)
        for i, ids in enumerate(all_ids):
            result[i, :len(ids)] = ids
        return result

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to text string.

        Strips BOS/EOS tokens and joins phoneme graphemes.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
        Returns:
            Decoded text string
        """
        id_list = ids.flatten().tolist()
        # Strip BOS and EOS
        id_list = [i for i in id_list if i not in (BOS, EOS, PAD)]
        # Map IDs to graphemes
        parts = []
        for pid in id_list:
            phoneme = ID_TO_PHONEME.get(pid, "?")
            grapheme = PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
            parts.append(grapheme)
        return "".join(parts)

    def decode_phonemes(self, ids: np.ndarray) -> list[str]:
        """Convert phoneme ID array to phoneme string list.

        Returns the actual ARPAbet phoneme names (e.g., ["HH", "EH", "L", "OW"]).
        Useful for debugging and visualization.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
        Returns:
            List of phoneme strings (BOS/EOS/PAD stripped)
        """
        id_list = ids.flatten().tolist()
        # Strip BOS, EOS, PAD
        id_list = [i for i in id_list if i not in (BOS, EOS, PAD)]
        return [ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Shows the mapping from text → phonemes → IDs → decoded text.

        Args:
            text: Input text string
        Returns:
            Multi-line string showing the encoding pipeline
        """
        ids = self.encode(text)
        phonemes = self.decode_phonemes(ids)
        decoded = self.decode(ids)

        lines = [
            f"Input:     {text!r}",
            f"Phonemes:  {' '.join(phonemes)}",
            f"IDs:       {ids.flatten().tolist()}",
            f"Decoded:   {decoded!r}",
        ]
        return "\n".join(lines)

    def score_pronunciation(self, target: str, spoken: str) -> dict:
        """Score how well a spoken word matches the target pronunciation.

        Uses phoneme-level comparison to evaluate pronunciation accuracy.
        Useful for language learning applications.

        Args:
            target: The correct word/phrase
            spoken: The spoken word/phrase to evaluate
        Returns:
            Dictionary with score (0.0-1.0), accuracy, and detailed comparison
        """
        target_ids = self.encode(target).flatten()
        spoken_ids = self.encode(spoken).flatten()

        # Strip BOS/EOS for comparison
        target_phonemes = [ID_TO_PHONEME.get(i, "?") for i in target_ids if i not in (BOS, EOS, PAD)]
        spoken_phonemes = [ID_TO_PHONEME.get(i, "?") for i in spoken_ids if i not in (BOS, EOS, PAD)]

        # Calculate similarity using longest common subsequence
        lcs_len = self._lcs_length(target_phonemes, spoken_phonemes)
        target_len = len(target_phonemes)
        spoken_len = len(spoken_phonemes)

        # Precision: how many spoken phonemes are correct
        precision = lcs_len / spoken_len if spoken_len > 0 else 0.0

        # Recall: how many target phonemes were produced
        recall = lcs_len / target_len if target_len > 0 else 0.0

        # F1 score: harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "score": f1,
            "precision": precision,
            "recall": recall,
            "target_phonemes": target_phonemes,
            "spoken_phonemes": spoken_phonemes,
            "target_len": target_len,
            "spoken_len": spoken_len,
        }

    def _lcs_length(self, a: list, b: list) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    @property
    def vocab_size(self) -> int:
        return NUM_PHONEMES
