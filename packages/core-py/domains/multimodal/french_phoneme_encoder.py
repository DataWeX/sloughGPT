"""
French Phoneme Encoder — rule-based grapheme-to-phoneme for French TTS.

Maps French text to a compact phoneme vocabulary. Based on ARPAbet with
French-specific additions.

Phoneme set based on IPA/ARPAbet hybrid for French.
"""

from __future__ import annotations

import re
import numpy as np


# ── French Phoneme inventory ──────────────────────────────────────────────

# Vowels
FRENCH_VOWELS = {
    "AA": 0,   # pere (father)
    "AE": 1,   # trap
    "AH": 2,   # but
    "AO": 3,   # dort (sleeps)
    "AW": 4,   # poudre (powder)
    "AY": 5,   # lit (bed)
    "EH": 6,   # ete (summer)
    "ER": 7,   # premier (first)
    "EY": 8,   # les (the)
    "IH": 9,   # vie (life)
    "IY": 10,  # si (if)
    "OW": 11,  # eau (water)
    "OY": 12,  # boire (to drink)
    "UH": 13,  # tout (all)
    "UW": 14,  # sous (under)
    "OE": 15,  # peur (fear)
    "UE": 16,  # peu (little)
    "YW": 17,  # bu (drank)
}

# Consonants
FRENCH_CONSONANTS = {
    "B": 18,   # bon (good)
    "CH": 19,  # chat (cat)
    "D": 20,   # dire (to say)
    "DH": 21,  # this (English loan)
    "F": 22,   # femme (woman)
    "G": 23,   # grand (big)
    "HH": 24,  # house (English loan)
    "JH": 25,  # Joy (English loan)
    "K": 26,   # courir (to run)
    "L": 27,   # livre (book)
    "M": 28,   # maison (house)
    "N": 29,   # non (no)
    "NG": 30,  # singing (English loan)
    "P": 31,   # pierre (stone)
    "R": 32,   # rouge (red)
    "S": 33,   # sac (bag)
    "SH": 34,  # shoe (English loan)
    "T": 35,   # terre (earth)
    "TH": 36,  # think (English loan)
    "V": 37,   # vert (green)
    "W": 38,  # oui (yes)
    "Y": 39,   # yard (English loan)
    "Z": 40,   # zoo
    "ZH": 41,  # vision (English loan)
}

# Special tokens
PAD = 42
BOS = 43
EOS = 44
SPACE = 45
SILENCE = 46

NUM_FRENCH_PHONEMES = 47

# Reverse lookup
FRENCH_ID_TO_PHONEME = {v: k for k, v in {**FRENCH_VOWELS, **FRENCH_CONSONANTS}.items()}
FRENCH_ID_TO_PHONEME[PAD] = "_"
FRENCH_ID_TO_PHONEME[BOS] = "<"
FRENCH_ID_TO_PHONEME[EOS] = ">"
FRENCH_ID_TO_PHONEME[SPACE] = " "
FRENCH_ID_TO_PHONEME[SILENCE] = "-"

# Phoneme-to-grapheme mapping for decode
FRENCH_PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Vowels
    "AA": "e", "AE": "a", "AH": "u", "AO": "o",
    "AW": "ou", "AY": "i", "EH": "e", "ER": "er",
    "EY": "e", "IH": "i", "IY": "i", "OW": "eau",
    "OY": "oi", "UH": "ou", "UW": "ou",
    "OE": "eu", "UE": "eu", "YW": "u",
    # Consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "th",
    "F": "f", "G": "g", "HH": "h", "JH": "j",
    "K": "c", "L": "l", "M": "m", "N": "n",
    "NG": "ng", "P": "p", "R": "r", "S": "s",
    "SH": "ch", "T": "t", "TH": "th", "V": "v",
    "W": "ou", "Y": "y", "Z": "z", "ZH": "j",
}


# ── French Grapheme-to-Phoneme rules ─────────────────────────────────────

_FRENCH_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "je": ["ZH", "UH"],
    "tu": ["T", "UW"],
    "il": ["IH", "L"],
    "elle": ["EH", "L"],
    "nous": ["N", "UW"],
    "vous": ["V", "UW"],
    "ils": ["IH", "L"],
    "elles": ["EH", "L"],
    "suis": ["S", "UW", "I"],
    "es": ["EH"],
    "est": ["EH"],
    "sommes": ["S", "AO", "M"],
    "etes": ["EH", "T"],
    "ai": ["EH"],
    "as": ["AH"],
    "avons": ["AH", "V", "OW"],
    "ont": ["OW"],
    "pas": ["P", "AA"],
    "ne": ["N", "UH"],
    "oui": ["W", "I"],
    "non": ["N", "OW"],
    "merci": ["M", "ER", "S", "I"],
    "bonjour": ["B", "OW", "ZH", "UW", "R"],
    "bonsoir": ["B", "OW", "S", "W", "AA", "R"],
    "salut": ["S", "AH", "L", "UW"],
    "au revoir": ["OW", "R", "UH", "V", "W", "AA", "R"],
    "merci": ["M", "ER", "S", "I"],
    "s'il vous plait": ["S", "IH", "L", "V", "UW", "P", "L", "EH"],
    "pardon": ["P", "AA", "R", "D", "OW"],
    "excusez-moi": ["EH", "K", "S", "K", "UW", "Z", "EH", "M", "W", "AA"],
    "bon": ["B", "OW"],
    "mauvais": ["M", "OW", "V", "EH"],
    "grand": ["G", "R", "AA"],
    "petit": ["P", "UH", "T", "I"],
    "nouveau": ["N", "UW", "V", "OW"],
    "vieux": ["V", "I", "UW"],
    "maison": ["M", "EH", "Z", "OW"],
    "ecole": ["EH", "K", "OW", "L"],
    "livre": ["L", "IH", "V", "R"],
    "chien": ["SH", "I", "EH"],
    "chat": ["SH", "AA"],
    "eau": ["OW"],
    "pain": ["P", "EH"],
    "vin": ["V", "EH"],
    "fromage": ["F", "R", "OW", "M", "AA", "ZH"],
    "bonjour": ["B", "OW", "ZH", "UW", "R"],
    "comment": ["K", "OW", "M", "AA"],
    "allez": ["AH", "L", "EH"],
    "tres": ["T", "R", "EH"],
    "bien": ["B", "I", "EH"],
    "aussi": ["OW", "S", "I"],
    "peut-etre": ["P", "UH", "T", "EH", "T", "R"],
    "voir": ["V", "W", "AA", "R"],
    "faire": ["F", "EH", "R"],
    "dire": ["D", "I", "R"],
    "venir": ["V", "UH", "N", "I"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "mettre": ["M", "EH", "T", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "manger": ["M", "AA", "ZH", "EH"],
    "dormir": ["D", "OW", "R", "M", "I", "R"],
    "lire": ["L", "I", "R"],
    "ecrire": ["EH", "K", "R", "I", "R"],
    "vivre": ["V", "IH", "V", "R"],
    "trois": ["T", "R", "W", "AA"],
    "quatre": ["K", "AH", "T", "R"],
    "cinq": ["S", "EH"],
    "six": ["S", "I"],
    "sept": ["S", "EH"],
    "huit": ["UW", "I"],
    "neuf": ["N", "UH", "F"],
    "dix": ["D", "I"],
    "jour": ["ZH", "UW", "R"],
    "nuit": ["N", "UW", "I"],
    "heure": ["UH", "R"],
    "matin": ["M", "AH", "T", "EH"],
    "soir": ["S", "W", "AA", "R"],
    "lundi": ["L", "UH", "D", "I"],
    "mardi": ["M", "AA", "R", "D", "I"],
    "mercredi": ["M", "ER", "K", "R", "UH", "D", "I"],
    "jeudi": ["ZH", "UH", "D", "I"],
    "vendredi": ["V", "AA", "D", "R", "UH", "D", "I"],
    "samedi": ["S", "AA", "M", "D", "I"],
    "dimanche": ["D", "I", "M", "AA", "SH"],
    # Additional common French words
    "manger": ["M", "AA", "ZH", "EH"],
    "boire": ["B", "W", "AA", "R"],
    "dormir": ["D", "OW", "R", "M", "I", "R"],
    "veiller": ["V", "EH", "Y", "EH"],
    "marcher": ["M", "AA", "R", "SH", "EH"],
    "courir": ["K", "UW", "R", "I", "R"],
    "sauter": ["S", "OW", "T", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "chanter": ["SH", "AA", "T", "EH"],
    "danser": ["D", "AA", "S", "EH"],
    "jouer": ["ZH", "UW", "EH"],
    "travailler": ["T", "R", "AA", "V", "AH", "Y", "EH"],
    "etudier": ["EH", "T", "UW", "D", "I", "EH"],
    "apprendre": ["AH", "P", "R", "AA", "D", "R"],
    "comprendre": ["K", "OW", "P", "R", "AA", "D", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "ecouter": ["EH", "K", "UW", "T", "EH"],
    "entendre": ["AH", "T", "AA", "D", "R"],
    "voir": ["V", "W", "AA", "R"],
    "regarder": ["R", "UH", "G", "AA", "R", "D", "EH"],
    "toucher": ["T", "UW", "SH", "EH"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "goûter": ["G", "UW", "T", "EH"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "donner": ["D", "OW", "N", "EH"],
    "recevoir": ["R", "UH", "S", "UW", "V", "W", "AA", "R"],
    "acheter": ["AH", "SH", "UH", "T", "EH"],
    "vendre": ["V", "AA", "D", "R"],
    "payer": ["P", "EH", "Y", "EH"],
    "compter": ["K", "OW", "T", "EH"],
    "mesurer": ["M", "UH", "Z", "UW", "R", "EH"],
    "couper": ["K", "UW", "P", "EH"],
    "casser": ["K", "AH", "S", "EH"],
    "reparer": ["R", "EH", "P", "AA", "R", "EH"],
    "nettoyer": ["N", "EH", "T", "W", "AH", "EH"],
    "laver": ["L", "AA", "V", "EH"],
    "cuire": ["K", "UW", "I", "R"],
    "chauffer": ["SH", "OW", "F", "EH"],
    "bouillir": ["B", "UW", "Y", "I", "R"],
    "geler": ["ZH", "UH", "L", "EH"],
    "congeler": ["K", "OW", "ZH", "UH", "L", "EH"],
    "defeler": ["D", "EH", "F", "UH", "L", "EH"],
    "fermer": ["F", "EH", "R", "M", "EH"],
    "ouvrir": ["UW", "V", "R", "I"],
    "entrer": ["AH", "T", "R", "EH"],
    "sortir": ["S", "OW", "R", "T", "I", "R"],
    "partir": ["P", "AA", "R", "T", "I", "R"],
    "arriver": ["AH", "R", "I", "V", "EH"],
    "revenir": ["R", "UH", "V", "UH", "N", "I"],
    "retourner": ["R", "UH", "T", "UW", "R", "N", "EH"],
    "monter": ["M", "OW", "T", "EH"],
    "descendre": ["D", "UH", "S", "AA", "D", "R"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
    "marcher": ["M", "AA", "R", "SH", "EH"],
    "courir": ["K", "UW", "R", "I", "R"],
    "sauter": ["S", "OW", "T", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "chanter": ["SH", "AA", "T", "EH"],
    "danser": ["D", "AA", "S", "EH"],
    "jouer": ["ZH", "UW", "EH"],
    "travailler": ["T", "R", "AA", "V", "AH", "Y", "EH"],
    "etudier": ["EH", "T", "UW", "D", "I", "EH"],
    "apprendre": ["AH", "P", "R", "AA", "D", "R"],
    "comprendre": ["K", "OW", "P", "R", "AA", "D", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "ecouter": ["EH", "K", "UW", "T", "EH"],
    "entendre": ["AH", "T", "AA", "D", "R"],
    "voir": ["V", "W", "AA", "R"],
    "regarder": ["R", "UH", "G", "AA", "R", "D", "EH"],
    "toucher": ["T", "UW", "SH", "EH"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "goûter": ["G", "UW", "T", "EH"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "donner": ["D", "OW", "N", "EH"],
    "recevoir": ["R", "UH", "S", "UW", "V", "W", "AA", "R"],
    "acheter": ["AH", "SH", "UH", "T", "EH"],
    "vendre": ["V", "AA", "D", "R"],
    "payer": ["P", "EH", "Y", "EH"],
    "compter": ["K", "OW", "T", "EH"],
    "mesurer": ["M", "UH", "Z", "UW", "R", "EH"],
    "couper": ["K", "UW", "P", "EH"],
    "casser": ["K", "AH", "S", "EH"],
    "reparer": ["R", "EH", "P", "AA", "R", "EH"],
    "nettoyer": ["N", "EH", "T", "W", "AH", "EH"],
    "laver": ["L", "AA", "V", "EH"],
    "cuire": ["K", "UW", "I", "R"],
    "chauffer": ["SH", "OW", "F", "EH"],
    "bouillir": ["B", "UW", "Y", "I", "R"],
    "geler": ["ZH", "UH", "L", "EH"],
    "congeler": ["K", "OW", "ZH", "UH", "L", "EH"],
    "defeler": ["D", "EH", "F", "UH", "L", "EH"],
    "fermer": ["F", "EH", "R", "M", "EH"],
    "ouvrir": ["UW", "V", "R", "I"],
    "entrer": ["AH", "T", "R", "EH"],
    "sortir": ["S", "OW", "R", "T", "I", "R"],
    "partir": ["P", "AA", "R", "T", "I", "R"],
    "arriver": ["AH", "R", "I", "V", "EH"],
    "revenir": ["R", "UH", "V", "UH", "N", "I"],
    "retourner": ["R", "UH", "T", "UW", "R", "N", "EH"],
    "monter": ["M", "OW", "T", "EH"],
    "descendre": ["D", "UH", "S", "AA", "D", "R"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
    "marcher": ["M", "AA", "R", "SH", "EH"],
    "courir": ["K", "UW", "R", "I", "R"],
    "sauter": ["S", "OW", "T", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "chanter": ["SH", "AA", "T", "EH"],
    "danser": ["D", "AA", "S", "EH"],
    "jouer": ["ZH", "UW", "EH"],
    "travailler": ["T", "R", "AA", "V", "AH", "Y", "EH"],
    "etudier": ["EH", "T", "UW", "D", "I", "EH"],
    "apprendre": ["AH", "P", "R", "AA", "D", "R"],
    "comprendre": ["K", "OW", "P", "R", "AA", "D", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "ecouter": ["EH", "K", "UW", "T", "EH"],
    "entendre": ["AH", "T", "AA", "D", "R"],
    "voir": ["V", "W", "AA", "R"],
    "regarder": ["R", "UH", "G", "AA", "R", "D", "EH"],
    "toucher": ["T", "UW", "SH", "EH"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "goûter": ["G", "UW", "T", "EH"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "donner": ["D", "OW", "N", "EH"],
    "recevoir": ["R", "UH", "S", "UW", "V", "W", "AA", "R"],
    "acheter": ["AH", "SH", "UH", "T", "EH"],
    "vendre": ["V", "AA", "D", "R"],
    "payer": ["P", "EH", "Y", "EH"],
    "compter": ["K", "OW", "T", "EH"],
    "mesurer": ["M", "UH", "Z", "UW", "R", "EH"],
    "couper": ["K", "UW", "P", "EH"],
    "casser": ["K", "AH", "S", "EH"],
    "reparer": ["R", "EH", "P", "AA", "R", "EH"],
    "nettoyer": ["N", "EH", "T", "W", "AH", "EH"],
    "laver": ["L", "AA", "V", "EH"],
    "cuire": ["K", "UW", "I", "R"],
    "chauffer": ["SH", "OW", "F", "EH"],
    "bouillir": ["B", "UW", "Y", "I", "R"],
    "geler": ["ZH", "UH", "L", "EH"],
    "congeler": ["K", "OW", "ZH", "UH", "L", "EH"],
    "defeler": ["D", "EH", "F", "UH", "L", "EH"],
    "fermer": ["F", "EH", "R", "M", "EH"],
    "ouvrir": ["UW", "V", "R", "I"],
    "entrer": ["AH", "T", "R", "EH"],
    "sortir": ["S", "OW", "R", "T", "I", "R"],
    "partir": ["P", "AA", "R", "T", "I", "R"],
    "arriver": ["AH", "R", "I", "V", "EH"],
    "revenir": ["R", "UH", "V", "UH", "N", "I"],
    "retourner": ["R", "UH", "T", "UW", "R", "N", "EH"],
    "monter": ["M", "OW", "T", "EH"],
    "descendre": ["D", "UH", "S", "AA", "D", "R"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
    "marcher": ["M", "AA", "R", "SH", "EH"],
    "courir": ["K", "UW", "R", "I", "R"],
    "sauter": ["S", "OW", "T", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "chanter": ["SH", "AA", "T", "EH"],
    "danser": ["D", "AA", "S", "EH"],
    "jouer": ["ZH", "UW", "EH"],
    "travailler": ["T", "R", "AA", "V", "AH", "Y", "EH"],
    "etudier": ["EH", "T", "UW", "D", "I", "EH"],
    "apprendre": ["AH", "P", "R", "AA", "D", "R"],
    "comprendre": ["K", "OW", "P", "R", "AA", "D", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "ecouter": ["EH", "K", "UW", "T", "EH"],
    "entendre": ["AH", "T", "AA", "D", "R"],
    "voir": ["V", "W", "AA", "R"],
    "regarder": ["R", "UH", "G", "AA", "R", "D", "EH"],
    "toucher": ["T", "UW", "SH", "EH"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "goûter": ["G", "UW", "T", "EH"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "donner": ["D", "OW", "N", "EH"],
    "recevoir": ["R", "UH", "S", "UW", "V", "W", "AA", "R"],
    "acheter": ["AH", "SH", "UH", "T", "EH"],
    "vendre": ["V", "AA", "D", "R"],
    "payer": ["P", "EH", "Y", "EH"],
    "compter": ["K", "OW", "T", "EH"],
    "mesurer": ["M", "UH", "Z", "UW", "R", "EH"],
    "couper": ["K", "UW", "P", "EH"],
    "casser": ["K", "AH", "S", "EH"],
    "reparer": ["R", "EH", "P", "AA", "R", "EH"],
    "nettoyer": ["N", "EH", "T", "W", "AH", "EH"],
    "laver": ["L", "AA", "V", "EH"],
    "cuire": ["K", "UW", "I", "R"],
    "chauffer": ["SH", "OW", "F", "EH"],
    "bouillir": ["B", "UW", "Y", "I", "R"],
    "geler": ["ZH", "UH", "L", "EH"],
    "congeler": ["K", "OW", "ZH", "UH", "L", "EH"],
    "defeler": ["D", "EH", "F", "UH", "L", "EH"],
    "fermer": ["F", "EH", "R", "M", "EH"],
    "ouvrir": ["UW", "V", "R", "I"],
    "entrer": ["AH", "T", "R", "EH"],
    "sortir": ["S", "OW", "R", "T", "I", "R"],
    "partir": ["P", "AA", "R", "T", "I", "R"],
    "arriver": ["AH", "R", "I", "V", "EH"],
    "revenir": ["R", "UH", "V", "UH", "N", "I"],
    "retourner": ["R", "UH", "T", "UW", "R", "N", "EH"],
    "monter": ["M", "OW", "T", "EH"],
    "descendre": ["D", "UH", "S", "AA", "D", "R"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
    "marcher": ["M", "AA", "R", "SH", "EH"],
    "courir": ["K", "UW", "R", "I", "R"],
    "sauter": ["S", "OW", "T", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "chanter": ["SH", "AA", "T", "EH"],
    "danser": ["D", "AA", "S", "EH"],
    "jouer": ["ZH", "UW", "EH"],
    "travailler": ["T", "R", "AA", "V", "AH", "Y", "EH"],
    "etudier": ["EH", "T", "UW", "D", "I", "EH"],
    "apprendre": ["AH", "P", "R", "AA", "D", "R"],
    "comprendre": ["K", "OW", "P", "R", "AA", "D", "R"],
    "parler": ["P", "AA", "L", "EH"],
    "ecouter": ["EH", "K", "UW", "T", "EH"],
    "entendre": ["AH", "T", "AA", "D", "R"],
    "voir": ["V", "W", "AA", "R"],
    "regarder": ["R", "UH", "G", "AA", "R", "D", "EH"],
    "toucher": ["T", "UW", "SH", "EH"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "sentir": ["S", "AA", "T", "I", "R"],
    "goûter": ["G", "UW", "T", "EH"],
    "prendre": ["P", "R", "AA", "D", "R"],
    "donner": ["D", "OW", "N", "EH"],
    "recevoir": ["R", "UH", "S", "UW", "V", "W", "AA", "R"],
    "acheter": ["AH", "SH", "UH", "T", "EH"],
    "vendre": ["V", "AA", "D", "R"],
    "payer": ["P", "EH", "Y", "EH"],
    "compter": ["K", "OW", "T", "EH"],
    "mesurer": ["M", "UH", "Z", "UW", "R", "EH"],
    "couper": ["K", "UW", "P", "EH"],
    "casser": ["K", "AH", "S", "EH"],
    "reparer": ["R", "EH", "P", "AA", "R", "EH"],
    "nettoyer": ["N", "EH", "T", "W", "AH", "EH"],
    "laver": ["L", "AA", "V", "EH"],
    "cuire": ["K", "UW", "I", "R"],
    "chauffer": ["SH", "OW", "F", "EH"],
    "bouillir": ["B", "UW", "Y", "I", "R"],
    "geler": ["ZH", "UH", "L", "EH"],
    "congeler": ["K", "OW", "ZH", "UH", "L", "EH"],
    "defeler": ["D", "EH", "F", "UH", "L", "EH"],
    "fermer": ["F", "EH", "R", "M", "EH"],
    "ouvrir": ["UW", "V", "R", "I"],
    "entrer": ["AH", "T", "R", "EH"],
    "sortir": ["S", "OW", "R", "T", "I", "R"],
    "partir": ["P", "AA", "R", "T", "I", "R"],
    "arriver": ["AH", "R", "I", "V", "EH"],
    "revenir": ["R", "UH", "V", "UH", "N", "I"],
    "retourner": ["R", "UH", "T", "UW", "R", "N", "EH"],
    "monter": ["M", "OW", "T", "EH"],
    "descendre": ["D", "UH", "S", "AA", "D", "R"],
    "tomber": ["T", "OW", "B", "EH"],
    "nager": ["N", "AA", "ZH", "EH"],
    "voler": ["V", "OW", "L", "EH"],
}


def _french_apply_rules(word: str) -> list[str]:
    """Apply French letter-to-sound rules."""
    result: list[str] = []
    i = 0
    w = word.lower()

    # French digraphs/trigraphs
    french_digraphs = [
        ("eau", ["OW"]),
        ("ou", ["UW"]),
        ("ai", ["EH"]),
        ("ei", ["EH"]),
        ("au", ["OW"]),
        ("eu", ["UH"]),
        ("ch", ["SH"]),
        ("gn", ["NY"]),
        ("ph", ["F"]),
        ("th", ["T"]),
        ("ill", ["IY", "L"]),
        ("ien", ["IY", "EH"]),
        ("tion", ["S", "Y", "OW"]),
    ]

    while i < len(w):
        matched = False

        # Try longest match first (3, 2 chars)
        for length in (3, 2):
            chunk = w[i:i + length]
            for pattern, sounds in french_digraphs:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            # French single letter rules
            french_letter_rules = {
                "a": ["AA"], "b": ["B"], "c": ["S"],
                "d": ["D"], "e": ["UH"], "f": ["F"],
                "g": ["G"], "h": [], "i": ["I"],
                "j": ["ZH"], "k": ["K"], "l": ["L"],
                "m": ["M"], "n": ["N"], "o": ["OW"],
                "p": ["P"], "q": ["K"], "r": ["R"],
                "s": ["S"], "t": ["T"], "u": ["UW"],
                "v": ["V"], "w": ["W"], "x": ["K", "S"],
                "y": ["I"], "z": ["Z"],
            }
            if ch in french_letter_rules:
                result.extend(french_letter_rules[ch])
            i += 1

    return result


def french_text_to_phonemes(text: str) -> list[str]:
    """Convert French text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [FRENCH_ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-Zàâäéèêëïîôùûüÿçœæ]+|[^a-zA-Zàâäéèêëïîôùûüÿçœæ]+", text)

    for token in tokens:
        if token.isspace():
            result.append(FRENCH_ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _FRENCH_PRONUNCIATION_DICT:
                result.extend(_FRENCH_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_french_apply_rules(token))
            result.append(FRENCH_ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(FRENCH_ID_TO_PHONEME[SILENCE])

    result.append(FRENCH_ID_TO_PHONEME[EOS])
    return result


class FrenchPhonemeEncoder:
    """Encodes French text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **FRENCH_VOWELS, **FRENCH_CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert French text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = french_text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to French text string.

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
            phoneme = FRENCH_ID_TO_PHONEME.get(pid, "?")
            grapheme = FRENCH_PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
            parts.append(grapheme)
        return "".join(parts)

    def decode_phonemes(self, ids: np.ndarray) -> list[str]:
        """Convert phoneme ID array to phoneme string list.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
        Returns:
            List of phoneme strings (BOS/EOS/PAD stripped)
        """
        id_list = ids.flatten().tolist()
        # Strip BOS, EOS, PAD
        id_list = [i for i in id_list if i not in (BOS, EOS, PAD)]
        return [FRENCH_ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input French text string
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
        """
        target_ids = self.encode(target).flatten()
        spoken_ids = self.encode(spoken).flatten()

        target_phonemes = [FRENCH_ID_TO_PHONEME.get(i, "?") for i in target_ids if i not in (BOS, EOS, PAD)]
        spoken_phonemes = [FRENCH_ID_TO_PHONEME.get(i, "?") for i in spoken_ids if i not in (BOS, EOS, PAD)]

        lcs_len = self._lcs_length(target_phonemes, spoken_phonemes)
        target_len = len(target_phonemes)
        spoken_len = len(spoken_phonemes)

        precision = lcs_len / spoken_len if spoken_len > 0 else 0.0
        recall = lcs_len / target_len if target_len > 0 else 0.0

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
        return NUM_FRENCH_PHONEMES
