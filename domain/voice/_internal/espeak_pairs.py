"""Espeak dataset adapter — text → waveform pairs for TTS training.

The only paired text+speech source we control offline: espeak synthesizes a
waveform for any phrase, giving (text, audio) supervision for the domain
SpectrogramDecoder. This is a DATASET adapter — training itself stays in
TTSEngine.train_step/train_epoch (one loop, never a second one).

Contract: mono float32 in [-1, 1] at TTSEngine.sample_rate (22050) — espeak's
default -w rate — so target mels line up with the vocoder's mel basis.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 22050
_ESPEAK_CANDIDATES = ("espeak-ng", "espeak")

# Phoneme-covering starter corpus: short phrases spanning English consonant
# clusters, fricatives, nasals, liquids, vowels and numbers — the diversity a
# tiny decoder needs before it overfits to one cadence. Drivers may pass their
# own list; this is the shared default so train script and benchmark see the
# same data.
DEFAULT_CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "How vexingly quick daft zebras jump.",
    "The five boxing wizards jump quickly.",
    "Bright vixens jump; dozy fowl quack.",
    "Jinxed wizards pluck ivy from the large quilt box.",
    "Two driven jocks help fax my big quiz.",
    "The rain in Spain stays mainly in the plain.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "A proper copper coffee pot belongs in the kitchen.",
    "We promptly judged antique ivory buckles for the next prize.",
    "Few black taxis drive up major roads on quiet nights.",
    "My girl wove six dozen plaid jackets before she quit.",
    "Crazy Frederick bought many very exquisite opal jewels.",
    "The job requires extra pluck and zeal from every young wage earner.",
    "Whenever the city council met, the mayor would drone on for hours.",
    "Sixty sixth graders were quizzed on geometric proofs.",
    "Blowzy night-frumps vexed Jack Q.",
    "Artistic sketches of zen blends were drawn on cheap newsprint.",
    "The existence of consciousness is a question for physics.",
    "Numbers: one two three four five six seven eight nine ten.",
    "Hello, how are you doing today?",
    "Good morning and welcome to the system.",
    "Please read this sentence out loud.",
    "Artificial intelligence should sound natural.",
    "The temperature outside is twenty two degrees.",
    "Open the door and close the window.",
    "Where is the nearest train station?",
    "I would like a cup of black coffee.",
    "Learning requires patience and repetition.",
    "Voice interfaces must speak clearly and slowly.",
    "The alphabet: a b c d e f g h i j k l m n o p q r s t u v w x y z.",
    "Budget vending cups for six shy unexpected guests.",
    "Chug, grog, and a quick buzz from a frozen fizzy drink.",
    "With purposeful strides the hiker crossed the rugged ridge.",
    "A gentle breeze rippled across the silver lake.",
    "The old clock ticked slowly in the dim hallway.",
    "Children laughed and chased each other through the garden.",
    "Bold plans demand steady hands and patient work.",
]

# Hold-out slice lives at the END of the corpus so drivers split with
# `corpus[:-n]` / `corpus[-n:]` and keep the default split stable.
DEFAULT_HOLDOUT = 6


def espeak_bin() -> str | None:
    """Locate an espeak binary, preferring espeak-ng."""
    for name in _ESPEAK_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def espeak_available() -> bool:
    return espeak_bin() is not None


def render(text: str, *, sample_rate: int = SAMPLE_RATE, speed: int = 150) -> np.ndarray:
    """Synthesize `text` with espeak into a float32 waveform at `sample_rate`.

    Raises RuntimeError when no espeak binary is installed and ValueError on
    empty text — callers distinguish "environment missing" from "bad input".
    """
    binary = espeak_bin()
    if binary is None:
        raise RuntimeError("espeak binary not found (install espeak or espeak-ng)")
    if not text or not text.strip():
        raise ValueError("cannot render empty text")
    with tempfile.TemporaryDirectory(prefix="espeak_pairs_") as td:
        wav_path = os.path.join(td, "utterance.wav")
        proc = subprocess.run(
            [binary, "-w", wav_path, "-s", str(speed), text],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not os.path.exists(wav_path):
            stderr = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"espeak failed (rc={proc.returncode}): {stderr}")
        with wave.open(wav_path, "rb") as w:
            channels = w.getnchannels()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and pcm.size > 1:
        duration = pcm.size / float(rate)
        src_t = np.linspace(0.0, duration, pcm.size, endpoint=False)
        dst_t = np.linspace(0.0, duration, int(duration * sample_rate), endpoint=False)
        pcm = np.interp(dst_t, src_t, pcm).astype(np.float32)
    return np.ascontiguousarray(pcm, dtype=np.float32)


def build_pairs(texts: list[str], **render_kwargs) -> list[tuple[str, np.ndarray]]:
    """Render a corpus into (text, waveform) training pairs, order preserved."""
    return [(t, render(t, **render_kwargs)) for t in texts]


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_HOLDOUT",
    "SAMPLE_RATE",
    "build_pairs",
    "espeak_available",
    "espeak_bin",
    "render",
]
