"""
Morphologically-aware tokenizer — pure Python, no external dependencies.

Own BPE implementation that reads tokenizer.json directly. Morphological
analysis is linguistic rule-based reasoning (prefixes, suffixes, roots).

Usage:
    from domains.infrastructure.morph_tokenizer import MorphTokenizer
    tok = MorphTokenizer.from_pretrained("gpt2")

    ids = tok.encode("unhappiness")
    text = tok.decode(ids)

    morphemes = tok.decompose("unhappiness")  # ["un", "happy", "ness"]
    root = tok.stem("running")  # "run"
    forms = tok.generate_forms("run")  # ["runs", "running", "ran", "runner"]
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("man.infrastructure.morph_tokenizer")


# ── English morphology rules ─────────────────────────────────────────────────

_SUFFIXES = [
    ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
    ("ation", "ate"), ("tion", "te"), ("sion", "de"), ("ment", ""),
    ("ness", "y"), ("able", ""), ("ible", ""), ("ful", ""), ("ous", ""),
    ("ive", ""), ("less", ""), ("ally", ""), ("ical", ""), ("ish", ""),
    ("ly", ""), ("er", ""), ("or", ""), ("ist", ""), ("ize", ""), ("ise", ""),
    ("ies", "y"), ("es", "e"), ("es", ""), ("s", ""),
    ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
]

_PREFIXES = [
    "un", "re", "in", "im", "dis", "en", "em", "non",
    "over", "mis", "out", "sub", "pre", "inter", "fore",
    "de", "trans", "super", "semi", "anti", "auto",
    "bi", "tri", "multi", "poly", "mono", "uni",
    "micro", "macro", "hyper", "ultra", "mini",
]

_ROOT_FORMS = {
    "ran": "run", "running": "run", "runs": "run", "runner": "run",
    "went": "go", "going": "go", "goes": "go", "gone": "go",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "was": "be", "were": "be", "been": "be", "being": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "having": "have",
    "did": "do", "does": "do", "doing": "do", "done": "do",
    "said": "say", "says": "say", "saying": "say",
    "made": "make", "makes": "make", "making": "make",
    "took": "take", "taken": "take", "takes": "take", "taking": "take",
    "came": "come", "comes": "come", "coming": "come",
    "gave": "give", "given": "give", "gives": "give", "giving": "give",
    "found": "find", "finds": "find", "finding": "find",
    "knew": "know", "known": "know", "knows": "know", "knowing": "know",
    "thought": "think", "thinks": "think", "thinking": "think",
    "saw": "see", "seen": "see", "sees": "see", "seeing": "see",
    "got": "get", "gets": "get", "getting": "get",
    "became": "become", "becomes": "become", "becoming": "become",
    "left": "leave", "leaves": "leave", "leaving": "leave",
    "kept": "keep", "keeps": "keep", "keeping": "keep",
    "held": "hold", "holds": "hold", "holding": "hold",
    "stood": "stand", "stands": "stand", "standing": "stand",
    "lost": "lose", "loses": "lose", "losing": "lose",
    "met": "meet", "meets": "meet", "meeting": "meet",
    "paid": "pay", "pays": "pay", "paying": "pay",
    "sat": "sit", "sits": "sit", "sitting": "sit",
    "spoke": "speak", "spoken": "speak", "speaks": "speak", "speaking": "speak",
    "wrote": "write", "written": "write", "writes": "write", "writing": "write",
    "read": "read", "reads": "read", "reading": "read",
    "led": "lead", "leads": "lead", "leading": "lead",
    "fed": "feed", "feeds": "feed", "feeding": "feed",
    "grew": "grow", "grown": "grow", "grows": "grow", "growing": "grow",
    "drew": "draw", "drawn": "draw", "draws": "draw", "drawing": "draw",
    "threw": "throw", "thrown": "throw", "throws": "throw", "throwing": "throw",
    "blew": "blow", "blown": "blow", "blows": "blow", "blowing": "blow",
    "flew": "fly", "flown": "fly", "flies": "fly", "flying": "fly",
    "broke": "break", "broken": "break", "breaks": "break", "breaking": "break",
    "chose": "choose", "chosen": "choose", "chooses": "choose", "choosing": "choose",
    "drove": "drive", "driven": "drive", "drives": "drive", "driving": "drive",
    "rode": "ride", "ridden": "ride", "rides": "ride", "riding": "ride",
    "woke": "wake", "woken": "wake", "wakes": "wake", "waking": "wake",
    "bore": "bear", "born": "bear", "borne": "bear", "bears": "bear",
    "tore": "tear", "torn": "tear", "tears": "tear",
    "wore": "wear", "worn": "wear", "wears": "wear",
    "began": "begin", "begun": "begin", "begins": "begin",
    "rang": "ring", "rung": "ring", "rings": "ring",
    "sang": "sing", "sung": "sing", "sings": "sing",
    "sank": "sink", "sunk": "sink", "sinks": "sink",
    "drank": "drink", "drunk": "drink", "drinks": "drink",
    "clung": "cling", "clings": "cling",
    "hung": "hang", "hangs": "hang",
    "dug": "dig", "digs": "dig",
    "spun": "spin", "spins": "spin",
    "won": "win", "wins": "win",
    "swam": "swim", "swum": "swim", "swims": "swim",
}

_IRREGULAR_FORMS = {
    "run": ["ran", "runs", "running", "runner"],
    "go": ["went", "goes", "going", "gone"],
    "be": ["is", "are", "was", "were", "been", "being"],
    "have": ["has", "had", "having"],
    "do": ["does", "did", "doing", "done"],
    "say": ["says", "said", "saying"],
    "make": ["makes", "made", "making"],
    "take": ["takes", "took", "taken", "taking"],
    "come": ["comes", "came", "coming"],
    "see": ["sees", "saw", "seen", "seeing"],
    "know": ["knows", "knew", "known", "knowing"],
    "get": ["gets", "got", "getting"],
    "give": ["gives", "gave", "given", "giving"],
    "find": ["finds", "found", "finding"],
    "think": ["thinks", "thought", "thinking"],
    "tell": ["tells", "told", "telling"],
    "become": ["becomes", "became", "becoming"],
    "leave": ["leaves", "left", "leaving"],
    "put": ["puts", "putting"],
    "keep": ["keeps", "kept", "keeping"],
    "let": ["lets", "letting"],
    "begin": ["begins", "began", "begun", "beginning"],
    "show": ["shows", "showed", "shown", "showing"],
    "hear": ["hears", "heard", "hearing"],
    "play": ["plays", "played", "playing"],
    "move": ["moves", "moved", "moving"],
    "live": ["lives", "lived", "living"],
    "hold": ["holds", "held", "holding"],
    "bring": ["brings", "brought", "bringing"],
    "write": ["writes", "wrote", "written", "writing"],
    "sit": ["sits", "sat", "sitting"],
    "stand": ["stands", "stood", "standing"],
    "lose": ["loses", "lost", "losing"],
    "pay": ["pays", "paid", "paying"],
    "meet": ["meets", "met", "meeting"],
    "lead": ["leads", "led", "leading"],
    "speak": ["speaks", "spoke", "spoken", "speaking"],
    "read": ["reads", "reading"],
    "grow": ["grows", "grew", "grown", "growing"],
    "win": ["wins", "won", "winning"],
    "buy": ["buys", "bought", "buying"],
    "fall": ["falls", "fell", "fallen", "falling"],
    "eat": ["eats", "ate", "eaten", "eating"],
    "break": ["breaks", "broke", "broken", "breaking"],
    "drive": ["drives", "drove", "driven", "driving"],
    "wear": ["wears", "wore", "worn", "wearing"],
    "rise": ["rises", "rose", "risen", "rising"],
    "choose": ["chooses", "chose", "chosen", "choosing"],
    "fight": ["fights", "fought", "fighting"],
    "catch": ["catches", "caught", "catching"],
    "cut": ["cuts", "cutting"],
    "build": ["builds", "built", "building"],
    "send": ["sends", "sent", "sending"],
    "sell": ["sells", "sold", "selling"],
    "die": ["dies", "died", "dying"],
    "spend": ["spends", "spent", "spending"],
}


# ── Pure Python BPE ──────────────────────────────────────────────────────────

# GPT-2 byte-to-unicode mapping
def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    """GPT-2 style byte-to-unicode mapping. Printable bytes stay as-is,
    others map to unicode 256+. Returns (byte2unicode, unicode2byte)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    byte2unicode = dict(zip(bs, [chr(c) for c in cs]))
    unicode2byte = {v: k for k, v in byte2unicode.items()}
    return byte2unicode, unicode2byte


_BYTE2UNICODE, _UNICODE2BYTE = _bytes_to_unicode()


class MorphTokenizer:
    """Own BPE tokenizer + morphological analysis. No external dependencies."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        eos_token_id: int = 50256,
        byte_level: bool = False,
        byte_fallback: bool = False,
        model_id: str = "",
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.eos_token_id = eos_token_id
        self.byte_level = byte_level
        self.byte_fallback = byte_fallback
        self.model_id = model_id
        self._root_cache: Dict[str, str] = {}
        self._morpheme_cache: Dict[str, List[str]] = {}

    @classmethod
    def from_pretrained(cls, model_id: str) -> "MorphTokenizer":
        """Load from tokenizer.json — our own parser, no HF tokenizers lib."""
        from domains.infrastructure.safetensors_loader import _get_model_dir
        model_dir = _get_model_dir(model_id)

        tokenizer_path = None
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                candidate = snap / "tokenizer.json"
                if candidate.exists():
                    tokenizer_path = candidate
                    break
        if tokenizer_path is None:
            tokenizer_path = model_dir / "tokenizer.json"
        if tokenizer_path is None or not tokenizer_path.exists():
            raise FileNotFoundError(f"No tokenizer.json for {model_id}")

        with open(tokenizer_path) as f:
            tok_data = json.load(f)

        # Parse vocab
        raw_vocab = tok_data["model"]["vocab"]
        if isinstance(raw_vocab, dict):
            vocab = raw_vocab
        else:
            vocab = {item[0]: item[1] for item in raw_vocab}

        # Parse merges
        raw_merges = tok_data["model"]["merges"]
        merges = [tuple(m.split(" ", 1)) for m in raw_merges]

        # Detect byte-level BPE (GPT-2, Qwen2, etc)
        byte_level = False
        byte_fallback = False
        pretok = tok_data.get("pre_tokenizer") or {}
        if pretok.get("type") == "ByteLevel":
            byte_level = True
        # Also check if vocab contains byte-mapped chars
        if "\xc3" in vocab or "\xc4" in vocab:
            byte_level = True
        # Detect SentencePiece ByteFallback (TinyLlama, LLaMA, Mistral)
        decoder = tok_data.get("decoder") or {}
        decoders = decoder.get("decoders", [])
        for d in decoders:
            if d.get("type") == "ByteFallback":
                byte_fallback = True
                break
        # ByteFallback overrides byte_level (different encoding scheme)
        if byte_fallback:
            byte_level = False

        # EOS token
        eos = tok_data.get("model", {}).get("eos_token_id", 50256)
        if isinstance(eos, list):
            eos = eos[0]
        # Also check post-processor for end_token_id
        post = tok_data.get("post_processor", {})
        if "end_token_id" in post:
            eos = post["end_token_id"]

        logger.info("Loaded tokenizer %s (vocab=%d, merges=%d, byte_level=%s, byte_fallback=%s)",
                     model_id, len(vocab), len(merges), byte_level, byte_fallback)
        return cls(vocab=vocab, merges=merges, eos_token_id=eos,
                   byte_level=byte_level, byte_fallback=byte_fallback,
                   model_id=model_id)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── BPE encode ──────────────────────────────────────────────────────

    def _bpe_encode(self, text: str) -> List[int]:
        """BPE encode a single word/token (no whitespace splitting)."""
        # Convert to initial tokens
        if self.byte_level:
            # Byte-level: convert to GPT-2 byte-to-unicode chars
            tokens = [_BYTE2UNICODE[b] for b in text.encode("utf-8")]
        else:
            # Character-level: split into chars
            tokens = list(text)

        # Iteratively merge highest-priority pairs
        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (i, pair)

            if best_pair is None or best_rank == float("inf"):
                break

            i, (a, b) = best_pair
            tokens = tokens[:i] + [a + b] + tokens[i + 2:]

        # Map to IDs
        ids = []
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            elif self.byte_level:
                # Fallback: encode bytes individually
                for byte_val in t.encode("utf-8"):
                    char = _BYTE2UNICODE.get(byte_val, chr(byte_val))
                    ids.append(self.vocab.get(char, self.eos_token_id))
            else:
                ids.append(self.eos_token_id)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs via BPE."""
        return self._bpe_encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for tid in ids:
            t = self.inv_vocab.get(int(tid), "")
            tokens.append(t)

        if self.byte_level:
            result = []
            for t in tokens:
                for ch in t:
                    byte_val = _UNICODE2BYTE.get(ch, ord(ch))
                    result.append(byte_val)
            return bytes(result).decode("utf-8", errors="replace")
        elif self.byte_fallback:
            # SentencePiece ByteFallback: <0xHH> tokens → bytes, ▁ → space
            result = bytearray()
            for t in tokens:
                if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                    # Hex byte token
                    try:
                        result.append(int(t[3:5], 16))
                    except ValueError:
                        result.extend(t.encode("utf-8"))
                elif t == "\u2581":  # ▁ = space prefix
                    result.append(0x20)
                else:
                    result.extend(t.encode("utf-8"))
            return result.decode("utf-8", errors="replace")
        else:
            return "".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Return the string tokens (not IDs) for inspection."""
        if self.byte_level:
            byte_chars = [_BYTE2UNICODE[b] for b in text.encode("utf-8")]
        else:
            byte_chars = list(text)
        return self._apply_bpe(byte_chars)

    def _apply_bpe(self, tokens: List[str]) -> List[str]:
        """Apply BPE merges to a list of token strings, return merged tokens."""
        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (i, pair)
            if best_pair is None or best_rank == float("inf"):
                break
            i, (a, b) = best_pair
            tokens = tokens[:i] + [a + b] + tokens[i + 2:]
        return tokens

    # ── Morpheme decomposition (linguistic rule-based) ───────────────────

    def decompose(self, word: str) -> List[str]:
        """Break a word into morphemes using linguistic rules."""
        if word in self._morpheme_cache:
            return self._morpheme_cache[word]

        original = word
        morphemes = []

        for prefix in sorted(_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                morphemes.append(prefix)
                word = word[len(prefix):]
                break

        suffix_parts = []
        remaining = word
        for suffix, stem_form in _SUFFIXES:
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 1:
                suffix_parts.append(suffix)
                base = remaining[:-len(suffix)]
                if stem_form and len(stem_form) == 1:
                    remaining = base[:-1] + stem_form if base else base
                elif stem_form:
                    remaining = base + stem_form
                else:
                    remaining = base
                # Handle doubled consonant (running -> runn -> run)
                if (len(remaining) >= 3 and remaining[-1] == remaining[-2]
                        and remaining[-1] not in "aeiou"):
                    remaining = remaining[:-1]
                break

        stem = _ROOT_FORMS.get(remaining, remaining)
        if stem:
            morphemes.append(stem)
        morphemes.extend(reversed(suffix_parts))

        if not morphemes or (len(morphemes) == 1 and morphemes[0] == original):
            morphemes = [original]

        self._morpheme_cache[original] = morphemes
        return morphemes

    def stem(self, word: str) -> str:
        """Extract the root/stem of a word."""
        if word in _ROOT_FORMS:
            return _ROOT_FORMS[word]
        if word in self._root_cache:
            return self._root_cache[word]

        morphemes = self.decompose(word)
        root = word
        for m in morphemes:
            if m not in _PREFIXES and m not in [s for s, _ in _SUFFIXES]:
                if len(m) > len(root) or root == word:
                    root = m

        if root == word or len(root) <= 2:
            for suffix, stem_form in _SUFFIXES:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    candidate = word[:-len(suffix)]
                    if stem_form:
                        candidate = candidate + stem_form
                    if len(candidate) > len(root):
                        root = candidate
                    break

        if root in _ROOT_FORMS:
            root = _ROOT_FORMS[root]

        self._root_cache[word] = root
        return root

    def generate_forms(self, root: str) -> List[str]:
        """Generate related word forms from a root."""
        forms = {root}
        if root in _IRREGULAR_FORMS:
            forms.update(_IRREGULAR_FORMS[root])
        else:
            forms.update([
                root + "s", root + "es", root + "ed", root + "ing",
                root + "er", root + "est", root + "ly",
            ])
            if root.endswith("e"):
                forms.add(root[:-1] + "ing")

        # y → i before suffixes that need it
        stem_i = root[:-1] + "i" if root.endswith("y") and len(root) > 1 else None

        for affix in ["ful", "less", "ness", "ment", "tion", "able", "ous", "ive", "ly", "er", "or", "ist", "ize"]:
            forms.add(root + affix)
            if stem_i:
                forms.add(stem_i + affix)
        for prefix in ["un", "re", "in", "im", "dis", "en", "over", "mis", "out", "pre"]:
            forms.add(prefix + root)

        return sorted(forms, key=lambda x: (-len(x), x))

    def find_related(self, word: str, max_results: int = 10) -> List[str]:
        """Find words related by shared root/morphology."""
        root = self.stem(word)
        related = set()
        forms = self.generate_forms(root)
        related.update(forms)
        related.discard(word)
        return sorted(related, key=lambda x: (len(x), x))[:max_results]

    def root_distance(self, word1: str, word2: str) -> int:
        """Morphological distance: 0=same, 1=same root, 2=related, 3=unrelated."""
        if word1 == word2:
            return 0
        root1, root2 = self.stem(word1), self.stem(word2)
        if root1 == root2:
            return 1
        if len(root1) == len(root2) and sum(a != b for a, b in zip(root1, root2)) <= 1:
            return 2
        return 3

    def are_related(self, word1: str, word2: str, max_distance: int = 2) -> bool:
        return self.root_distance(word1, word2) <= max_distance

    def decompose_batch(self, words: List[str]) -> Dict[str, List[str]]:
        return {w: self.decompose(w) for w in words}

    def stem_batch(self, words: List[str]) -> Dict[str, str]:
        return {w: self.stem(w) for w in words}

    def build_root_index(self, words: List[str]) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for word in words:
            index[self.stem(word)].append(word)
        return dict(index)

    def vocabulary_coverage(self, words: List[str]) -> float:
        covered = sum(1 for w in words if w in self.vocab)
        return covered / len(words) if words else 0.0

    def morphological_diversity(self, words: List[str]) -> float:
        roots = set(self.stem(w) for w in words)
        return len(roots) / len(words) if words else 0.0
