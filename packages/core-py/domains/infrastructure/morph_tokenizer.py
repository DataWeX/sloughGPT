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
import numpy as np
from domains.shared import find_repo_root

logger = logging.getLogger("slo.infrastructure.morph_tokenizer")


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


class _NumpyTokenTensor:
    """Minimal tensor wrapper exposing ``.to(device)`` and ``.shape``.

    Used by :meth:`MorphTokenizer.__call__` to return HF-compatible token
    tensors backed by numpy arrays.  ``.to()`` is a no-op (CPU-only); callers
    that need the actual array can access ``._data``.
    """
    __slots__ = ("_data",)

    def __init__(self, data: "np.ndarray") -> None:
        self._data = data

    def to(self, _device: str) -> "_NumpyTokenTensor":
        return self

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def dtype(self) -> Any:
        return self._data.dtype

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)


class MorphTokenizer:
    """Own BPE tokenizer + morphological analysis. No external dependencies."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        eos_token_id: int = 50256,
        pad_token_id: Optional[int] = None,
        byte_level: bool = False,
        byte_fallback: bool = False,
        model_id: str = "",
        added_tokens: Optional[Dict[str, int]] = None,
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.byte_level = byte_level
        self.byte_fallback = byte_fallback
        self.model_id = model_id
        self._root_cache: Dict[str, str] = {}
        self._morpheme_cache: Dict[str, List[str]] = {}
        self._chat_template: Optional[str] = None
        self._chat_template_jinja = False
        # Special tokens added beyond BPE vocab (chat template tokens like im_start, im_end)
        self.added_tokens: Dict[str, int] = added_tokens or {}
        self._added_token_patterns = self._build_added_token_patterns()

    def _build_added_token_patterns(self):
        """Build regex patterns for matching added tokens, longest first."""
        import re as _re
        if not self.added_tokens:
            return []
        # Sort by length descending so longest match wins
        sorted_tokens = sorted(self.added_tokens.keys(), key=len, reverse=True)
        # Escape for regex, match literally
        pattern = "|".join(_re.escape(t) for t in sorted_tokens)
        return _re.compile(pattern) if pattern else None

    def chat_stop_ids(self) -> Tuple[int, ...]:
        """Resolve chat-template turn-end stop ids for generation.

        Returns the ids of standard chat-template markers that terminate a
        model turn (``<|im_end|>``, ``<|endoftext|>``, ``<|im_start|>``) when
        defined by this tokenizer's ``added_tokens``, plus ``eos_token_id``.
        Markers absent from the tokenizer are skipped, so non-chat models
        get a set containing only the regular EOS token.

        Returns:
            Tuple of token ids that should stop generation.
        """
        _CHAT_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
        stop_ids = {self.eos_token_id} if self.eos_token_id is not None else set()
        for marker in _CHAT_STOP_MARKERS:
            tok_id = self.added_tokens.get(marker)
            if tok_id is not None:
                stop_ids.add(tok_id)
        return tuple(sorted(stop_ids))

    @classmethod
    def from_pretrained(cls, model_id: str) -> "MorphTokenizer":
        """Load from tokenizer.json — our own parser, no HF tokenizers lib."""
        from pathlib import Path

        model_slug = model_id.replace("/", "--")

        # Search multiple cache locations
        search_dirs = []
        # Direct local directory (e.g. a fine-tuned model dir with tokenizer.json)
        direct = Path(model_id)
        if direct.is_dir():
            search_dirs.append(direct)
        # Standard HF cache
        import os
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        search_dirs.append(Path(hf_home) / "hub" / f"models--{model_slug}")
        # Project-local cache (models/hf-cache/hub/)
        for candidate in [
            find_repo_root(Path(__file__).resolve()) / "models" / "hf-cache" / "hub" / f"models--{model_slug}",
            Path("models/hf-cache/hub") / f"models--{model_slug}",
        ]:
            if candidate.exists() and candidate not in search_dirs:
                search_dirs.append(candidate)

        tokenizer_path = None
        for model_dir in search_dirs:
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    candidate = snap / "tokenizer.json"
                    if candidate.exists():
                        tokenizer_path = candidate
                        break
            if tokenizer_path is None:
                candidate = model_dir / "tokenizer.json"
                if candidate.exists():
                    tokenizer_path = candidate
            if tokenizer_path is not None:
                break

        if tokenizer_path is None or not tokenizer_path.exists():
            raise FileNotFoundError(f"No tokenizer.json for {model_id} (searched {len(search_dirs)} dirs)")

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

        # Parse added tokens (chat template tokens like im_start, im_end)
        added_tokens = {}
        for at in tok_data.get("added_tokens", []):
            content_str = at.get("content", "")
            tid = at.get("id")
            if content_str and tid is not None:
                added_tokens[content_str] = tid
        if added_tokens:
            logger.info("Loaded %d added tokens for %s: %s",
                        len(added_tokens), model_id, list(added_tokens.keys())[:5],
                        extra={"tag": "INFRA"})

        logger.info("Loaded tokenizer %s (vocab=%d, merges=%d, byte_level=%s, byte_fallback=%s, added=%d)",
                     model_id, len(vocab), len(merges), byte_level, byte_fallback, len(added_tokens),
                     extra={"tag": "INFRA"})

        instance = cls(vocab=vocab, merges=merges, eos_token_id=eos,
                       byte_level=byte_level, byte_fallback=byte_fallback,
                       model_id=model_id, added_tokens=added_tokens)

        # Extract chat template — check tokenizer.json first, then tokenizer_config.json
        chat_tpl = tok_data.get("chat_template")
        if not chat_tpl:
            # tokenizer_config.json is usually in same dir as tokenizer.json
            tok_config_path = tokenizer_path.parent / "tokenizer_config.json"
            if not tok_config_path.exists():
                # Try parent (model dir / snapshot dir)
                tok_config_path = tokenizer_path.parent.parent / "tokenizer_config.json"
            if not tok_config_path.exists():
                # Broader search: walk snapshots
                tok_model_dir = tokenizer_path.parent.parent
                snapshots = tok_model_dir / "snapshots"
                if snapshots.exists():
                    for snap in snapshots.iterdir():
                        candidate = snap / "tokenizer_config.json"
                        if candidate.exists():
                            tok_config_path = candidate
                            break
            if tok_config_path.exists():
                try:
                    with open(tok_config_path) as f:
                        tok_config_data = json.load(f)
                    chat_tpl = tok_config_data.get("chat_template")
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("chat template parse failed: %s", exc)
        if chat_tpl:
            instance._chat_template = chat_tpl
            instance._chat_template_jinja = True
            logger.info("Loaded chat template for %s", model_id,
                extra={"tag": "INFRA"})

        return instance

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


    # -- Chat template -----------------------------------------------------------

    def apply_chat_template(self, messages):
        if not messages:
            return ''
        if self._chat_template:
            return self._render_chat_template(messages)
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            c = msg.get('content', '')
            if role == 'system':
                parts.append('System: ' + c + chr(10))
            elif role == 'user':
                parts.append('User: ' + c + chr(10))
            elif role == 'assistant':
                parts.append('Assistant: ' + c + chr(10))
        parts.append('Assistant:')
        return ''.join(parts)

    def _render_chat_template(self, messages):
        template = self._chat_template
        system_msg = ''
        conversation = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                system_msg = content
            else:
                conversation.append({'role': role, 'content': content})

        # Qwen/Llama ChatML style
        ims = chr(60) + '|im_start|' + chr(62)
        ime = chr(60) + '|im_end|' + chr(62)
        if 'im_start' in template:
            parts = []
            if system_msg:
                parts.append(ims + 'system' + chr(10) + system_msg + ime + chr(10))
            for msg in conversation:
                parts.append(ims + msg['role'] + chr(10) + msg['content'] + ime + chr(10))
            parts.append(ims + 'assistant' + chr(10))
            return ''.join(parts)

        # Generic for-loop template
        if 'for message in messages' in template:
            rendered = ''
            for msg in conversation:
                rendered += msg['role'] + chr(10) + msg['content'] + chr(10)
            result = rendered + 'assistant' + chr(10)
            if system_msg:
                result = 'system' + chr(10) + system_msg + chr(10) + chr(10) + result
            return result

        # Default ChatML
        parts = []
        if system_msg:
            parts.append(ims + 'system' + chr(10) + system_msg + ime + chr(10))
        for msg in conversation:
            parts.append(ims + msg['role'] + chr(10) + msg['content'] + ime + chr(10))
        parts.append(ims + 'assistant' + chr(10))
        return ''.join(parts)
# ── BPE encode ──────────────────────────────────────────────────────

    def _normalize_sentencepiece(self, text: str) -> str:
        """Apply SentencePiece normalization: replace spaces with \u2581, prepend \u2581."""
        return "▁" + text.replace(" ", "▁")

    def _bpe_encode(self, text: str) -> List[int]:
        """BPE encode text, handling byte_level and byte_fallback modes."""
        # Apply SentencePiece normalization for byte_fallback models
        if self.byte_fallback:
            text = self._normalize_sentencepiece(text)

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
            elif self.byte_fallback:
                # SentencePiece ByteFallback: encode each byte as <0xHH> token
                for byte_val in t.encode('utf-8'):
                    hex_token = f"<0x{byte_val:02X}>"
                    if hex_token in self.vocab:
                        ids.append(self.vocab[hex_token])
                    else:
                        ids.append(self.eos_token_id)
            elif self.byte_level:
                # Fallback: encode bytes individually
                for byte_val in t.encode('utf-8'):
                    char = _BYTE2UNICODE.get(byte_val, chr(byte_val))
                    ids.append(self.vocab.get(char, self.eos_token_id))
            else:
                ids.append(self.eos_token_id)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs via BPE, respecting special tokens."""
        if not self._added_token_patterns:
            return self._bpe_encode(text)
        # Split text by special tokens, keeping the tokens
        parts = self._added_token_patterns.split(text)
        ids: List[int] = []
        import re as _re
        all_tokens = self._added_token_patterns.findall(text)
        for i, part in enumerate(parts):
            if part:
                ids.extend(self._bpe_encode(part))
            if i < len(all_tokens):
                ids.append(self.added_tokens[all_tokens[i]])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        inv_added = {v: k for k, v in self.added_tokens.items()}
        tokens = []
        for tid in ids:
            tid = int(tid)
            if tid in inv_added:
                tokens.append(inv_added[tid])
            else:
                t = self.inv_vocab.get(tid, "")
                tokens.append(t)

        if self.byte_level:
            result = []
            for t in tokens:
                for ch in t:
                    byte_val = _UNICODE2BYTE.get(ch, ord(ch))
                    result.append(byte_val)
            return bytes(result).decode("utf-8", errors="replace")
        elif self.byte_fallback:
            # SentencePiece ByteFallback: <0xHH> tokens -> bytes
            # \u2581 in tokens means space-before-word (not at text start)
            result = bytearray()
            first = True
            for t in tokens:
                if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                    try:
                        result.append(int(t[3:5], 16))
                    except ValueError:
                        result.extend(t.encode("utf-8"))
                else:
                    # Strip \u2581 prefix; add space only if NOT the first token
                    cleaned = t.lstrip("\u2581")
                    n_stripped = len(t) - len(cleaned)
                    if n_stripped > 0 and not first:
                        result.extend(b" " * n_stripped)
                    first = False
                    result.extend(cleaned.encode("utf-8"))
            return result.decode("utf-8", errors="replace")
        else:
            return "".join(tokens)

    # ── HuggingFace-compatible __call__ API ──────────────────────────────

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """HuggingFace-compatible tokenizer call.

        Accepts ``text`` (str) and returns a dict with ``input_ids`` and
        ``attention_mask`` wrapped in :class:`_NumpyTokenTensor` objects
        that expose ``.to(device)`` (no-op for numpy/CPU) and ``.shape``.

        Args:
            text: Input text to tokenize.
            return_tensors: Ignored (always returns numpy-backed tensors).

        Returns:
            ``{"input_ids": _NumpyTokenTensor, "attention_mask": _NumpyTokenTensor}``
        """
        ids = self.encode(text)
        mask = [1] * len(ids)
        input_ids = _NumpyTokenTensor(np.array([ids], dtype=np.int64))
        attention_mask = _NumpyTokenTensor(np.array([mask], dtype=np.int64))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

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
