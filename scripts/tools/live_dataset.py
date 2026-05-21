"""Live JSONL streaming dataset.

The dataset watches a directory for ``*.jsonl`` files. Each line in a JSONL file is
expected to be a mapping with at least the keys ``prompt`` and ``completion``. The
two strings are concatenated (with a newline separator) and tokenised using a
simple character‑level tokenizer (the same as used in ``train.py``). The token
stream is then sliced into fixed‑size blocks of length ``block_size``. For each
block we return a tuple ``(input_ids, target_ids)`` where ``target_ids`` is the
input shifted by one token – exactly what a causal language model expects.

The implementation is deliberately lightweight: on every ``__iter__`` call we:
1. Scan the directory for new ``*.jsonl`` files.
2. Load all lines, tokenise, and extend an internal buffer.
3. Yield blocks from the buffer until it is exhausted, then repeat the scan.

This design works both for a static dataset (all files present at start) and for a
live‑updating scenario where new JSONL files appear while training is running.
"""

import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

import torch


class SimpleTokenizer:
    """Character‑level tokenizer compatible with the one in ``train.py``.

    It builds a vocabulary from the *initial* corpus that is passed during
    construction.  The vocabulary is frozen – new characters encountered later
    will raise a ``KeyError``; this mirrors the behaviour of many simple demos.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


from torch.utils.data import IterableDataset

class LiveJSONLDataset(IterableDataset):
    """Iterable dataset that streams token blocks from JSONL files.

    Parameters
    ----------
    data_dir: str or Path
        Directory that contains ``*.jsonl`` files.
    block_size: int
        Length of each token block fed to the model.
    buffer_size: int, optional
        Number of tokens to keep in memory before yielding blocks. Larger buffers
        reduce the number of filesystem scans but use more RAM. Default is
        ``block_size * 100``.
    """

    def __init__(self, data_dir: str, block_size: int, buffer_size: Optional[int] = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.buffer_size = buffer_size or block_size * 100
        self._tokenizer: Optional[SimpleTokenizer] = None
        self._buffer: List[int] = []
        self._seen_files: set[Path] = set()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_tokenizer(self, sample_text: str) -> None:
        """Create the ``SimpleTokenizer`` from a sample of text.
        """
        self._tokenizer = SimpleTokenizer(sample_text)

    def _load_new_files(self) -> None:
        """Read any JSONL files that have not been processed yet.
        The content of each file is concatenated into a single string consisting
        of ``prompt`` + ``\n`` + ``completion`` for every line.
        """
        assert self._tokenizer is not None, "Tokenizer must be initialised before loading files"
        for file_path in self.data_dir.glob("*.jsonl"):
            if file_path in self._seen_files:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # Skip malformed lines – they are not critical for training
                            continue
                        prompt = obj.get("prompt", "")
                        completion = obj.get("completion", "")
                        text = f"{prompt}\n{completion}\n"
                        self._buffer.extend(self._tokenizer.encode(text))
                self._seen_files.add(file_path)
            except OSError:
                # If the file disappears while we are reading, just ignore it
                continue

    def _maybe_yield_block(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the next ``(input, target)`` block if enough tokens are cached.
        """
        if len(self._buffer) < self.block_size + 1:
            return None
        # Extract a block of size ``block_size`` and the shifted target
        block = self._buffer[: self.block_size]
        target = self._buffer[1 : self.block_size + 1]
        # Remove the used tokens from the front of the buffer
        self._buffer = self._buffer[1:]
        x = torch.tensor(block, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y

    # ---------------------------------------------------------------------
    # Public iterator protocol
    # ---------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Initialise tokenizer on first iteration using a quick sample of the data
        if self._tokenizer is None:
            # Grab a small snippet from the first available file (if any)
            sample_text = ""
            for p in self.data_dir.glob("*.jsonl"):
                try:
                    with open(p, "r", encoding="utf-8") as fp:
                        for line in fp:
                            obj = json.loads(line.strip())
                            sample_text += obj.get("prompt", "") + "\n" + obj.get("completion", "") + "\n"
                            if len(sample_text) > 1000:
                                break
                except Exception:
                    continue
                if sample_text:
                    break
            if not sample_text:
                raise RuntimeError("No JSONL data found to build tokenizer")
            self._build_tokenizer(sample_text)

        # Main streaming loop – continue indefinitely (training can stop via external
        # epoch logic). We keep yielding blocks until the process is terminated.
        while True:
            # Try to produce a block from the current buffer
            maybe = self._maybe_yield_block()
            if maybe is not None:
                yield maybe
                continue

            # If we don't have enough tokens, load more files (or wait for new ones)
            self._load_new_files()

            # If after loading we still cannot produce a block, the buffer is too
            # small – truncate it to free memory and sleep briefly.
            if len(self._buffer) < self.block_size + 1:
                # Avoid busy‑waiting; pause a short time before rescanning.
                import time

                time.sleep(0.5)
                continue
