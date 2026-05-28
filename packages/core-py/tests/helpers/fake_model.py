"""
Fake model for testing process-level isolation in worker subprocesses.

Uses numpy arrays instead of torch to avoid subprocess import hangs.
"""
import time
import numpy as np


class FakeTokenizer:
    """Replaces a HuggingFace tokenizer in worker subprocess tests."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

    def decode(self, token_ids, **kwargs):
        return self._reply

    def __getattr__(self, name):
        return 0


class FakeTestModel:
    """A pickleable test model that runs inside the worker subprocess."""

    def __init__(self, reply: str = "hello from worker", delay: float = 0.0):
        self._reply = reply
        self._delay = delay
        self.tokenizer = FakeTokenizer()
        self.tokenizer._reply = reply
        self.tokenizer.decode = lambda ids, **kw: reply
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 0

    def generate(self, **kwargs):
        if self._delay > 0:
            time.sleep(self._delay)
        input_ids = kwargs.get("input_ids")
        prompt_len = input_ids.shape[1] if input_ids is not None else 5
        return np.zeros((1, prompt_len + 5), dtype=np.int64)

    def to(self, device):
        return self

    @property
    def device(self):
        return "cpu"
