"""torch.cuda stub — numpy-backed (CPU-only)."""

is_available = lambda: False


class Event:
    def __init__(self): pass
    def wait(self): pass
    def record(self, stream=None): pass
    def elapsed_time(self, end_event): return 0.0


class Stream:
    def __init__(self, device=None, priority=0): pass
    def wait_event(self, event): pass
    def synchronize(self): pass


class default_stream:
    synchronize = lambda: None
