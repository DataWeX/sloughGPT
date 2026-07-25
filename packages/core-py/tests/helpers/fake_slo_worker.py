"""
Fake SloNet worker for testing process-level isolation.

Replaces _slo_worker_main with a fake that uses the same Queue protocol
but no actual model loading. The subprocess imports this module instead
of domains.inference.slonet_provider.
"""
import os
import queue
import time
import multiprocessing as mp


def fake_slo_generate(slnc_path, model_id, req_q, resp_q, hb_q, worker_id):
    """Fake SloNet generate — echoes back without loading a real model."""
    hb_q.put_nowait(("ready", os.getpid()))

    while True:
        try:
            cmd, payload = req_q.get(timeout=0.5)
        except queue.Empty:
            hb_q.put_nowait(("alive", os.getpid()))
            continue

        if cmd == "stop":
            break

        if cmd == "generate":
            prompt, kwargs = payload
            try:
                resp_q.put_nowait(("result", {
                    "text": f"fake-slo({model_id}): {prompt}",
                    "tokens_generated": 1,
                    "elapsed_ms": 1.0,
                }))
            except Exception:
                pass

        if cmd == "generate_stream":
            prompt, kwargs = payload
            try:
                for word in f"fake-slo({model_id}): {prompt}".split():
                    resp_q.put_nowait(("token", word + " "))
                resp_q.put_nowait(("result", {
                    "text": "",
                    "tokens_generated": 5,
                    "elapsed_ms": 1.0,
                }))
            except Exception:
                pass

    hb_q.put_nowait(("dead", os.getpid()))
