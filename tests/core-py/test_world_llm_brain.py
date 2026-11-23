"""Stage B tests — LLMBrain (token-space) and the world worker pool.

Covers the brain contract (registry, serialization, parsing, neutral
fallback, no RNG consumption), the non-genome breeding rules (spawn_child
inherits whole; the flat genome refuses), worker-pool seeded parity, and
the Stage-B CLI surface (--brain llm, --runs/--workers sweep).

No mock anywhere (user ruling 2026-11-21): the model path runs against a
REAL local InferenceEngine — already-running at ``$SLO_TEST_ENGINE_PORT``
or ``127.0.0.1:9100``, else spawned as a subprocess (skipped when CI has
no engine) — the transport-failure path against a real dead port
(connection refused), and answer parsing as the pure function it is.
"""

import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from domain.shell._internal.evolution import Genome
from domain.shell._internal.llm_brain import (
    LLMBrain,
    LocalEngineClient,
    get_default_client,
    make_client,
    reset_default_client,
)
from domain.shell._internal.model_brain import BRAIN_KINDS, brain_from_dict, make_brain
from domain.shell._internal.simulation import SimBaby, WorldParams
from domain.shell._internal.world_driver import WorldDriver, main
from domain.shell._internal.world_workers import run_world_runs


def _dead_client() -> LocalEngineClient:
    """A real client pointed at a closed port — connection refused, no listener."""
    return LocalEngineClient("127.0.0.1", 1)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _engine_alive(port: int) -> bool:
    from domain.infrastructure._internal.inference_client import InferenceClient

    try:
        client = InferenceClient(host="127.0.0.1", port=port, connect_timeout=2.0)
        alive = bool(client.connect())
        client.disconnect()
        return alive
    except Exception:  # noqa: BLE001 — any transport answer means "not usable"
        return False


@pytest.fixture(scope="module")
def engine_port():
    """A real local InferenceEngine, or skip (no mock anywhere).

    Resolution order: ``$SLO_TEST_ENGINE_PORT`` → an already-running engine
    on ``127.0.0.1:9100`` → spawn the engine as a subprocess. CI ships no
    model weights and starts no engine, so it skips: tests that need a real
    model are local-only by design (user ruling — real engine in tests,
    skip or fail without it).
    """
    env_port = os.environ.get("SLO_TEST_ENGINE_PORT")
    if env_port:
        port = int(env_port)
        if _engine_alive(port):
            yield port
            return
        pytest.skip(f"SLO_TEST_ENGINE_PORT={port} is not answering")
    if _engine_alive(9100):
        yield 9100
        return
    if os.environ.get("CI"):
        pytest.skip("no running engine in CI — real-model tests are local-only")

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "domain.infrastructure._internal.inference_engine",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()
    deadline = time.monotonic() + 90.0
    ready = False
    while time.monotonic() < deadline:
        if any(f"ENGINE_READY port={port}" in line for line in lines):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    if not ready:
        proc.kill()
        pytest.skip(f"engine did not become ready: {' | '.join(lines[-3:])}")
    yield port
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _small_params(**kw) -> WorldParams:
    return WorldParams(grid_size=(8, 6, 8), generate_world=True, start_agents=2, **kw)


def _spec(seed: int, kind: str = "ticks", **kw) -> dict:
    spec = {"seed": seed, "params": _small_params(world_seed=seed), "kind": kind}
    spec.update(kw)
    return spec


# ── registry ────────────────────────────────────────────────────────────────


def test_llm_registered_in_brain_kinds():
    assert "llm" in BRAIN_KINDS
    brain = make_brain("llm", 5, 3, hidden_units=4, n_embed=16, n_layer=2)
    assert isinstance(brain, LLMBrain)
    assert brain.input_dim == 5
    assert brain.output_dim == 3
    assert brain.hidden_units == 0  # weight-space concept dropped by design


def test_llm_roundtrip_through_brain_from_dict():
    brain = LLMBrain(4, 2)
    payload = brain.to_dict()
    assert payload == {"kind": "llm", "input_dim": 4, "output_dim": 2}
    rebuilt = brain_from_dict(payload)
    assert isinstance(rebuilt, LLMBrain)
    assert (rebuilt.input_dim, rebuilt.output_dim) == (4, 2)


def test_brain_from_dict_unknown_kind_fails_loud():
    with pytest.raises(ValueError, match="unknown brain kind"):
        brain_from_dict({"kind": "nope", "input_dim": 1, "output_dim": 1})


def test_default_client_is_the_real_local_engine():
    """Until configured, the process default is the real local engine (no mock)."""
    reset_default_client()
    assert isinstance(get_default_client(), LocalEngineClient)
    reset_default_client()


# ── decision contract (hermetic — no model needed) ──────────────────────────


def test_prompt_varies_with_observation():
    """The owned half of decision variance: obs → byte-stable, distinct requests.

    That two prompts yield two different ANSWERS is the model's contract,
    not ours (Test Rule — owned code only); what we own is that the same
    observation always sends the identical request and a different one
    never does.
    """
    brain = LLMBrain(3, 3, client=_dead_client())
    zero = brain._render_prompt(np.zeros(3, dtype=np.float32))
    assert zero == brain._render_prompt(np.zeros(3, dtype=np.float32))
    one = brain._render_prompt(np.ones(3, dtype=np.float32))
    assert zero != one


def test_parse_pads_and_clips():
    brain = LLMBrain(3, 4, client=_dead_client())
    gates = brain._parse_gates("1.5, -0.2, 0.7")
    # taken in order, clipped to [0,1], missing 4th gate padded neutral
    assert gates.dtype == np.float32
    assert np.allclose(gates, [1.0, 0.0, 0.7, 0.5])


def test_unparseable_answer_is_neutral():
    brain = LLMBrain(2, 3, client=_dead_client())
    assert np.allclose(brain._parse_gates("no numbers here"), [0.5, 0.5, 0.5])


def test_dead_engine_steers_neutral():
    """Real transport failure (connection refused) → neutral, never a raised tick."""
    brain = LLMBrain(2, 2, client=_dead_client())
    gates = brain.forward(np.zeros(2, dtype=np.float32))
    assert np.allclose(gates, [0.5, 0.5])


def test_update_is_noop():
    brain = LLMBrain(3, 2, client=_dead_client())
    before = brain.to_dict()
    brain.update(np.ones(3, dtype=np.float32), np.array([1.0, -1.0]), lr=0.5)
    assert brain.to_dict() == before


def test_consumes_no_rng():
    """Enabling an LLM brain must not perturb the seeded world's RNG stream."""
    before = np.random.get_state()
    brain = LLMBrain(4, 3, client=_dead_client())
    x = np.zeros(4, dtype=np.float32)
    brain.forward(x)
    brain.update(x, np.ones(3, dtype=np.float32))
    brain_from_dict(brain.to_dict())
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_counts_model_calls():
    brain = LLMBrain(2, 2, client=_dead_client())
    assert brain.calls == 0
    brain.forward(np.zeros(2, dtype=np.float32))
    brain.forward(np.zeros(2, dtype=np.float32))
    assert brain.calls == 2


# ── real engine (the model path — no canned answers) ────────────────────────


class TestRealEngine:
    """End-to-end against a real local InferenceEngine (fixture skips if absent)."""

    def test_forward_deterministic_on_same_observation(self, engine_port):
        client = LocalEngineClient("127.0.0.1", engine_port)
        x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        a = LLMBrain(3, 3, client=client).forward(x)
        b = LLMBrain(3, 3, client=client).forward(x)
        # temperature=0.0 → greedy argmax (slonet._sample_token), so the
        # identical request must come back as identical gates.
        assert np.array_equal(a, b)
        assert a.shape == (3,)
        assert np.all((0.0 <= a) & (a <= 1.0))

    def test_forward_decides_from_the_observation(self, engine_port):
        """Obs → prompt → real model → parsed gates, for both extremes."""
        brain = LLMBrain(3, 3, client=LocalEngineClient("127.0.0.1", engine_port))
        for x in (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)):
            gates = brain.forward(x)
            assert gates.shape == (3,)
            assert np.all((0.0 <= gates) & (gates <= 1.0))
        assert brain.calls == 2


# ── breeding rules (non-genome tier) ────────────────────────────────────────


def test_spawn_child_inherits_llm_brain_whole():
    baby = SimBaby(params=WorldParams(brain_kind="llm"), initial_energy=150.0)
    assert isinstance(baby.perceptron_cells, LLMBrain)
    child = baby.spawn_child(position=np.array([2.0, 2.0, 2.0]))
    assert isinstance(child.perceptron_cells, LLMBrain)
    assert child.perceptron_cells.to_dict() == baby.perceptron_cells.to_dict()


def test_genome_from_baby_refuses_llm_brain():
    baby = SimBaby(params=WorldParams(brain_kind="llm"), initial_energy=150.0)
    with pytest.raises(NotImplementedError, match="llm"):
        Genome.from_baby(baby)


def test_genome_apply_to_refuses_llm_brain():
    params = WorldParams(brain_kind="llm")
    baby = SimBaby(params=params, initial_energy=150.0)
    genome = Genome.random(params, np.random.default_rng(0))
    with pytest.raises(NotImplementedError, match="llm"):
        genome.apply_to(baby)


# ── model client factory ────────────────────────────────────────────────────


def test_make_client_by_backend():
    local = make_client("local", "127.0.0.1:9999")
    assert local.name == "local"
    assert (local.host, local.port) == ("127.0.0.1", 9999)
    api = make_client("api", "http://example.invalid/v1")
    assert api.name == "api"
    assert api.base_url == "http://example.invalid/v1"
    # the mock backend is gone by user ruling — it must fail loud, not revive
    with pytest.raises(ValueError, match="unknown model backend"):
        make_client("mock")
    with pytest.raises(ValueError, match="unknown model backend"):
        make_client("nope")


# ── worker pool: seeded parity ──────────────────────────────────────────────


def test_workers_ticks_parity_serial_vs_pool():
    specs = [_spec(seed) for seed in (7, 8, 9)]
    serial = run_world_runs(specs, max_workers=1)
    pooled = run_world_runs(specs, max_workers=2)
    assert serial == pooled  # bit-identical: each run seeds its own process
    assert [r["seed"] for r in pooled] == [7, 8, 9]  # input order preserved


def test_workers_match_inline_world_driver():
    spec = _spec(7, ticks=12)
    direct = WorldDriver(spec["params"], seed=7).run_ticks(12)[-1]
    via_pool = run_world_runs([_spec(7, ticks=12)], max_workers=2)[0]["snapshot"]
    assert via_pool == direct


def test_workers_evolution_parity_serial_vs_pool():
    specs = [
        _spec(seed, kind="evolution", generations=2, population=4, ticks_per_generation=6)
        for seed in (5, 6)
    ]

    def _normalize(results):
        # Genome has no __eq__ (identity semantics elsewhere), so compare its
        # tensors as lists; everything else is plain data and must match exactly.
        normalized = []
        for r in results:
            r = dict(r)
            if r.get("kind") == "evolution":
                summary = dict(r["summary"])
                genome = summary.pop("best_genome", None)
                summary["best_genome_tensors"] = (
                    None if genome is None else {k: v.tolist() for k, v in genome.tensors.items()}
                )
                r["summary"] = summary
            normalized.append(r)
        return normalized

    serial = run_world_runs(specs, max_workers=1)
    pooled = run_world_runs(specs, max_workers=2)
    assert _normalize(serial) == _normalize(pooled)


def test_workers_empty_and_unknown_kind():
    assert run_world_runs([]) == []
    with pytest.raises(ValueError, match="unknown world run kind"):
        run_world_runs([{"seed": 1, "params": _small_params(), "kind": "bogus"}], max_workers=1)


# ── CLI surface ─────────────────────────────────────────────────────────────


def test_cli_llm_smoke(capsys):
    """--brain llm runs end-to-end hermetically: dead port → neutral steering.

    The wiring (flag → make_client → set_default_client → babies) is what
    this proves; the model itself is covered by TestRealEngine above.
    """
    code = main(
        [
            "--grid",
            "8,6,8",
            "--babies",
            "2",
            "--seed",
            "7",
            "--ticks",
            "6",
            "--brain",
            "llm",
            "--model-backend",
            "local",
            "--model-url",
            "127.0.0.1:1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "total_energy" in out  # tick table printed


def test_cli_runs_sweep_prints_one_line_per_seed(capsys):
    code = main(
        [
            "--grid",
            "8,6,8",
            "--babies",
            "2",
            "--seed",
            "7",
            "--ticks",
            "8",
            "--runs",
            "3",
            "--workers",
            "2",
        ]
    )
    assert code == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert [ln.split()[0] for ln in lines] == ["7", "8", "9"]
    assert all("ticks" in ln for ln in lines)


def test_cli_rejects_unknown_model_backend(capsys):
    code = main(["--brain", "llm", "--model-backend", "bogus", "--ticks", "1"])
    assert code == 2  # argparse rejects the choice before any world builds
