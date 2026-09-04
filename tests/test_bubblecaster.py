import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bubblecaster as bc  # noqa: E402


def _make_star(x=bc.CASTER_X, y=8.0, **kw):
    return bc.Star(x=x, y=y, radius=bc.STAR_RADIUS, **kw)


def test_cast_vertical_hits_high_star():
    star = _make_star(y=8.0)
    cast = bc.Cast(angle=88.0, power=12.0, sigil="◆", target_index=0)
    result = bc.simulate_cast([star], cast)
    assert result["hit"] is True
    assert result["hit_index"] == 0


def test_weak_cast_falls_short():
    star = _make_star(y=8.0)
    cast = bc.Cast(angle=65.0, power=bc.MIN_POWER, sigil="◇", target_index=0)
    result = bc.simulate_cast([star], cast)
    assert result["hit"] is False
    assert result["hit_index"] is None


def test_append_and_count_records(monkeypatch, tmp_path):
    data_dir = tmp_path / "bubblecaster"
    monkeypatch.setattr(bc, "DATA_DIR", data_dir)
    monkeypatch.setattr(bc, "DATA_PATH", data_dir / "casts.jsonl")

    state = bc.GameState(stars=[_make_star()])
    cast = bc.Cast(angle=88.0, power=12.0, sigil="✦", target_index=0)
    result = bc.simulate_cast(state.stars, cast)

    assert bc.append_record(state, cast, result, 2) == 1
    assert bc.append_record(state, cast, result, 2) == 2

    lines = (data_dir / "casts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["event"] == "cast"
    assert rec["session"] == state.session_id
    assert rec["cast"]["power"] == 12.0
    assert rec["outcome"]["hit"] is True


def test_fit_aim_model_learns_linear_power():
    # dx -> power relationship is approximately linear; feed noisy samples
    records = []
    for dx, power in [(5, 3), (10, 6), (15, 8), (20, 11), (30, 16)]:
        records.append({
            "cast": {"target_index": 0, "power": power, "angle": 60.0, "sigil": "◆"},
            "sky": [bc.Star(x=bc.CASTER_X + dx, y=6.0, radius=bc.STAR_RADIUS)],
        })
    model = bc.fit_aim_model(records)
    assert model is not None
    a, b = model
    # Predicted power at dx=10 should be near the observed value of 6
    assert abs(a * 10 + b - 6) < 3


def test_fit_aim_model_needs_enough_samples():
    assert bc.fit_aim_model([]) is None
    thin = [
        {"cast": {"target_index": 0, "power": 5.0}, "sky": [bc.Star(x=bc.CASTER_X + 8, y=6.0)]}
        for _ in range(3)
    ]
    assert bc.fit_aim_model(thin) is None
