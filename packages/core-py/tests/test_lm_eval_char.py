"""Tests for domains/training/lm_eval_char.py — char-LM perplexity eval.

The main path is exercised end-to-end against a real tiny ``SloughGPTModel``
on the numpy SloNet stack (no PyTorch, no weights, no network). CLI wrapper
tests monkeypatch only the thin ``evaluate_...`` boundary.
"""

import math

import numpy as np
import pytest

import domains.training.lm_eval_char as ev
from domains.models import SloughGPTModel

VOCAB = 64
EMBED = 32
BLOCK = 8


def _make_model():
    return SloughGPTModel(vocab_size=VOCAB, n_embed=EMBED, n_layer=2, n_head=4,
                          block_size=BLOCK, dropout=0.1)


def _bundle(model):
    sd = {k: np.asarray(v) for k, v in model.state_dict().items()}
    stoi = {chr(ord("a") + i): i for i in range(26)}
    itos = {i: chr(ord("a") + i) for i in range(26)}
    return {
        "model_state_dict": sd,
        "stoi": stoi,
        "itos": itos,
        "training_info": {"vocab_size": VOCAB, "n_embed": EMBED, "n_layer": 2,
                          "n_head": 4, "block_size": BLOCK},
    }


class TestEvaluateSoulCharLm:
    def _write_soul(self, tmp_path, model, *, stoi=None, itos=None, chars=None):
        from domains.inference.slo_format import SloProfile, save_soul

        soul = SloProfile(name="eval-soul")
        cfg = {"vocab_size": VOCAB, "n_embed": EMBED, "n_layer": 2,
               "n_head": 4, "block_size": BLOCK}
        soul.metadata["config"] = cfg
        soul.metadata["vocab_size"] = VOCAB
        if stoi is not None:
            soul.metadata["stoi"] = stoi
        if itos is not None:
            soul.metadata["itos"] = itos
        if chars is not None:
            soul.metadata["chars"] = chars
        path = tmp_path / "tiny.soul"
        save_soul(model, str(path), soul_profile=soul)
        return path

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ev.evaluate_soul_char_lm("nope.soul", str(tmp_path / "missing.txt"))

    def test_invalid_soul_raises(self, tmp_path):
        p = tmp_path / "bad.soul"
        p.write_bytes(b"NOTSOUL")
        t = tmp_path / "t.txt"
        t.write_text("abc", encoding="utf-8")
        with pytest.raises(ValueError):
            ev.evaluate_soul_char_lm(str(p), str(t))

    def test_end_to_end_metrics(self, tmp_path):
        model = _make_model()
        stoi = {chr(ord("a") + i): i for i in range(26)}
        itos = {i: chr(ord("a") + i) for i in range(26)}
        soul_path = self._write_soul(tmp_path, model, stoi=stoi, itos=itos)
        text = "thequickbrownfoxjumpsoverthelazydog"
        p = tmp_path / "eval.txt"
        p.write_text(text, encoding="utf-8")
        out = ev.evaluate_soul_char_lm(str(soul_path), str(p))
        assert out["block_size"] == BLOCK
        assert out["vocab_size"] == VOCAB
        assert out["num_chars_skipped"] == 0
        assert out["warnings"] == []
        expected_n = ((len(text) - 1) // BLOCK) * BLOCK
        assert out["num_token_positions"] == expected_n
        assert out["mean_loss"] >= 0
        assert math.isclose(out["perplexity"], math.exp(out["mean_loss"]), rel_tol=1e-9)

    def test_builds_vocab_from_chars_metadata(self, tmp_path):
        model = _make_model()
        chars = [chr(ord("a") + i) for i in range(26)]
        soul_path = self._write_soul(tmp_path, model, chars=chars)
        p = tmp_path / "eval.txt"
        p.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
        out = ev.evaluate_soul_char_lm(str(soul_path), str(p))
        assert out["num_chars_skipped"] == 0
        assert out["warnings"] == []

    def test_eval_text_fallback_warning(self, tmp_path):
        model = _make_model()
        soul_path = self._write_soul(tmp_path, model)
        p = tmp_path / "eval.txt"
        p.write_text("hello world", encoding="utf-8")
        out = ev.evaluate_soul_char_lm(str(soul_path), str(p))
        assert any("no stoi/itos/chars" in w for w in out["warnings"])
        assert out["vocab_size"] == VOCAB

    def test_skipped_chars_warning(self, tmp_path):
        model = _make_model()
        stoi = {chr(ord("a") + i): i for i in range(26)}
        soul_path = self._write_soul(tmp_path, model, stoi=stoi)
        p = tmp_path / "mixed.txt"
        p.write_text("abcdefghijXYZklmnopqrstuvwxyz", encoding="utf-8")
        out = ev.evaluate_soul_char_lm(str(soul_path), str(p))
        assert out["num_chars_skipped"] == 3
        assert any("Skipped 3 characters" in w for w in out["warnings"])

    def test_max_chars_truncation_warning(self, tmp_path):
        model = _make_model()
        stoi = {"a": 0}
        soul_path = self._write_soul(tmp_path, model, stoi=stoi)
        p = tmp_path / "long.txt"
        p.write_text("a" * 100, encoding="utf-8")
        out = ev.evaluate_soul_char_lm(str(soul_path), str(p), max_chars=25)
        assert any("truncated from 100 to 25" in w for w in out["warnings"])
        assert out["num_token_positions"] == ((25 - 1) // BLOCK) * BLOCK

    def test_too_short_text_raises(self, tmp_path):
        model = _make_model()
        stoi = {"a": 0}
        soul_path = self._write_soul(tmp_path, model, stoi=stoi)
        p = tmp_path / "short.txt"
        p.write_text("ab", encoding="utf-8")
        with pytest.raises(ValueError):
            ev.evaluate_soul_char_lm(str(soul_path), str(p))


class TestMain:
    def test_json_output(self, tmp_path, monkeypatch, capsys):
        import json as json_lib
        monkeypatch.setattr(
            ev,
            "evaluate_soul_char_lm",
            lambda *a, **k: {"mean_loss": 2.0, "perplexity": math.exp(2), "num_token_positions": 8,
                             "num_chars_skipped": 0, "block_size": 8, "vocab_size": 64, "warnings": []},
        )
        monkeypatch.setattr("sys.argv", ["lm_eval_char", "--checkpoint", "c.soul",
                                         "--data", "d.txt", "--json"])
        ev.main()
        payload = json_lib.loads(capsys.readouterr().out)
        assert payload["perplexity"] == pytest.approx(math.exp(2))

    def test_json_output_inf_perplexity(self, monkeypatch, capsys):
        import json as json_lib
        monkeypatch.setattr(
            ev,
            "evaluate_soul_char_lm",
            lambda *a, **k: {"mean_loss": 200.0, "perplexity": float("inf"), "num_token_positions": 8,
                             "num_chars_skipped": 0, "block_size": 8, "vocab_size": 64, "warnings": []},
        )
        monkeypatch.setattr("sys.argv", ["lm_eval_char", "--checkpoint", "c.soul",
                                         "--data", "d.txt", "--json"])
        ev.main()
        payload = json_lib.loads(capsys.readouterr().out)
        assert payload["perplexity"] is None

    def test_error_exits_one(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise ValueError("bad checkpoint")

        monkeypatch.setattr(ev, "evaluate_soul_char_lm", _boom)
        monkeypatch.setattr("sys.argv", ["lm_eval_char", "--checkpoint", "c.soul", "--data", "d.txt"])
        with pytest.raises(SystemExit) as exc:
            ev.main()
        assert exc.value.code == 1
        assert "bad checkpoint" in capsys.readouterr().err
