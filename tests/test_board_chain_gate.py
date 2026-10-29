"""Gate tests for scripts/check_board_chain.py — the missing CI verifier.

Card 20261029: merge 5e5c8ce36 shipped 975 duplicate board rows and a
chain broken at every index because NOTHING ran verify_chain against the
real .kanban/board.jsonl (only tmp-dir unit tests). These tests pin each
corruption symptom the gate exists to catch, plus the healthy-board path
so the gate cannot go red on a well-formed board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "app-planner" / "src"))

import check_board_chain as gate  # noqa: E402
from app_planner.store import PlannerStore  # noqa: E402


def _write(board_file: Path, lines: list[str]) -> Path:
    board_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return board_file


def _sealed(tmp_path: Path, titles: list[str]) -> Path:
    """A healthy board: schema first, cards chained by compute_chains()."""
    store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
    for title in titles:
        store.add_card(title)
    store.compute_chains()
    assert store.verify_chain() == []
    return tmp_path / "board.jsonl"


def _lines(board_file: Path) -> list[str]:
    return board_file.read_text(encoding="utf-8").splitlines()


class TestHealthyBoard:
    def test_sealed_board_passes(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a", "b", "c"])
        assert gate.check_board(board) == []

    def test_missing_file_is_nothing_to_guard(self, tmp_path: Path):
        assert gate.check_board(tmp_path / "board.jsonl") == []

    def test_real_repo_board_passes(self):
        """The gate must be green on the repo's actual board — a red gate
        here would block every push, so its own base state is asserted."""
        board = REPO_ROOT / ".kanban" / "board.jsonl"
        assert board.exists()
        assert gate.check_board(board) == []


class TestCorruptionSymptoms:
    """One test per symptom merge 5e5c8ce36 exhibited."""

    def test_duplicate_card_id(self, tmp_path: Path):
        board = _sealed(tmp_path, ["unique"])
        lines = _lines(board)
        card_line = next(l for l in lines if '"id"' in l and '"schema"' not in l)
        _write(board, lines + [card_line])  # append a second copy of the card
        findings = gate.check_board(board)
        assert any("duplicate card id" in f for f in findings), findings

    def test_schema_header_at_eof(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a"])
        lines = _lines(board)
        schema_idx = next(i for i, l in enumerate(lines) if '"schema"' in l)
        reordered = lines[:schema_idx] + lines[schema_idx + 1 :] + [lines[schema_idx]]
        _write(board, reordered)
        findings = gate.check_board(board)
        assert any("schema header on line" in f for f in findings), findings

    def test_missing_schema_header(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a"])
        lines = [l for l in _lines(board) if '"schema"' not in l]
        _write(board, lines)
        findings = gate.check_board(board)
        assert any("missing" in f and "schema" in f for f in findings), findings

    def test_unsealed_card_breaks_chain(self, tmp_path: Path):
        """A card appended without compute_chains() (chain_index=-1) must
        fail — sealing before landing is the whole point of the gate."""
        board = _sealed(tmp_path, ["a", "b"])
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        store.add_card("never sealed")  # add_card does NOT auto-seal
        findings = gate.check_board(board)
        assert any("hash chain broken" in f for f in findings), findings

    def test_tampered_card_content_breaks_chain(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a", "b"])
        lines = _lines(board)
        tampered = [
            l.replace('"a"', '"TAMPERED"') if '"id"' in l and '"schema"' not in l else l
            for l in lines
        ]
        _write(board, tampered)
        findings = gate.check_board(board)
        assert any("hash chain broken" in f for f in findings), findings

    def test_unparseable_line(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a"])
        _write(board, _lines(board) + ["{not json"])
        findings = gate.check_board(board)
        assert any("unparseable JSON" in f for f in findings), findings

    def test_not_an_object_line(self, tmp_path: Path):
        board = _sealed(tmp_path, ["a"])
        _write(board, _lines(board) + ["[1, 2, 3]"])
        findings = gate.check_board(board)
        assert any("not a JSON object" in f for f in findings), findings

    def test_corruption_does_not_report_chain_noise(self, tmp_path: Path):
        """Structural findings short-circuit the chain check — one clear
        diagnosis, not a 2041-line stack of index noise (the merge-era file
        would produce an unreadable wall otherwise)."""
        board = _sealed(tmp_path, ["a"])
        lines = _lines(board)
        card_line = next(l for l in lines if '"id"' in l and '"schema"' not in l)
        _write(board, lines + [card_line])
        findings = gate.check_board(board)
        assert not any("hash chain broken" in f for f in findings), findings


class TestMainExitCodes:
    def test_exit_0_healthy(self, tmp_path: Path, capsys):
        board = _sealed(tmp_path, ["ok"])
        assert gate.main([str(gate.__file__), str(board)]) == 0
        assert capsys.readouterr().err == ""

    def test_exit_1_corrupted(self, tmp_path: Path, capsys):
        board = _sealed(tmp_path, ["ok"])
        lines = _lines(board)
        card_line = next(l for l in lines if '"id"' in l and '"schema"' not in l)
        _write(board, lines + [card_line])
        assert gate.main([str(gate.__file__), str(board)]) == 1
        err = capsys.readouterr().err
        assert "FAIL" in err and "duplicate card id" in err
        assert "no-verify" in err  # prints the documented bypass

    def test_default_target_is_repo_board(self, tmp_path: Path):
        """No-arg default must point at the real board, not cwd-relative."""
        out = gate.main([str(gate.__file__)])
        assert out == (0 if gate.check_board(REPO_ROOT / ".kanban" / "board.jsonl") == [] else 1)


class TestMergeUnionReplay:
    def test_replay_of_5e5c8ce36_shape_is_caught(self, tmp_path: Path):
        """Reproduce the corruption's SHAPE (two parents union-merged):
        parent B's lines appended after parent A's, schema line dragged to
        EOF. The gate must name all three symptoms."""
        board_a = _sealed(tmp_path / "a", ["card-1", "card-2"])
        lines_a = _lines(board_a)
        # second parent: same card-1 id, different content (the union-merge case)
        card2 = json.loads(next(l for l in lines_a if '"card-2"' in l))
        card2["title"] = "card-2 (other side, newer)"
        other = [l for l in lines_a if '"card-2"' not in l]
        other.append(json.dumps(card2, sort_keys=True, ensure_ascii=False))

        merged = tmp_path / "merged"
        merged.mkdir()
        merged_board = merged / "board.jsonl"
        # union merge shape: A's lines + B's extra lines, schema ends up last
        union = lines_a + [l for l in other if '"schema"' not in l or l is other[0]]
        schema = next(l for l in lines_a if '"schema"' in l)
        union = [l for l in union if '"schema"' not in l] + [schema]
        _write(merged_board, union)

        findings = gate.check_board(merged_board)
        joined = "\n".join(findings)
        assert "duplicate card id" in joined
        assert "schema header on line" in joined
        assert gate.main([str(gate.__file__), str(merged_board)]) == 1
