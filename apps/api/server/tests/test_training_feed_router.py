"""Tests for the training data feed endpoints (``/training/feed``).

Serves the live corpus (API conversations) as a paginated JSON feed with an
offset cursor, so training consumers can point a URL at the flow and read only
records written since their last position.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from training import feeds
from training.execution import _feed_spec
from training.router import router
from training.schemas import TrainingRequest

app = FastAPI()
register_app_error_handler(app)
app.include_router(router)
client = TestClient(app)


def _corpus_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"model": "test", "captured_at": "2026-09-22T00:00:00+00:00"},
    }


@pytest.fixture(autouse=True)
def _feed_sources(tmp_path, monkeypatch) -> Iterator[Path]:
    """Point feed sources at a temp corpus with 3 valid records + 1 bad line."""
    corpus = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps(_corpus_record("first user query", "first assistant reply")),
        "this line is not json at all",
        json.dumps(_corpus_record("second user query", "second assistant reply")),
        json.dumps(_corpus_record("third user query", "third assistant reply")),
    ]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        feeds,
        "_SOURCES",
        {
            "api-conversations": str(corpus),
            "feedback": str(tmp_path / "missing.jsonl"),
        },
    )
    yield corpus


# ── GET /training/feed ────────────────────────────────────────────────────────


class TestTrainingFeedRead:
    def test_returns_all_records_by_default(self):
        resp = client.get("/training/feed")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "api-conversations"
        assert data["total"] == 4
        assert data["offset"] == 0
        assert data["next_offset"] == 4
        assert len(data["records"]) == 3  # malformed line skipped

    def test_offset_sequences_through_feed(self):
        resp = client.get("/training/feed", params={"offset": 2})
        data = resp.json()["data"]
        assert data["offset"] == 2
        assert data["next_offset"] == 4
        assert len(data["records"]) == 2

    def test_limit_slices_records(self):
        resp = client.get("/training/feed", params={"offset": 2, "limit": 2})
        data = resp.json()["data"]
        assert data["next_offset"] == 4
        assert len(data["records"]) == 2
        assert data["records"][0]["messages"][0]["content"] == "second user query"

    def test_limit_zero_returns_metadata_only(self):
        resp = client.get("/training/feed", params={"offset": 0, "limit": 0})
        data = resp.json()["data"]
        assert data["records"] == []
        assert data["total"] == 4

    def test_offset_beyond_end_returns_empty_records(self):
        resp = client.get("/training/feed", params={"offset": 10})
        data = resp.json()["data"]
        assert data["records"] == []
        assert data["next_offset"] == 4

    def test_other_source_by_name(self):
        resp = client.get("/training/feed", params={"source": "feedback"})
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 0

    def test_directory_source_aggregates_jsonl_shards(self, tmp_path, monkeypatch):
        shard_dir = tmp_path / "response_shards"
        shard_dir.mkdir()
        (shard_dir / "responses_20260729.jsonl").write_text(
            json.dumps(_corpus_record("older user query", "older assistant answer")) + "\n",
            encoding="utf-8",
        )
        (shard_dir / "responses_20260730.jsonl").write_text(
            json.dumps(_corpus_record("newer user query", "newer assistant answer")) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(feeds, "_SOURCES", {"response-logs": str(shard_dir)})

        data = client.get("/training/feed", params={"source": "response-logs"}).json()["data"]
        assert data["total"] == 2
        assert len(data["records"]) == 2
        assert data["records"][0]["messages"][0]["content"] == "older user query"

        page = client.get("/training/feed", params={"source": "response-logs", "offset": 1}).json()[
            "data"
        ]
        assert page["next_offset"] == 2
        assert len(page["records"]) == 1
        assert page["records"][0]["messages"][0]["content"] == "newer user query"

    def test_unknown_source_is_400(self):
        resp = client.get("/training/feed", params={"source": "not-a-source"})
        assert resp.status_code == 400

    def test_negative_offset_is_400(self):
        assert client.get("/training/feed", params={"offset": -1}).status_code == 400

    def test_limit_over_max_is_400(self):
        assert (
            client.get("/training/feed", params={"limit": feeds._MAX_RECORDS + 1}).status_code
            == 400
        )


# ── GET /training/feeds ───────────────────────────────────────────────────────


class TestTrainingFeedsList:
    def test_lists_sources_with_totals(self):
        resp = client.get("/training/feeds")
        assert resp.status_code == 200
        feeds_map = {f["name"]: f for f in resp.json()["data"]["feeds"]}
        assert "api-conversations" in feeds_map
        assert feeds_map["api-conversations"]["total"] == 4
        assert feeds_map["api-conversations"]["source"] == "api-conversations"
        # missing file still listed with total 0
        assert feeds_map["feedback"]["total"] == 0

    def test_rows_carry_trainable_spec(self):
        resp = client.get("/training/feeds")
        feeds_map = {f["name"]: f for f in resp.json()["data"]["feeds"]}
        assert feeds_map["api-conversations"]["spec"] == "feed:api-conversations"
        assert feeds_map["feedback"]["spec"] == "feed:feedback"

    def test_directory_source_lists_total_and_mtime(self, tmp_path, monkeypatch):
        shard_dir = tmp_path / "response_shards"
        shard_dir.mkdir()
        (shard_dir / "responses_20260729.jsonl").write_text(
            json.dumps(_corpus_record("older user query", "older assistant answer")) + "\n",
            encoding="utf-8",
        )
        (shard_dir / "responses_20260730.jsonl").write_text(
            json.dumps(_corpus_record("newer user query", "newer assistant answer")) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(feeds, "_SOURCES", {"response-logs": str(shard_dir)})

        resp = client.get("/training/feeds")
        feeds_map = {f["name"]: f for f in resp.json()["data"]["feeds"]}
        assert feeds_map["response-logs"]["total"] == 2
        assert feeds_map["response-logs"]["last_updated"] is not None


# ── start_training feed short-circuit (_feed_spec) ─────────────────────────────


class TestFeedSpec:
    """The /training/start feed spec: point a feed URL at the training flow."""

    def _req(self, **kwargs):
        defaults = {"name": "feed job", "model": "slonet"}
        defaults.update(kwargs)
        return TrainingRequest(**defaults)

    def test_feed_source_tuple(self):
        spec = _feed_spec(self._req(dataset="feed:api-conversations"))
        assert spec == ("feed:api-conversations", "api-conversations", None, "feed")

    def test_http_url_uses_path_for_stem(self):
        spec = _feed_spec(self._req(dataset="http://localhost:8000/training/feed/corpus.jsonl"))
        assert spec[0] == "http://localhost:8000/training/feed/corpus.jsonl"
        assert spec[1] == "corpus_jsonl"
        assert spec[2] is None
        assert spec[3] == "feed"

    def test_plain_dataset_is_not_a_feed(self):
        assert _feed_spec(self._req(dataset="data/foo/input.txt")) is None

    def test_manifest_request_is_not_a_feed(self):
        # Exactly one of dataset / manifest_uri / dataset_ref is enforced by the
        # request model; a manifest-resolved job must not be treated as a feed.
        spec = _feed_spec(self._req(dataset=None, manifest_uri="data/manifests/corpus.json"))
        assert spec is None


# ── feed dataset validation (fail fast at the API boundary) ───────────────────


class TestFeedDatasetValidation:
    """A feed typo is a 400 with the valid specs — not a thread-death later."""

    def test_registered_source_passes(self):
        assert feeds.validate_feed_dataset("feed:feedback") == "feed:feedback"

    def test_non_feed_spec_passes_through(self):
        assert feeds.validate_feed_dataset("data/foo/input.txt") == "data/foo/input.txt"

    def test_empty_source_is_400(self):
        from domain.infrastructure._internal.errors import AppError

        with pytest.raises(AppError) as exc:
            feeds.validate_feed_dataset("feed:")
        assert exc.value.http_status == 400
        assert "feed:api-conversations" in exc.value.message

    def test_unknown_source_is_400_listing_valid_specs(self):
        from domain.infrastructure._internal.errors import AppError

        with pytest.raises(AppError) as exc:
            feeds.validate_feed_dataset("feed:not-a-real-source")
        assert exc.value.http_status == 400
        assert "feed:api-conversations" in exc.value.message
        assert "feed:feedback" in exc.value.message

    def test_resolve_registered_source(self):
        assert feeds.resolve_feed_source("feedback") is not None

    def test_resolve_resolvable_unregistered_source(self):
        # The corpus lives at data/api_conversations/ (underscore spelling);
        # the unregistered spelling must still resolve, mirroring the trainer.
        assert feeds.resolve_feed_source("api_conversations") is not None

    def test_resolve_typo_returns_none(self):
        assert feeds.resolve_feed_source("totally-not-a-corpus") is None

    def test_feed_spec_rejects_unknown_source_before_enqueue(self):
        from domain.infrastructure._internal.errors import AppError

        req = TrainingRequest(name="typo", model="slonet", dataset="feed:not-a-real-source")
        with pytest.raises(AppError) as exc:
            _feed_spec(req)
        assert exc.value.http_status == 400

    def test_start_with_unknown_feed_is_400_and_creates_no_job(self):
        from training.jobs import training_jobs

        before = set(training_jobs.keys())
        resp = client.post(
            "/training/start",
            json={"name": "typo job", "model": "slonet", "dataset": "feed:not-a-real-source"},
        )
        assert resp.status_code == 400
        assert "Unknown feed source" in resp.text
        assert set(training_jobs.keys()) == before


# ── live snapshot builder (/training/live-stream) ─────────────────────────────


class TestLiveSnapshot:
    """build_live_snapshot: per-frame deltas + job change detection."""

    @staticmethod
    def _rows(totals: dict[str, int]) -> list[dict]:
        return [
            {"name": name, "source": name, "spec": f"feed:{name}", "total": total}
            for name, total in sorted(totals.items())
        ]

    def test_first_frame_is_baseline_changed(self, monkeypatch):
        from training import live_stream

        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({"feedback": 7}))
        monkeypatch.setattr(live_stream, "training_jobs", {})
        snap, changed = live_stream.build_live_snapshot(None)
        assert changed is True  # initial frame worth rendering
        assert snap["totals"] == {"feedback": 7}
        assert snap["sources"][0]["delta"] == 0
        assert snap["jobs"] == {"active": [], "count": 0}

    def test_growth_produces_delta_and_changed(self, monkeypatch):
        from training import live_stream

        monkeypatch.setattr(live_stream, "training_jobs", {})
        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({"feedback": 7}))
        prev, _ = live_stream.build_live_snapshot(None)

        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({"feedback": 9}))
        snap, changed = live_stream.build_live_snapshot(prev)
        assert changed is True
        assert snap["sources"][0]["delta"] == 2
        assert snap["totals"]["feedback"] == 9

    def test_static_frame_reports_unchanged(self, monkeypatch):
        from training import live_stream

        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({"feedback": 7}))
        monkeypatch.setattr(live_stream, "training_jobs", {})
        prev, _ = live_stream.build_live_snapshot(None)
        snap, changed = live_stream.build_live_snapshot(prev)
        assert changed is False
        assert snap["sources"][0]["delta"] == 0

    def test_job_appearance_marks_changed(self, monkeypatch):
        from training import live_stream

        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({"feedback": 7}))
        monkeypatch.setattr(live_stream, "training_jobs", {})
        prev, _ = live_stream.build_live_snapshot(None)

        monkeypatch.setattr(
            live_stream,
            "training_jobs",
            {
                "job-1": {
                    "id": "job-1",
                    "name": "live job",
                    "status": "running",
                    "dataset": "feed:feedback",
                    "progress": 40,
                }
            },
        )
        snap, changed = live_stream.build_live_snapshot(prev)
        assert changed is True
        assert snap["jobs"]["count"] == 1
        assert snap["jobs"]["active"][0]["dataset"] == "feed:feedback"

    def test_terminal_jobs_are_excluded(self, monkeypatch):
        from training import live_stream

        monkeypatch.setattr(live_stream, "list_feed_sources", lambda: self._rows({}))
        monkeypatch.setattr(
            live_stream,
            "training_jobs",
            {
                "done": {"id": "done", "status": "completed", "progress": 100},
                "dead": {"id": "dead", "status": "failed", "progress": 10},
            },
        )
        snap, _ = live_stream.build_live_snapshot(None)
        assert snap["jobs"]["active"] == []
        assert snap["jobs"]["count"] == 0

    def test_live_stream_endpoint_emits_envelope(self, monkeypatch):
        # Drive the generator directly (repo SSE-test idiom): one frame, then
        # a scripted disconnect — proves wiring, headers, and envelope shape.
        import asyncio
        from unittest.mock import MagicMock

        from training import live_stream

        monkeypatch.setattr(live_stream, "LIVE_STREAM_INTERVAL", 0.01)

        # FastAPI may wrap includes as lazy _IncludedRouter — walk nested routers.
        def _paths(router_obj):
            for entry in router_obj.routes:
                path = getattr(entry, "path", None)
                if path:
                    yield path
                inner = getattr(entry, "original_router", None)
                if inner is not None:
                    yield from _paths(inner)

        assert "/training/live-stream" in list(_paths(router))
        checks = {"n": 0}

        async def fake_is_disconnected() -> bool:
            checks["n"] += 1
            return checks["n"] > 1  # first cycle runs; second breaks the loop

        mock_request = MagicMock()
        mock_request.is_disconnected = fake_is_disconnected

        async def run() -> tuple[str, str, list[dict]]:
            response = await live_stream.training_live_stream(
                request=mock_request, auth_user=None
            )
            frames: list[dict] = []
            async for chunk in response.body_iterator:
                text = chunk if isinstance(chunk, str) else chunk.decode()
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(json.loads(line[len("data: ") :]))
            return (
                response.media_type,
                response.headers.get("Cache-Control", ""),
                frames,
            )

        media_type, cache_header, frames = asyncio.run(run())
        assert media_type == "text/event-stream"
        assert cache_header == "no-cache"
        assert frames, "expected at least one SSE frame"
        frame = frames[0]
        assert frame["stream"] == "training-live"
        assert frame["status"] == "working"
        assert "sources" in frame["data"]
        assert "jobs" in frame["data"]


# ── POST /training/start with a feed dataset (full API journey) ────────────────


class _FakeTrainer:
    """Records the data_path and kwargs the endpoint feeds to SloughGPTTrainer, runs nothing."""

    captured: list[str] = []
    captured_kwargs: list[dict] = []

    def __init__(self, data_path, **kwargs) -> None:
        _FakeTrainer.captured.append(data_path)
        _FakeTrainer.captured_kwargs.append(kwargs)

    def train(self, **kwargs) -> dict:
        return {"best_eval_loss": 0.5}

    def save(self, path: str) -> None:  # noqa: ARG002 - stub, must not touch disk
        pass


class _FakeTracker:
    def end_run(self) -> None:
        pass


async def _async_noop(*args, **kwargs) -> None:  # noqa: ARG002
    return None


class TestTrainingStartFeed:
    """Feed dataset spec flows through /training/start end to end.

    The trainer itself is stubbed (hermetic — no compute, no models/ writes);
    the real endpoint pre-flight, feed short-circuit, job enqueue, background
    executor dispatch, and SloughGPTTrainer construction are all exercised.
    """

    @pytest.mark.parametrize(
        "dataset, stem",
        [
            ("feed:api-conversations", "api-conversations"),
            ("http://localhost:8000/training/feed/api-conversations", "api-conversations"),
        ],
    )
    def test_feed_start_trains(self, dataset, stem, monkeypatch):
        from training import execution as exec_mod
        from training.jobs import training_jobs

        import domain.training as domain_training
        from domain.training._internal import wandb_helpers

        monkeypatch.setattr(domain_training, "SloughGPTTrainer", _FakeTrainer)
        monkeypatch.setattr(
            wandb_helpers, "create_training_tracker_for_api_job", lambda **kw: _FakeTracker()
        )
        monkeypatch.setattr(exec_mod, "notify_training_event", _async_noop)
        monkeypatch.setattr(exec_mod, "notify_push", lambda **kw: None)
        _FakeTrainer.captured.clear()
        _FakeTrainer.captured_kwargs.clear()

        resp = client.post(
            "/training/start",
            json={"name": "feed journey", "model": "slonet", "dataset": dataset},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "started"
        job_id = payload["job_id"]

        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = training_jobs.get(job_id, {}).get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(0.02)
        assert status == "completed", f"job did not complete: {status}"

        job = training_jobs[job_id]
        assert job["data_source"] == "feed"
        assert job["data_path"] == dataset
        assert job["output_checkpoint_stem"] == stem
        assert _FakeTrainer.captured == [dataset]

    def test_feed_start_passes_refresh_interval(self, monkeypatch):
        from training import execution as exec_mod
        from training.jobs import training_jobs

        import domain.training as domain_training
        from domain.training._internal import wandb_helpers

        monkeypatch.setattr(domain_training, "SloughGPTTrainer", _FakeTrainer)
        monkeypatch.setattr(
            wandb_helpers, "create_training_tracker_for_api_job", lambda **kw: _FakeTracker()
        )
        monkeypatch.setattr(exec_mod, "notify_training_event", _async_noop)
        monkeypatch.setattr(exec_mod, "notify_push", lambda **kw: None)
        _FakeTrainer.captured.clear()
        _FakeTrainer.captured_kwargs.clear()

        resp = client.post(
            "/training/start",
            json={
                "name": "feed live",
                "model": "slonet",
                "dataset": "feed:api-conversations",
                "feed_refresh_interval": 30,
            },
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = training_jobs.get(job_id, {}).get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(0.02)
        assert status == "completed", f"job did not complete: {status}"

        kw = _FakeTrainer.captured_kwargs[-1]
        assert kw["feed_refresh_interval"] == 30.0


class TestFeedLiveRefreshSchema:
    def test_refresh_interval_defaults_off(self):
        req = TrainingRequest(name="x", model="slonet", dataset="feed:api-conversations")
        assert req.feed_refresh_interval == 0.0

    def test_refresh_interval_passthrough(self):
        req = TrainingRequest(
            name="x",
            model="slonet",
            dataset="feed:api-conversations",
            feed_refresh_interval=30,
        )
        assert req.feed_refresh_interval == 30.0

    def test_refresh_interval_rejects_negative(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            TrainingRequest(
                name="x",
                model="slonet",
                dataset="feed:api-conversations",
                feed_refresh_interval=-1,
            )
