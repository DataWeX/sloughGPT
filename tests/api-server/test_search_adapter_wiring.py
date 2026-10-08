"""Default-construction wiring — the gap that let two live bugs through:
the memory adapter never called ``_resolve()`` (contract fakes inject
the service, masking it) and the files adapter called ``get_db()``
without the required database name. Contract fakes prove matching;
these tests prove the lazy default resolution against the real import
paths.
"""

from __future__ import annotations

from types import SimpleNamespace

from domain.search.adapters.files import FilesAdapter
from domain.search.adapters.memory import MemoryAdapter
from domain.search.adapters.training import TrainingJobsAdapter
from domain.search.types import SearchContext

CTX = SearchContext(user_id="u1")


async def test_memory_adapter_resolves_default_service(monkeypatch):
    import domain.memory._internal.service as svc_mod

    fake = SimpleNamespace(
        retrieve=lambda q, limit=None: [
            {"id": "m1", "content": "alpha fact", "topic": "t", "score": 1.0}
        ]
    )
    monkeypatch.setattr(svc_mod, "get_memory_service", lambda *a, **k: fake)

    adapter = MemoryAdapter()  # default construction — no injection
    hits = await adapter.search("alpha", 5, CTX)
    assert len(hits) == 1
    assert hits[0].store == "memory"
    assert hits[0].title == "alpha fact"


async def test_memory_adapter_handles_service_returning_none(monkeypatch):
    """Some deployments expose no memory service — empty, not crash."""
    import domain.memory._internal.service as svc_mod

    monkeypatch.setattr(svc_mod, "get_memory_service", lambda *a, **k: None)
    adapter = MemoryAdapter()
    # Unavailable (defensive) = same as disabled: no matches, no raise.
    assert await adapter.search("anything", 5, CTX) == []


async def test_training_adapter_resolves_job_store(monkeypatch):
    """Default resolution must find the REAL job store — a phantom import
    (domain.training.repository never existed) made this store partial
    live while contract fakes kept passing."""
    import training.job_store as job_store_mod

    calls: list[str] = []

    class FakeJobStore:
        def list_by_workspace(self, workspace_id, status=None):
            calls.append(workspace_id)
            return [{"id": "j1", "name": "fine-tune llama", "status": "completed"}]

    monkeypatch.setattr(job_store_mod, "JobStore", FakeJobStore)

    adapter = TrainingJobsAdapter()  # default construction — no injection
    hits = await adapter.search("fine-tune", 5, SearchContext(user_id="u", workspace_id="ws9"))
    assert calls == ["ws9"]
    assert len(hits) == 1
    assert hits[0].id == "j1"
    assert hits[0].locator == "route:/training"


async def test_files_adapter_uses_uploads_mogdb(tmp_path, monkeypatch):
    calls: list[str] = []

    class _FakeCol:
        def find(self):
            return [
                {
                    "file_id": "f1",
                    "filename": "report.pdf",
                    "original_name": "report.pdf",
                    "extension": ".pdf",
                    "tags": [],
                }
            ]

    def fake_get_db(db_name, **kw):
        calls.append(db_name)
        return SimpleNamespace(collection=lambda name: _FakeCol())

    import infrastructure.db_pool as db_pool

    monkeypatch.setattr(db_pool, "get_db", fake_get_db)

    adapter = FilesAdapter()  # default construction — no docs_factory
    adapter._uploads_dir = tmp_path
    (tmp_path / "report.pdf").write_bytes(b"%PDF")

    hits = await adapter.search("report", 5, CTX)
    assert calls == ["uploads_mogdb"], "must use the same db name as legacy /files"
    assert len(hits) == 1
    assert hits[0].title == "report.pdf"
