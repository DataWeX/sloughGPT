"""
Tests for the MogDB-backed WebhookStore.

Covers register/get/list/unregister semantics, active-only listing with
event filtering, secret signing, stats, and the in-memory delivery log
trimming (which matches the original store's behaviour).
"""

from datetime import datetime

from training.webhooks import WebhookStore


def _mk_store(tmp_path):
    return WebhookStore(str(tmp_path / "webhooks.db"))


def _register(store, url="https://example.com/hook", events=("training.started",), secret=None):
    return store.register(url, list(events), secret=secret, description="test hook")


def test_register_generates_secret_and_id(tmp_path):
    store = _mk_store(tmp_path)
    wid = _register(store)

    hook = store.get(wid)
    assert hook is not None
    assert hook.id == wid
    assert len(wid) == 16
    assert len(hook.secret) == 32
    assert hook.url == "https://example.com/hook"
    assert hook.events == ["training.started"]
    assert hook.is_active
    assert isinstance(hook.created_at, datetime)


def test_register_keeps_provided_secret(tmp_path):
    store = _mk_store(tmp_path)
    wid = _register(store, secret="my-secret")

    assert store.get(wid).secret == "my-secret"


def test_get_missing_webhook_returns_none(tmp_path):
    store = _mk_store(tmp_path)

    assert store.get("does-not-exist") is None


def test_list_returns_active_only_most_recent_first(tmp_path):
    store = _mk_store(tmp_path)
    a = _register(store, url="https://a.example/hook")
    b = _register(store, url="https://b.example/hook")

    store.unregister(a)

    hooks = store.list()
    assert [h.id for h in hooks] == [b]


def test_list_filters_by_event(tmp_path):
    store = _mk_store(tmp_path)
    _register(store, events=("training.started",))
    _register(store, events=("training.completed",))

    hooks = store.list(event_filter="training.completed")
    assert len(hooks) == 1
    assert hooks[0].events == ["training.completed"]


def test_unregister_removes_and_returns_bool(tmp_path):
    store = _mk_store(tmp_path)
    wid = _register(store)

    assert store.unregister(wid) is True
    assert store.unregister(wid) is False
    assert store.get(wid) is None


def test_get_secret_and_sign_payload(tmp_path):
    store = _mk_store(tmp_path)
    wid = _register(store, secret="s3cret")

    assert store.get_secret(wid) == "s3cret"
    sig = store.sign_payload(wid, '{"event": "training.started"}')
    assert sig.startswith("sha256=")
    assert sig == store.sign_payload(wid, '{"event": "training.started"}')
    assert sig != store.sign_payload(wid, '{"event": "other"}')
    assert store.sign_payload("missing", "x") is None


def test_get_stats_counts(tmp_path):
    store = _mk_store(tmp_path)
    a = _register(store)
    b = _register(store)
    store.unregister(a)

    stats = store.get_stats()
    assert stats["total_webhooks"] == 1
    assert stats["active_webhooks"] == 1
    assert stats["total_deliveries"] == 0
    assert stats["success_rate"] == "N/A"


def test_delivery_log_trimmed_at_max_size(tmp_path):
    store = _mk_store(tmp_path)
    wid = _register(store)
    store._max_log_size = 3

    for i in range(5):
        store._add_delivery(_delivery(store, wid, str(i)))

    assert len(store.delivery_log) == 3
    assert store.delivery_log[0].payload == "2"


def _delivery(store, wid, payload):
    from training.webhooks import WebhookDelivery

    return WebhookDelivery(
        id=f"d-{payload}",
        webhook_id=wid,
        event="training.started",
        payload=payload,
    )
