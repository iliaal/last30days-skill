"""Timeout architecture: shared X-chain deadline and bounded stream waits."""

import threading
import time
from unittest import mock

from lib import health, pipeline, schema, xai_x


def test_stream_timeouts_are_bounded():
    for name in (
        "STREAM_FUTURE_TIMEOUT_SECONDS",
        "DISCOVERY_FUTURE_TIMEOUT_SECONDS",
        "THIN_RETRY_FUTURE_TIMEOUT_SECONDS",
        "X_CHAIN_DEADLINE_SECONDS",
    ):
        value = getattr(pipeline, name)
        assert 0 < value <= 600, f"{name}={value} must be a finite wall-clock bound"


def test_fetch_x_backend_forwards_deadline(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pipeline.grok_x, "search_x",
        lambda *a, **k: (seen.update(grok=k.get("deadline")), {"items": []})[1],
    )
    monkeypatch.setattr(
        pipeline.xai_x, "search_x",
        lambda *a, **k: (seen.update(xai=k.get("deadline_monotonic")), {"items": []})[1],
    )
    monkeypatch.setattr(pipeline.xai_x, "parse_x_response", lambda r: [])
    monkeypatch.setattr(
        pipeline.xquik, "search_xquik",
        lambda *a, **k: (seen.update(xquik=k.get("deadline")), {"items": []})[1],
    )
    monkeypatch.setattr(pipeline.xquik, "parse_xquik_response", lambda r: [])
    monkeypatch.setattr(
        pipeline.x_api, "search_x",
        lambda *a, **k: (seen.update(xapi=k.get("deadline")), {"items": []})[1],
    )
    deadline = time.monotonic() + 90
    cfg = {"XAI_API_KEY": "k", "X_BEARER_TOKEN": "b"}
    monkeypatch.setattr(pipeline.env, "get_xquik_token", lambda c: "q")
    pipeline._fetch_x_backend("grok", "q", "2026-08-01", "2026-08-31", "quick", cfg, deadline=deadline)
    assert seen["grok"] == deadline
    pipeline._fetch_x_backend("xai", "q", "2026-08-01", "2026-08-31", "quick", cfg, deadline=deadline)
    assert seen["xai"] == deadline
    pipeline._fetch_x_backend("xquik", "q", "2026-08-01", "2026-08-31", "quick", cfg, deadline=deadline)
    assert seen["xquik"] == deadline
    pipeline._fetch_x_backend("xapi", "q", "2026-08-01", "2026-08-31", "quick", cfg, deadline=deadline)
    assert seen["xapi"] == deadline


def test_x_chain_skips_past_deadline(monkeypatch):
    monkeypatch.setattr(pipeline.env, "x_backend_chain", lambda c: ["xapi", "xquik"])
    calls = []
    monkeypatch.setattr(
        pipeline, "_fetch_x_backend",
        lambda *a, **k: (calls.append(a[0]), ([], "boom"))[1],
    )
    # First monotonic() call sets the deadline, the second (chain check) is past it.
    monkeypatch.setattr(
        pipeline.time, "monotonic",
        mock.Mock(side_effect=[1000.0, 2000.0, 2000.0]),
    )
    plan = mock.Mock()
    plan.domain = "topic"
    items, err = pipeline._fetch_discovery_source(
        "x", plan, from_date="2026-08-01", to_date="2026-08-31",
        depth="quick", mock=False, config={}, keyword_gate=True,
    )
    assert items == []
    assert "budget exhausted" in (err or "")
    assert calls == [], "no backend may start past the shared deadline"


def test_enrich_cancel_skips_network(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    subquery = schema.SubQuery(
        label="primary", search_query="topic",
        ranking_query="topic", sources=["reddit"], weight=1.0,
    )
    items, artifact = pipeline._retrieve_stream_impl(
        topic="topic", subquery=subquery, source="reddit",
        config={"_enrich_cancel": cancel}, depth="quick",
        date_range=("2026-08-01", "2026-08-31"),
        runtime=mock.Mock(), mock=True,
    )
    assert items == [] and artifact == {}


def test_discovery_timeout_records_timeout_partial(monkeypatch):
    monkeypatch.setattr(pipeline, "DISCOVERY_FUTURE_TIMEOUT_SECONDS", 0.05)
    def slow(*a, **k):
        time.sleep(0.4)
        return [], None
    monkeypatch.setattr(pipeline, "_fetch_discovery_source", slow)
    plan = mock.Mock()
    plan.sources = ["reddit"]
    plan.domain = "topic"
    start = time.monotonic()
    bundle = pipeline.nominate_candidates(
        plan, from_date="2026-08-01", to_date="2026-08-31",
        depth="quick", mock=False, config={}, lookback_days=30,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 1.5, f"hung lane must not stall nominate (took {elapsed:.2f}s)"
    outcome = bundle.source_status["reddit"]
    assert outcome.state == health.TIMEOUT


def test_xai_single_attempt_with_deadline(monkeypatch):
    seen = {}
    def fake_post(url, payload, headers=None, **kwargs):
        seen.update(kwargs)
        return {"output": []}
    monkeypatch.setattr(pipeline.xai_x.http, "post", fake_post)
    deadline = time.monotonic() + 90
    xai_x.search_x("k", "m", "topic", "2026-08-01", "2026-08-31", deadline_monotonic=deadline)
    assert seen.get("retries") == 1
    assert seen.get("deadline_monotonic") == deadline
