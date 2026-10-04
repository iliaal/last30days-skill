"""Shared findings retain every topic's membership and history."""

import sqlite3
from datetime import datetime, timedelta

import pytest

import briefing
import store


@pytest.fixture
def shared_findings(tmp_path):
    path = tmp_path / "research.db"
    with store.scoped_db(path):
        store.init_db()
        topics = {name: store.add_topic(name) for name in ("A", "B")}
        runs = {}
        for name in topics:
            runs[name] = store.record_run(topics[name]["id"])
            store.store_findings(runs[name], topics[name]["id"], [
                {
                    "source": "reddit",
                    "source_url": "https://example.test/shared",
                    "source_title": "Shared research",
                    "engagement_score": 10,
                },
                {
                    "source": "reddit",
                    "source_url": f"https://example.test/{name}",
                    "source_title": f"Exclusive {name}",
                    "engagement_score": 5,
                },
            ])
        yield path, topics, runs


@pytest.mark.parametrize("removed", ["A", "B"])
def test_remove_topic_preserves_other_topic_history(shared_findings, removed):
    path, topics, runs = shared_findings
    survivor = "B" if removed == "A" else "A"
    sightings = store.get_sightings_for_run(topics[survivor]["id"], runs[survivor])

    assert store.remove_topic(removed)

    assert store.get_sightings_for_run(topics[survivor]["id"], runs[survivor]) == sightings
    findings = store.get_new_findings(topics[survivor]["id"])
    assert {item["source_url"] for item in findings} == {
        "https://example.test/shared", f"https://example.test/{survivor}",
    }
    shared = next(item for item in findings if item["source_title"] == "Shared research")
    assert shared["run_id"] == runs[survivor]
    assert shared["sighting_count"] == 2
    assert store.search_findings("Shared")[0]["id"] == shared["id"]
    assert store.get_topic(removed) is None
    assert not store.remove_topic(removed)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT id FROM research_runs").fetchall() == [(runs[survivor],)]

    assert store.remove_topic(survivor)
    assert store.get_stats()["total_findings"] == 0


def test_topic_reads_include_shared_findings_once(shared_findings):
    _, topics, _ = shared_findings
    run = store.record_run(topics["B"]["id"])
    store.store_findings(run, topics["B"]["id"], [{
        "source": "reddit", "source_url": "https://example.test/shared",
        "engagement_score": 10,
    }])

    assert {t["name"]: t["finding_count"] for t in store.list_topics()} == {"A": 2, "B": 2}
    for topic in topics.values():
        assert len(store.get_new_findings(topic["id"])) == 2
    trending = {item["name"]: item for item in store.get_trending()}
    assert {name: item["new_findings"] for name, item in trending.items()} == {"A": 2, "B": 2}
    assert {name: item["total_engagement"] for name, item in trending.items()} == {"A": 15, "B": 15}


def test_removal_preserves_membership_between_first_and_latest_topic(shared_findings):
    _, topics, _ = shared_findings
    topic_c = store.add_topic("C")
    run_c = store.record_run(topic_c["id"])
    store.store_findings(run_c, topic_c["id"], [{
        "source": "reddit", "source_url": "https://example.test/shared",
    }])

    assert store.remove_topic("A")

    assert len(store.get_new_findings(topics["B"]["id"])) == 2
    shared = store.get_new_findings(topic_c["id"])
    assert len(shared) == 1
    assert shared[0]["source_url"] == "https://example.test/shared"
    assert shared[0]["run_id"] == run_c
    assert store.remove_topic("C")
    assert len(store.get_new_findings(topics["B"]["id"])) == 2


def test_removal_preserves_other_topic_delta(shared_findings):
    _, topics, _ = shared_findings
    run = store.record_run(topics["B"]["id"])
    store.store_findings(run, topics["B"]["id"], [{
        "source": "reddit", "source_url": "https://example.test/shared",
    }])
    before = store.compute_topic_delta(topics["B"]["id"])
    assert before["continued"] == 1
    assert before["dropped"] == 1

    assert store.remove_topic("A")

    assert store.compute_topic_delta(topics["B"]["id"]) == before


def test_weekly_briefing_counts_shared_findings_in_both_periods(shared_findings, monkeypatch):
    path, _, _ = shared_findings
    monkeypatch.setattr(briefing, "BRIEFS_DIR", path.parent / "briefs")
    last_week = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE findings SET first_seen = ? WHERE source_url = ?",
            (last_week, "https://example.test/shared"),
        )

    result = briefing.generate_weekly()

    assert result["status"] == "ok"
    assert len(result["topics"]) == 2
    for topic in result["topics"]:
        assert topic["this_week_count"] == 1
        assert topic["last_week_count"] == 1
        assert topic["engagement_change_pct"] == -50.0


def test_topic_findings_date_range_excludes_upper_boundary(shared_findings):
    path, topics, _ = shared_findings
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE findings SET first_seen = '2026-01-01'")
        conn.execute(
            "UPDATE findings SET first_seen = '2026-01-08' WHERE source_url = ?",
            ("https://example.test/shared",),
        )

    findings = store.get_new_findings(topics["B"]["id"], "2026-01-01", before="2026-01-08")

    assert [item["source_url"] for item in findings] == ["https://example.test/B"]
    assert store.get_new_findings(topics["B"]["id"], before="2026-01-01") == []


@pytest.mark.parametrize("removed", ["A", "B"])
def test_remove_topic_preserves_legacy_aggregate_membership(shared_findings, removed):
    path, topics, runs = shared_findings
    survivor = "B" if removed == "A" else "A"
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM finding_sightings")

    assert store.remove_topic(removed)

    findings = store.get_new_findings(topics[survivor]["id"])
    assert {item["source_url"] for item in findings} == {
        "https://example.test/shared", f"https://example.test/{survivor}",
    }
    shared = next(item for item in findings if item["source_title"] == "Shared research")
    assert shared["run_id"] == (runs["B"] if survivor == "B" else None)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_remove_topic_rolls_back_when_deletion_fails(shared_findings):
    path, topics, runs = shared_findings
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT * FROM findings ORDER BY id").fetchall()
        conn.execute("""CREATE TRIGGER reject_topic_delete BEFORE DELETE ON topics
                        BEGIN SELECT RAISE(ABORT, 'deletion blocked'); END""")

    with pytest.raises(sqlite3.IntegrityError, match="deletion blocked"):
        store.remove_topic("A")

    assert store.get_topic("A") == topics["A"]
    assert len(store.get_sightings_for_run(topics["A"]["id"], runs["A"])) == 2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM findings ORDER BY id").fetchall() == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
