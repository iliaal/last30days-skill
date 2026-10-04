import json
import sys

import pytest

import last30days as cli
from lib import planner


SINGLE_TOPICS = [
    "CI/CD",
    "I/O performance",
    "openai/openai-python",
    "https://example.com",
    "https://example.com/React/Vue",
    "https://example.com/vs/topic",
    "https://example.com/vs.Vue",
    "React vsVue",
    "compare CI/CD workflows",
]

COMPARISONS = [
    ("React/Vue/Svelte", ["React", "Vue", "Svelte"]),
    ("React vs Vue", ["React", "Vue"]),
    ("React vs. Vue", ["React", "Vue"]),
    ("React vs.Vue", ["React", "Vue"]),
    ("React VS.Vue", ["React", "Vue"]),
    ("React versus Vue", ["React", "Vue"]),
    ("React compared to Vue", ["React", "Vue"]),
    ("difference between React and Vue", ["React", "Vue"]),
    ("React/Vue/Svelte for CI/CD", ["React", "Vue", "Svelte"]),
    ("CI/CD vs I/O", ["CI/CD", "I/O"]),
    ("CI/CD vs.I/O", ["CI/CD", "I/O"]),
    ("openai/openai-python vs anthropics/anthropic-sdk-python",
     ["openai/openai-python", "anthropics/anthropic-sdk-python"]),
]


@pytest.mark.parametrize("topic", SINGLE_TOPICS)
@pytest.mark.parametrize("uncapped", [False, True])
def test_slashes_without_comparison_intent_do_not_produce_entities(topic, uncapped):
    assert planner._comparison_entities(topic, uncapped=uncapped) == []


@pytest.mark.parametrize("topic,entities", COMPARISONS)
def test_comparison_entities_preserve_slashes_inside_explicit_entities(topic, entities):
    assert planner._comparison_entities(topic) == entities


@pytest.fixture
def run_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.env, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.env, "get_config", lambda **kwargs: {})
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LAST30DAYS_STORE", "")

    def run(topic):
        monkeypatch.setattr(sys, "argv", [
            "last30days", topic, "--mock", "--quick", "--search=reddit",
            "--no-browser-cookies", "--emit=json", "--json-profile=raw",
            "--save-dir", "",
        ])
        result = cli.main()
        captured = capsys.readouterr()
        assert result == 0, captured.err
        return json.loads(captured.out)

    return run


@pytest.mark.parametrize("topic", SINGLE_TOPICS)
def test_main_preserves_slash_topics_as_single_research_runs(topic, run_cli):
    payload = run_cli(topic)
    assert payload.get("comparison") is not True
    assert payload["topic"] == topic
    assert payload["query_plan"]["raw_topic"] == topic


@pytest.mark.parametrize("topic,entities", COMPARISONS)
def test_main_keeps_supported_comparison_forms(topic, entities, run_cli):
    payload = run_cli(topic)
    assert payload["comparison"] is True
    assert payload["entities"] == entities
    assert [entry["report"]["topic"] for entry in payload["reports"]] == entities
