import pytest

from lib import cluster, fusion, hiring_signals, normalize, render, rerank, schema


def _jobs(count):
    return normalize.normalize_source_items(
        "jobs",
        [
            {
                "id": f"job-{index}",
                "title": f"Software Engineer {index}",
                "description": "Build Acme backend infrastructure.",
                "url": f"https://boards.greenhouse.io/acme/jobs/{index}",
                "provider": "greenhouse",
                "department": "Engineering",
                "date": "2026-06-01",
            }
            for index in range(count)
        ],
        "2026-05-17",
        "2026-06-16",
    )


def _plan(sources):
    return schema.QueryPlan(
        intent="company",
        freshness_mode="relaxed",
        cluster_mode="story",
        raw_topic="Acme",
        subqueries=[schema.SubQuery(
            label="primary",
            search_query="Acme",
            ranking_query="What is Acme hiring for?",
            sources=sources,
        )],
        source_weights={source: 1.0 for source in sources},
    )


def test_job_provider_does_not_cap_roles_or_exempt_social_author():
    jobs = _jobs(6)
    posts = [
        schema.SourceItem(
            item_id=f"post-{index}",
            source="x",
            title=f"Acme update {index}",
            body="Acme update",
            url=f"https://x.com/greenhouse/status/{index}",
            author="greenhouse",
            relevance_hint=0.8,
        )
        for index in range(6)
    ]

    candidates = fusion.weighted_rrf(
        {("primary", "jobs"): jobs, ("primary", "x"): posts},
        _plan(["jobs", "x"]),
        pool_limit=40,
    )

    assert {item.item_id for item in candidates if item.source == "jobs"} == {
        item.item_id for item in jobs
    }
    assert len([item for item in candidates if item.source == "x"]) == 3


def _report(jobs, pool_limit):
    plan = _plan(["jobs"])
    candidates = fusion.weighted_rrf(
        {("primary", "jobs"): jobs}, plan, pool_limit=pool_limit,
    )
    ranked = rerank.rerank_candidates(
        topic="Acme", plan=plan, candidates=candidates,
        provider=None, model=None, shortlist_size=pool_limit,
    )
    return schema.Report(
        topic="Acme",
        range_from="2026-05-17",
        range_to="2026-06-16",
        generated_at="2026-06-16T00:00:00+00:00",
        provider_runtime=schema.ProviderRuntime(
            reasoning_provider=None, planner_model=None, rerank_model=None,
        ),
        query_plan=plan,
        ranked_candidates=ranked,
        clusters=cluster.cluster_candidates(ranked, plan),
        items_by_source={"jobs": jobs},
        errors_by_source={},
        artifacts={"hiring_signals": hiring_signals.analyze(
            jobs, explicit=True, topic="Acme",
        )},
    )


@pytest.mark.parametrize("renderer", [
    lambda report: render.render_compact(report, cluster_limit=1),
    lambda report: render.render_context(report, cluster_limit=1),
    render.render_for_html,
    lambda report: render.render_compact(report, register="exec", cluster_limit=1),
], ids=["compact", "context", "html", "exec"])
@pytest.mark.parametrize("pool_limit", [5, 40])
def test_hiring_analysis_uses_board_beyond_ranking_and_display_limits(renderer, pool_limit):
    jobs = _jobs(40)
    jobs[-1].title = "Founding Research Scientist, Human Simulation"
    jobs[-1].body = "Acme is founding a human simulation research team."
    report = _report(jobs, pool_limit=pool_limit)

    assert len(report.ranked_candidates) == pool_limit
    if pool_limit == 5:
        assert all(candidate.item_id != jobs[-1].item_id for candidate in report.ranked_candidates)
    text = renderer(report)

    assert "company-size tier: growth" in text
    assert "Founding Research Scientist, Human Simulation" in text
    assert "evidence: 39 roles" in text


@pytest.mark.parametrize("rejection", ["cluster", "candidate"])
def test_hiring_analysis_excludes_rejected_roles_from_full_board(rejection):
    jobs = _jobs(6)
    jobs[-1].title = "Founding Research Scientist, Human Simulation"
    report = _report(jobs, pool_limit=6)
    assert jobs[-1].title in render.render_compact(report)

    rejected = next(
        candidate for candidate in report.ranked_candidates
        if candidate.item_id == jobs[-1].item_id
    )
    if rejection == "cluster":
        next(
            cluster for cluster in report.clusters
            if rejected.candidate_id in cluster.candidate_ids
        ).score = 0
    else:
        rejected.explanation = "fallback-local-score (entity-miss demotion)"

    text = render.render_compact(report)

    assert "## Hiring Signals" in text
    assert jobs[-1].title not in text
    assert "evidence: 5 roles" in text
