from __future__ import annotations

import json

from app.core.workflow.dataset_synthesis.sampler import StratifiedHybridSampler


def test_shs_targeted_sampling_uses_feasibility_probe(monkeypatch) -> None:
    sampler = StratifiedHybridSampler()
    samples = [
        json.dumps({"nodes": [{"elementId": "n1", "labels": ["A"], "properties": {}}], "relationships": []}),
        json.dumps(
            {
                "nodes": [
                    {"elementId": "n1", "labels": ["A"], "properties": {}},
                    {"elementId": "n2", "labels": ["B"], "properties": {}},
                ],
                "relationships": [
                    {
                        "elementId": "r1",
                        "type": "KNOWS",
                        "start_node_elementId": "n1",
                        "end_node_elementId": "n2",
                        "properties": {"score": 1.2},
                    }
                ],
            }
        ),
    ]
    state = {"idx": 0}

    def _fake_random_subgraph(self, graph_db, max_depth, max_nodes, max_edges):
        idx = min(state["idx"], len(samples) - 1)
        state["idx"] += 1
        return samples[idx]

    monkeypatch.setattr(
        "app.core.workflow.dataset_synthesis.sampler.RandomWalkSampler.get_random_subgraph",
        _fake_random_subgraph,
    )

    selected = sampler.get_targeted_subgraph(
        graph_db=object(),
        max_depth=2,
        max_nodes=5,
        max_edges=5,
        required_intents=["query.ranking.topk"],
    )
    obj = json.loads(selected)
    assert len(obj["nodes"]) == 2
    assert len(obj["relationships"]) == 1


def test_shs_sampling_metrics_update_with_acceptance(monkeypatch) -> None:
    sampler = StratifiedHybridSampler()

    payload = json.dumps(
        {
            "nodes": [
                {"elementId": "n1", "labels": ["A"], "properties": {}},
                {"elementId": "n2", "labels": ["B"], "properties": {}},
            ],
            "relationships": [
                {
                    "elementId": "r1",
                    "type": "KNOWS",
                    "start_node_elementId": "n1",
                    "end_node_elementId": "n2",
                    "properties": {"amount": 10},
                }
            ],
        }
    )

    monkeypatch.setattr(
        "app.core.workflow.dataset_synthesis.sampler.RandomWalkSampler.get_random_subgraph",
        lambda self, graph_db, max_depth, max_nodes, max_edges: payload,
    )

    sampler.get_targeted_subgraph(
        graph_db=object(),
        max_depth=2,
        max_nodes=5,
        max_edges=5,
        required_intents=["query.path.shortest"],
    )
    sampler.register_acceptance(["query.path.shortest"], accepted_rows_count=2)
    metrics = sampler.get_sampling_metrics()
    assert "intent_sampling" in metrics
    assert "query.path.shortest" in metrics["intent_sampling"]
    assert metrics["intent_sampling"]["query.path.shortest"]["attempts"] >= 1
    assert metrics["intent_sampling"]["query.path.shortest"]["accepted_rate"] > 0
