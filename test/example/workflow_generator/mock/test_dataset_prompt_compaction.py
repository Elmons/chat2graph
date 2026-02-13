from __future__ import annotations

import json

from app.core.prompt.data_synthesis import generate_query_tv_template
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.model import Row
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo


class _NoopModelService:
    async def generate(self, sys_prompt: str, messages, tools=None, tool_call_ctx=None):  # pragma: no cover
        raise AssertionError("not used in this test")


class _DummySampler:
    def get_random_subgraph(self, graph_db, max_depth: int, max_nodes: int, max_edges: int) -> str:  # pragma: no cover
        return "{}"


def _make_generator(monkeypatch) -> SamplingDatasetGenerator:
    monkeypatch.setattr(
        "app.core.workflow.dataset_synthesis.generator.ModelServiceFactory.create",
        lambda *args, **kwargs: _NoopModelService(),
    )
    return SamplingDatasetGenerator(
        graph_db=object(),  # type: ignore[arg-type]
        sampler=_DummySampler(),  # type: ignore[arg-type]
        strategy="query",
    )


def test_generate_prompt_compaction_keeps_key_info(monkeypatch) -> None:
    gen = _make_generator(monkeypatch)
    info = GraphTaskTypesInfo(strategy="query")
    subgraph_obj = {
        "nodes": [
            {"elementId": "n1", "labels": ["Person"], "properties": {"name": "Alice", "age": 30}},
            {"elementId": "n2", "labels": ["Person"], "properties": {"name": "Bob", "age": 31}},
        ],
        "relationships": [
            {
                "elementId": "r1",
                "type": "KNOWS",
                "start_node_elementId": "n1",
                "end_node_elementId": "n2",
                "properties": {"since": 2020},
            }
        ],
    }
    subgraph_pretty = json.dumps(subgraph_obj, ensure_ascii=False, indent=2)

    compact_prompt = gen._build_generate_pairs_prompt(
        task_types_info=info,
        subgraph=subgraph_pretty,
        task_description="desc",
        nums=1,
    )
    legacy_prompt = generate_query_tv_template.format(
        task_description="desc",
        subgraph=subgraph_pretty,
        num_pairs=1,
        task_level_info=info.get_tasks_info(),
        task_statistic_info=info.get_count_info(),
    )

    # Prompt gets smaller after removing verbose examples/pretty JSON.
    assert len(compact_prompt) < len(legacy_prompt)

    # Key taxonomy and subgraph facts remain present.
    assert "L1 Simple Query Tasks" in compact_prompt
    assert "Entity Attribute and Label Query" in compact_prompt
    assert '"elementId":"n1"' in compact_prompt
    assert '"type":"KNOWS"' in compact_prompt
    assert '"start_node_elementId":"n1"' in compact_prompt
    assert '"end_node_elementId":"n2"' in compact_prompt

    # Verbose examples are removed from compact level info.
    assert "What is the department of employee Zhang San?" not in compact_prompt


def test_filter_prompt_uses_compact_json_dataset(monkeypatch) -> None:
    gen = _make_generator(monkeypatch)
    rows = [
        Row(
            level="L1",
            task_type="query",
            task_subtype="Entity Attribute and Label Query",
            task="Find Alice",
            verifier="MATCH (n {name:'Alice'}) RETURN n",
        )
    ]
    subgraph_pretty = json.dumps(
        {"nodes": [{"elementId": "n1", "labels": ["Person"], "properties": {"name": "Alice"}}], "relationships": []},
        ensure_ascii=False,
        indent=2,
    )

    prompt = gen._build_filter_prompt(task_desc="desc", subgraph=subgraph_pretty, dataset=rows)

    # Dataset is serialized as compact JSON, not Python object repr.
    assert "Row(" not in prompt
    assert '"task":"Find Alice"' in prompt
    assert '"verifier":"MATCH (n {name:\'Alice\'}) RETURN n"' in prompt
    assert '"elementId":"n1"' in prompt


def test_subgraph_compression_minifies_without_semantic_change(monkeypatch) -> None:
    gen = _make_generator(monkeypatch)
    raw_subgraph = json.dumps(
        {
            "nodes": [
                {
                    "elementId": "4:5033fde7-9b4f-45a1-b744-7453cad94c1f:15347",
                    "labels": ["Account"],
                    "properties": {
                        "name": "alice_account",
                        "description": "x" * 180,
                    },
                },
                {
                    "elementId": "4:5033fde7-9b4f-45a1-b744-7453cad94c1f:16054",
                    "labels": ["Account"],
                    "properties": {"name": "bob_account"},
                },
            ],
            "relationships": [
                {
                    "elementId": "4:5033fde7-9b4f-45a1-b744-7453cad94c1f:20001",
                    "type": "TRANSFER",
                    "start_node_elementId": "4:5033fde7-9b4f-45a1-b744-7453cad94c1f:15347",
                    "end_node_elementId": "4:5033fde7-9b4f-45a1-b744-7453cad94c1f:16054",
                    "properties": {"amount": 100, "remark": "r" * 200},
                }
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    compressed = gen._compress_subgraph_for_prompt(raw_subgraph)
    # length reduction should be visible for pretty-json input.
    assert len(compressed) < len(raw_subgraph)
    # semantic content remains unchanged (no id/property mapping or truncation).
    assert json.loads(compressed) == json.loads(raw_subgraph)
