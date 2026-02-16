from __future__ import annotations

from typing import Any

import pytest

from app.core.common.type import MessageSourceType
from app.core.model.message import ModelMessage
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator


class _DummySampler:
    def __init__(self, subgraph: str = "{\"nodes\": [], \"relationships\": []}"):
        self._subgraph = subgraph

    def get_random_subgraph(self, graph_db: Any, max_depth: int, max_nodes: int, max_edges: int) -> str:
        return self._subgraph


class _QueueModelService:
    def __init__(self, payloads: list[str]):
        self._payloads = list(payloads)
        self.call_count = 0

    async def generate(self, sys_prompt: str, messages: list[ModelMessage], tools=None, tool_call_ctx=None) -> ModelMessage:  # type: ignore[override]
        self.call_count += 1
        if not self._payloads:
            raise AssertionError("No queued payloads left for generate()")
        payload = self._payloads.pop(0)
        job_id = messages[-1].get_job_id() if messages else "job"
        return ModelMessage(
            payload=payload,
            job_id=job_id,
            step=1,
            source_type=MessageSourceType.ACTOR,
        )


@pytest.mark.asyncio
async def test_identify_strategy_is_query_only_and_does_not_call_llm(mocker) -> None:
    llm = _QueueModelService(payloads=[])
    mocker.patch(
        "app.core.reasoner.model_service_factory.ModelServiceFactory.create",
        return_value=llm,
    )

    gen = SamplingDatasetGenerator(graph_db=object(), sampler=_DummySampler())
    strategy = await gen.identify_strategy("any task desc")
    assert strategy == "query"
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_generate_query_only_end_to_end_with_mock_llm(mocker) -> None:
    generate_payload = """
[
  {
    "level":"L1",
    "task_subtype":"attribute_filtering",
    "task":"Find nodes with name = 'Alice'",
    "verifier":"MATCH (n {name: 'Alice'}) RETURN n",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup"],
    "global_verifier":"MATCH (n {name: 'Alice'}) RETURN n",
    "expected_global":"alice_result"
  },
  {
    "level":"L2",
    "task_subtype":"multi_hop",
    "task":"Check if Alice is connected to Bob within 2 hops",
    "verifier":"MATCH (a {name:'Alice'}), (b {name:'Bob'}) RETURN shortestPath((a)-[*..2]-(b)) IS NOT NULL",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup","query.path.shortest"],
    "global_verifier":"MATCH (a {name:'Alice'}), (b {name:'Bob'}) RETURN shortestPath((a)-[*..2]-(b)) IS NOT NULL",
    "expected_global":"true"
  }
]
""".strip()

    # Filter step can return a subset (or even the same set).
    filter_payload = """
[
  {
    "level":"L1",
    "task_subtype":"attribute_filtering",
    "task":"Find nodes with name = 'Alice'",
    "verifier":"MATCH (n {name: 'Alice'}) RETURN n",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup"],
    "global_verifier":"MATCH (n {name: 'Alice'}) RETURN n",
    "expected_global":"alice_result"
  }
]
""".strip()

    llm = _QueueModelService(payloads=[generate_payload, filter_payload])
    mocker.patch(
        "app.core.reasoner.model_service_factory.ModelServiceFactory.create",
        return_value=llm,
    )

    gen = SamplingDatasetGenerator(
        graph_db=object(),
        sampler=_DummySampler(),
        strategy=None,
        nums_per_subgraph=1,
    )

    dataset = await gen.generate(task_desc="desc", dataset_name="ds", size=1)
    assert dataset.name == "ds"
    assert dataset.task_desc == "desc"
    assert len(dataset.data) == 1
    assert dataset.protocol_version == "row_v2_1"
    assert dataset.data[0].task_type == "query"
    assert dataset.data[0].answer_scope == "global_graph"
    assert dataset.data[0].global_verifier is not None
    assert dataset.data[0].expected_global is not None
    assert "accepted_rows" in dataset.qa_gate_stats
    assert llm.call_count == 2  # generate_pairs + filter


@pytest.mark.asyncio
async def test_qa_gate_rejects_implicit_full_enumeration(mocker) -> None:
    generate_payload = """
[
  {
    "level":"L1",
    "task_subtype":"attribute_filtering",
    "task":"List all friends of Alice",
    "verifier":"MATCH (a {name:'Alice'})-[:KNOWS]->(f) RETURN f",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup"],
    "global_verifier":"MATCH (a {name:'Alice'})-[:KNOWS]->(f) RETURN f",
    "expected_global":"[...]"
  }
]
""".strip()
    # Filter step echoes same row; salvage should rewrite it into bounded form.
    filter_payload = generate_payload

    llm = _QueueModelService(payloads=[generate_payload, filter_payload])
    mocker.patch(
        "app.core.reasoner.model_service_factory.ModelServiceFactory.create",
        return_value=llm,
    )

    gen = SamplingDatasetGenerator(
        graph_db=object(),
        sampler=_DummySampler(),
        strategy=None,
        nums_per_subgraph=1,
    )
    dataset = await gen.generate(task_desc="desc", dataset_name="ds", size=1)
    assert len(dataset.data) == 1
    assert "top 5 results only" in dataset.data[0].task.lower()
    assert "limit 5" in (dataset.data[0].global_verifier or "").lower()


@pytest.mark.asyncio
async def test_generate_pairs_rejects_non_query_task_type(mocker) -> None:
    llm = _QueueModelService(payloads=["[]"])
    mocker.patch(
        "app.core.reasoner.model_service_factory.ModelServiceFactory.create",
        return_value=llm,
    )
    gen = SamplingDatasetGenerator(graph_db=object(), sampler=_DummySampler())

    with pytest.raises(ValueError, match="query-only"):
        # runtime-only check (even though types are query-only now)
        await gen.generate_pairs(  # type: ignore[arg-type]
            task_type="non-query",
            task_types_info=object(),  # will not be reached
            subgraph="{}",
            task_description="desc",
            nums=1,
        )


@pytest.mark.asyncio
async def test_generate_20_rows_with_mock_pipeline(mocker) -> None:
    batch_payload = """
[
  {
    "level":"L1",
    "task_subtype":"attribute_filtering",
    "task":"Find Alice",
    "verifier":"MATCH (n {name:'Alice'}) RETURN n",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup"],
    "global_verifier":"MATCH (n {name:'Alice'}) RETURN n",
    "expected_global":"alice_result"
  },
  {
    "level":"L1",
    "task_subtype":"attribute_filtering",
    "task":"Find Bob",
    "verifier":"MATCH (n {name:'Bob'}) RETURN n",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup"],
    "global_verifier":"MATCH (n {name:'Bob'}) RETURN n",
    "expected_global":"bob_result"
  },
  {
    "level":"L2",
    "task_subtype":"multi_hop",
    "task":"Find path from Alice to Bob within 2 hops",
    "verifier":"MATCH (a {name:'Alice'}), (b {name:'Bob'}) RETURN shortestPath((a)-[*..2]-(b))",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup","query.path.shortest"],
    "global_verifier":"MATCH (a {name:'Alice'}), (b {name:'Bob'}) RETURN shortestPath((a)-[*..2]-(b))",
    "expected_global":"path_result"
  },
  {
    "level":"L2",
    "task_subtype":"ranking",
    "task":"Top 1 transfer target of Alice",
    "verifier":"MATCH (a {name:'Alice'})-[:TRANSFER]->(t) RETURN t ORDER BY t.amount DESC LIMIT 1",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup","query.ranking.topk"],
    "global_verifier":"MATCH (a {name:'Alice'})-[:TRANSFER]->(t) RETURN t ORDER BY t.amount DESC LIMIT 1",
    "expected_global":"top_result"
  },
  {
    "level":"L2",
    "task_subtype":"aggregation",
    "task":"Count transfers from Alice",
    "verifier":"MATCH (a {name:'Alice'})-[:TRANSFER]->(t) RETURN count(t)",
    "generation_scope":"local_subgraph",
    "answer_scope":"global_graph",
    "intent_set":["query.lookup","query.aggregation.count"],
    "global_verifier":"MATCH (a {name:'Alice'})-[:TRANSFER]->(t) RETURN count(t)",
    "expected_global":"3"
  }
]
""".strip()

    # 20 rows with nums_per_subgraph=5 means 4 rounds, each round has generate + filter.
    llm_payloads = [batch_payload for _ in range(8)]
    llm = _QueueModelService(payloads=llm_payloads)
    mocker.patch(
        "app.core.reasoner.model_service_factory.ModelServiceFactory.create",
        return_value=llm,
    )

    gen = SamplingDatasetGenerator(
        graph_db=object(),
        sampler=_DummySampler(),
        strategy="query",
        nums_per_subgraph=5,
    )

    dataset = await gen.generate(task_desc="desc", dataset_name="ds20", size=20)
    assert dataset.name == "ds20"
    assert len(dataset.data) == 20
    assert dataset.qa_gate_stats["accepted_rows"] == 20
    assert dataset.qa_gate_stats["data_retention_rate_pct"] > 0
    assert llm.call_count == 8
