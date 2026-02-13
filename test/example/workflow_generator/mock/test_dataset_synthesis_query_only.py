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
  {"level":"L1","task_subtype":"attribute_filtering","task":"Find nodes with name = 'Alice'","verifier":"MATCH (n {name: 'Alice'}) RETURN n"},
  {"level":"L2","task_subtype":"multi_hop","task":"Check if Alice is connected to Bob within 2 hops","verifier":"MATCH (a {name:'Alice'}), (b {name:'Bob'}) RETURN shortestPath((a)-[*..2]-(b)) IS NOT NULL"}
]
""".strip()

    # Filter step can return a subset (or even the same set).
    filter_payload = """
[
  {"level":"L1","task_subtype":"attribute_filtering","task":"Find nodes with name = 'Alice'","verifier":"MATCH (n {name: 'Alice'}) RETURN n"}
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
    assert dataset.data[0].task_type == "query"
    assert llm.call_count == 2  # generate_pairs + filter


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
