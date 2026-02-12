import json
from pathlib import Path
import random
import time
import types
from typing import Iterable, List

import pytest

from app.core.common.type import MessageSourceType
from app.core.model.message import ModelMessage
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.model import Row
from app.core.workflow.dataset_synthesis.sampler import RandomWalkSampler, SubGraphSampler
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo
from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset


class SequenceModelService:
    """Deterministic LLM stub returning payloads in sequence."""

    def __init__(self, payloads: Iterable[str]):
        self._payloads: List[str] = list(payloads)
        self.calls = 0

    async def generate(self, sys_prompt, messages, tools=None, tool_call_ctx=None):
        if self.calls >= len(self._payloads):
            raise AssertionError("SequenceModelService has no payload left to consume")
        payload = self._payloads[self.calls]
        self.calls += 1
        message = messages[-1]
        return ModelMessage(
            payload=payload,
            job_id=message.get_job_id(),
            step=message.get_step() + 1,
            source_type=MessageSourceType.MODEL,
        )


class StubSampler(SubGraphSampler):
    """Sampler stub that returns pre-seeded subgraph strings."""

    def __init__(self, results: Iterable[str] = ("{\"nodes\": [], \"relationships\": []}",)):
        self._results: List[str] = list(results)
        if not self._results:
            self._results.append("{\"nodes\": [], \"relationships\": []}")
        self.calls = 0

    def get_random_subgraph(self, graph_db, max_depth, max_nodes, max_edges) -> str:
        if self.calls >= len(self._results):
            return self._results[-1]
        result = self._results[self.calls]
        self.calls += 1
        return result


def _create_generator(monkeypatch, *, sampler=None, llm_payloads=None, strategy=None):
    llm = SequenceModelService(llm_payloads or [""])
    monkeypatch.setattr(
        "app.core.workflow.dataset_synthesis.generator.ModelServiceFactory.create",
        lambda *args, **kwargs: llm,
    )
    generator = SamplingDatasetGenerator(
        graph_db=object(),
        sampler=sampler or StubSampler(),
        strategy=strategy,
        max_depth=1,
        max_noeds=3,
        max_edges=3,
        nums_per_subgraph=2,
    )
    return generator, llm


def test_extract_pairs_parses_valid_rows(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    payload = (
        '{"level": "L1", "task_subtype": "SubtypeA", "task": "Q1", "verifier": "A1"}'
        '\n'
        '{"level": "L2", "task_subtype": "SubtypeB", "task": "Q2", "verifier": "A2"}'
    )
    rows = generator.extract_pairs("query", payload)
    assert len(rows) == 2
    assert rows[0].task_subtype == "SubtypeA"
    assert rows[0].task_type == "query"
    assert rows[0].task == "Q1"
    assert rows[1].task_subtype == "SubtypeB"
    assert rows[1].task_type == "query"
    assert rows[1].level == "L2"
    assert rows[1].task == "Q2"
    


def test_extract_pairs_skips_invalid_objects(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    payload = '{"level": "L1", "task": "missing fields"}'
    rows = generator.extract_pairs("query", payload)
    assert rows == []


@pytest.mark.asyncio
async def test_identify_strategy_uses_llm_response(monkeypatch):
    generator, llm = _create_generator(monkeypatch, llm_payloads=["This is clearly a query task"])
    strategy = await generator.identify_strategy("desc")
    assert strategy == "query"
    # Query-only mode: strategy identification should not call LLM.
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_identify_strategy_returns_cached_strategy(monkeypatch):
    generator, llm = _create_generator(monkeypatch, strategy="non-query")
    result = await generator.identify_strategy("ignored")
    assert result == "query"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_generate_pairs_returns_rows(monkeypatch):
    payload = (
        '{"level": "L1", "task_subtype": "SubtypeA", "task": "Q1", "verifier": "A1"}'
        '\n'
        '{"level": "L1", "task_subtype": "SubtypeB", "task": "Q2", "verifier": "A2"}'
    )
    generator, _ = _create_generator(monkeypatch, llm_payloads=[payload])
    info = GraphTaskTypesInfo(strategy="query")
    rows = await generator.generate_pairs(
        task_type="query",
        task_types_info=info,
        subgraph="{\"nodes\": [], \"relationships\": []}",
        task_description="desc",
        nums=2,
    )
    assert len(rows) == 2
    assert all(row.task_type == "query" for row in rows)
    assert {row.task for row in rows} == {"Q1", "Q2"}
    assert {row.task_subtype for row in rows} == {"SubtypeA", "SubtypeB"}
    assert {row.verifier for row in rows} == {"A1", "A2"}


def test_get_task_type_from_strategy_direct(monkeypatch):
    generator, _ = _create_generator(monkeypatch, strategy="query")
    assert generator.get_task_type_from_strategy("query") == "query"


def test_get_task_type_from_strategy_mixed(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    assert generator.get_task_type_from_strategy("mixed") == "query"

@pytest.mark.asyncio
async def test_filter_returns_filtered_rows(monkeypatch):
    response = (
        '{"level": "L2", "task_subtype": "SubtypeB", "task": "Q", "verifier": "A"}'
        '\n'
        '{"level": "L3", "task_subtype": "SubtypeC", "task": "Task", "verifier": "Verify"}'
    )
    generator, _ = _create_generator(monkeypatch, llm_payloads=[response])
    dataset = [
        Row(
            level="L1",
            task_type="query",
            task_subtype="SubtypeA",
            task="Question",
            verifier="Answer",
        ),
        Row(
            level="L2",
            task_type="query",
            task_subtype="SubtypeB",
            task="Q",
            verifier="A",
        ),
        Row(
            level="L3",
            task_type="query",
            task_subtype="SubtypeC",
            task="Task",
            verifier="Verify",
        ),
    ]
    result = await generator.filter(
        task_type="query",
        task_desc="desc",
        subgraph="{\"nodes\": [], \"relationships\": []}",
        dataset=dataset,
    )
    assert len(result) == 2
    assert result[0].level == "L2"
    assert result[0].task == "Q"
    assert result[1].level == "L3"


@pytest.mark.asyncio
async def test_generate_compiles_dataset(monkeypatch):
    generator, _ = _create_generator(monkeypatch, sampler=StubSampler(["stub-subgraph"]))

    async def fake_identify(self, task_desc):
        return "query"

    async def fake_generate_pairs(
        self,
        task_type,
        task_types_info,
        subgraph,
        task_description,
        nums,
    ):
        return [
            Row(
                level="L1",
                task_type=task_type,
                task_subtype="Subtype",
                task="task1",
                verifier="verify1",
            ),
            Row(
                level="L1",
                task_type=task_type,
                task_subtype="Subtype",
                task="task2",
                verifier="verify2",
            ),
        ]

    async def fake_filter(self, task_type, task_desc, subgraph, dataset):
        return dataset

    generator.identify_strategy = types.MethodType(fake_identify, generator)
    generator.generate_pairs = types.MethodType(fake_generate_pairs, generator)
    generator.filter = types.MethodType(fake_filter, generator)

    dataset = await generator.generate(
        task_desc="desc",
        dataset_name="final",
        size=2,
    )

    assert dataset.name == "final"
    assert len(dataset.data) == 2
    assert {row.task for row in dataset.data} == {"task1", "task2"}


def test_random_walk_sampler_serializes_graph(monkeypatch):
    sampler = RandomWalkSampler()

    def fake_get_random_subgraph(self, graph_db, max_depth, max_nodes, max_edges):
        nodes = [
            {"node_id": "1", "labels": ["Person"], "properties": {"name": "Alice"}},
            {"node_id": "2", "labels": ["Person"], "properties": {"name": "Bob"}},
        ]
        rels = [
            {
                "rel_id": "r1",
                "rel_type": "KNOWS",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
            }
        ]
        return nodes, rels

    monkeypatch.setattr(RandomWalkSampler, "_get_random_subgraph", fake_get_random_subgraph)
    result = sampler.get_random_subgraph(graph_db=object(), max_depth=1, max_nodes=5, max_edges=5)
    graph = json.loads(result)
    assert len(graph["nodes"]) == 2
    assert graph["relationships"][0]["type"] == "KNOWS"


def test_load_workflow_train_dataset_reads_ratio():
    data_file = Path(__file__).parent / "data_example.json"
    with open(data_file, encoding="utf-8") as f:
        raw = json.load(f)

    dataset = load_workflow_train_dataset(
        task_desc="desc",
        path=str(data_file),
        ratio=0.5,
    )

    expected_len = int(len(raw) * 0.5)
    assert len(dataset.data) == expected_len
    assert dataset.task_desc == "desc"
    assert dataset.name == "test"
    assert all(isinstance(row, Row) for row in dataset.data)
