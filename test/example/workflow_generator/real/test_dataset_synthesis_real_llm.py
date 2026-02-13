from __future__ import annotations

import os

import pytest

from app.core.common.system_env import SystemEnv
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo


class _DummySampler:
    def get_random_subgraph(self, graph_db, max_depth: int, max_nodes: int, max_edges: int) -> str:
        return "{\"nodes\": [], \"relationships\": []}"


def _has_llm_config() -> bool:
    return bool(SystemEnv.LLM_NAME and SystemEnv.LLM_ENDPOINT and SystemEnv.LLM_APIKEY)


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_generate_pairs_real_llm_smoke() -> None:
    if os.getenv("CHAT2GRAPH_RUN_REAL_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_REAL_LLM_TESTS=1 to enable real LLM tests")
    if not _has_llm_config():
        pytest.skip("LLM not configured (LLM_NAME/LLM_ENDPOINT/LLM_APIKEY missing)")

    gen = SamplingDatasetGenerator(graph_db=object(), sampler=_DummySampler(), strategy="query")  # type: ignore[arg-type]
    task_types_info = GraphTaskTypesInfo(strategy="query")

    rows = await gen.generate_pairs(
        task_type="query",
        task_types_info=task_types_info,
        subgraph="{\"nodes\": [], \"relationships\": []}",
        task_description="Generate one simple query QA pair about this subgraph.",
        nums=1,
    )
    assert isinstance(rows, list)

