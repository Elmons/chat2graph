from __future__ import annotations

import json
from pathlib import Path

from app.core.workflow.dataset_synthesis.model import WorkflowTrainDataset
from app.core.workflow.workflow_generator.mcts_workflow_generator.generator import (
    MCTSWorkflowGenerator,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    AgenticConfigSection,
    WorkflowLogFormat,
)


class _NoopSelector:
    def select(self, top_k: int, logs: dict):  # pragma: no cover
        raise AssertionError("not used")


class _NoopExpander:
    async def expand(self, task_tesc: str, current_config: dict, round_context):  # pragma: no cover
        raise AssertionError("not used")


class _NoopEvaluator:
    async def evaluate_workflow(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not used")


def test_log_save_writes_edges_json(tmp_path: Path) -> None:
    dataset = WorkflowTrainDataset(name="ds", task_desc="d", data=[])
    gen = MCTSWorkflowGenerator(
        db=object(),  # type: ignore[arg-type]
        dataset=dataset,
        selector=_NoopSelector(),  # type: ignore[arg-type]
        expander=_NoopExpander(),  # type: ignore[arg-type]
        evaluator=_NoopEvaluator(),  # type: ignore[arg-type]
        optimize_grain=[AgenticConfigSection.EXPERTS, AgenticConfigSection.OPERATORS],
        optimized_path=str(tmp_path / "ws"),
        max_rounds=1,
        top_k=1,
    )
    gen.logs = {
        1: WorkflowLogFormat(
            round_number=1,
            parent_round=None,
            score=0.0,
            reflection="",
            modifications=[],
            feedbacks=[],
        ),
        2: WorkflowLogFormat(
            round_number=2,
            parent_round=1,
            score=1.0,
            reflection="",
            modifications=["m"],
            feedbacks=[],
        ),
    }
    gen.log_save()

    edges_path = Path(gen.optimized_path) / "log" / "edges.json"
    edges = json.loads(edges_path.read_text(encoding="utf-8"))
    assert edges == [{"parent_round": 1, "child_round": 2}]

