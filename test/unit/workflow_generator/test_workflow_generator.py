from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock

import pytest

from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo
from app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator import (
    Evaluator,
    LLMEvaluator,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.expander import (
    Expander,
    LLMExpander,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.generator import (
    MCTSWorkflowGenerator,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    AgenticConfigSection,
    OptimizeAction,
    OptimizeActionType,
    OptimizeObject,
    OptimizeResp,
    ReflectResult,
    WorkflowLogFormat,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.selector import (
    MixedProbabilitySelector,
    Selector,
)

ACTIONS_YAML = """actions:
  - &action_one
    name: action_one
    desc: First action
"""


OPERATORS_YAML = """operators:
  - &operator_one
    instruction: Original instruction
    output_schema: Original schema
    actions:
      - action_one
"""


EXPERTS_YAML = """experts:
  - profile:
      name: ExpertOne
      desc: Handles queries
    reasoner:
      actor_name: ExpertOne
      thinker_name: ExpertOne
    workflow:
      - - operator_one
"""


OPERATORS_UPDATE = """operators:
  - &operator_new
    instruction: Updated instruction
    output_schema: Updated schema
    actions:
      - action_one
"""


EXPERTS_UPDATE = """experts:
  - profile:
      name: ExpertOne
      desc: Handles queries
    reasoner:
      actor_name: ExpertOne
      thinker_name: ExpertOne
    workflow:
      - - operator_new
"""


def _make_round_context(round_number: int = 1) -> WorkflowLogFormat:
    return WorkflowLogFormat(
        round_number=round_number,
        score=0.5,
        modifications=["initial tweak"],
        reflection="reflection",
        feedbacks=[],
    )


def _make_current_config() -> Dict[str, str]:
    return {
        AgenticConfigSection.ACTIONS.value: ACTIONS_YAML,
        AgenticConfigSection.OPERATORS.value: OPERATORS_YAML,
        AgenticConfigSection.EXPERTS.value: EXPERTS_YAML,
    }


def _make_dataset() -> WorkflowTrainDataset:
    rows = [
        Row(
            level="L1",
            task_type="query",
            task_subtype="SubtypeA",
            task="Task-1",
            verifier="Answer-1",
        ),
        Row(
            level="L1",
            task_type="query",
            task_subtype="SubtypeB",
            task="Task-2",
            verifier="Answer-2",
        ),
        Row(
            level="L1",
            task_type="query",
            task_subtype="SubtypeC",
            task="Task-3",
            verifier="Answer-3",
        ),
    ]
    return WorkflowTrainDataset(name="demo", task_desc="Handle queries", data=rows)


class _StubSelector(Selector):
    def __init__(self):
        self.calls: List[Dict[int, WorkflowLogFormat]] = []

    def select(self, top_k: int, logs: Dict[int, WorkflowLogFormat]) -> WorkflowLogFormat:  # noqa: D401
        self.calls.append(logs)
        return logs[1]


class _StubExpander(Expander):
    async def expand(self, task_tesc, current_config, round_context):  # noqa: D401
        action = OptimizeAction(
            action_type=OptimizeActionType.MODIFY,
            optimize_object=OptimizeObject.OPERATOR,
            reason="update operators",
        )
        resp = OptimizeResp(
            modifications=["mod-1"],
            new_configs={
                AgenticConfigSection.OPERATORS.value: OPERATORS_UPDATE,
                AgenticConfigSection.EXPERTS.value: EXPERTS_UPDATE,
            },
        )
        return [action], resp


class _StubEvaluator(Evaluator):
    def __init__(self):
        self.calls: List[int] = []

    async def evaluate_workflow(  # noqa: D401
        self,
        optimized_path: str,
        round_num: int,
        parent_round: int,
        dataset: List[Row],
        modifications: List[str],
    ) -> tuple[float, str]:
        self.calls.append(round_num)
        return float(round_num), f"reflection-{round_num}"


class _FlatScoreEvaluator(Evaluator):
    def __init__(self, score: float = 1.0):
        self.score = score
        self.calls: List[int] = []

    async def evaluate_workflow(  # noqa: D401
        self,
        optimized_path: str,
        round_num: int,
        parent_round: int,
        dataset: List[Row],
        modifications: List[str],
    ) -> tuple[float, str]:
        self.calls.append(round_num)
        return self.score, f"reflection-{round_num}"


def test_mixed_probability_selector_select(monkeypatch):
    selector = MixedProbabilitySelector()
    logs = {
        1: WorkflowLogFormat(
            round_number=1,
            score=0.8,
            modifications=[],
            reflection="r1",
            feedbacks=[],
        ),
        2: WorkflowLogFormat(
            round_number=2,
            score=0.3,
            modifications=[],
            reflection="r2",
            feedbacks=[],
        ),
    }

    captured = {}

    def fake_choice(size, p):
        captured["probabilities"] = p
        return 0

    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.selector.np.random.choice",
        fake_choice,
    )

    selected = selector.select(top_k=2, logs=logs)

    assert selected.round_number == 1
    assert pytest.approx(sum(captured["probabilities"])) == 1


@pytest.mark.asyncio
async def test_llm_expander_expand_returns_combined_configs(monkeypatch):
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.expander.ModelServiceFactory.create",
        lambda *args, **kwargs: None,
    )

    expander = LLMExpander()

    mocked_generate = AsyncMock(
        side_effect=[
            [
                {
                    "action_type": OptimizeActionType.MODIFY.value,
                    "optimize_object": OptimizeObject.OPERATOR.value,
                    "reason": "Improve operator",
                },
                {
                    "action_type": OptimizeActionType.MODIFY.value,
                    "optimize_object": OptimizeObject.EXPERT.value,
                    "reason": "Align experts",
                },
            ],
            {
                "modifications": ["update operators"],
                "new_configs": {
                    AgenticConfigSection.OPERATORS.value: OPERATORS_UPDATE,
                },
            },
            {
                "modifications": ["update experts"],
                "new_configs": {
                    AgenticConfigSection.EXPERTS.value: EXPERTS_UPDATE,
                },
            },
        ]
    )
    monkeypatch.setattr(expander, "_generate", mocked_generate)

    actions, response = await expander.expand(
        task_tesc="Configure agent",
        current_config=_make_current_config(),
        round_context=_make_round_context(),
    )

    assert [action.optimize_object for action in actions] == [
        OptimizeObject.OPERATOR,
        OptimizeObject.EXPERT,
    ]
    assert response.new_configs[AgenticConfigSection.OPERATORS.value] == OPERATORS_UPDATE
    assert response.new_configs[AgenticConfigSection.EXPERTS.value] == EXPERTS_UPDATE
    assert mocked_generate.await_count == 3


@pytest.mark.asyncio
async def test_llm_expander_expand_raises_when_no_actions(monkeypatch):
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.expander.ModelServiceFactory.create",
        lambda *args, **kwargs: None,
    )

    expander = LLMExpander()
    expander.max_retry = 2
    mocked_generate = AsyncMock(return_value=[])
    monkeypatch.setattr(expander, "_generate", mocked_generate)

    with pytest.raises(Exception, match="failed while get optimize actions"):
        await expander.expand(
            task_tesc="Configure agent",
            current_config=_make_current_config(),
            round_context=_make_round_context(),
        )

    assert mocked_generate.await_count == expander.max_retry


@pytest.mark.asyncio
async def test_llm_evaluator_evaluate_workflow(monkeypatch, tmp_path):
    dataset = _make_dataset().data[:2]
    optimized_path = tmp_path / "workspace"

    evaluator = LLMEvaluator(need_reflect=True)
    evaluator._scoring_batch_size = 1
    mocked_scoring = AsyncMock(return_value=3)
    mocked_reflect = AsyncMock(
        return_value=ReflectResult(failed_reason=["f"], optimize_suggestion=["s"])
    )

    async def _fake_run_dataset(
        self,
        *,
        workflow_path,
        rows,
        graph_db_config=None,
        reset_state=True,
    ):
        records = []
        for idx, row in enumerate(rows):
            records.append(
                SimpleNamespace(
                    task=row.task,
                    verifier=row.verifier,
                    model_output=f"output-{idx}",
                    error="",
                    latency_ms=10.0,
                    tokens=0,
                )
            )
        return records

    evaluator._llm_scoring = mocked_scoring
    evaluator._reflect = mocked_reflect
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator.WorkflowRunner.run_dataset",
        _fake_run_dataset,
    )
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator.load_execute_result",
        lambda path: [],
    )

    avg_score, reflection_json = await evaluator.evaluate_workflow(
        optimized_path=str(optimized_path),
        round_num=1,
        parent_round=-1,
        dataset=dataset,
        modifications=["mod"],
    )

    assert avg_score == 3
    reflection = json.loads(reflection_json)
    assert reflection == {"failed_reason": ["f"], "optimize_suggestion": ["s"]}
    results_file = optimized_path / "round1" / "results.json"
    assert results_file.exists()
    assert mocked_scoring.await_count == len(dataset)
    mocked_reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcts_generator_generate_rounds(monkeypatch, tmp_path):
    dataset = _make_dataset()
    base_path = Path(__file__).resolve().parents[3]
    init_template = (
        base_path
        / "app"
        / "core"
        / "workflow"
        / "workflow_generator"
        / "mcts_workflow_generator"
        / "init_template"
        / "basic_template.yml"
    )

    stub_selector = _StubSelector()
    stub_expander = _StubExpander()
    stub_evaluator = _StubEvaluator()

    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.time.time",
        lambda: 1_700_000_000,
    )
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.validate_candidate_config",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )

    generator = MCTSWorkflowGenerator(
        db=None,
        dataset=dataset,
        selector=stub_selector,
        expander=stub_expander,
        evaluator=stub_evaluator,
        optimize_grain=[AgenticConfigSection.OPERATORS, AgenticConfigSection.EXPERTS],
        init_template_path=str(init_template),
        max_rounds=2,
        optimized_path=str(tmp_path),
        top_k=2,
        max_retries=1,
    )

    max_score, optimal_round = await generator._generate_rounds()

    assert max_score == 2
    assert optimal_round == 2
    assert stub_selector.calls
    assert stub_evaluator.calls == [1, 2]
    assert 2 in generator.logs


@pytest.mark.asyncio
async def test_mcts_generator_early_stop_on_no_improvement(monkeypatch, tmp_path):
    dataset = _make_dataset()
    base_path = Path(__file__).resolve().parents[3]
    init_template = (
        base_path
        / "app"
        / "core"
        / "workflow"
        / "workflow_generator"
        / "mcts_workflow_generator"
        / "init_template"
        / "basic_template.yml"
    )

    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.validate_candidate_config",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )

    flat_evaluator = _FlatScoreEvaluator(score=1.0)
    generator = MCTSWorkflowGenerator(
        db=None,
        dataset=dataset,
        selector=_StubSelector(),
        expander=_StubExpander(),
        evaluator=flat_evaluator,
        optimize_grain=[AgenticConfigSection.OPERATORS, AgenticConfigSection.EXPERTS],
        init_template_path=str(init_template),
        max_rounds=5,
        optimized_path=str(tmp_path),
        top_k=2,
        max_retries=1,
        no_improvement_patience=2,
    )

    max_score, optimal_round = await generator._generate_rounds()

    assert max_score == 1.0
    assert optimal_round == 1
    assert flat_evaluator.calls == [1, 2, 3]
    assert 3 in generator.logs
    assert 4 not in generator.logs


@pytest.mark.asyncio
async def test_mcts_generator_resume_from_existing_logs(monkeypatch, tmp_path):
    dataset = _make_dataset()
    base_path = Path(__file__).resolve().parents[3]
    init_template = (
        base_path
        / "app"
        / "core"
        / "workflow"
        / "workflow_generator"
        / "mcts_workflow_generator"
        / "init_template"
        / "basic_template.yml"
    )

    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.time.time",
        lambda: 1_700_000_020,
    )
    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.validate_candidate_config",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )

    first_evaluator = _StubEvaluator()
    first_generator = MCTSWorkflowGenerator(
        db=None,
        dataset=dataset,
        selector=_StubSelector(),
        expander=_StubExpander(),
        evaluator=first_evaluator,
        optimize_grain=[AgenticConfigSection.OPERATORS, AgenticConfigSection.EXPERTS],
        init_template_path=str(init_template),
        max_rounds=2,
        optimized_path=str(tmp_path),
        top_k=2,
        max_retries=1,
    )
    await first_generator._generate_rounds()
    assert first_evaluator.calls == [1, 2]

    resumed_evaluator = _StubEvaluator()
    resumed_generator = MCTSWorkflowGenerator(
        db=None,
        dataset=dataset,
        selector=_StubSelector(),
        expander=_StubExpander(),
        evaluator=resumed_evaluator,
        optimize_grain=[AgenticConfigSection.OPERATORS, AgenticConfigSection.EXPERTS],
        init_template_path=str(init_template),
        max_rounds=4,
        optimized_path=first_generator.optimized_path,
        top_k=2,
        max_retries=1,
        resume=True,
    )
    max_score, optimal_round = await resumed_generator._generate_rounds()

    assert resumed_evaluator.calls == [3, 4]
    assert max_score == 4
    assert optimal_round == 4
    assert all(r in resumed_generator.logs for r in [1, 2, 3, 4])


def test_mcts_generator_load_config_dict(monkeypatch, tmp_path):
    dataset = _make_dataset()
    base_path = Path(__file__).resolve().parents[3]
    init_template = (
        base_path
        / "app"
        / "core"
        / "workflow"
        / "workflow_generator"
        / "mcts_workflow_generator"
        / "init_template"
        / "basic_template.yml"
    )

    monkeypatch.setattr(
        "app.core.workflow.workflow_generator.mcts_workflow_generator.generator.time.time",
        lambda: 1_700_000_001,
    )

    generator = MCTSWorkflowGenerator(
        db=None,
        dataset=dataset,
        selector=_StubSelector(),
        expander=_StubExpander(),
        evaluator=_StubEvaluator(),
        optimize_grain=[AgenticConfigSection.OPERATORS, AgenticConfigSection.EXPERTS],
        init_template_path=str(init_template),
        max_rounds=1,
        optimized_path=str(tmp_path),
        top_k=1,
        max_retries=1,
    )

    generator.init_workflow()
    config = generator.load_config_dict(1, skip_section=[AgenticConfigSection.PLUGIN])

    assert "app" in config
    assert "plugin" not in config


def test_graph_task_types_info_snapshot():
    """Smoke test to ensure GraphTaskTypesInfo exposes task metadata."""

    info = GraphTaskTypesInfo()

    tasks_info = info.get_tasks_info()
    counts_info = info.get_count_info()

    assert "L1" in tasks_info
    assert "L1" in counts_info
