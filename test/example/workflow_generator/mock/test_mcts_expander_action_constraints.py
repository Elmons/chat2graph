from __future__ import annotations

import pytest

from app.core.workflow.workflow_generator.mcts_workflow_generator.expander import LLMExpander
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import WorkflowLogFormat


@pytest.mark.asyncio
async def test_get_optimize_actions_injects_allowed_action_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    expander = LLMExpander(main_expert_name="Main Expert")
    captured = {"prompt": ""}

    async def _fake_generate(prompt: str, job_id: str, filter, extra_messages):  # noqa: A002
        captured["prompt"] = prompt
        return [{"action_type": "modify", "optimize_object": "operator", "reason": "r"}]

    monkeypatch.setattr(expander, "_generate", _fake_generate)

    actions_yaml = """
actions:
  - &schema_getter_action
    name: "schema_getter"
    desc: "Get schema"
""".strip()

    context = WorkflowLogFormat(
        round_number=1,
        parent_round=None,
        score=0.0,
        reflection="",
        modifications=[],
        feedbacks=[],
    )

    result = await expander._get_optimize_actions(
        task_desc="task",
        current_actions=actions_yaml,
        current_operators="operators: []",
        current_experts="experts: []",
        round_context=context,
        optimization_mode="conservative",
        change_budget="local-only",
        max_actions=2,
    )

    assert result
    assert "Allowed Actions (Strict Whitelist)" in captured["prompt"]
    assert "schema_getter_action" in captured["prompt"]
