from __future__ import annotations

from pathlib import Path

from app.core.workflow.workflow_generator.mcts_workflow_generator.validator import (
    infer_single_expert_name,
    validate_workflow_yaml,
)


def test_validator_accepts_repo_basic_template(tmp_path: Path) -> None:
    # base_template doesn't include tools/actions/toolkit; validator expects full workflow.yml.
    # Validate the assembled round1 workflow produced by MCTSWorkflowGenerator.
    from app.core.workflow.dataset_synthesis.model import WorkflowTrainDataset
    from app.core.workflow.workflow_generator.mcts_workflow_generator.generator import (
        MCTSWorkflowGenerator,
    )
    from app.core.workflow.workflow_generator.mcts_workflow_generator.model import AgenticConfigSection

    dataset = WorkflowTrainDataset(name="ds", task_desc="d", data=[])
    gen = MCTSWorkflowGenerator(
        db=object(),  # type: ignore[arg-type]
        dataset=dataset,
        selector=object(),  # type: ignore[arg-type]
        expander=object(),  # type: ignore[arg-type]
        evaluator=object(),  # type: ignore[arg-type]
        optimize_grain=[AgenticConfigSection.EXPERTS, AgenticConfigSection.OPERATORS],
        optimized_path=str(tmp_path / "ws"),
        max_rounds=1,
        top_k=1,
    )
    # write assembled workflow.yml
    gen.init_workflow()
    round1 = Path(gen.optimized_path) / "round1" / "workflow.yml"
    result = validate_workflow_yaml(round1)
    assert result.ok, result.errors


def test_validator_rejects_multiple_tails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text(
        """
app: {name: "x"}
plugin: {workflow_platform: "BUILTIN"}
reasoner: {type: "DUAL"}
tools: []
actions: []
toolkit: []
operators:
  - &op1 {instruction: "1", output_schema: "a:1", actions: []}
  - &op2 {instruction: "2", output_schema: "a:2", actions: []}
  - &op3 {instruction: "3", output_schema: "a:3", actions: []}
experts:
  - profile: {name: "Main Expert", desc: "d"}
    workflow:
      - [*op1, *op2]
      - [*op1, *op3]
""".lstrip(),
        encoding="utf-8",
    )
    result = validate_workflow_yaml(bad)
    assert not result.ok
    assert any("tail" in e for e in result.errors)


def test_validator_rejects_cycle(tmp_path: Path) -> None:
    bad = tmp_path / "cycle.yml"
    bad.write_text(
        """
app: {name: "x"}
plugin: {workflow_platform: "BUILTIN"}
reasoner: {type: "DUAL"}
tools: []
actions: []
toolkit: []
operators:
  - &op1 {instruction: "1", output_schema: "a:1", actions: []}
  - &op2 {instruction: "2", output_schema: "a:2", actions: []}
experts:
  - profile: {name: "Main Expert", desc: "d"}
    workflow:
      - [*op1, *op2, *op1]
""".lstrip(),
        encoding="utf-8",
    )
    result = validate_workflow_yaml(bad)
    assert not result.ok
    assert any("cycle" in e for e in result.errors)


def test_validator_accepts_custom_main_expert_name(tmp_path: Path) -> None:
    ok = tmp_path / "ok.yml"
    ok.write_text(
        """
app: {name: "x"}
plugin: {workflow_platform: "BUILTIN"}
reasoner: {type: "DUAL"}
tools: []
actions: []
toolkit: []
operators:
  - &op1 {instruction: "1", output_schema: "a:1", actions: []}
  - &op2 {instruction: "2", output_schema: "a:2", actions: []}
experts:
  - profile: {name: "Entry", desc: "d"}
    workflow:
      - [*op1, *op2]
""".lstrip(),
        encoding="utf-8",
    )
    result = validate_workflow_yaml(ok, main_expert_name="Entry")
    assert result.ok, result.errors


def test_infer_single_expert_name(tmp_path: Path) -> None:
    ok = tmp_path / "ok.yml"
    ok.write_text(
        """
app: {name: "x"}
plugin: {workflow_platform: "BUILTIN"}
reasoner: {type: "DUAL"}
operators:
  - &op1 {instruction: "1", output_schema: "a:1", actions: []}
experts:
  - profile: {name: "Entry", desc: "d"}
    workflow:
      - [*op1]
""".lstrip(),
        encoding="utf-8",
    )
    assert infer_single_expert_name(ok) == "Entry"

    bad = tmp_path / "bad.yml"
    bad.write_text(
        """
app: {name: "x"}
experts:
  - profile: {name: "A"}
  - profile: {name: "B"}
""".lstrip(),
        encoding="utf-8",
    )
    assert infer_single_expert_name(bad) is None
