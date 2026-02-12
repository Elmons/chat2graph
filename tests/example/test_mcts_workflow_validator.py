from __future__ import annotations

from pathlib import Path

from app.core.workflow.workflow_generator.mcts_workflow_generator.validator import (
    validate_workflow_yaml,
)


def test_validator_accepts_repo_basic_template() -> None:
    path = Path(
        "app/core/workflow/workflow_generator/mcts_workflow_generator/init_template/basic_template.yml"
    )
    result = validate_workflow_yaml(path)
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

