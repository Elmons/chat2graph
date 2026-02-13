from __future__ import annotations

from pathlib import Path

from app.core.workflow.workflow_generator.mcts_workflow_generator.config_assembler import (
    assemble_workflow_file,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.constraints import (
    WorkflowConstraints,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.validator import (
    validate_candidate_config,
)


def _build_valid_workflow(tmp_path: Path) -> Path:
    out = tmp_path / "workflow.yml"
    return assemble_workflow_file(
        base_template_path=(
            "app/core/workflow/workflow_generator/mcts_workflow_generator/"
            "init_template/base_template.yml"
        ),
        toolset_path="app/core/sdk/toolsets/graph_only.yml",
        candidate_sections={},
        output_path=out,
    )


def test_validate_candidate_config_runs_parse_and_dry_run(monkeypatch, tmp_path: Path) -> None:
    workflow_path = _build_valid_workflow(tmp_path)

    called = {"load": 0}

    def _fake_load(_path):
        called["load"] += 1
        return object()

    monkeypatch.setattr(
        "app.core.sdk.agentic_service.AgenticService.load",
        staticmethod(_fake_load),
    )

    constraints = WorkflowConstraints(
        main_expert_name="Main Expert",
        require_agentic_parse=True,
        require_agentic_service_dry_run=True,
    )
    result = validate_candidate_config(workflow_path, constraints=constraints)

    assert result.ok, result.errors
    assert called["load"] == 1


def test_validate_candidate_config_reports_structural_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text("app: {name: x}\nexperts: []\n", encoding="utf-8")

    constraints = WorkflowConstraints(require_agentic_parse=False, require_agentic_service_dry_run=False)
    result = validate_candidate_config(bad, constraints=constraints)

    assert not result.ok
    assert "structure" in result.stage_errors
