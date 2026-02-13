from __future__ import annotations

from pathlib import Path

from app.core.workflow.workflow_generator.mcts_workflow_generator.config_assembler import (
    assemble_workflow_file,
    build_action_constraints,
    collect_allowed_action_refs,
)


def test_assemble_workflow_file_merges_base_toolset_and_candidate(tmp_path: Path) -> None:
    out = tmp_path / "workflow.yml"
    candidate_sections = {
        "operators": (
            "operators:\n"
            "  - &qa_operator\n"
            "    instruction: |\n"
            "      You are a strict QA operator.\n"
            "    output_schema: |\n"
            "      answer: text\n"
            "    actions: []\n"
        ),
        "experts": (
            "experts:\n"
            "  - profile:\n"
            "      name: \"Main Expert\"\n"
            "      desc: \"Single entry expert\"\n"
            "    workflow:\n"
            "      - - *qa_operator\n"
        ),
    }

    assembled = assemble_workflow_file(
        base_template_path=(
            "app/core/workflow/workflow_generator/mcts_workflow_generator/"
            "init_template/base_template.yml"
        ),
        toolset_path="app/core/sdk/toolsets/graph_only.yml",
        candidate_sections=candidate_sections,
        output_path=out,
    )

    text = assembled.read_text(encoding="utf-8")
    assert "tools:" in text
    assert "actions:" in text
    assert "toolkit:" in text
    assert "You are a strict QA operator." in text


def test_collect_allowed_action_refs_and_prompt_constraints() -> None:
    actions_section = """
actions:
  - &schema_getter_action
    name: "schema_getter"
    desc: "Get graph schema"
  - &query_action
    name: "query_execution"
    desc: "Run cypher"
""".strip()

    allowed = collect_allowed_action_refs(actions_section)
    assert "schema_getter_action" in allowed
    assert "query_action" in allowed
    assert "schema_getter" in allowed
    assert "query_execution" in allowed

    constraints = build_action_constraints(actions_section)
    assert "schema_getter_action" in constraints
    assert "action.name=schema_getter" in constraints
