from __future__ import annotations

from pathlib import Path

import yaml

from app.core.model.agentic_config import AgenticConfig


def test_mcts_init_template_has_single_entry_expert_and_no_mcp_tools() -> None:
    path = Path(
        "app/core/workflow/workflow_generator/mcts_workflow_generator/init_template/basic_template.yml"
    )
    config = AgenticConfig.from_yaml(str(path))
    assert len(config.experts) == 1
    assert config.experts[0].profile.name == "Main Expert"

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tools = raw.get("tools", [])
    assert all(isinstance(t, dict) for t in tools)
    assert all(t.get("type", "LOCAL_TOOL") != "MCP" for t in tools)

