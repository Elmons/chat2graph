from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Set

import yaml

from app.core.workflow.workflow_generator.mcts_workflow_generator.model import AgenticConfigSection


def _load_sections(path: str | Path) -> Dict[str, str]:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    result: Dict[str, str] = {}
    for section in AgenticConfigSection:
        name = str(section.value)
        pattern = re.compile(rf"(^|\n){name}:(.*?)(?=\n\w+:|\Z)", re.DOTALL)
        match = pattern.search(content)
        if match:
            result[name] = match.group(0).strip()
    return result


def _section_order() -> List[AgenticConfigSection]:
    return [
        AgenticConfigSection.APP,
        AgenticConfigSection.PLUGIN,
        AgenticConfigSection.REASONER,
        AgenticConfigSection.TOOLS,
        AgenticConfigSection.ACTIONS,
        AgenticConfigSection.TOOLKIT,
        AgenticConfigSection.OPERATORS,
        AgenticConfigSection.EXPERTS,
        AgenticConfigSection.KNOWLEDGEBASE,
        AgenticConfigSection.MEMORY,
        AgenticConfigSection.ENV,
    ]


def _required_sections() -> Sequence[AgenticConfigSection]:
    return (
        AgenticConfigSection.APP,
        AgenticConfigSection.PLUGIN,
        AgenticConfigSection.REASONER,
        AgenticConfigSection.TOOLS,
        AgenticConfigSection.ACTIONS,
        AgenticConfigSection.TOOLKIT,
        AgenticConfigSection.OPERATORS,
        AgenticConfigSection.EXPERTS,
    )


def assemble_config_sections(
    *,
    base_sections: Mapping[str, str],
    toolset_sections: Mapping[str, str],
    candidate_sections: Mapping[str, str],
) -> Dict[str, str]:
    """Merge config sections, with candidate sections overriding defaults."""
    merged: Dict[str, str] = {}
    merged.update(base_sections)
    merged.update(toolset_sections)
    merged.update(candidate_sections)
    return merged


def write_assembled_config(
    *,
    output_path: str | Path,
    merged_sections: Mapping[str, str],
    section_order: Sequence[AgenticConfigSection] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_sections = list(section_order or _section_order())
    with path.open("w", encoding="utf-8") as f:
        first = True
        for section in ordered_sections:
            name = str(section.value)
            body = merged_sections.get(name)
            if not body:
                continue
            if not first:
                f.write("\n\n")
            f.write(body.strip())
            first = False
        f.write("\n")
    return path


def validate_assembled_sections(merged_sections: Mapping[str, str]) -> List[str]:
    errors: List[str] = []
    for section in _required_sections():
        name = str(section.value)
        if not merged_sections.get(name):
            errors.append(f"missing `{name}` section in assembled workflow")
    return errors


def assemble_workflow_file(
    *,
    base_template_path: str | Path,
    toolset_path: str | Path,
    candidate_sections: Mapping[str, str],
    output_path: str | Path,
) -> Path:
    base_sections = _load_sections(base_template_path)
    toolset_sections = _load_sections(toolset_path)
    merged = assemble_config_sections(
        base_sections=base_sections,
        toolset_sections=toolset_sections,
        candidate_sections=candidate_sections,
    )
    errors = validate_assembled_sections(merged)
    if errors:
        raise ValueError(f"assemble_workflow_file failed: {errors}")
    return write_assembled_config(output_path=output_path, merged_sections=merged)


def assemble_workflow_file_from_candidate_yaml(
    *,
    base_template_path: str | Path,
    toolset_path: str | Path,
    candidate_yaml_path: str | Path,
    output_path: str | Path,
) -> Path:
    candidate_sections = _load_sections(candidate_yaml_path)
    return assemble_workflow_file(
        base_template_path=base_template_path,
        toolset_path=toolset_path,
        candidate_sections=candidate_sections,
        output_path=output_path,
    )


def build_action_constraints(actions_section: str) -> str:
    """Build a compact action whitelist text for prompts."""
    anchor_pattern = re.compile(r"-\s*&([A-Za-z0-9_]+)\s*\n")
    anchors: List[str] = []

    def _capture(match: re.Match[str]) -> str:
        anchors.append(match.group(1))
        return "-\n"

    text_wo_anchor = anchor_pattern.sub(_capture, actions_section)
    text_cleaned = re.sub(r"\*([A-Za-z0-9_]+)", r"\1", text_wo_anchor)
    parsed = yaml.safe_load(text_cleaned) or {}
    actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
    lines: List[str] = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        anchor = anchors[idx] if idx < len(anchors) else ""
        name = action.get("name", "")
        desc = str(action.get("desc", "")).strip().replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        if anchor and name:
            lines.append(f"- {anchor} (action.name={name}): {desc}")
        elif anchor:
            lines.append(f"- {anchor}: {desc}")
        elif name:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def collect_allowed_action_refs(actions_section: str) -> Set[str]:
    """Collect allowed action identifiers (both YAML anchors and action.name)."""
    allowed: Set[str] = set()
    anchor_pattern = re.compile(r"-\s*&([A-Za-z0-9_]+)\s*\n")
    anchors = anchor_pattern.findall(actions_section)
    allowed.update(a.strip() for a in anchors if a.strip())

    text_wo_anchor = anchor_pattern.sub("-\n", actions_section)
    text_cleaned = re.sub(r"\*([A-Za-z0-9_]+)", r"\1", text_wo_anchor)
    parsed = yaml.safe_load(text_cleaned) or {}
    actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
    for action in actions:
        if isinstance(action, dict):
            name = action.get("name")
            if isinstance(name, str) and name.strip():
                allowed.add(name.strip())
    return allowed


def serialize_graph_db_config(config: object) -> str:
    """Best-effort serialization helper used by run metadata."""
    if config is None:
        return ""
    if hasattr(config, "to_dict"):
        try:
            return json.dumps(config.to_dict(), ensure_ascii=False)
        except Exception:
            return str(config)
    return str(config)
