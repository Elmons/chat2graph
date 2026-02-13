from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from app.core.common.logger import Chat2GraphLogger
from app.core.common.type import WorkflowPlatformType
from app.core.workflow.workflow_generator.mcts_workflow_generator.constraints import (
    WorkflowConstraints,
)

logger = Chat2GraphLogger.get_logger(__name__)


@dataclass(frozen=True)
class WorkflowValidationResult:
    ok: bool
    errors: List[str]


@dataclass(frozen=True)
class CandidateValidationResult:
    ok: bool
    errors: List[str]
    stage_errors: Dict[str, List[str]]


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _canonical_operator(op: Dict[str, Any]) -> Tuple[str, str, Tuple[str, ...]]:
    instruction = _as_str(op.get("instruction"))
    output_schema = _as_str(op.get("output_schema"))
    actions_raw = op.get("actions", [])
    action_names: List[str] = []
    if isinstance(actions_raw, list):
        for a in actions_raw:
            if isinstance(a, str):
                name = a.strip()
                if name:
                    action_names.append(name)
            elif isinstance(a, dict):
                name = a.get("name")
                if isinstance(name, str) and name.strip():
                    action_names.append(name.strip())
    actions = tuple(action_names)
    return (instruction, output_schema, actions)


def _build_edges(workflow: Iterable[Iterable[Dict[str, Any]]]) -> List[Tuple[Tuple, Tuple]]:
    edges: List[Tuple[Tuple, Tuple]] = []
    for chain in workflow:
        chain_list = list(chain)
        for i in range(len(chain_list) - 1):
            edges.append((_canonical_operator(chain_list[i]), _canonical_operator(chain_list[i + 1])))
    return edges


def validate_workflow_yaml(
    yaml_path: str | Path,
    *,
    main_expert_name: str = "Main Expert",
    expected_workflow_platform: Optional[WorkflowPlatformType] = None,
    require_single_expert: bool = True,
    require_single_tail: bool = True,
) -> WorkflowValidationResult:
    """Validate workflow YAML for single-expert execution and DAG constraints.

    Hard constraints for this refactor iteration:
    - Exactly one expert, named ``main_expert_name``.
    - Expert workflow forms a DAG and has exactly one tail (out-degree == 0).
    - Expert workflow operators must exist in the top-level ``operators`` list.
    """
    path = Path(yaml_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return WorkflowValidationResult(ok=False, errors=[f"YAML parse failed: {e}"])

    if not isinstance(raw, dict):
        return WorkflowValidationResult(ok=False, errors=["YAML root must be a mapping"])

    errors: List[str] = []

    if expected_workflow_platform is not None:
        plugin = raw.get("plugin", {})
        workflow_platform = None
        if isinstance(plugin, dict):
            workflow_platform = plugin.get("workflow_platform")
        if workflow_platform != expected_workflow_platform.value:
            errors.append(
                "workflow platform mismatch: "
                f"expected {expected_workflow_platform.value!r}, got {workflow_platform!r}"
            )

    experts = raw.get("experts", [])
    if not isinstance(experts, list):
        errors.append("`experts` must be a list")
        return WorkflowValidationResult(ok=False, errors=errors)
    if require_single_expert and len(experts) != 1:
        errors.append(f"single-expert mode requires exactly 1 expert, got {len(experts)}")
        return WorkflowValidationResult(ok=False, errors=errors)
    if len(experts) == 0:
        errors.append("`experts` must be non-empty")
        return WorkflowValidationResult(ok=False, errors=errors)
    expert0 = experts[0]
    if not isinstance(expert0, dict):
        errors.append("expert must be a mapping")
        return WorkflowValidationResult(ok=False, errors=errors)
    profile = expert0.get("profile", {})
    if not isinstance(profile, dict):
        errors.append("expert.profile must be a mapping")
        return WorkflowValidationResult(ok=False, errors=errors)
    if require_single_expert and profile.get("name") != main_expert_name:
        errors.append(
            f"single-expert mode requires expert profile.name == {main_expert_name!r}, "
            f"got {profile.get('name')!r}"
        )

    operators = raw.get("operators", [])
    if not isinstance(operators, list) or not all(isinstance(o, dict) for o in operators):
        errors.append("`operators` must be a list of mappings")
        return WorkflowValidationResult(ok=False, errors=errors)
    if len(operators) == 0:
        errors.append("`operators` must be non-empty in single-expert mode")
        return WorkflowValidationResult(ok=False, errors=errors)

    operator_keys = {"instruction", "output_schema", "actions"}
    canonical_ops: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}

    # action name registry (for operator action reference checks)
    actions_section = raw.get("actions", [])
    action_name_set: set[str] = set()
    if isinstance(actions_section, list):
        for a in actions_section:
            if isinstance(a, dict):
                name = a.get("name")
                if isinstance(name, str) and name.strip():
                    action_name_set.add(name.strip())

    for idx, op in enumerate(operators):
        missing = [k for k in operator_keys if k not in op]
        if missing:
            errors.append(f"operators[{idx}] missing fields: {missing}")
            continue
        if not isinstance(op.get("instruction"), str) or not _as_str(op.get("instruction")).strip():
            errors.append(f"operators[{idx}].instruction must be a non-empty string")
        if (
            not isinstance(op.get("output_schema"), str)
            or not _as_str(op.get("output_schema")).strip()
        ):
            errors.append(f"operators[{idx}].output_schema must be a non-empty string")
        if not isinstance(op.get("actions"), list):
            errors.append(f"operators[{idx}].actions must be a list")
        else:
            # Validate action references when actions are declared in YAML.
            for a in op.get("actions", []):
                if isinstance(a, dict) and "name" in a:
                    name = a.get("name")
                    if isinstance(name, str) and name.strip():
                        if action_name_set and name.strip() not in action_name_set:
                            errors.append(
                                f"operators[{idx}].actions references unknown action {name!r}"
                            )
                    else:
                        errors.append(f"operators[{idx}].actions contains invalid action mapping")
                elif isinstance(a, str):
                    # tolerate string-only action names if present
                    if action_name_set and a.strip() and a.strip() not in action_name_set:
                        errors.append(f"operators[{idx}].actions references unknown action {a!r}")
                else:
                    # allow empty list, but reject unrecognized structures
                    if a is not None:
                        errors.append(f"operators[{idx}].actions contains invalid action ref")

        key = _canonical_operator(op)
        if key in canonical_ops:
            errors.append(
                "duplicate operator definition detected (same instruction/output_schema/actions)"
            )
        canonical_ops[key] = idx

    workflow = expert0.get("workflow", [])
    if not isinstance(workflow, list) or not all(isinstance(x, list) for x in workflow):
        errors.append("expert.workflow must be a 2D list")
        return WorkflowValidationResult(ok=False, errors=errors)
    if len(workflow) == 0:
        errors.append("expert.workflow must be non-empty")
        return WorkflowValidationResult(ok=False, errors=errors)

    workflow_ops: List[Tuple[str, str, Tuple[str, ...]]] = []
    for chain in workflow:
        for op in chain:
            if not isinstance(op, dict):
                errors.append("workflow operator must be a mapping (YAML anchor alias should resolve)")
                continue
            if any(k not in op for k in operator_keys):
                errors.append("workflow operator missing required fields")
                continue
            workflow_ops.append(_canonical_operator(op))

    for op_key in workflow_ops:
        if op_key not in canonical_ops:
            errors.append("workflow references an operator not present in top-level `operators`")
            break

    # DAG checks (on canonical node identity)
    nodes = set(workflow_ops)
    edges = _build_edges(workflow)  # type: ignore[arg-type]

    out_deg: Dict[Tuple, int] = {n: 0 for n in nodes}
    in_deg: Dict[Tuple, int] = {n: 0 for n in nodes}
    adj: Dict[Tuple, List[Tuple]] = {n: [] for n in nodes}
    for a, b in edges:
        if a not in nodes or b not in nodes:
            continue
        adj[a].append(b)
        out_deg[a] += 1
        in_deg[b] += 1

    # cycle detection via Kahn
    queue = [n for n in nodes if in_deg.get(n, 0) == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for nxt in adj.get(n, []):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    if visited != len(nodes):
        errors.append("workflow must be a DAG (cycle detected)")

    tails = [n for n in nodes if out_deg.get(n, 0) == 0]
    if require_single_tail and len(tails) != 1:
        errors.append(f"workflow must have exactly 1 tail operator, got {len(tails)}")

    return WorkflowValidationResult(ok=len(errors) == 0, errors=errors)


def validate_candidate_config(
    yaml_path: str | Path,
    *,
    constraints: Optional[WorkflowConstraints] = None,
) -> CandidateValidationResult:
    """Validate a candidate workflow with structural and optional dry-run checks."""
    constraints = constraints or WorkflowConstraints()
    stage_errors: Dict[str, List[str]] = {}

    structural = validate_workflow_yaml(
        yaml_path=yaml_path,
        main_expert_name=constraints.main_expert_name,
        expected_workflow_platform=constraints.workflow_platform,
        require_single_expert=constraints.require_single_expert,
        require_single_tail=constraints.require_single_tail,
    )
    if not structural.ok:
        stage_errors["structure"] = structural.errors

    if constraints.require_agentic_parse and not stage_errors.get("structure"):
        try:
            from app.core.model.agentic_config import AgenticConfig

            AgenticConfig.from_yaml(yaml_path)
        except Exception as e:
            stage_errors["agentic_parse"] = [f"AgenticConfig.from_yaml failed: {e}"]

    if constraints.require_agentic_service_dry_run and not stage_errors.get("structure"):
        try:
            from app.core.common.runtime_reset import reset_runtime_state
            from app.core.sdk.agentic_service import AgenticService

            reset_runtime_state()
            AgenticService.load(yaml_path)
        except Exception as e:
            stage_errors["agentic_load"] = [f"AgenticService.load dry-run failed: {e}"]
        finally:
            try:
                from app.core.common.runtime_reset import reset_runtime_state

                reset_runtime_state()
            except Exception:
                logger.debug("runtime reset skipped after candidate dry-run", exc_info=True)

    all_errors: List[str] = []
    for errs in stage_errors.values():
        all_errors.extend(errs)

    return CandidateValidationResult(
        ok=len(all_errors) == 0,
        errors=all_errors,
        stage_errors=stage_errors,
    )


def infer_single_expert_name(yaml_path: str | Path) -> str | None:
    """Infer the entry expert name when (and only when) exactly one expert exists."""
    path = Path(yaml_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    experts = raw.get("experts", [])
    if not isinstance(experts, list) or len(experts) != 1:
        return None

    expert0 = experts[0]
    if not isinstance(expert0, dict):
        return None

    profile = expert0.get("profile", {})
    if not isinstance(profile, dict):
        return None

    name = profile.get("name")
    return name if isinstance(name, str) and name.strip() else None
