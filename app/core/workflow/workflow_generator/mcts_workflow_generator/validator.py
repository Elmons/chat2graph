from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class WorkflowValidationResult:
    ok: bool
    errors: List[str]


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _canonical_operator(op: Dict[str, Any]) -> Tuple[str, str, Tuple[str, ...]]:
    instruction = _as_str(op.get("instruction"))
    output_schema = _as_str(op.get("output_schema"))
    actions_raw = op.get("actions", [])
    actions: Tuple[str, ...]
    if isinstance(actions_raw, list) and all(isinstance(x, str) for x in actions_raw):
        actions = tuple(actions_raw)
    else:
        actions = tuple()
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

    experts = raw.get("experts", [])
    if not isinstance(experts, list):
        errors.append("`experts` must be a list")
        return WorkflowValidationResult(ok=False, errors=errors)
    if len(experts) != 1:
        errors.append(f"single-expert mode requires exactly 1 expert, got {len(experts)}")
        return WorkflowValidationResult(ok=False, errors=errors)
    expert0 = experts[0]
    if not isinstance(expert0, dict):
        errors.append("expert must be a mapping")
        return WorkflowValidationResult(ok=False, errors=errors)
    profile = expert0.get("profile", {})
    if not isinstance(profile, dict):
        errors.append("expert.profile must be a mapping")
        return WorkflowValidationResult(ok=False, errors=errors)
    if profile.get("name") != main_expert_name:
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
    for idx, op in enumerate(operators):
        missing = [k for k in operator_keys if k not in op]
        if missing:
            errors.append(f"operators[{idx}] missing fields: {missing}")
            continue
        key = _canonical_operator(op)
        if key in canonical_ops:
            errors.append("duplicate operator definition detected (same instruction/output_schema/actions)")
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
    if len(tails) != 1:
        errors.append(f"workflow must have exactly 1 tail operator, got {len(tails)}")

    return WorkflowValidationResult(ok=len(errors) == 0, errors=errors)

