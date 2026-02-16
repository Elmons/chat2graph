import json
import re
from typing import Iterable

from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.dataset_synthesis.task_subtypes import (
    get_query_subtype_meta,
    get_subtype_canonical_intents,
    resolve_query_subtype_id,
)


ENGINE_CAPABILITY_MATRIX: dict[str, dict[str, set[str]]] = {
    "neo4j": {
        "unsupported_keywords": set(),
    },
    "neptune": {
        "unsupported_keywords": {
            "CALL ",
            "shortestPath(",
        },
    },
    "unknown": {
        "unsupported_keywords": set(),
    },
}

READ_ONLY_FORBIDDEN_KEYWORDS = {
    " CREATE ",
    " MERGE ",
    " DELETE ",
    " DETACH DELETE ",
    " SET ",
    " REMOVE ",
    " DROP ",
}

KNOWN_REL_TYPES = {
    "TRANSFER",
    "DEPOSIT",
    "REPAY",
    "OWN",
    "INVEST",
    "INVESTED_IN",
    "WITHDRAW",
    "APPLY",
    "GUARANTEE",
    "SIGNIN",
}

TRANSACTION_REL_TYPES = {"TRANSFER", "DEPOSIT", "WITHDRAW", "REPAY"}

# QueryTaxonomy v2: only strongly-detectable intents participate in strict op checks.
ENFORCED_QUERY_INTENTS = {
    "query.path.shortest",
    "query.path.reachability",
    "query.path.constrained",
    "query.cycle.exists",
    "query.motif.triangle_count",
    "query.similarity.shared_neighbors",
    "query.ranking.topk",
    "query.aggregation.count",
    "query.aggregation.group_count",
    "query.topology.degree",
}

# Alias map from legacy/noisy labels to canonical SamplingIntentV2 labels.
INTENT_ALIAS_MAP = {
    # legacy canonical intents
    "query": "query.lookup",
    "path.shortest": "query.path.shortest",
    "path.exists": "query.path.reachability",
    "path.existence": "query.path.reachability",
    "path.multi_hop": "query.path.reachability",
    "hop.multi": "query.path.reachability",
    "ranking.topk": "query.ranking.topk",
    "ranking": "query.ranking.topk",
    "ranking.importance": "query.ranking.topk",
    "aggregation.count": "query.aggregation.count",
    "aggregation.group_count": "query.aggregation.group_count",
    "aggregation.group": "query.aggregation.group_count",
    "query.aggregation.group": "query.aggregation.group_count",
    "query.group": "query.aggregation.group_count",
    "query.aggregate.group": "query.aggregation.group_count",
    "query.grouping": "query.aggregation.group_count",
    "query.aggregation": "query.aggregation.count",
    "path.constrained": "query.path.constrained",
    "path.constraint": "query.path.constrained",
    "cycle.exists": "query.cycle.exists",
    "cycle.detect": "query.cycle.exists",
    "motif.triangle_count": "query.motif.triangle_count",
    "query.motif.triangle": "query.motif.triangle_count",
    "triangle.count": "query.motif.triangle_count",
    "similarity.shared_neighbors": "query.similarity.shared_neighbors",
    "shared_neighbors": "query.similarity.shared_neighbors",
    "shared.neighbors": "query.similarity.shared_neighbors",
    "degree": "query.topology.degree",
    "topology.degree": "query.topology.degree",
    # noisy labels from prior real runs
    "entity.attribute": "query.lookup",
    "attribute.get": "query.lookup",
    "attribute.query": "query.lookup",
    "attribute.filter": "query.filter.single",
    "filter.attribute": "query.filter.single",
    "filter.combined": "query.filter.combined",
    "pattern.match": "query.filter.combined",
    "relation.exist": "query.neighbor",
    "relationship.exists": "query.neighbor",
    "relationship.existence": "query.neighbor",
    "relationship.incoming": "query.neighbor",
    "relationship.count": "query.aggregation.count",
    "reasoning.single_step": "query.reasoning.single_step",
    "reasoning.multi_step": "query.reasoning.chain",
    "reasoning.chain": "query.reasoning.chain",
    "chain.reason": "query.reasoning.chain",
    # split/noisy tokens produced by llm in real runs
    "query.topology": "query.topology.degree",
    "query.degree": "query.topology.degree",
    "query.ranking": "query.ranking.topk",
    "query.topk": "query.ranking.topk",
    "query.count": "query.aggregation.count",
    "query.pattern": "query.filter.combined",
    "query.relationship.existence": "query.neighbor",
}


def looks_like_query(text: str | None) -> bool:
    if not text:
        return False
    head = text.strip()
    if not head:
        return False
    return bool(
        re.match(
            r"^(MATCH|OPTIONAL MATCH|WITH|CALL|UNWIND|RETURN|PROFILE|EXPLAIN)\b",
            head,
            flags=re.IGNORECASE,
        )
    )


def _normalized_query_text(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def is_generic_global_verifier(query: str | None) -> bool:
    normalized = _normalized_query_text(query or "")
    return normalized in {
        "match (n) return n",
        "match (n) return n limit 10",
        "match (n) return n limit 5",
    }


def _task_prefers_ranking(task: str) -> bool:
    low = (task or "").lower()
    return bool(
        re.search(
            r"\b(top|largest|highest|lowest|smallest|most|least|rank)\b",
            low,
        )
    )


def _query_needs_deterministic_order(task: str, query: str) -> bool:
    low_task = (task or "").lower()
    low_query = f" {(query or '').lower()} "
    has_limit = " limit " in low_query
    has_order = " order by " in low_query
    if not has_limit or has_order:
        return False
    if "count(" in low_query:
        return False
    if _task_prefers_ranking(low_task):
        return True
    if "list all" in low_task:
        return True
    return False


def _canonical_intent(intent: str) -> str:
    low = (intent or "").strip().lower()
    if not low:
        return ""
    return INTENT_ALIAS_MAP.get(low, low)


def normalize_intent_set(
    raw: object,
    task_subtype: str,
    *,
    subtype_id: str | None = None,
    level: str | None = None,
) -> list[str]:
    if isinstance(raw, list):
        intents = [_canonical_intent(str(item)) for item in raw]
    elif isinstance(raw, str):
        intents = [_canonical_intent(token) for token in raw.split(",")]
    else:
        intents = []

    intents = [intent for intent in intents if intent]
    if not intents:
        subtype_intents = get_subtype_canonical_intents(subtype_id or task_subtype, level=level) or []
        if subtype_intents:
            intents = subtype_intents

    if not intents:
        low_subtype = (task_subtype or "").lower()
        inferred: list[str] = ["query.lookup"]
        if "path" in low_subtype or "hop" in low_subtype:
            inferred.append("query.path.reachability")
        if "shortest" in low_subtype:
            inferred.append("query.path.shortest")
        if "constrained" in low_subtype:
            inferred.append("query.path.constrained")
        if "cycle" in low_subtype:
            inferred.append("query.cycle.exists")
        if "triangle" in low_subtype or "motif" in low_subtype:
            inferred.append("query.motif.triangle_count")
        if "similarity" in low_subtype or "shared" in low_subtype:
            inferred.append("query.similarity.shared_neighbors")
        if "rank" in low_subtype or "top" in low_subtype:
            inferred.append("query.ranking.topk")
        if "group" in low_subtype and "count" in low_subtype:
            inferred.append("query.aggregation.group_count")
        elif "count" in low_subtype or "aggregation" in low_subtype:
            inferred.append("query.aggregation.count")
        if "degree" in low_subtype or "topology" in low_subtype:
            inferred.append("query.topology.degree")
        intents = inferred

    # keep order stable while de-duplicating
    return list(dict.fromkeys(intents))


def _intent_features(intents: list[str]) -> dict[str, bool]:
    normalized = {str(intent).lower() for intent in intents}
    return {
        "has_path": bool(normalized.intersection({"query.path.reachability", "query.path.shortest"})),
        "has_shortest_path": "query.path.shortest" in normalized,
        "has_ranking": "query.ranking.topk" in normalized,
        "has_aggregation_count": "query.aggregation.count" in normalized,
        "has_aggregation_group_count": "query.aggregation.group_count" in normalized,
        "has_topology_degree": "query.topology.degree" in normalized,
    }


def normalize_row_protocol(row: Row) -> Row:
    if not row.generation_scope:
        row.generation_scope = "local_subgraph"

    row.task_subtype_id = (
        row.task_subtype_id
        or resolve_query_subtype_id(row.task_subtype, level=row.level)
    )

    row.intent_set = normalize_intent_set(
        row.intent_set,
        row.task_subtype,
        subtype_id=row.task_subtype_id,
        level=row.level,
    )

    if not row.global_verifier and looks_like_query(row.verifier):
        row.global_verifier = row.verifier

    # Keep backward compatibility:
    # - query-style verifier -> default global answer scope
    # - plain text verifier -> local scope (legacy rows without executable verifier)
    if not row.answer_scope:
        row.answer_scope = "global_graph" if row.global_verifier else "local_subgraph"

    if not row.expected_global:
        if row.answer_scope == "global_graph":
            if row.global_verifier and not looks_like_query(row.global_verifier):
                row.expected_global = row.verifier
        else:
            row.expected_global = row.verifier
    return row


def check_engine_query_compatibility(query: str, engine_hint: str | None) -> tuple[bool, str]:
    """Validate query compatibility against read-only and engine constraints."""
    normalized = (query or "").strip().lower()
    if not normalized:
        return False, "empty_global_verifier"

    # Basic read-only safety: dataset synthesis verifiers must be query-only.
    upper_query = f" {query.strip().upper()} "
    for keyword in READ_ONLY_FORBIDDEN_KEYWORDS:
        if keyword in upper_query:
            return False, f"read_only_violation_{keyword.strip().lower()}"

    # Basic syntactic sanity checks (cheap static checks).
    if upper_query.count("(") != upper_query.count(")"):
        return False, "syntax_unbalanced_parentheses"
    if "MATCH " not in upper_query and "CALL " not in upper_query and "RETURN " not in upper_query:
        return False, "syntax_missing_read_clause"

    # Ban unbounded variable-length traversals in synthesized verifiers.
    if re.search(r"\[[^\]]*\*(?!\.\.)[^\]]*\]", query, flags=re.IGNORECASE):
        return False, "path_missing_hop_bound"

    # Neo4j 5+ no longer supports pattern expression inside size(...).
    if re.search(r"size\s*\(\s*\(.*\)\s*--?\s*\(.*\)\s*\)", query, flags=re.IGNORECASE):
        return False, "syntax_legacy_size_pattern_expression"

    # Guard against shadowing: MATCH p=shortestPath((p:...)-[...]-(...)).
    shadow_match = re.search(
        r"match\s+([a-zA-Z_]\w*)\s*=\s*shortestpath\s*\(\s*\(\s*([a-zA-Z_]\w*)\s*:",
        query,
        flags=re.IGNORECASE,
    )
    if shadow_match and shadow_match.group(1).lower() == shadow_match.group(2).lower():
        return False, "syntax_variable_shadowing"

    engine = infer_engine_family(engine_hint)
    unsupported = ENGINE_CAPABILITY_MATRIX[engine]["unsupported_keywords"]
    for keyword in unsupported:
        if keyword in upper_query:
            return False, f"dialect_unsupported_{keyword.strip().lower()}"

    return True, "ok"


def infer_engine_family(engine_hint: str | None) -> str:
    hint = (engine_hint or "").lower()
    if "neo4j" in hint:
        return "neo4j"
    if "neptune" in hint:
        return "neptune"
    return "unknown"


def detect_query_operations(query: str) -> set[str]:
    normalized = (query or "").lower()
    ops: set[str] = set()

    if "shortestpath(" in normalized:
        ops.add("query.path.shortest")
    if re.search(r"\[[^\]]*\*\.\.\d+[^\]]*\]", normalized):
        ops.add("query.path.reachability")
    if (
        "relationships(p)" in normalized
        and "type(rel)" in normalized
        and re.search(r"\[[^\]]*\*\.\.\d+[^\]]*\]", normalized)
    ):
        ops.add("query.path.constrained")
    if (
        "has_cycle" in normalized
        or re.search(r"\((\w+)\)\s*-\[\*\.\.\d+\]\s*-\(\1\)", normalized)
    ):
        ops.add("query.cycle.exists")

    if "order by" in normalized and "limit" in normalized:
        ops.add("query.ranking.topk")
    if "shared_neighbors" in normalized or "common_neighbors" in normalized:
        ops.add("query.similarity.shared_neighbors")

    has_count = "count(" in normalized
    if has_count:
        if " group by " in normalized or re.search(r"\bwith\b.*count\(", normalized):
            ops.add("query.aggregation.group_count")
        else:
            ops.add("query.aggregation.count")
    if "triangle_count" in normalized or re.search(r"\(\w+\)\s*--\s*\(\w+\)\s*--\s*\(\w+\)\s*--\s*\(\w+\)", normalized):
        ops.add("query.motif.triangle_count")

    if " as degree" in normalized:
        ops.add("query.topology.degree")

    if "match " in normalized:
        ops.add("query.lookup")
    return ops


def _subtype_constraint_mismatch_reason(row: Row) -> str | None:
    query = row.global_verifier or row.verifier or ""
    if row.answer_scope == "local_subgraph" and not looks_like_query(query):
        return None
    meta = get_query_subtype_meta(row.task_subtype_id or row.task_subtype, level=row.level)
    if not meta:
        return None
    low_query = query.lower()
    tags = set(meta.constraint_tags or [])
    if "requires_order_limit" in tags:
        intents = set(
            normalize_intent_set(
                row.intent_set,
                row.task_subtype,
                subtype_id=row.task_subtype_id,
                level=row.level,
            )
        )
        should_enforce_order_limit = (
            "query.ranking.topk" in intents
            or _task_prefers_ranking(row.task)
        )
        if should_enforce_order_limit and not ("order by" in low_query and "limit" in low_query):
            return "missing_order_limit"
    if "requires_group_by" in tags:
        has_grouping = " group by " in low_query or re.search(r"\bwith\b.*count\(", low_query)
        if not has_grouping:
            # Cypher implicit grouping in RETURN clause:
            # RETURN key, count(...)
            has_grouping = bool(
                re.search(r"\breturn\b[^;]*,\s*count\s*\(", low_query)
                or re.search(r"\breturn\b[^;]*count\s*\([^;]*,\s*", low_query)
            )
        if not has_grouping:
            return "missing_group_by"
    return None


def _extract_task_relation_types(task: str) -> set[str]:
    upper = task.upper()
    rel_types = {token for token in KNOWN_REL_TYPES if re.search(rf"\b{token}\b", upper)}
    if "financial transaction" in task.lower():
        rel_types.update(TRANSACTION_REL_TYPES)
    return rel_types


def _literal_bound_to_label(query: str, label: str, literal: str) -> bool:
    lit = re.escape(literal.lower())
    # Prefer explicit property-map binding on labeled node.
    if re.search(rf":{label}\s*\{{[^}}]*{lit}[^}}]*\}}", query):
        return True
    # Fallback: label + literal + common identifying fields.
    field_tokens = {
        "account": ["nickname", "id"],
        "company": ["companyname", "id"],
        "person": ["personname", "id"],
        "medium": ["id", "mediumtype"],
        "loan": ["id", "loanname"],
    }
    fields = field_tokens.get(label, [])
    return (
        f":{label}" in query
        and literal.lower() in query
        and any(field in query for field in fields)
    )


def _task_query_semantic_mismatch_reason(row: Row) -> str | None:
    original_task = (row.task or "").strip()
    task = original_task.lower()
    query = f" {(row.global_verifier or row.verifier or '').lower()} "
    if not task or not query.strip():
        return None

    exact_hops_match = re.search(
        r"\bexactly\s+(\d+|one|two|three|four|five|six)\s*hops?\b",
        task,
    )
    if exact_hops_match:
        token = exact_hops_match.group(1)
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        hops = int(token) if token.isdigit() else word_to_num.get(token, 0)
        if hops <= 0:
            return None
        has_exact_hops = bool(
            re.search(rf"\[\s*\*\s*{hops}\s*\.\.\s*{hops}\s*\]", query)
            or re.search(rf"\blength\s*\(\s*\w+\s*\)\s*=\s*{hops}\b", query)
        )
        if not has_exact_hops:
            return "path_exact_hops_not_enforced"
    length_exact_match = re.search(
        r"\blength\s+exactly\s+(\d+|one|two|three|four|five|six)\b",
        task,
    )
    if not length_exact_match:
        length_exact_match = re.search(
            r"\bcycle\s+of\s+length\s+(\d+|one|two|three|four|five|six)\b",
            task,
        )
    if length_exact_match:
        token = length_exact_match.group(1)
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        hops = int(token) if token.isdigit() else word_to_num.get(token, 0)
        if hops > 0:
            has_exact_hops = bool(
                re.search(rf"\[\s*\*\s*{hops}\s*\.\.\s*{hops}\s*\]", query)
                or re.search(rf"\blength\s*\(\s*\w+\s*\)\s*=\s*{hops}\b", query)
            )
            if not has_exact_hops:
                return "path_exact_hops_not_enforced"
    hyphen_hops_match = re.search(
        r"\b(\d+|one|two|three|four|five|six)\s*-\s*hops?\b",
        task,
    )
    if hyphen_hops_match:
        token = hyphen_hops_match.group(1)
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        hops = int(token) if token.isdigit() else word_to_num.get(token, 0)
        if hops > 0:
            has_exact_hops = bool(
                re.search(rf"\[\s*\*\s*{hops}\s*\.\.\s*{hops}\s*\]", query)
                or re.search(rf"\blength\s*\(\s*\w+\s*\)\s*=\s*{hops}\b", query)
            )
            if not has_exact_hops:
                return "path_exact_hops_not_enforced"

    if "first hop is" in task and "second hop is" in task:
        # Expect explicit step-wise constraints to be encoded.
        has_step_constraints = bool(
            "relationships(p)[0]" in query
            or re.search(r"-\[\s*\w+\s*:\s*[a-z_]+\s*\]->\([^)]*\)-\[\s*\w+\s*\]->", query)
        )
        if not has_step_constraints:
            return "path_step_constraints_not_encoded"
        if "second hop is not" in task and not re.search(r"\b(type\s*\(\s*\w+\s*\)\s*(?:<>|!=)\s*'?[a-z_]+|not\s+type\s*\()", query):
            return "path_step_constraints_not_encoded"

    followed_match = re.search(
        r"\b([A-Za-z_]+)\s+followed by\s+(?:a|an)?\s*([A-Za-z_]+)\b",
        original_task,
    )
    if "path" in task and followed_match:
        first_rel = followed_match.group(1).upper()
        second_rel = followed_match.group(2).upper()
        has_chain_pattern = bool(
            re.search(r"-\[\s*\w+\s*:\s*[a-z_]+\s*\]->\([^)]*\)-\[\s*\w+\s*:\s*[a-z_]+\s*\]->", query)
            or "relationships(p)[0]" in query
        )
        if not has_chain_pattern:
            return "path_step_constraints_not_encoded"
        if first_rel.lower() not in query or second_rel.lower() not in query:
            return "path_step_constraints_not_encoded"

    ordered_match = re.search(
        r"\b([A-Za-z_]+)\s+and\s+([A-Za-z_]+)\s+relationships?\s+in\s+that\s+order\b",
        original_task,
    )
    if "path" in task and ordered_match:
        first_rel = ordered_match.group(1).upper()
        second_rel = ordered_match.group(2).upper()
        has_chain_pattern = bool(
            re.search(r"-\[\s*\w+\s*:\s*[a-z_]+\s*\]->\([^)]*\)-\[\s*\w+\s*:\s*[a-z_]+\s*\]->", query)
            or "relationships(p)[0]" in query
        )
        if not has_chain_pattern:
            return "path_step_constraints_not_encoded"
        if first_rel.lower() not in query or second_rel.lower() not in query:
            return "path_step_constraints_not_encoded"

    if "shortest path" in task and (
        "path length" in task
        or "in hops" in task
        or "relationship count" in task
        or "number of relationships" in task
    ):
        return_clause = ""
        return_match = re.search(r"\breturn\b(.+)", query)
        if return_match:
            return_clause = return_match.group(1)
        if "length(" not in return_clause and "size(relationships(" not in return_clause:
            return "shortest_path_length_not_returned"

    if "email domain" in task:
        if "email" not in query:
            return "email_domain_filter_not_encoded"
        domain_match = re.search(r"email\s+domain\s+['\"]?([a-z0-9.-]+\.[a-z]{2,})['\"]?", task)
        if domain_match:
            domain = domain_match.group(1).lower()
            has_domain_logic = (
                f"@{domain}" in query
                or domain in query and ("split(" in query or "ends with" in query or "contains" in query)
            )
            if not has_domain_logic:
                return "email_domain_filter_not_encoded"

    if " email " in f" {task} " and re.search(r"email\s+['\"]", task):
        if "email" not in query:
            return "task_query_semantic_mismatch"

    if "accountlevel" in task and "accountlevel" not in query:
        return "task_query_semantic_mismatch"

    for m in re.finditer(r"company[^'\"]*['\"]([^'\"]+)['\"]", original_task, flags=re.IGNORECASE):
        literal = m.group(1)
        if literal and not _literal_bound_to_label(query, "company", literal):
            return "task_query_semantic_mismatch"

    for m in re.finditer(r"medium[^'\"]*id[^'\"]*['\"]([^'\"]+)['\"]", original_task, flags=re.IGNORECASE):
        literal = m.group(1)
        if literal and not _literal_bound_to_label(query, "medium", literal):
            return "task_query_semantic_mismatch"

    for m in re.finditer(r"medium[^'\"]*mediumtype[^'\"]*['\"]([^'\"]+)['\"]", original_task, flags=re.IGNORECASE):
        literal = m.group(1)
        if literal and not (
            ":medium" in query and literal.lower() in query and "mediumtype" in query
        ):
            return "task_query_semantic_mismatch"

    for m in re.finditer(
        r"\b([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*)'s\s+account\b",
        original_task,
    ):
        literal = m.group(1)
        if literal and literal.lower() not in query:
            return "task_query_semantic_mismatch"

    has_signin_phrase = ("signin" in task) or ("sign in" in task) or ("signs in" in task)
    if "login type" in task and has_signin_phrase and "freqlogintype" not in query:
        return "task_query_semantic_mismatch"

    if "freqlogintype" in task and re.search(r"\bwhat is (its|the)\s+freqlogintype\b", task):
        return_clause = ""
        return_match = re.search(r"\breturn\b(.+)", query)
        if return_match:
            return_clause = return_match.group(1)
        if "freqlogintype" not in return_clause:
            return "task_query_semantic_mismatch"

    is_boolean_task = (
        (task.startswith("if ") and bool(re.search(r"\b(can|should|does)\b", task)))
        or task.startswith("has ")
        or task.startswith("does ")
    )
    if is_boolean_task:
        return_clause = ""
        return_match = re.search(r"\breturn\b(.+)", query)
        if return_match:
            return_clause = return_match.group(1)
        boolean_like = bool(
            "case when" in query
            or re.search(r"count\s*\([^)]*\)\s*(?:>|<|=)", return_clause)
            or re.search(r"\b(true|false)\b", return_clause)
        )
        if not boolean_like:
            return "boolean_task_not_boolean_query"
        if "blocked" in task and "transfer" in task and "isblocked" not in query:
            return "task_query_semantic_mismatch"
        quoted_literals = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", original_task)]
        quoted_literals = [literal for literal in quoted_literals if literal]
        if len(quoted_literals) >= 2:
            if any(literal.lower() not in query for literal in quoted_literals[:3]):
                return "task_query_semantic_mismatch"
        guarantee_pairs = re.findall(
            r"\b([A-Z][a-z]+)\s+guarantees\s+([A-Z][a-z]+)\b",
            original_task,
        )
        if guarantee_pairs:
            names: list[str] = []
            for src, dst in guarantee_pairs:
                names.append(src)
                names.append(dst)
            dedup_names = list(dict.fromkeys(names))
            if len(dedup_names) >= 2 and any(name.lower() not in query for name in dedup_names[:3]):
                return "task_query_semantic_mismatch"

    if (
        ("total amount" in task or "sum" in task or "total deposited" in task)
        and ("deposit" in task or "withdraw" in task or "transfer" in task or "repay" in task)
    ):
        sum_match = re.search(r"sum\s*\(\s*(\w+)\.amount\s*\)", query)
        if sum_match:
            alias = sum_match.group(1)
            if not re.search(rf"-\[\s*{re.escape(alias)}\s*:", query):
                return "amount_aggregation_not_on_relationship"

    if (
        ("shared neighbor" in task or "shared neighboring" in task or "share a common" in task or "share common" in task)
        and ("same source" in task or "withdraw" in task or "transfer" in task)
    ):
        if ("withdraw" in task and "withdraw" not in query) or ("transfer" in task and "transfer" not in query):
            return "shared_neighbor_scope_not_encoded"

    if "using only" in task:
        rel_types = _extract_task_relation_types(row.task or "")
        if rel_types:
            has_rel_constraints = (
                "relationships(p)" in query
                or bool(re.search(r"\[:[a-z_|]+\]", query))
            )
            has_rel_tokens = any(rel.lower() in query for rel in rel_types)
            if not has_rel_constraints and not has_rel_tokens:
                return "path_relation_scope_not_encoded"

    return None


def intent_verifier_alignment_ok(row: Row) -> bool:
    intents = set(
        normalize_intent_set(
            row.intent_set,
            row.task_subtype,
            subtype_id=row.task_subtype_id,
            level=row.level,
        )
    )
    if not intents:
        return True
    query = row.global_verifier or row.verifier or ""
    if row.answer_scope == "local_subgraph" and not looks_like_query(query):
        # Legacy local-answer rows may carry free-form verifier text.
        return True
    ops = detect_query_operations(query)
    requested_ops = {intent for intent in intents if intent in ENFORCED_QUERY_INTENTS}
    if not requested_ops:
        return ("query.lookup" in ops) or (row.answer_scope == "local_subgraph")
    return requested_ops.issubset(ops)


def qa_gate_reason(row: Row, engine_hint: str | None = None) -> str | None:
    row = normalize_row_protocol(row)
    global_verifier = row.global_verifier or ""

    if row.generation_scope != "local_subgraph":
        return "invalid_generation_scope"

    if row.answer_scope == "global_graph":
        if not global_verifier:
            return "missing_global_verifier"
        if not row.expected_global:
            return "missing_expected_global"

    if global_verifier and not looks_like_query(global_verifier):
        return "global_verifier_not_query"
    if global_verifier and is_generic_global_verifier(global_verifier):
        return "generic_global_verifier"

    low_task = (row.task or "").lower()

    def _has_boundary_signal(task_text: str, query_text: str) -> bool:
        # Natural-language boundary terms that usually turn open enumeration
        # into a bounded/computable question.
        if re.search(
            r"\b("
            r"top|within|limit|at most|at least|exactly|latest|first|last|"
            r"largest|highest|lowest|smallest|most|least|"
            r"\d+|between|before|after|during|in\s+\d{4}"
            r")\b",
            task_text,
        ):
            return True
        query_low = (query_text or "").lower()
        # Execution-level boundaries: LIMIT and COUNT are strong signals.
        if " limit " in f" {query_low} ":
            return True
        if "count(" in query_low:
            return True
        return False

    has_boundary_signal = _has_boundary_signal(low_task, global_verifier)
    if re.search(r"\b(all|list|every)\b", low_task):
        if not has_boundary_signal:
            return "implicit_full_enumeration"
    # Catch open-ended WH-style enumeration requests that imply full listing.
    # Example: "Which accounts did X transfer money to?"
    if re.search(r"\b(which|who)\b", low_task) and re.search(
        r"\b(did|are|were|has|have)\b", low_task
    ):
        likely_open_set = bool(
            re.search(
                r"\b(accounts?|users?|people|transactions?|relationships?|neighbors?)\b",
                low_task,
            )
        ) and bool(re.search(r"\b(to|from|with|by|through)\b", low_task))
        if likely_open_set and not has_boundary_signal:
            return "implicit_full_enumeration"

    if global_verifier:
        if _query_needs_deterministic_order(row.task, global_verifier):
            return "missing_order_for_bounded_list"
        compatible, reason = check_engine_query_compatibility(
            query=global_verifier,
            engine_hint=engine_hint,
        )
        if not compatible:
            return reason

    semantic_mismatch_reason = _task_query_semantic_mismatch_reason(row)
    if semantic_mismatch_reason:
        return semantic_mismatch_reason

    subtype_constraint_reason = _subtype_constraint_mismatch_reason(row)
    if subtype_constraint_reason:
        return subtype_constraint_reason

    if not intent_verifier_alignment_ok(row):
        return "intent_verifier_mismatch"
    return None


def qa_gate_filter(
    rows: Iterable[Row],
    engine_hint: str | None = None,
) -> tuple[list[Row], dict[str, int]]:
    accepted: list[Row] = []
    reject_stats: dict[str, int] = {}
    for row in rows:
        reason = qa_gate_reason(row=row, engine_hint=engine_hint)
        if reason is None:
            accepted.append(row)
            continue
        reject_stats[reason] = reject_stats.get(reason, 0) + 1
    return accepted, reject_stats


def load_workflow_train_dataset(
    task_desc: str, path: str, ratio: float = 1.0
) -> WorkflowTrainDataset:
    """Load a workflow training dataset from a JSON file."""
    assert 0 <= ratio <= 1.0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        dataset: list[Row] = []

        for qa in data:
            dataset.append(Row.model_validate(qa))
        return WorkflowTrainDataset(
            name="test", task_desc=task_desc, data=dataset[: int(len(dataset) * ratio)]
        )
