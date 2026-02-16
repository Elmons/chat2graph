import json
import math
import re
from typing import Dict, List

from app.core.workflow.dataset_synthesis.model import (
    GENERATOR_STRATEGY,
    TASK_LEVEL,
    LevelInfo,
    Row,
    SubTaskType,
)

# SamplingIntentV2 (query-only): canonical core intents used by sampling/QA/evaluation.
SAMPLING_CORE_INTENTS_V2: tuple[str, ...] = (
    "query.lookup",
    "query.neighbor",
    "query.filter.single",
    "query.filter.combined",
    "query.reasoning.single_step",
    "query.reasoning.chain",
    "query.path.reachability",
    "query.path.shortest",
    "query.aggregation.count",
    "query.aggregation.group_count",
    "query.ranking.topk",
    "query.topology.degree",
    "query.path.constrained",
    "query.cycle.exists",
    "query.motif.triangle_count",
    "query.similarity.shared_neighbors",
)

# Constraint tags are not sampled directly; they are used by QA gates and template routing.
SAMPLING_CONSTRAINT_TAGS_V2: tuple[str, ...] = (
    "requires_hop_bound",
    "requires_order_limit",
    "requires_group_by",
    "forbid_unbounded_enumeration",
    "global_answer_required",
)


def _q(
    *,
    level: TASK_LEVEL,
    subtype_id: str,
    name: str,
    desc: str,
    examples: list[str],
    canonical_intents: list[str],
    constraint_tags: list[str],
    required_query_features: list[str],
    forbidden_patterns: list[str],
    target_ratio_min: float,
    target_ratio_max: float,
) -> SubTaskType:
    return SubTaskType(
        level=level,
        subtype_id=subtype_id,
        name=name,
        desc=desc,
        examples=examples,
        canonical_intents=canonical_intents,
        constraint_tags=constraint_tags,
        required_query_features=required_query_features,
        forbidden_patterns=forbidden_patterns,
        target_ratio_min=target_ratio_min,
        target_ratio_max=target_ratio_max,
    )


class QueryTaskSubtypes:
    """QueryTaxonomy v2 for dataset synthesis (query-only).

    The taxonomy exposes stable `subtype_id` and keeps human-readable display names
    for prompts. Sampling, counting and QA alignment should use `subtype_id`.
    """

    l1_tasks = [
        _q(
            level="L1",
            subtype_id="q_l1_attr_lookup",
            name="Entity Attribute and Label Query",
            desc=(
                "Retrieve direct attributes/labels for one entity (0-hop lookup) from "
                "the global graph."
            ),
            examples=["What is the department of employee Zhang San?"],
            canonical_intents=["query.lookup"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["match", "return"],
            forbidden_patterns=["order by", "shortestpath(", " group by "],
            target_ratio_min=0.06,
            target_ratio_max=0.22,
        ),
        _q(
            level="L1",
            subtype_id="q_l1_neighbor_query",
            name="Direct Relationship and Neighbor Query",
            desc=(
                "Check direct relationship existence or list bounded one-hop neighbors "
                "for a given entity."
            ),
            examples=[
                "Is there a cooperation relationship between Company A and Company B?"
            ],
            canonical_intents=["query.neighbor"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["match", "return"],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.06,
            target_ratio_max=0.20,
        ),
        _q(
            level="L1",
            subtype_id="q_l1_filter_single",
            name="Simple Attribute Filtering",
            desc=(
                "Filter entities by one explicit predicate on attributes or labels."
            ),
            examples=["Which products have inventory > 100?"],
            canonical_intents=["query.filter.single"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["where"],
            forbidden_patterns=[" group by ", "shortestpath("],
            target_ratio_min=0.06,
            target_ratio_max=0.20,
        ),
        _q(
            level="L1",
            subtype_id="q_l1_reasoning_single_step",
            name="Single-Step Intuitive Reasoning Query",
            desc=(
                "Infer one-step conclusions from direct relationships (single-hop reasoning)."
            ),
            examples=["Who is the direct supervisor of employee Chen Qi?"],
            canonical_intents=["query.reasoning.single_step"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["match", "return"],
            forbidden_patterns=["shortestpath(", " group by "],
            target_ratio_min=0.05,
            target_ratio_max=0.18,
        ),
    ]

    l2_tasks = [
        _q(
            level="L2",
            subtype_id="q_l2_path_reachability",
            name="Multi-Hop Relationship and Reachability Query",
            desc=(
                "Reason over bounded multi-hop paths (>=2 hops) to determine reachability "
                "or retrieve constrained path evidence."
            ),
            examples=["Can Supplier X reach Customer Y within 3 hops?"],
            canonical_intents=["query.path.reachability", "query.reasoning.chain"],
            constraint_tags=[
                "requires_hop_bound",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["*..n", "match"],
            forbidden_patterns=["[*]-", "shortestpath((a)-[*]-(b))"],
            target_ratio_min=0.08,
            target_ratio_max=0.22,
        ),
        _q(
            level="L2",
            subtype_id="q_l2_filter_combined",
            name="Pattern-Based and Combined Filtering Query",
            desc=(
                "Combine multiple predicates/pattern constraints across entities."
            ),
            examples=[
                "Which projects are managed by employees in Beijing and tenure > 2 years?"
            ],
            canonical_intents=["query.filter.combined"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["where", "and/or"],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.07,
            target_ratio_max=0.20,
        ),
        _q(
            level="L2",
            subtype_id="q_l2_reasoning_chain",
            name="Multi-Step Chain Reasoning Query",
            desc=(
                "Follow bounded chain relationships across multiple entities."
            ),
            examples=["Who is the supervisor of employee M's supervisor?"],
            canonical_intents=["query.reasoning.chain"],
            constraint_tags=[
                "requires_hop_bound",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["match", "where"],
            forbidden_patterns=["[*]-"],
            target_ratio_min=0.06,
            target_ratio_max=0.18,
        ),
        _q(
            level="L2",
            subtype_id="q_l2_aggregation_count",
            name="Scoped Aggregation Count Query",
            desc=(
                "Compute bounded count aggregations for explicitly scoped entity sets."
            ),
            examples=["How many transfers are initiated by Alice?"],
            canonical_intents=["query.aggregation.count"],
            constraint_tags=["forbid_unbounded_enumeration", "global_answer_required"],
            required_query_features=["count("],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.05,
            target_ratio_max=0.16,
        ),
        _q(
            level="L2",
            subtype_id="q_l2_aggregation_group_count",
            name="Scoped Group Aggregation Query",
            desc=(
                "Group entities by explicit keys and aggregate per group."
            ),
            examples=["Count transfer records by relation type for Alice's neighborhood."],
            canonical_intents=["query.aggregation.group_count"],
            constraint_tags=[
                "requires_group_by",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["count(", "group by/with"],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.05,
            target_ratio_max=0.16,
        ),
        _q(
            level="L2",
            subtype_id="q_l2_ranking_topk",
            name="Ranking Top-K Query",
            desc=(
                "Rank candidates with explicit order-by metric and limit K."
            ),
            examples=["Top 3 transfer targets of Alice by transfer amount."],
            canonical_intents=["query.ranking.topk"],
            constraint_tags=[
                "requires_order_limit",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["order by", "limit"],
            forbidden_patterns=[],
            target_ratio_min=0.06,
            target_ratio_max=0.18,
        ),
    ]

    l3_tasks = [
        _q(
            level="L3",
            subtype_id="q_l3_path_shortest",
            name="Path Analysis (Shortest Path)",
            desc=(
                "Use bounded shortest path style queries with explicit endpoints."
            ),
            examples=["What is the shortest path from Store A to Store B within 4 hops?"],
            canonical_intents=["query.path.shortest"],
            constraint_tags=[
                "requires_hop_bound",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["shortestpath(", "*..n"],
            forbidden_patterns=["shortestpath((a)-[*]-(b))"],
            target_ratio_min=0.10,
            target_ratio_max=0.25,
        ),
        _q(
            level="L3",
            subtype_id="q_l3_topology_degree",
            name="Local Topological Degree Query",
            desc=(
                "Compute local topological degree-like metrics in a bounded subspace."
            ),
            examples=["Which node has the highest degree around account Alice?"],
            canonical_intents=["query.topology.degree", "query.ranking.topk"],
            constraint_tags=[
                "requires_order_limit",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["count(", "order by", "limit"],
            forbidden_patterns=["size(("],
            target_ratio_min=0.10,
            target_ratio_max=0.22,
        ),
        _q(
            level="L3",
            subtype_id="q_l3_path_constrained",
            name="Constrained Path Query",
            desc=(
                "Evaluate bounded paths constrained by relation types or path predicates."
            ),
            examples=[
                "Is there a path from Account A to Loan B within 3 hops using only TRANSFER/REPAY?"
            ],
            canonical_intents=["query.path.constrained", "query.path.reachability"],
            constraint_tags=[
                "requires_hop_bound",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["*..n", "relationships(p)", "type(rel)"],
            forbidden_patterns=["[*]-", "shortestpath((a)-[*]-(b))"],
            target_ratio_min=0.0,
            target_ratio_max=0.16,
        ),
        _q(
            level="L3",
            subtype_id="q_l3_cycle_exists",
            name="Cycle Existence Query",
            desc=(
                "Determine whether bounded cycles exist around a target entity."
            ),
            examples=["Does account A participate in any cycle within 4 hops?"],
            canonical_intents=["query.cycle.exists", "query.path.reachability"],
            constraint_tags=[
                "requires_hop_bound",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["*..n", "count(p) > 0"],
            forbidden_patterns=["[*]-", "shortestpath((a)-[*]-(b))"],
            target_ratio_min=0.0,
            target_ratio_max=0.14,
        ),
        _q(
            level="L3",
            subtype_id="q_l3_motif_triangle_count",
            name="Motif Triangle Counting Query",
            desc=(
                "Count closed triads/triangles in the neighborhood of a target entity."
            ),
            examples=["How many triangles exist around account A?"],
            canonical_intents=["query.motif.triangle_count", "query.aggregation.count"],
            constraint_tags=[
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["count(", "(a)--(b)--(c)--(a)"],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.0,
            target_ratio_max=0.14,
        ),
        _q(
            level="L3",
            subtype_id="q_l3_similarity_shared_neighbors",
            name="Structural Similarity (Shared Neighbors) Query",
            desc=(
                "Measure structural similarity by counting shared neighbors and ranking."
            ),
            examples=["Top 3 accounts most similar to account A by shared neighbors."],
            canonical_intents=["query.similarity.shared_neighbors", "query.ranking.topk"],
            constraint_tags=[
                "requires_order_limit",
                "forbid_unbounded_enumeration",
                "global_answer_required",
            ],
            required_query_features=["count(", "order by", "limit"],
            forbidden_patterns=["shortestpath("],
            target_ratio_min=0.0,
            target_ratio_max=0.16,
        ),
    ]

    L1 = LevelInfo(
        level="L1",
        name="Simple Query Tasks",
        desc="0-1 hop lookups/filters/reasoning without aggregation or graph algorithms.",
        subtasks=l1_tasks,
    )
    L2 = LevelInfo(
        level="L2",
        name="Complex Query Tasks (No Algorithms)",
        desc="Bounded multi-hop, combined predicates, aggregation and ranking queries.",
        subtasks=l2_tasks,
    )
    L3 = LevelInfo(
        level="L3",
        name="Query Tasks Requiring Strong Structural Constraints",
        desc="Shortest-path and topology-style tasks with strict bounded constraints.",
        subtasks=l3_tasks,
    )

    REGISTER_LIST = [L1, L2, L3]


SUBTYPES_MAP = {
    "query": [*QueryTaskSubtypes.REGISTER_LIST],
}

_QUERY_SUBTYPE_META_BY_ID: dict[str, SubTaskType] = {
    subtype.subtype_id: subtype
    for level in QueryTaskSubtypes.REGISTER_LIST
    for subtype in level.subtasks
}


def _normalize_subtype_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


_DEFAULT_SUBTYPE_BY_LEVEL: dict[str, str] = {
    "L1": "q_l1_attr_lookup",
    "L2": "q_l2_filter_combined",
    "L3": "q_l3_path_shortest",
}

_SUBTYPE_ALIAS_MAP: dict[str, str] = {
    # Current ids and aliases.
    "q_l1_attr_lookup": "q_l1_attr_lookup",
    "q_l1_neighbor_query": "q_l1_neighbor_query",
    "q_l1_filter_single": "q_l1_filter_single",
    "q_l1_reasoning_single_step": "q_l1_reasoning_single_step",
    "q_l2_path_reachability": "q_l2_path_reachability",
    "q_l2_filter_combined": "q_l2_filter_combined",
    "q_l2_reasoning_chain": "q_l2_reasoning_chain",
    "q_l2_aggregation_count": "q_l2_aggregation_count",
    "q_l2_aggregation_group_count": "q_l2_aggregation_group_count",
    "q_l2_ranking_topk": "q_l2_ranking_topk",
    "q_l3_path_shortest": "q_l3_path_shortest",
    "q_l3_topology_degree": "q_l3_topology_degree",
    "q_l3_path_constrained": "q_l3_path_constrained",
    "q_l3_cycle_exists": "q_l3_cycle_exists",
    "q_l3_motif_triangle_count": "q_l3_motif_triangle_count",
    "q_l3_similarity_shared_neighbors": "q_l3_similarity_shared_neighbors",
    # Legacy display names.
    "entity_attribute_and_label_query": "q_l1_attr_lookup",
    "direct_relationship_and_neighbor_query": "q_l1_neighbor_query",
    "simple_attribute_filtering": "q_l1_filter_single",
    "single_step_intuitive_reasoning_query": "q_l1_reasoning_single_step",
    "multi_hop_relationship_and_path_query": "q_l2_path_reachability",
    "pattern_based_and_combined_filtering_query": "q_l2_filter_combined",
    "multi_step_chain_reasoning_query": "q_l2_reasoning_chain",
    "path_analysis": "q_l3_path_shortest",
    "local_topological_index_calculation": "q_l3_topology_degree",
    "constrained_path_query": "q_l3_path_constrained",
    "constrained_path": "q_l3_path_constrained",
    "cycle_existence_query": "q_l3_cycle_exists",
    "cycle_detection_query": "q_l3_cycle_exists",
    "cycle_detection": "q_l3_cycle_exists",
    "motif_triangle_counting_query": "q_l3_motif_triangle_count",
    "triangle_motif_count": "q_l3_motif_triangle_count",
    "structural_similarity_shared_neighbors_query": "q_l3_similarity_shared_neighbors",
    "shared_neighbor_similarity": "q_l3_similarity_shared_neighbors",
    "local_node_importance_ranking": "q_l2_ranking_topk",
    # Short aliases from tests and prior runs.
    "attribute_filtering": "q_l1_filter_single",
    "multi_hop": "q_l2_path_reachability",
    "ranking": "q_l2_ranking_topk",
    "aggregation": "q_l2_aggregation_count",
    "topology_degree": "q_l3_topology_degree",
}


def resolve_query_subtype_id(
    subtype: str | None,
    *,
    level: TASK_LEVEL | None = None,
) -> str | None:
    if not subtype:
        return _DEFAULT_SUBTYPE_BY_LEVEL.get(level or "") if level else None
    low = (subtype or "").strip().lower()
    if low in _QUERY_SUBTYPE_META_BY_ID:
        return low
    normalized = _normalize_subtype_key(subtype)
    resolved = _SUBTYPE_ALIAS_MAP.get(normalized)
    if resolved:
        return resolved
    return None


def get_query_subtype_meta(subtype: str | None, *, level: TASK_LEVEL | None = None) -> SubTaskType | None:
    resolved = resolve_query_subtype_id(subtype, level=level)
    if not resolved:
        return None
    return _QUERY_SUBTYPE_META_BY_ID.get(resolved)


def get_subtype_canonical_intents(subtype: str | None, *, level: TASK_LEVEL | None = None) -> list[str]:
    meta = get_query_subtype_meta(subtype, level=level)
    if not meta:
        return []
    return list(meta.canonical_intents or [])


class GraphTaskTypesInfo:
    """Structured task taxonomy and subtype counters for synthesis."""

    def __init__(
        self,
        strategy: GENERATOR_STRATEGY = "query",
    ):
        if strategy is None:
            strategy = "query"
        self.strategy = strategy
        self.tasks_info = SUBTYPES_MAP[str(strategy)]
        self.count_info: Dict[str, Dict[str, int]] = {}
        self.subtype_meta_by_id: dict[str, SubTaskType] = {
            subtype.subtype_id: subtype
            for level_info in self.tasks_info
            for subtype in level_info.subtasks
        }
        self.level_by_subtype_id: dict[str, TASK_LEVEL] = {
            subtype.subtype_id: subtype.level
            for subtype in self.subtype_meta_by_id.values()
        }

        for level_info in self.tasks_info:
            self.count_info[level_info.level] = {}
            for subtask in level_info.subtasks:
                self.count_info[level_info.level][subtask.subtype_id] = 0

    def resolve_subtype_id(self, subtype: str | None, *, level: TASK_LEVEL | None = None) -> str | None:
        return resolve_query_subtype_id(subtype=subtype, level=level)

    def get_subtype_meta(self, subtype: str | None, *, level: TASK_LEVEL | None = None) -> SubTaskType | None:
        subtype_id = self.resolve_subtype_id(subtype, level=level)
        if not subtype_id:
            return None
        return self.subtype_meta_by_id.get(subtype_id)

    def update(self, rows: List[Row]):
        """Update counters and normalize rows with stable subtype ids."""
        for row in rows:
            subtype_id = row.task_subtype_id or self.resolve_subtype_id(row.task_subtype, level=row.level)
            if not subtype_id:
                self.add(level=row.level, subtask=None)
                continue
            row.task_subtype_id = subtype_id
            self.add(level=row.level, subtask=subtype_id)

    def add(self, level: TASK_LEVEL, subtask: str | None):
        """Increment counter for the specified level and subtype id."""
        subtype_id = self.resolve_subtype_id(subtask, level=level) if subtask else None
        if level in self.count_info and subtype_id and subtype_id in self.count_info[level]:
            self.count_info[level][subtype_id] += 1
            return
        if level in self.count_info:
            if "unknown" not in self.count_info[level]:
                self.count_info[level]["unknown"] = 0
            self.count_info[level]["unknown"] += 1
            return
        print(f"[GraphTaskInfos]unknown level {level}")

    def total_rows(self) -> int:
        total = 0
        for level_counts in self.count_info.values():
            total += sum(level_counts.values())
        return total

    def subtype_counts_flat(self) -> dict[str, int]:
        flat: dict[str, int] = {}
        for level_counts in self.count_info.values():
            for subtype_id, cnt in level_counts.items():
                flat[subtype_id] = flat.get(subtype_id, 0) + int(cnt)
        return flat

    def coverage_gaps(self, *, projected_total: int) -> dict[str, int]:
        """Return subtype coverage gap under target min ratios for projected dataset size."""
        projected_total = max(projected_total, 1)
        counts = self.subtype_counts_flat()
        gaps: dict[str, int] = {}
        for subtype_id, meta in self.subtype_meta_by_id.items():
            target_min_count = int(math.ceil(projected_total * float(meta.target_ratio_min)))
            current = int(counts.get(subtype_id, 0))
            gaps[subtype_id] = max(0, target_min_count - current)
        return gaps

    def get_tasks_info(self) -> str:
        """Get human-readable taxonomy description for prompts."""
        tasks_info = ""
        for level_info in self.tasks_info:
            tasks_info += f"#### {level_info.level}: {level_info.name}\n"
            tasks_info += f"description: {level_info.desc}\n"
            tasks_info += "subtask types: \n"
            for subtask in level_info.subtasks:
                tasks_info += (
                    f" - {subtask.subtype_id} ({subtask.name}): {subtask.desc}\n"
                )
                tasks_info += (
                    f" - intents: {', '.join(subtask.canonical_intents)}\n"
                )
                if subtask.constraint_tags:
                    tasks_info += (
                        f" - constraints: {', '.join(subtask.constraint_tags)}\n"
                    )
                tasks_info += " - examples:\n"
                for idx, example in enumerate(subtask.examples, 1):
                    tasks_info += f"   {idx}. {example}\n"
                tasks_info += "\n"
            tasks_info += "---\n\n"
        return tasks_info

    def get_count_info(self) -> str:
        """Get subtype counters as JSON text."""
        return json.dumps(self.count_info, indent=4, ensure_ascii=False)
