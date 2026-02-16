from abc import ABC, abstractmethod
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import string
from typing import Any, Dict

from app.core.common.system_env import SystemEnv
from app.core.common.type import MessageSourceType
from app.core.model.message import ModelMessage
from app.core.prompt.data_synthesis import (
    filter_prompt_template,
    generate_query_tv_template,
)
from app.core.reasoner.model_service_factory import ModelService, ModelServiceFactory
from app.core.toolkit.graph_db.graph_db import GraphDb
from app.core.workflow.dataset_synthesis.model import (
    GENERATOR_STRATEGY,
    TASK_TYPES,
    Row,
    WorkflowTrainDataset,
)
from app.core.workflow.dataset_synthesis.sampler import (
    StratifiedHybridSampler,
    SubGraphSampler,
)
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo
from app.core.workflow.dataset_synthesis.task_subtypes import get_query_subtype_meta
from app.core.workflow.dataset_synthesis.utils import (
    check_engine_query_compatibility,
    is_generic_global_verifier,
    intent_verifier_alignment_ok,
    looks_like_query,
    normalize_intent_set,
    normalize_row_protocol,
    qa_gate_reason,
)


class DatasetGenerator(ABC):
    """Unified interface for dataset generators in the dataset synthesis subsystem.

    Purpose:
      Provide a single, extensible interface for dataset synthesis so that
      different generation strategies and implementations can be added later
      without changing the surrounding orchestration code.

    Contract:
      Implementations must provide an asynchronous `generate` method.

    Parameters (for the generate method):
      - task_desc: a human-readable description that guides the synthesis.
      - dataset_name: name or identifier for the generated dataset.
      - size: the dataset scale (i.e., the desired number of examples to generate).

    Returns:
      WorkflowTrainDataset: the generated dataset.
    """

    @abstractmethod
    async def generate(
        self, task_desc: str, dataset_name: str, size: int
    ) -> WorkflowTrainDataset: ...


class SamplingDatasetGenerator(DatasetGenerator):
    SOFT_REJECT_REASONS = {
        "implicit_full_enumeration",
        "near_duplicate_task",
        "llm_filter_drop",
        "missing_global_verifier",
        "path_missing_hop_bound",
        "missing_order_limit",
        "missing_order_for_bounded_list",
        "missing_group_by",
        "intent_verifier_mismatch",
        "generic_global_verifier",
        "singular_question_multi_rows",
        "boolean_query_multi_rows",
        "scalar_query_multi_rows",
        "entity_not_found_for_id_scoped_question",
        "direction_scope_mismatch",
        "task_entity_not_anchored_in_query",
        "unanchored_path_query",
        "global_verifier_execution_failed",
        "path_exact_hops_not_enforced",
        "path_step_constraints_not_encoded",
        "path_relation_scope_not_encoded",
        "email_domain_filter_not_encoded",
        "shared_neighbor_scope_not_encoded",
        "task_query_semantic_mismatch",
        "shortest_path_length_not_returned",
        "boolean_task_not_boolean_query",
        "amount_aggregation_not_on_relationship",
        "all_null_result_rows",
    }

    """Subgraph-sampling-based implementation of DatasetGenerator.

    Description:
      This generator samples subgraphs from a GraphDb instance and uses a
      ModelService (LLM) to synthesize training examples from each sampled
      subgraph.

    Key constructor parameters:
      - graph_db: GraphDb client/connection used to access the graph database.
      - sampler: an instantiated SubGraphSampler used to extract subgraphs
                 from the provided graph_db (replaces the previous sampler_cls).
      - strategy: generation strategy. See GENERATOR_STRATEGY; typical values:
          * "query"     — generate only query-type tasks
      - max_depth / max_nodes / max_edges: limits controlling sampled subgraph size.
      - nums_per_subgraph: number of examples requested per sampled subgraph.

    Notes:
      - The sampler argument must be a SubGraphSampler instance (it performs
        subgraph extraction against graph_db).
      - GraphDb refers to the graph database connection/client used by sampler.
      - The generator handles strategy identification, pair generation and
        post-generation filtering via the LLM.
    """

    def __init__(
        self,
        graph_db: GraphDb,
        sampler: SubGraphSampler,
        strategy: GENERATOR_STRATEGY = None,
        max_depth: int = 2,
        max_noeds: int = 10,
        max_edges: int = 20,
        nums_per_subgraph: int = 10,
        enable_task_dedup: bool = False,
    ):
        super().__init__()
        self.graph_db = graph_db
        self._llm: ModelService = ModelServiceFactory.create(
            model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
        )
        if not sampler:
            sampler = StratifiedHybridSampler()
        self.sampler: SubGraphSampler = sampler
        self.max_depth = max_depth
        self.max_nodes = max_noeds
        self.max_edges = max_edges
        self.strategy = strategy
        self.nums_per_subgraph = nums_per_subgraph
        self.enable_task_dedup = enable_task_dedup
        self._qa_reject_stats: dict[str, int] = {}
        self._qa_accept_count: int = 0
        self._qa_candidate_count: int = 0
        self._qa_global_executable_count: int = 0
        self._qa_scope_aligned_count: int = 0
        self._qa_intent_aligned_count: int = 0
        self._qa_local_global_consistent_count: int = 0
        self._progress_output_dir: Path | None = None
        self._progress_written_rows: int = 0
        self._progress_reject_stats: dict[str, int] = {}
        self._progress_reject_written_rows: int = 0
        self._progress_candidate_written_rows: int = 0
        self._progress_decision_written_rows: int = 0
        self._progress_salvaged_written_rows: int = 0
        self._salvage_attempt_count: int = 0
        self._salvage_success_count: int = 0
        self._task_fingerprint_set: set[str] = set()

    def _debug_enabled(self) -> bool:
        return bool(SystemEnv.DATASET_SYNTHESIS_DEBUG)

    def _debug_limit(self) -> int:
        limit = int(SystemEnv.DATASET_SYNTHESIS_DEBUG_MAX_CHARS)
        return limit if limit > 0 else 2000

    def _debug_print(self, message: str) -> None:
        if self._debug_enabled():
            print(f"[SamplingDatasetGenerator][debug] {message}")

    def _truncate(self, text: str) -> str:
        limit = self._debug_limit()
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...<truncated {len(text) - limit} chars>"

    def _subgraph_brief(self, subgraph: str) -> str:
        try:
            payload = json.loads(subgraph)
            nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            rels = payload.get("relationships", []) if isinstance(payload, dict) else []
            labels: set[str] = set()
            rel_types: set[str] = set()
            for node in nodes[:50]:
                if isinstance(node, dict):
                    for label in node.get("labels", [])[:5]:
                        labels.add(str(label))
            for rel in rels[:100]:
                if isinstance(rel, dict) and rel.get("type"):
                    rel_types.add(str(rel["type"]))
            return (
                f"nodes={len(nodes)}, relationships={len(rels)}, "
                f"labels={sorted(labels)[:5]}, rel_types={sorted(rel_types)[:8]}"
            )
        except Exception:
            return f"raw_chars={len(subgraph)}"

    @staticmethod
    def _task_fingerprint(task: str) -> str:
        text = (task or "").lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        # Normalize common near-synonyms to reduce paraphrase duplicates.
        replacements = {
            "find": "query",
            "show": "query",
            "list": "query",
            "which": "query",
            "what": "query",
            "top ": "top",
            "within ": "within",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    @staticmethod
    def _candidate_id(attempt: int, index: int, row: Row) -> str:
        basis = "|".join(
            [
                row.level,
                row.task_subtype_id or row.task_subtype or "",
                row.task or "",
                row.global_verifier or "",
            ]
        )
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
        return f"a{attempt:03d}_c{index:03d}_{digest}"

    @staticmethod
    def _token_jaccard(a: str, b: str) -> float:
        sa = set(a.split())
        sb = set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _deduplicate_tasks(
        self,
        *,
        pairs: list[Row],
        existing_rows: list[Row],
    ) -> tuple[list[Row], int, list[tuple[Row, str]]]:
        if not pairs:
            return pairs, 0, []

        accepted: list[Row] = []
        rejected = 0
        rejected_rows: list[tuple[Row, str]] = []
        existing_fingerprints = [self._task_fingerprint(row.task) for row in existing_rows]
        local_fingerprints: list[str] = []

        for row in pairs:
            fp = self._task_fingerprint(row.task)
            duplicate = fp in self._task_fingerprint_set

            if not duplicate:
                for prev in local_fingerprints:
                    if self._token_jaccard(fp, prev) >= 0.9:
                        duplicate = True
                        break
            if not duplicate:
                for prev in existing_fingerprints[-50:]:
                    if self._token_jaccard(fp, prev) >= 0.92:
                        duplicate = True
                        break

            if duplicate:
                rejected += 1
                rejected_rows.append((row, "near_duplicate_task"))
                continue

            accepted.append(row)
            local_fingerprints.append(fp)
            self._task_fingerprint_set.add(fp)

        return accepted, rejected, rejected_rows

    @classmethod
    def _reject_risk_level(cls, reason: str) -> str:
        return "soft" if reason in cls.SOFT_REJECT_REASONS else "hard"

    def _append_rejected_rows(
        self,
        *,
        rejected_rows: list[tuple[Row, str]],
        attempt: int,
        stage: str,
    ) -> None:
        if not rejected_rows:
            return
        for _, reason in rejected_rows:
            self._progress_reject_stats[reason] = self._progress_reject_stats.get(reason, 0) + 1

        if self._progress_output_dir is None:
            return
        out = self._progress_output_dir
        reject_jsonl = out / "progress_rejections.jsonl"
        reject_jsonl.touch(exist_ok=True)
        updated_at = datetime.now().isoformat(timespec="seconds")
        with open(reject_jsonl, "a", encoding="utf-8") as f:
            for row, reason in rejected_rows:
                payload = normalize_row_protocol(row).model_dump()
                f.write(
                    json.dumps(
                        {
                            "updated_at": updated_at,
                            "attempt": attempt,
                            "stage": stage,
                            "reason": reason,
                            "risk_level": self._reject_risk_level(reason),
                            "row": payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._progress_reject_written_rows += 1

    def _append_raw_candidates(
        self,
        *,
        candidates: list[tuple[str, Row]],
        attempt: int,
    ) -> None:
        if not candidates or self._progress_output_dir is None:
            return
        out = self._progress_output_dir
        raw_jsonl = out / "progress_raw_candidates.jsonl"
        raw_jsonl.touch(exist_ok=True)
        updated_at = datetime.now().isoformat(timespec="seconds")
        with open(raw_jsonl, "a", encoding="utf-8") as f:
            for candidate_id, row in candidates:
                f.write(
                    json.dumps(
                        {
                            "updated_at": updated_at,
                            "attempt": attempt,
                            "candidate_id": candidate_id,
                            "row": normalize_row_protocol(row).model_dump(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._progress_candidate_written_rows += 1

    def _append_decision_rows(
        self,
        *,
        decisions: list[dict[str, Any]],
    ) -> None:
        if not decisions or self._progress_output_dir is None:
            return
        out = self._progress_output_dir
        decision_jsonl = out / "progress_decisions.jsonl"
        decision_jsonl.touch(exist_ok=True)
        with open(decision_jsonl, "a", encoding="utf-8") as f:
            for item in decisions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                self._progress_decision_written_rows += 1

    def _append_salvaged_rows(
        self,
        *,
        salvaged_rows: list[dict[str, Any]],
    ) -> None:
        if not salvaged_rows or self._progress_output_dir is None:
            return
        out = self._progress_output_dir
        salvage_jsonl = out / "progress_salvaged.jsonl"
        salvage_jsonl.touch(exist_ok=True)
        with open(salvage_jsonl, "a", encoding="utf-8") as f:
            for item in salvaged_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                self._progress_salvaged_written_rows += 1

    def set_progress_output(self, output_dir: str | Path | None) -> None:
        """Enable incremental progress persistence to a local directory."""
        if output_dir is None:
            self._progress_output_dir = None
            self._progress_written_rows = 0
            self._progress_reject_written_rows = 0
            self._progress_candidate_written_rows = 0
            self._progress_decision_written_rows = 0
            self._progress_salvaged_written_rows = 0
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._progress_output_dir = out
        self._progress_written_rows = 0
        self._progress_reject_written_rows = 0
        self._progress_candidate_written_rows = 0
        self._progress_decision_written_rows = 0
        self._progress_salvaged_written_rows = 0

    def _persist_progress(
        self,
        *,
        dataset: list[Row],
        task_types_info: GraphTaskTypesInfo,
        attempt: int,
        max_attempts: int,
        target_size: int,
        status: str,
        accepted_in_attempt: int = 0,
    ) -> None:
        if self._progress_output_dir is None:
            return
        out = self._progress_output_dir
        dataset_jsonl = out / "progress_dataset.jsonl"
        stats_json = out / "progress_stats.json"
        events_jsonl = out / "progress_events.jsonl"
        reject_jsonl = out / "progress_rejections.jsonl"
        raw_jsonl = out / "progress_raw_candidates.jsonl"
        decisions_jsonl = out / "progress_decisions.jsonl"
        salvage_jsonl = out / "progress_salvaged.jsonl"
        dataset_jsonl.touch(exist_ok=True)
        events_jsonl.touch(exist_ok=True)
        reject_jsonl.touch(exist_ok=True)
        raw_jsonl.touch(exist_ok=True)
        decisions_jsonl.touch(exist_ok=True)
        salvage_jsonl.touch(exist_ok=True)

        # Append only the delta rows to keep I/O small and inspectable mid-run.
        if len(dataset) > self._progress_written_rows:
            with open(dataset_jsonl, "a", encoding="utf-8") as f:
                for row in dataset[self._progress_written_rows :]:
                    f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
            self._progress_written_rows = len(dataset)

        candidates = max(1, self._qa_candidate_count)
        stats = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "target_size": target_size,
            "current_size": len(dataset),
            "accepted_in_attempt": accepted_in_attempt,
            "qa_gate_stats": {
                "accepted_rows": self._qa_accept_count,
                "rejected_rows": sum(self._qa_reject_stats.values()),
                "global_executable_rate_pct": round(
                    100.0 * self._qa_global_executable_count / candidates, 4
                ),
                "scope_alignment_rate_pct": round(
                    100.0 * self._qa_scope_aligned_count / candidates, 4
                ),
                "intent_verifier_alignment_rate_pct": round(
                    100.0 * self._qa_intent_aligned_count / candidates, 4
                ),
                "local_global_consistency_rate_pct": round(
                    100.0 * self._qa_local_global_consistent_count / candidates, 4
                ),
                "data_retention_rate_pct": round(100.0 * self._qa_accept_count / candidates, 4),
                "reject_breakdown": dict(self._qa_reject_stats),
                "reject_risk_breakdown": {
                    "soft": sum(
                        value
                        for reason, value in self._qa_reject_stats.items()
                        if self._reject_risk_level(reason) == "soft"
                    ),
                    "hard": sum(
                        value
                        for reason, value in self._qa_reject_stats.items()
                        if self._reject_risk_level(reason) == "hard"
                    ),
                },
            },
            "reject_audit": {
                "rejected_rows_written": self._progress_reject_written_rows,
                "reject_breakdown": dict(self._progress_reject_stats),
            },
            "decision_audit": {
                "raw_candidates_written": self._progress_candidate_written_rows,
                "decisions_written": self._progress_decision_written_rows,
                "salvaged_rows_written": self._progress_salvaged_written_rows,
                "salvage_attempts": self._salvage_attempt_count,
                "salvage_successes": self._salvage_success_count,
                "salvage_success_rate_pct": round(
                    100.0 * self._salvage_success_count / max(1, self._salvage_attempt_count),
                    4,
                ),
            },
            "subtype_count_info": task_types_info.count_info,
            "sampling_stats": (
                self.sampler.get_sampling_metrics()
                if hasattr(self.sampler, "get_sampling_metrics")
                else {}
            ),
        }

        tmp = stats_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(stats_json)
        with open(events_jsonl, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "updated_at": stats["updated_at"],
                        "status": status,
                        "attempt": attempt,
                        "current_size": len(dataset),
                        "accepted_in_attempt": accepted_in_attempt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    @staticmethod
    def _minify_json_text(raw: str) -> str:
        """Return minified JSON text when possible; otherwise return original text."""
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return raw

    @staticmethod
    def _compress_subgraph_for_prompt(subgraph_raw: str) -> str:
        """Compress subgraph JSON only at formatting level.

        Keep all original ids/properties unchanged. We only remove whitespace to
        reduce prompt size.
        """
        return SamplingDatasetGenerator._minify_json_text(subgraph_raw)

    @staticmethod
    def _build_compact_task_level_info(task_types_info: GraphTaskTypesInfo) -> str:
        """Build compact level/subtype description without verbose examples."""
        parts: list[str] = []
        for level_info in task_types_info.tasks_info:
            parts.append(f"{level_info.level} {level_info.name}: {level_info.desc}")
            for subtask in level_info.subtasks:
                parts.append(f"- {subtask.name}: {subtask.desc}")
        return "\n".join(parts)

    @staticmethod
    def _build_compact_task_stat_info(task_types_info: GraphTaskTypesInfo) -> str:
        """Build compact statistics JSON (no indentation) for prompts."""
        return json.dumps(task_types_info.count_info, ensure_ascii=False, separators=(",", ":"))

    def _build_generate_pairs_prompt(
        self,
        *,
        task_types_info: GraphTaskTypesInfo,
        subgraph: str,
        task_description: str,
        nums: int,
    ) -> str:
        """Build a compact prompt for generate_pairs while preserving key semantics."""
        subgraph_compact = self._compress_subgraph_for_prompt(subgraph)
        level_info_compact = self._build_compact_task_level_info(task_types_info)
        stat_info_compact = self._build_compact_task_stat_info(task_types_info)

        prompt = generate_query_tv_template.format(
            task_description=task_description,
            subgraph=subgraph_compact,
            num_pairs=nums,
            task_level_info=level_info_compact,
            task_statistic_info=stat_info_compact,
        )

        self._debug_print(
            "generate_pairs prompt chars="
            f"{len(prompt)} (subgraph={len(subgraph_compact)}, "
            f"level_info={len(level_info_compact)}, stat_info={len(stat_info_compact)})"
        )
        return prompt

    def _build_filter_prompt(
        self,
        *,
        task_desc: str,
        subgraph: str,
        dataset: list[Row],
    ) -> str:
        """Build a compact filter prompt with JSON-serialized dataset/subgraph."""
        subgraph_compact = self._compress_subgraph_for_prompt(subgraph)
        dataset_compact = json.dumps(
            [row.model_dump() for row in dataset],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = filter_prompt_template.format(
            task_desc=task_desc,
            subgraph=subgraph_compact,
            dataset=dataset_compact,
        )
        self._debug_print(
            "filter prompt chars="
            f"{len(prompt)} (subgraph={len(subgraph_compact)}, dataset={len(dataset_compact)})"
        )
        return prompt

    @staticmethod
    def _intent_features_from_row(row: Row) -> dict[str, Any]:
        intents = {
            intent.lower()
            for intent in normalize_intent_set(
                row.intent_set,
                row.task_subtype,
                subtype_id=row.task_subtype_id,
                level=row.level,
            )
        }
        return {
            "has_path": bool(
                intents.intersection({"query.path.reachability", "query.path.shortest"})
            ),
            "has_shortest_path": "query.path.shortest" in intents,
            "has_reachability_path": "query.path.reachability" in intents,
            "has_constrained_path": "query.path.constrained" in intents,
            "has_cycle_exists": "query.cycle.exists" in intents,
            "has_triangle_motif": "query.motif.triangle_count" in intents,
            "has_similarity_shared_neighbors": "query.similarity.shared_neighbors" in intents,
            "has_ranking": "query.ranking.topk" in intents,
            "has_aggregation_count": "query.aggregation.count" in intents,
            "has_aggregation_group_count": "query.aggregation.group_count" in intents,
            "has_topology_degree": "query.topology.degree" in intents,
        }

    def _build_intent_plan(
        self,
        *,
        task_desc: str,
        task_types_info: GraphTaskTypesInfo,
        remaining: int,
    ) -> dict[str, Any]:
        if remaining <= 2:
            core_intents = ["query.lookup", "query.filter.single"]
            return {
                "task_desc": task_desc,
                "core_intents": core_intents,
                "neighbor_intents": ["query.filter.single"],
                "difficulty_plan": {
                    "remaining_examples": remaining,
                    "focus_levels": ["L1"],
                },
                "capability_constraints": [
                    "query_only_mode",
                    "forbid_unbounded_enumeration",
                    "global_answer_required",
                ],
                "template_routes": {
                    "query.lookup": "template.lookup_basic",
                    "query.filter.single": "template.filter_single",
                },
                "coverage_basis": ["tail_fill_mode"],
                "coverage_gap": {},
            }

        counts = task_types_info.count_info
        subtype_counts = task_types_info.subtype_counts_flat()
        projected_total = task_types_info.total_rows() + max(remaining, 1)
        subtype_gaps = task_types_info.coverage_gaps(projected_total=projected_total)
        sorted_subtypes = sorted(
            subtype_counts.items(),
            key=lambda item: (
                -int(subtype_gaps.get(item[0], 0)),
                int(item[1]),
            ),
        )
        focus_subtypes = [name for name, _ in sorted_subtypes[:4]]

        core_intents: list[str] = []
        constraint_tags: set[str] = set()
        for subtype_id in focus_subtypes:
            meta = task_types_info.get_subtype_meta(subtype_id)
            if not meta:
                continue
            for intent in meta.canonical_intents:
                if intent not in core_intents:
                    core_intents.append(intent)
            for tag in meta.constraint_tags:
                constraint_tags.add(tag)
        if not core_intents:
            core_intents = ["query.lookup"]

        # L3 boost: when current L3 coverage is low, force at least one L3-family
        # intent into the core plan so new advanced subtypes are actually sampled.
        l3_total = sum(int(v) for v in counts.get("L3", {}).values())
        l3_target_min = max(2, int(projected_total * 0.2))
        if l3_total < l3_target_min:
            l3_priority_intents = [
                "query.path.shortest",
                "query.path.constrained",
                "query.cycle.exists",
                "query.motif.triangle_count",
                "query.similarity.shared_neighbors",
                "query.topology.degree",
            ]
            for intent in l3_priority_intents:
                if intent not in core_intents:
                    core_intents.append(intent)
                if len(core_intents) >= 6:
                    break

        difficulty_plan = {
            "remaining_examples": remaining,
            "focus_levels": [level for level, _ in counts.items()],
        }
        template_routes = {
            "query.lookup": "template.lookup_basic",
            "query.neighbor": "template.neighbor_one_hop",
            "query.filter.single": "template.filter_single",
            "query.filter.combined": "template.filter_combined",
            "query.path.reachability": "template.path_reachability",
            "query.path.shortest": "template.path_shortest",
            "query.path.constrained": "template.path_constrained",
            "query.cycle.exists": "template.cycle_exists",
            "query.motif.triangle_count": "template.motif_triangle_count",
            "query.similarity.shared_neighbors": "template.similarity_shared_neighbors",
            "query.aggregation.count": "template.aggregation_count",
            "query.aggregation.group_count": "template.aggregation_group_count",
            "query.ranking.topk": "template.ranking_topk",
            "query.topology.degree": "template.topology_degree",
        }
        return {
            "task_desc": task_desc,
            "core_intents": core_intents,
            "neighbor_intents": [intent for intent in core_intents if intent != "query.lookup"],
            "difficulty_plan": difficulty_plan,
            "capability_constraints": ["query_only_mode", *sorted(constraint_tags)],
            "template_routes": {
                intent: template_routes[intent] for intent in core_intents if intent in template_routes
            },
            "coverage_basis": focus_subtypes,
            "coverage_gap": {subtype: int(subtype_gaps.get(subtype, 0)) for subtype in focus_subtypes},
        }

    def _prioritize_pairs_for_coverage(
        self,
        *,
        pairs: list[Row],
        task_types_info: GraphTaskTypesInfo,
        remaining: int,
    ) -> list[Row]:
        if not pairs:
            return pairs

        projected_total = task_types_info.total_rows() + max(remaining, 1)
        subtype_gaps = task_types_info.coverage_gaps(projected_total=projected_total)
        level_counts = {
            level: sum(int(v) for v in level_map.values())
            for level, level_map in task_types_info.count_info.items()
        }
        l3_target_min = max(1, int(projected_total * 0.2))

        scored: list[tuple[float, Row]] = []
        for row in pairs:
            normalized = normalize_row_protocol(row)
            subtype_id = normalized.task_subtype_id or task_types_info.resolve_subtype_id(
                normalized.task_subtype, level=normalized.level
            )
            subtype_gap = int(subtype_gaps.get(subtype_id or "", 0))
            level_bonus = 0.0
            if normalized.level == "L3" and level_counts.get("L3", 0) < l3_target_min:
                level_bonus = 3.0
            features = self._intent_features_from_row(normalized)
            operation_bonus = 0.0
            if (
                features["has_shortest_path"]
                or features["has_constrained_path"]
                or features["has_cycle_exists"]
                or features["has_triangle_motif"]
                or features["has_similarity_shared_neighbors"]
                or features["has_ranking"]
                or features["has_aggregation_group_count"]
            ):
                operation_bonus = 1.0
            score = float(subtype_gap) * 5.0 + level_bonus + operation_bonus
            scored.append((score, normalized))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]

    @staticmethod
    def _extract_task_literals(task: str) -> list[str]:
        return [item["value"] for item in SamplingDatasetGenerator._extract_literal_specs(task)]

    @staticmethod
    def _extract_literal_specs(task: str) -> list[dict[str, str | None]]:
        if not task:
            return []
        specs: list[dict[str, str | None]] = []
        seen_values: set[str] = set()
        for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", task):
            literal = match.group(1) or match.group(2) or ""
            if not literal:
                continue
            seen_values.add(literal)
            prefix = task[max(0, match.start() - 48) : match.start()].lower()
            entity_positions = {
                "account": prefix.rfind("account"),
                "loan": prefix.rfind("loan"),
                "company": prefix.rfind("company"),
                "medium": prefix.rfind("medium"),
                "person": prefix.rfind("person"),
            }
            last_entity = max(entity_positions.items(), key=lambda item: item[1])
            entity_key = last_entity[0] if last_entity[1] >= 0 else None
            field_hint = "name"
            if re.search(r"\bid\b", prefix) or re.fullmatch(r"\d{6,}", literal):
                field_hint = "id"
            elif "email" in prefix:
                field_hint = "email"
            elif "accountlevel" in prefix:
                field_hint = "accountLevel"
            elif "account type" in prefix or "accounttype" in prefix:
                field_hint = "accountType"
            elif "freqlogintype" in prefix or "frequent login type" in prefix:
                field_hint = "freqLoginType"
            elif "mediumtype" in prefix or "medium type" in prefix:
                field_hint = "mediumType"
            elif "nickname" in prefix:
                field_hint = "nickname"
            elif entity_key == "company":
                field_hint = "companyName"
            elif entity_key == "loan":
                field_hint = "loanName"
            elif entity_key == "person":
                field_hint = "personName"
            elif entity_key == "account":
                field_hint = "nickname"

            label_hint = None
            if entity_key == "account":
                label_hint = "Account"
            elif entity_key == "loan":
                label_hint = "Loan"
            elif entity_key == "company":
                label_hint = "Company"
            elif entity_key == "medium":
                label_hint = "Medium"
            elif entity_key == "person":
                label_hint = "Person"

            # Medium IDs in tasks are usually numeric-like ids.
            if label_hint == "Medium" and field_hint == "name":
                field_hint = "id"

            specs.append(
                {
                    "value": literal,
                    "field_hint": field_hint,
                    "label_hint": label_hint,
                }
            )

        # Capture unquoted numeric ids such as "ID 179018085187986221".
        for match in re.finditer(
            r"\b(?:id|ID)\s*(?:of\s+\w+\s*)?(?:=|is|:)?\s*([0-9]{6,})\b",
            task,
        ):
            literal = match.group(1)
            if not literal or literal in seen_values:
                continue
            prefix = task[max(0, match.start() - 48) : match.start()].lower()
            entity_positions = {
                "account": prefix.rfind("account"),
                "loan": prefix.rfind("loan"),
                "company": prefix.rfind("company"),
                "medium": prefix.rfind("medium"),
                "person": prefix.rfind("person"),
            }
            last_entity = max(entity_positions.items(), key=lambda item: item[1])
            entity_key = last_entity[0] if last_entity[1] >= 0 else None
            label_hint = None
            if entity_key == "account":
                label_hint = "Account"
            elif entity_key == "loan":
                label_hint = "Loan"
            elif entity_key == "medium":
                label_hint = "Medium"
            elif entity_key == "company":
                label_hint = "Company"
            elif entity_key == "person":
                label_hint = "Person"
            specs.append(
                {
                    "value": literal,
                    "field_hint": "id",
                    "label_hint": label_hint,
                }
            )
            seen_values.add(literal)

        # Capture unquoted possessive person names: "Berta Batko's account".
        for match in re.finditer(
            r"\b([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*)'s\s+account\b",
            task,
        ):
            literal = match.group(1).strip()
            if not literal or literal in seen_values:
                continue
            specs.append(
                {
                    "value": literal,
                    "field_hint": "personName",
                    "label_hint": "Person",
                }
            )
            seen_values.add(literal)
        return specs

    @staticmethod
    def _entity_match_predicate(alias: str, literal: str, field_hint: str | None) -> str:
        escaped = literal.replace("'", "\\'")
        if field_hint in {
            "id",
            "nickname",
            "name",
            "personName",
            "companyName",
            "loanName",
            "email",
            "accountLevel",
            "accountType",
            "freqLoginType",
            "mediumType",
        }:
            return f"toString({alias}.{field_hint})='{escaped}'"
        fields = [
            "id",
            "nickname",
            "name",
            "personName",
            "companyName",
            "loanName",
            "email",
            "accountLevel",
            "accountType",
            "freqLoginType",
            "mediumType",
        ]
        return " OR ".join([f"toString({alias}.{field})='{escaped}'" for field in fields])

    @staticmethod
    def _entity_where_clause(literal_specs: list[dict[str, str | None]]) -> str:
        if not literal_specs:
            return ""
        clauses: list[str] = []
        for spec in literal_specs[:3]:
            literal = str(spec.get("value") or "")
            field_hint = spec.get("field_hint")
            predicate = SamplingDatasetGenerator._entity_match_predicate(
                alias="n",
                literal=literal,
                field_hint=field_hint,
            )
            clauses.append(f"({predicate})")
        if not clauses:
            return ""
        return " OR ".join(clauses)

    def _build_verifier_ir(self, row: Row) -> dict[str, Any]:
        literal_specs = self._extract_literal_specs(row.task)
        literals = [str(item.get("value") or "") for item in literal_specs if item.get("value")]
        intents = [
            intent.lower()
            for intent in normalize_intent_set(
                row.intent_set,
                row.task_subtype,
                subtype_id=row.task_subtype_id,
                level=row.level,
            )
        ]
        text = (row.task or "").lower()

        topk = 3
        topk_match = re.search(r"\btop\s*(\d+)\b", text)
        if topk_match:
            topk = max(1, int(topk_match.group(1)))

        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        hop_bound = 4
        exact_hops = None
        hop_match = re.search(
            r"\b(\d+|one|two|three|four|five|six)(?:\s*-\s*|\s*)hops?\b",
            text,
        )
        if hop_match:
            hop_token = hop_match.group(1)
            hop_val = int(hop_token) if hop_token.isdigit() else word_to_num.get(hop_token, 4)
            hop_bound = max(1, hop_val)
            if "-" in hop_match.group(0):
                exact_hops = hop_bound
        exact_hops_match = re.search(
            r"\bexactly\s+(\d+|one|two|three|four|five|six)\s*hops?\b",
            text,
        )
        if not exact_hops_match:
            exact_hops_match = re.search(
                r"\blength\s+exactly\s+(\d+|one|two|three|four|five|six)\b",
                text,
            )
        if not exact_hops_match:
            exact_hops_match = re.search(
                r"\bcycle\s+of\s+length\s+(\d+|one|two|three|four|five|six)\b",
                text,
            )
        if exact_hops_match:
            exact_token = exact_hops_match.group(1)
            exact_val = int(exact_token) if exact_token.isdigit() else word_to_num.get(exact_token, 0)
            exact_hops = max(1, exact_val)
            hop_bound = exact_hops

        known_rel_types = {
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
        relation_types = sorted(
            {
                token
                for token in re.findall(r"\b[A-Z][A-Z_]{2,}\b", row.task or "")
                if token in known_rel_types
            }
        )
        rel_alias_map = {
            "transfer": "TRANSFER",
            "deposit": "DEPOSIT",
            "repay": "REPAY",
            "own": "OWN",
            "invest": "INVEST",
            "invested in": "INVESTED_IN",
            "withdraw": "WITHDRAW",
            "withdrawal": "WITHDRAW",
            "apply": "APPLY",
            "guarantee": "GUARANTEE",
            "signin": "SIGNIN",
            "sign in": "SIGNIN",
        }
        low_text = (row.task or "").lower()
        for keyword, rel_type in rel_alias_map.items():
            if re.search(rf"\b{re.escape(keyword)}\b", low_text):
                relation_types.append(rel_type)
        if "financial transaction" in low_text:
            relation_types.extend(["TRANSFER", "DEPOSIT", "WITHDRAW", "REPAY"])
        relation_types = sorted(set(relation_types))
        direction_hint = "both"
        if re.search(r"\b(out-degree|outgoing|from)\b", text):
            direction_hint = "out"
        elif re.search(r"\b(in-degree|incoming|to)\b", text):
            direction_hint = "in"

        first_rel_type = None
        second_rel_not_type = None
        second_rel_type = None
        step_match = re.search(
            r"first hop is ['\"]?([A-Za-z_]+)['\"]?.*second hop is not ['\"]?([A-Za-z_]+)['\"]?",
            row.task or "",
            flags=re.IGNORECASE,
        )
        if step_match:
            first_rel_type = step_match.group(1).upper()
            second_rel_not_type = step_match.group(2).upper()
        followed_match = re.search(
            r"\b([A-Za-z_]+)\s+followed by\s+(?:a|an)?\s*([A-Za-z_]+)\b",
            row.task or "",
            flags=re.IGNORECASE,
        )
        if followed_match:
            first_rel_type = followed_match.group(1).upper()
            second_rel_type = followed_match.group(2).upper()
            hop_bound = min(hop_bound, 2)
            if exact_hops is None:
                exact_hops = 2
        ordered_rel_match = re.search(
            r"\b([A-Za-z_]+)\s+and\s+([A-Za-z_]+)\s+relationships?\s+in\s+that\s+order\b",
            row.task or "",
            flags=re.IGNORECASE,
        )
        if ordered_rel_match:
            first_rel_type = ordered_rel_match.group(1).upper()
            second_rel_type = ordered_rel_match.group(2).upper()
            hop_bound = min(hop_bound, 2)
            if exact_hops is None:
                exact_hops = 2
        via_loan = bool(re.search(r"\bthrough\s+(?:a\s+)?loan\b", low_text))

        freq_login_type = None
        for pattern in [
            r"logs in via\s+([A-Za-z0-9_-]+)",
            r"uses\s+([A-Za-z0-9_-]+)\s+as the frequent login type",
            r"frequent login type\s+['\"]?([A-Za-z0-9_-]+)['\"]?",
            r"freqlogintype\s*(?:=|is)?\s*['\"]?([A-Za-z0-9_-]+)['\"]?",
        ]:
            match = re.search(pattern, row.task or "", flags=re.IGNORECASE)
            if match:
                freq_login_type = match.group(1).upper()
                break

        email_domain = None
        email_match = re.search(
            r"email\s+domain\s+['\"]?([A-Za-z0-9.-]+\.[A-Za-z]{2,})['\"]?",
            row.task or "",
            flags=re.IGNORECASE,
        )
        if email_match:
            email_domain = email_match.group(1).lower()

        same_source = bool(re.search(r"\bsame\s+source\b", low_text))

        return {
            "intents": intents,
            "literals": literals,
            "literal_specs": literal_specs,
            "topk": topk,
            "hop_bound": hop_bound,
            "exact_hops": exact_hops,
            "relation_types": relation_types,
            "direction_hint": direction_hint,
            "first_rel_type": first_rel_type,
            "second_rel_type": second_rel_type,
            "second_rel_not_type": second_rel_not_type,
            "via_loan": via_loan,
            "freq_login_type": freq_login_type,
            "email_domain": email_domain,
            "same_source": same_source,
            "task_text": row.task or "",
        }

    def _compile_global_verifier_from_ir(self, ir: dict[str, Any]) -> str:
        intents = {str(intent).lower() for intent in ir.get("intents", [])}
        literal_specs = ir.get("literal_specs") or []
        if not literal_specs:
            literal_specs = [
                {"value": str(item), "field_hint": "name", "label_hint": None}
                for item in ir.get("literals", [])
            ]
        literals = [str(item.get("value") or "") for item in literal_specs]
        topk = int(ir.get("topk", 3))
        hop_bound = int(ir.get("hop_bound", 4))
        relation_types = [str(item) for item in ir.get("relation_types", [])]
        direction_hint = str(ir.get("direction_hint", "both"))
        exact_hops_raw = ir.get("exact_hops")
        exact_hops = None if exact_hops_raw in (None, "") else max(1, int(exact_hops_raw))
        if exact_hops is not None:
            hop_bound = exact_hops
        first_rel_type = str(ir.get("first_rel_type") or "").upper()
        second_rel_type = str(ir.get("second_rel_type") or "").upper()
        second_rel_not_type = str(ir.get("second_rel_not_type") or "").upper()
        via_loan = bool(ir.get("via_loan"))
        freq_login_type = str(ir.get("freq_login_type") or "").upper()
        email_domain = str(ir.get("email_domain") or "").lower()
        same_source = bool(ir.get("same_source"))
        task_text = str(ir.get("task_text") or "")
        low_task = task_text.lower()

        known_rel_types = {
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
        entity_specs = [
            spec
            for spec in literal_specs
            if str(spec.get("value") or "").upper() not in known_rel_types
        ]
        if not entity_specs:
            entity_specs = literal_specs
        path_selector = (
            f"[*{exact_hops}..{exact_hops}]"
            if exact_hops is not None
            else f"[*..{hop_bound}]"
        )

        if "query.ranking.topk" in intents:
            if literals:
                specs = entity_specs[: len(literals)]
                cond = " OR ".join(
                    [
                        (
                            f"({self._entity_match_predicate('n', str(spec.get('value') or ''), spec.get('field_hint'))}) "
                            f"OR ({self._entity_match_predicate('m', str(spec.get('value') or ''), spec.get('field_hint'))})"
                        )
                        for spec in specs
                    ]
                )
                return (
                    f"MATCH (n)-[r]->(m) WHERE {cond} "
                    f"RETURN n,m ORDER BY toString(r) LIMIT {topk}"
                )
            return f"MATCH (n)-[r]->(m) RETURN n,m ORDER BY toString(r) LIMIT {topk}"

        if "query.path.shortest" in intents:
            if len(entity_specs) >= 2:
                a_spec, b_spec = entity_specs[0], entity_specs[1]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                b = str(b_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                b_field = str(b_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                wants_hop_length = (
                    "path length" in low_task
                    or "in hops" in low_task
                    or "relationship count" in low_task
                    or "number of relationships" in low_task
                )
                if relation_types:
                    rel_types_text = ", ".join([f"'{item}'" for item in relation_types])
                    if wants_hop_length:
                        return (
                            f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                            f"OPTIONAL MATCH p=shortestPath((a)-{path_selector}-(b)) "
                            f"WHERE p IS NULL OR all(rel IN relationships(p) WHERE type(rel) IN [{rel_types_text}]) "
                            "RETURN CASE WHEN p IS NULL THEN null ELSE length(p) END AS hop_count"
                        )
                    return (
                        f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                        f"MATCH p=shortestPath((a)-{path_selector}-(b)) "
                        f"WHERE all(rel IN relationships(p) WHERE type(rel) IN [{rel_types_text}]) "
                        "RETURN p"
                    )
                if wants_hop_length:
                    return (
                        f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                        f"OPTIONAL MATCH p=shortestPath((a)-{path_selector}-(b)) "
                        "RETURN CASE WHEN p IS NULL THEN null ELSE length(p) END AS hop_count"
                    )
                return (
                    f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                    f"RETURN shortestPath((a)-{path_selector}-(b))"
                )
            return ""

        if "query.path.reachability" in intents:
            if len(entity_specs) >= 2:
                a_spec, b_spec = entity_specs[0], entity_specs[1]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                b = str(b_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                b_field = str(b_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                where_clause = ""
                if relation_types:
                    rel_types_text = ", ".join([f"'{item}'" for item in relation_types])
                    where_clause = (
                        f" WHERE all(rel IN relationships(p) WHERE type(rel) IN [{rel_types_text}])"
                    )
                return (
                    f"MATCH p=(a{a_label} {{{a_field}:'{a}'}})-{path_selector}-(b{b_label} {{{b_field}:'{b}'}})"
                    f"{where_clause} "
                    "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS reachable"
                )
            return ""

        if "query.path.constrained" in intents:
            if first_rel_type and second_rel_type and entity_specs:
                a_spec = entity_specs[0]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                if len(entity_specs) >= 2:
                    b_spec = entity_specs[1]
                    b = str(b_spec.get("value") or "").replace("'", "\\'")
                    b_field = str(b_spec.get("field_hint") or "name")
                    b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                    target = f"(b{b_label} {{{b_field}:'{b}'}})"
                else:
                    target = "(b)"
                mid_node = "(mid:Loan)" if via_loan else "()"
                return (
                    f"MATCH (a{a_label} {{{a_field}:'{a}'}})-[r1:{first_rel_type}]->{mid_node}-[r2:{second_rel_type}]->{target} "
                    "RETURN CASE WHEN count(*) > 0 THEN true ELSE false END AS constrained_reachable"
                )
            if first_rel_type and second_rel_not_type and entity_specs:
                a_spec = entity_specs[0]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                if freq_login_type:
                    freq = freq_login_type.replace("'", "\\'")
                    target = f"(b:Account {{freqLoginType:'{freq}'}})"
                elif len(entity_specs) >= 2:
                    b_spec = entity_specs[1]
                    b = str(b_spec.get("value") or "").replace("'", "\\'")
                    b_field = str(b_spec.get("field_hint") or "name")
                    b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                    target = f"(b{b_label} {{{b_field}:'{b}'}})"
                else:
                    target = "(b)"
                return (
                    f"MATCH (a{a_label} {{{a_field}:'{a}'}})-[r1:{first_rel_type}]->()-[r2]->{target} "
                    f"WHERE type(r2) <> '{second_rel_not_type}' "
                    "RETURN CASE WHEN count(*) > 0 THEN true ELSE false END AS constrained_reachable"
                )
            if len(entity_specs) >= 2:
                a_spec, b_spec = entity_specs[0], entity_specs[1]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                b = str(b_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                b_field = str(b_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                where_clause = ""
                if relation_types:
                    rel_types_text = ", ".join([f"'{item}'" for item in relation_types])
                    where_clause = (
                        f" WHERE all(rel IN relationships(p) WHERE type(rel) IN [{rel_types_text}])"
                    )
                return (
                    f"MATCH p=(a{a_label} {{{a_field}:'{a}'}})-{path_selector}-(b{b_label} {{{b_field}:'{b}'}})"
                    f"{where_clause} "
                    "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS constrained_reachable"
                )
            return ""

        if "query.cycle.exists" in intents:
            if entity_specs:
                a_spec = entity_specs[0]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                return (
                    f"MATCH p=(a{a_label} {{{a_field}:'{a}'}})-{path_selector}-(a) "
                    "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS has_cycle"
                )
            return ""

        if "query.motif.triangle_count" in intents:
            if entity_specs:
                a_spec = entity_specs[0]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                return (
                    f"MATCH (n{a_label} {{{a_field}:'{a}'}})--(x)--(y)--(n) "
                    "WHERE elementId(x) < elementId(y) "
                    "RETURN count(DISTINCT [elementId(x), elementId(y)]) AS triangle_count"
                )
            return (
                "MATCH (n)--(x)--(y)--(n) "
                "WHERE elementId(x) < elementId(y) "
                "WITH n, count(DISTINCT [elementId(x), elementId(y)]) AS triangle_count "
                "RETURN n, triangle_count ORDER BY triangle_count DESC LIMIT 10"
            )

        if "query.similarity.shared_neighbors" in intents:
            if len(entity_specs) >= 2:
                a_spec, b_spec = entity_specs[0], entity_specs[1]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                b = str(b_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                b_field = str(b_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                b_label = f":{b_spec['label_hint']}" if b_spec.get("label_hint") else ""
                if relation_types:
                    rel = next(
                        (
                            candidate
                            for candidate in ["TRANSFER", "WITHDRAW", "DEPOSIT", "REPAY", "INVEST", "SIGNIN"]
                            if candidate in relation_types
                        ),
                        "",
                    )
                    if rel in {"TRANSFER", "WITHDRAW", "DEPOSIT", "REPAY"}:
                        if direction_hint == "in":
                            return (
                                f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                                f"MATCH (src)-[:{rel}]->(a) "
                                f"MATCH (src)-[:{rel}]->(b) "
                                "RETURN count(DISTINCT src) AS shared_neighbors"
                            )
                        if direction_hint == "out":
                            return (
                                f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                                f"MATCH (a)-[:{rel}]->(dst) "
                                f"MATCH (b)-[:{rel}]->(dst) "
                                "RETURN count(DISTINCT dst) AS shared_neighbors"
                            )
                if same_source and "WITHDRAW" in relation_types:
                    return (
                        f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                        "MATCH (src)-[:WITHDRAW]->(a) "
                        "MATCH (src)-[:WITHDRAW]->(b) "
                        "RETURN count(DISTINCT src) AS shared_neighbors"
                    )
                return (
                    f"MATCH (a{a_label} {{{a_field}:'{a}'}}), (b{b_label} {{{b_field}:'{b}'}}) "
                    "MATCH (a)--(x)--(b) "
                    "RETURN count(DISTINCT x) AS shared_neighbors"
                )
            if entity_specs:
                a_spec = entity_specs[0]
                a = str(a_spec.get("value") or "").replace("'", "\\'")
                a_field = str(a_spec.get("field_hint") or "name")
                a_label = f":{a_spec['label_hint']}" if a_spec.get("label_hint") else ""
                if "WITHDRAW" in relation_types:
                    if "how many" in low_task and ("share" in low_task or "common" in low_task):
                        return (
                            f"MATCH (a{a_label} {{{a_field}:'{a}'}})-[:WITHDRAW]->(x)<-[:WITHDRAW]-(b:Account) "
                            "WHERE elementId(a) <> elementId(b) "
                            "RETURN count(DISTINCT b) AS shared_accounts"
                        )
                    return (
                        f"MATCH (a{a_label} {{{a_field}:'{a}'}})-[:WITHDRAW]->(x)<-[:WITHDRAW]-(b:Account) "
                        "WHERE elementId(a) <> elementId(b) "
                        "WITH b, count(DISTINCT x) AS shared_neighbors "
                        f"RETURN b, shared_neighbors ORDER BY shared_neighbors DESC LIMIT {topk}"
                    )
                return (
                    f"MATCH (a{a_label} {{{a_field}:'{a}'}})--(x)--(b) "
                    "WHERE elementId(a) <> elementId(b) "
                    "WITH b, count(DISTINCT x) AS shared_neighbors "
                    f"RETURN b, shared_neighbors ORDER BY shared_neighbors DESC LIMIT {topk}"
                )
            return ""

        if "query.reasoning.chain" in intents:
            rel1 = relation_types[0] if relation_types else "INVEST"
            rel2 = relation_types[1] if len(relation_types) >= 2 else rel1
            if rel1 == "SIGNIN" and "same medium" in low_task:
                person_spec = next(
                    (
                        spec
                        for spec in entity_specs
                        if str(spec.get("field_hint") or "") == "personName"
                        or str(spec.get("label_hint") or "") == "Person"
                    ),
                    None,
                )
                if person_spec:
                    person_name = str(person_spec.get("value") or "").replace("'", "\\'")
                    return (
                        f"MATCH (p:Person {{personName:'{person_name}'}})-[:OWN]->(base:Account)<-[:SIGNIN]-(m:Medium)-[:SIGNIN]->(other:Account) "
                        "RETURN DISTINCT other.id AS accountId, other.nickname AS nickname, m.riskLevel AS riskLevel "
                        f"LIMIT {topk}"
                    )
            if entity_specs:
                target_spec = entity_specs[-1]
                target_value = str(target_spec.get("value") or "").replace("'", "\\'")
                target_field = str(target_spec.get("field_hint") or "name")
                target_label = f":{target_spec['label_hint']}" if target_spec.get("label_hint") else ""
                src_label = ":Company" if "which companies" in low_task else ""
                return_col = "src.companyName AS companyName" if "which companies" in low_task else "src"
                return (
                    f"MATCH (src{src_label})-[:{rel1}]->(mid)-[:{rel2}]->"
                    f"(target{target_label} {{{target_field}:'{target_value}'}}) "
                    f"RETURN DISTINCT {return_col} LIMIT {topk}"
                )
            return (
                f"MATCH (src)-[:{rel1}]->(mid)-[:{rel2}]->(target) "
                f"RETURN DISTINCT src LIMIT {topk}"
            )

        has_signin_phrase = ("signin" in low_task) or ("sign in" in low_task) or ("signs in" in low_task)
        if "login type" in low_task and has_signin_phrase:
            medium_type_spec = next(
                (
                    spec
                    for spec in entity_specs
                    if str(spec.get("field_hint") or "") == "mediumType"
                    or (
                        str(spec.get("label_hint") or "") == "Medium"
                        and str(spec.get("field_hint") or "") in {"name", "id"}
                    )
                ),
                None,
            )
            if medium_type_spec:
                medium_value = str(medium_type_spec.get("value") or "").replace("'", "\\'")
                if str(medium_type_spec.get("field_hint") or "") == "id":
                    return (
                        f"MATCH (m:Medium {{id:'{medium_value}'}})-[:SIGNIN]->(a:Account) "
                        "RETURN DISTINCT a.freqLoginType AS freqLoginType"
                    )
                return (
                    f"MATCH (m:Medium {{mediumType:'{medium_value}'}})-[:SIGNIN]->(a:Account) "
                    "RETURN DISTINCT a.freqLoginType AS freqLoginType"
                )

        if (
            ("total amount" in low_task or "sum" in low_task or "total deposited" in low_task)
            and any(rel in relation_types for rel in ["DEPOSIT", "WITHDRAW", "TRANSFER", "REPAY"])
        ):
            rel_type = next(
                rel
                for rel in ["DEPOSIT", "WITHDRAW", "TRANSFER", "REPAY"]
                if rel in relation_types
            )
            person_spec = next(
                (
                    spec
                    for spec in entity_specs
                    if str(spec.get("field_hint") or "") == "personName"
                    or str(spec.get("label_hint") or "") == "Person"
                ),
                None,
            )
            loan_spec = next(
                (
                    spec
                    for spec in entity_specs
                    if str(spec.get("label_hint") or "") == "Loan"
                    or (
                        str(spec.get("field_hint") or "") == "id"
                        and "loan" in low_task
                    )
                ),
                None,
            )
            account_level_spec = next(
                (
                    spec
                    for spec in entity_specs
                    if str(spec.get("field_hint") or "") == "accountLevel"
                ),
                None,
            )
            match_clauses: list[str] = []
            where_clauses: list[str] = []
            if person_spec:
                person_value = str(person_spec.get("value") or "").replace("'", "\\'")
                match_clauses.append(
                    f"(p:Person {{personName:'{person_value}'}})-[:OWN]->(a:Account)"
                )
            else:
                match_clauses.append("(a:Account)")
            if account_level_spec:
                level_value = str(account_level_spec.get("value") or "").replace("'", "\\'")
                where_clauses.append(f"toString(a.accountLevel)='{level_value}'")
            if loan_spec and rel_type == "DEPOSIT":
                loan_value = str(loan_spec.get("value") or "").replace("'", "\\'")
                loan_field = str(loan_spec.get("field_hint") or "id")
                match_clauses.append(f"(l:Loan {{{loan_field}:'{loan_value}'}})-[r:DEPOSIT]->(a)")
            else:
                match_clauses.append(f"()-[r:{rel_type}]->(a)")
            where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            return (
                f"MATCH {', '.join(match_clauses)}"
                f"{where_sql} "
                "RETURN sum(toFloat(r.amount)) AS total"
            )

        if (
            (low_task.startswith("if ") and re.search(r"\b(can|should|does)\b", low_task))
            or low_task.startswith("has ")
            or low_task.startswith("does ")
        ):
            if "blocked" in low_task and "transfer" in low_task:
                return (
                    "MATCH (a:Account)-[r:TRANSFER]->() "
                    "WHERE toString(a.isBlocked)='true' "
                    "RETURN count(r) > 0 AS answer"
                )
            rel = relation_types[0] if relation_types else "SIGNIN"
            if len(entity_specs) >= 3:
                s1, s2, s3 = entity_specs[0], entity_specs[1], entity_specs[2]
                v1 = str(s1.get("value") or "").replace("'", "\\'")
                v2 = str(s2.get("value") or "").replace("'", "\\'")
                v3 = str(s3.get("value") or "").replace("'", "\\'")
                f1 = str(s1.get("field_hint") or "name")
                f2 = str(s2.get("field_hint") or "name")
                f3 = str(s3.get("field_hint") or "name")
                l1 = f":{s1['label_hint']}" if s1.get("label_hint") else ""
                l2 = f":{s2['label_hint']}" if s2.get("label_hint") else ""
                l3 = f":{s3['label_hint']}" if s3.get("label_hint") else ""
                return (
                    f"MATCH (a{l1} {{{f1}:'{v1}'}})-[:{rel}]->(b{l2} {{{f2}:'{v2}'}})-[:{rel}]->(c{l3} {{{f3}:'{v3}'}}) "
                    "RETURN count(*) > 0 AS answer"
                )
            return f"MATCH ()-[r:{rel}]->() RETURN count(r) > 0 AS answer"

        if email_domain:
            domain = email_domain.replace("'", "\\'")
            return (
                "MATCH (a:Account) "
                f"WHERE toLower(split(toString(a.email), '@')[1]) = '{domain}' "
                "RETURN DISTINCT a.nickname ORDER BY a.nickname ASC LIMIT 200"
            )

        if freq_login_type and "freqlogintype" in low_task:
            freq = freq_login_type.replace("'", "\\'")
            if re.search(r"\bwhat is (its|the)\s+freqlogintype\b", low_task):
                return (
                    f"MATCH (a:Account {{freqLoginType:'{freq}'}}) "
                    "RETURN DISTINCT a.freqLoginType AS freqLoginType LIMIT 1"
                )
            return (
                f"MATCH (a:Account {{freqLoginType:'{freq}'}}) "
                "RETURN DISTINCT a.nickname AS nickname"
            )

        if "query.aggregation.group_count" in intents:
            return (
                "MATCH (n)-[r]->(m) "
                "WITH type(r) AS relation_type, count(*) AS relation_count "
                "RETURN relation_type, relation_count ORDER BY relation_count DESC LIMIT 10"
            )

        if "query.aggregation.count" in intents:
            return "MATCH ()-[r]->() RETURN count(r) AS total_relationships"

        if "query.topology.degree" in intents:
            where_clause = self._entity_where_clause(entity_specs)
            if direction_hint == "out":
                directed_pattern = "(n)-[r]->()"
            elif direction_hint == "in":
                directed_pattern = "()-[r]->(n)"
            else:
                directed_pattern = "(n)-[r]-()"
            if relation_types:
                rel_types_text = ", ".join([f"'{item}'" for item in relation_types])
                if where_clause:
                    return (
                        f"MATCH (n) WHERE {where_clause} "
                        f"OPTIONAL MATCH {directed_pattern} WHERE type(r) IN [{rel_types_text}] "
                        "RETURN count(r) AS degree"
                    )
                return (
                    f"MATCH (n) OPTIONAL MATCH {directed_pattern} WHERE type(r) IN [{rel_types_text}] "
                    "RETURN n, count(r) AS degree ORDER BY degree DESC LIMIT 10"
                )
            if where_clause:
                if direction_hint == "out":
                    degree_pattern = "(n)-->(m)"
                elif direction_hint == "in":
                    degree_pattern = "(m)-->(n)"
                else:
                    degree_pattern = "(n)--(m)"
                return (
                    f"MATCH (n) WHERE {where_clause} "
                    f"OPTIONAL MATCH {degree_pattern} "
                    "RETURN count(m) AS degree"
                )
            return (
                "MATCH (n)--(m) "
                "WITH n, count(m) AS degree "
                "RETURN n, degree ORDER BY degree DESC LIMIT 10"
            )

        return ""

    def _compile_global_verifier(
        self,
        row: Row,
        subgraph: str,
        *,
        prefer_existing: bool = True,
    ) -> str:
        row = normalize_row_protocol(row)
        if prefer_existing:
            if row.global_verifier:
                return row.global_verifier
            if looks_like_query(row.verifier):
                return row.verifier

        # Template-first route (intent-specific deterministic templates).
        literal_specs = self._extract_literal_specs(row.task)
        literals = [str(item.get("value") or "") for item in literal_specs if item.get("value")]
        features = self._intent_features_from_row(row)
        if features["has_ranking"]:
            if literals:
                specs = literal_specs[: len(literals)]
                cond = " OR ".join(
                    [
                        (
                            f"({self._entity_match_predicate('n', str(spec.get('value') or ''), spec.get('field_hint'))}) "
                            f"OR ({self._entity_match_predicate('m', str(spec.get('value') or ''), spec.get('field_hint'))})"
                        )
                        for spec in specs
                    ]
                )
                return (
                    f"MATCH (n)-[r]->(m) WHERE {cond} "
                    "RETURN n,m ORDER BY toString(r) LIMIT 3"
                )
            return "MATCH (n)-[r]->(m) RETURN n,m ORDER BY toString(r) LIMIT 3"
        ir = self._build_verifier_ir(row)
        common_ir = {
            "literals": ir.get("literals", []),
            "literal_specs": ir.get("literal_specs", []),
            "relation_types": ir.get("relation_types", []),
            "topk": ir.get("topk", 3),
            "hop_bound": ir.get("hop_bound", 4),
            "exact_hops": ir.get("exact_hops"),
            "direction_hint": ir.get("direction_hint", "both"),
            "first_rel_type": ir.get("first_rel_type"),
            "second_rel_type": ir.get("second_rel_type"),
            "second_rel_not_type": ir.get("second_rel_not_type"),
            "via_loan": ir.get("via_loan", False),
            "freq_login_type": ir.get("freq_login_type"),
            "email_domain": ir.get("email_domain"),
            "same_source": ir.get("same_source", False),
            "task_text": row.task or "",
        }
        if features["has_shortest_path"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.path.shortest"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_reachability_path"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.path.reachability"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_constrained_path"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.path.constrained"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_cycle_exists"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.cycle.exists"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_triangle_motif"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.motif.triangle_count"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_similarity_shared_neighbors"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.similarity.shared_neighbors"],
                    **common_ir,
                }
            )
            if compiled:
                return compiled
            return ""
        if features["has_aggregation_group_count"]:
            return (
                "MATCH (n)-[r]->(m) "
                "WITH type(r) AS relation_type, count(*) AS relation_count "
                "RETURN relation_type, relation_count ORDER BY relation_count DESC LIMIT 10"
            )
        if features["has_aggregation_count"]:
            return "MATCH ()-[r]->() RETURN count(r) AS total_relationships"
        if features["has_topology_degree"]:
            compiled = self._compile_global_verifier_from_ir(
                {
                    "intents": ["query.topology.degree"],
                    **common_ir,
                    "topk": 10,
                }
            )
            if compiled:
                return compiled

        # IR-first route (fallback when no template branch matched).
        compiled = self._compile_global_verifier_from_ir(
            {
                **ir,
                "task_text": row.task or "",
            }
        )
        if compiled:
            return compiled

        # No safe verifier can be compiled for this row.
        return ""

    def _repair_global_verifier_if_needed(self, row: Row, subgraph: str) -> Row:
        row = normalize_row_protocol(row)
        query = row.global_verifier or ""
        low_query = query.lower()
        compatible, compat_reason = check_engine_query_compatibility(
            query=query,
            engine_hint=type(self.graph_db).__name__,
        ) if query else (False, "missing_global_verifier")

        features = self._intent_features_from_row(row)
        subtype_meta = get_query_subtype_meta(
            row.task_subtype_id or row.task_subtype,
            level=row.level,
        )
        subtype_tags = set(subtype_meta.constraint_tags or []) if subtype_meta else set()
        needs_repair = (
            (not query)
            or (not looks_like_query(query))
            or is_generic_global_verifier(query)
            or (not compatible and compat_reason in {
                "path_missing_hop_bound",
                "syntax_variable_shadowing",
                "syntax_legacy_size_pattern_expression",
            })
            or (not intent_verifier_alignment_ok(row))
        )
        if features["has_ranking"] and not ("order by" in low_query and "limit" in low_query):
            needs_repair = True
        if features["has_aggregation_group_count"]:
            has_grouping = " group by " in low_query or re.search(r"\bwith\b.*count\(", low_query)
            if not has_grouping:
                needs_repair = True
        task_prefers_ranking = bool(
            re.search(r"\b(top|largest|highest|lowest|smallest|most|least|rank)\b", (row.task or "").lower())
        )
        if (
            "requires_order_limit" in subtype_tags
            and (features["has_ranking"] or task_prefers_ranking)
            and not ("order by" in low_query and "limit" in low_query)
        ):
            needs_repair = True
        if "requires_group_by" in subtype_tags:
            has_grouping = " group by " in low_query or re.search(r"\bwith\b.*count\(", low_query)
            if not has_grouping:
                needs_repair = True

        if not needs_repair:
            return row

        repaired = self._compile_global_verifier(
            row=row,
            subgraph=subgraph,
            prefer_existing=False,
        )
        if repaired and looks_like_query(repaired):
            row.global_verifier = repaired
            row = normalize_row_protocol(row)
        return row

    def _run_global_verifier(self, global_verifier: str) -> str | None:
        if not looks_like_query(global_verifier):
            return None
        try:
            conn = getattr(self.graph_db, "conn", None)
        except Exception:
            conn = None
        if conn is None:
            return None

        try:
            with conn.session() as session:
                result = session.run(global_verifier)
                rows = []
                for idx, record in enumerate(result):
                    if idx >= 200:
                        break
                    normalized_row: dict[str, Any] = {}
                    for key in record.keys():
                        normalized_row[key] = self._to_json_value(record[key])
                    rows.append(normalized_row)
            return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            self._debug_print(f"global verifier execution failed: {e}")
            return None

    @staticmethod
    def _parse_execution_rows(executed: str | None) -> list[dict[str, Any]]:
        if not executed:
            return []
        try:
            parsed = json.loads(executed)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    @classmethod
    def _runtime_semantic_reason(cls, row: Row, executed: str | None) -> str | None:
        rows = cls._parse_execution_rows(executed)
        task = (row.task or "").strip().lower()
        query = f" {(row.global_verifier or '').lower()} "

        if ("list all" in task or " top " in f" {task} ") and " limit " in query and " order by " not in query:
            if "count(" not in query:
                return "missing_order_for_bounded_list"

        unique_rows = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in rows}
        unique_count = len(unique_rows)
        if rows:
            non_null_exists = any(
                any(value is not None for value in record.values())
                for record in rows
            )
            if not non_null_exists:
                return "all_null_result_rows"

        scalar_task = bool(
            re.search(
                r"\b(how many|largest|highest|lowest|smallest|out-degree|in-degree|risk level|account type)\b",
                task,
            )
        ) or task.startswith("what is ")
        if scalar_task and len(rows) > 1:
            return "scalar_query_multi_rows"

        is_bool_task = (
            task.startswith("is there")
            or task.startswith("can we infer")
            or task.startswith("does ")
            or task.startswith("do ")
            or task.startswith("has ")
            or (task.startswith("if ") and bool(re.search(r"\b(can|should)\b", task)))
        )
        if is_bool_task and len(rows) > 1:
            return "boolean_query_multi_rows"

        singular_task = bool(
            re.search(r"^(which|what|who)\s+(account|company|loan|medium)\b", task)
        ) or task.startswith("who is ") or task.startswith("what is the ")
        if singular_task and unique_count > 1:
            return "singular_question_multi_rows"

        if (" id '" in task or ' id "' in task) and len(rows) == 0 and scalar_task:
            return "entity_not_found_for_id_scoped_question"

        if "total number of incoming and outgoing" in task:
            if "(n)-->(m)" in query or "(m)-->(n)" in query or "(n)-[r]->()" in query or "()-[r]->(n)" in query:
                return "direction_scope_mismatch"

        if "path" in task:
            if "match (a) with a limit 20" in query and "shortestpath((a)-[*..4]-(b))" in query:
                return "unanchored_path_query"
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
                    return "path_exact_hops_not_enforced"
                if not (
                    re.search(rf"\[\s*\*\s*{hops}\s*\.\.\s*{hops}\s*\]", query)
                    or re.search(rf"\blength\s*\(\s*\w+\s*\)\s*=\s*{hops}\b", query)
                ):
                    return "path_exact_hops_not_enforced"
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", row.task or "")
            literals = [a or b for a, b in quoted if (a or b)]
            rel_literals = {
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
            literals = [item for item in literals if item.upper() not in rel_literals]
            for m in re.finditer(
                r"\b(?:id|ID)\s*(?:of\s+\w+\s*)?(?:=|is|:)?\s*([0-9]{6,})\b",
                row.task or "",
            ):
                literals.append(m.group(1))
            if len(literals) >= 2:
                endpoint_literals = literals[:2]
                if any(str(literal).lower() not in query for literal in endpoint_literals):
                    return "task_entity_not_anchored_in_query"

        return None

    @classmethod
    def _to_json_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [cls._to_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): cls._to_json_value(v) for k, v in value.items()}

        # Neo4j node-like object.
        if hasattr(value, "labels") and hasattr(value, "items"):
            try:
                return {
                    "_type": "node",
                    "labels": sorted([str(label) for label in list(value.labels)]),
                    "properties": {str(k): cls._to_json_value(v) for k, v in dict(value).items()},
                }
            except Exception:
                pass

        # Neo4j relationship-like object.
        if hasattr(value, "type") and hasattr(value, "start_node") and hasattr(value, "end_node"):
            try:
                return {
                    "_type": "relationship",
                    "rel_type": str(value.type),
                    "properties": {str(k): cls._to_json_value(v) for k, v in dict(value).items()},
                }
            except Exception:
                pass

        # Neo4j path-like object.
        if hasattr(value, "nodes") and hasattr(value, "relationships"):
            try:
                return {
                    "_type": "path",
                    "nodes": [cls._to_json_value(node) for node in list(value.nodes)],
                    "relationships": [cls._to_json_value(rel) for rel in list(value.relationships)],
                }
            except Exception:
                pass

        return str(value)

    def _has_live_graph_session(self) -> bool:
        try:
            conn = getattr(self.graph_db, "conn", None)
        except Exception:
            return False
        return conn is not None and hasattr(conn, "session")

    def _qa_gate_row(self, row: Row) -> tuple[bool, str]:
        row = normalize_row_protocol(row)
        if row.answer_scope == "global_graph" and row.global_verifier:
            if not looks_like_query(row.global_verifier):
                return False, "global_verifier_not_query"
            if is_generic_global_verifier(row.global_verifier):
                return False, "generic_global_verifier"
            compatible, reason = check_engine_query_compatibility(
                query=row.global_verifier,
                engine_hint=type(self.graph_db).__name__,
            )
            if not compatible:
                return False, reason
            if self._has_live_graph_session():
                executed = self._run_global_verifier(row.global_verifier)
                if executed is None:
                    # Retry once with deterministic recompilation before reject.
                    row.global_verifier = self._compile_global_verifier(
                        row=row,
                        subgraph="",
                        prefer_existing=False,
                    )
                    compatible_retry, reason_retry = check_engine_query_compatibility(
                        query=row.global_verifier,
                        engine_hint=type(self.graph_db).__name__,
                    )
                    if not compatible_retry:
                        return False, reason_retry
                    executed = self._run_global_verifier(row.global_verifier)
                    if executed is None:
                        return False, "global_verifier_execution_failed"
                row.expected_global = executed
                semantic_reason = self._runtime_semantic_reason(row=row, executed=executed)
                if semantic_reason:
                    return False, semantic_reason

        reason = qa_gate_reason(row=row, engine_hint=type(self.graph_db).__name__)
        if reason is None:
            return True, "accepted"
        return False, reason

    def _qa_gate_filter(
        self, rows: list[Row]
    ) -> tuple[list[Row], dict[str, int], list[tuple[Row, str]]]:
        accepted: list[Row] = []
        reject_stats: dict[str, int] = {}
        rejected_rows: list[tuple[Row, str]] = []
        for row in rows:
            ok, reason = self._qa_gate_row(row)
            if not ok:
                reject_stats[reason] = reject_stats.get(reason, 0) + 1
                rejected_rows.append((row, reason))
                continue

            reason = qa_gate_reason(row=row, engine_hint=type(self.graph_db).__name__)
            if reason is None:
                accepted.append(row)
                continue
            reject_stats[reason] = reject_stats.get(reason, 0) + 1
            rejected_rows.append((row, reason))
        return accepted, reject_stats, rejected_rows

    @staticmethod
    def _append_limit_if_missing(query: str, limit: int = 5) -> str:
        low = (query or "").lower()
        if " limit " in f" {low} ":
            return query
        return query.rstrip() + f" LIMIT {limit}"

    @staticmethod
    def _append_order_limit_if_missing(query: str, limit: int = 5) -> str:
        updated = query.rstrip()
        low = updated.lower()
        if " order by " not in f" {low} ":
            updated += " ORDER BY 1 ASC"
        if " limit " not in f" {low} ":
            updated += f" LIMIT {limit}"
        return updated

    @staticmethod
    def _pluralize_task(task: str) -> str:
        updated = task or ""
        replacements = [
            ("Which account is", "Which accounts are"),
            ("Which account has", "Which accounts have"),
            ("Which company is", "Which companies are"),
            ("Which company owns", "Which companies own"),
            ("What is the nickname", "What are the nicknames"),
            ("What is the account", "What are the accounts"),
            ("Who is", "Who are"),
        ]
        for source, target in replacements:
            if updated.startswith(source):
                return target + updated[len(source) :]
        return updated

    def _rewrite_row_by_reason(self, row: Row, reason: str, subgraph: str) -> tuple[Row, str]:
        candidate = normalize_row_protocol(row.model_copy(deep=True))
        original_task = candidate.task.rstrip("?")
        original_query = candidate.global_verifier or ""

        if reason in {"implicit_full_enumeration", "llm_filter_drop"}:
            if not re.search(
                r"\b(top|within|limit|at most|exactly|\d+|latest|first|last|largest|highest|lowest)\b",
                original_task.lower(),
            ):
                candidate.task = f"{original_task} (top 5 results only)?"
            if original_query and looks_like_query(original_query):
                candidate.global_verifier = self._append_order_limit_if_missing(original_query, limit=5)
            if not candidate.global_verifier:
                candidate.global_verifier = self._compile_global_verifier(
                    row=candidate,
                    subgraph=subgraph,
                    prefer_existing=False,
                )
            candidate = normalize_row_protocol(candidate)
            return candidate, "bounded_enumeration_limit"

        if reason == "missing_order_limit":
            query = candidate.global_verifier or self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate.global_verifier = self._append_order_limit_if_missing(query, limit=5)
            if "top" not in candidate.task.lower():
                candidate.task = f"{original_task} (top 5)?"
            candidate = normalize_row_protocol(candidate)
            return candidate, "add_order_limit"

        if reason == "missing_group_by":
            candidate.global_verifier = self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate = normalize_row_protocol(candidate)
            return candidate, "recompile_group_count"

        if reason == "intent_verifier_mismatch":
            candidate.global_verifier = self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate = normalize_row_protocol(candidate)
            return candidate, "recompile_by_intent"

        if reason == "global_verifier_execution_failed":
            candidate.global_verifier = self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate = normalize_row_protocol(candidate)
            return candidate, "recompile_after_exec_failure"

        if reason == "missing_order_for_bounded_list":
            query = candidate.global_verifier or self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate.global_verifier = self._append_order_limit_if_missing(query, limit=5)
            candidate = normalize_row_protocol(candidate)
            return candidate, "add_deterministic_order_limit"

        if reason == "singular_question_multi_rows":
            candidate.task = self._pluralize_task(original_task)
            if candidate.task == original_task:
                candidate.task = f"{original_task} (top 5 results only)"
            if original_query and looks_like_query(original_query):
                candidate.global_verifier = self._append_order_limit_if_missing(original_query, limit=5)
            else:
                candidate.global_verifier = self._compile_global_verifier(
                    row=candidate,
                    subgraph=subgraph,
                    prefer_existing=False,
                )
            candidate = normalize_row_protocol(candidate)
            return candidate, "rewrite_to_plural_bounded"

        if reason in {
            "missing_global_verifier",
            "path_missing_hop_bound",
            "path_exact_hops_not_enforced",
            "path_step_constraints_not_encoded",
            "path_relation_scope_not_encoded",
            "email_domain_filter_not_encoded",
            "shared_neighbor_scope_not_encoded",
            "task_query_semantic_mismatch",
            "shortest_path_length_not_returned",
            "boolean_task_not_boolean_query",
            "amount_aggregation_not_on_relationship",
            "all_null_result_rows",
            "boolean_query_multi_rows",
            "scalar_query_multi_rows",
            "entity_not_found_for_id_scoped_question",
            "generic_global_verifier",
            "direction_scope_mismatch",
            "task_entity_not_anchored_in_query",
            "unanchored_path_query",
        }:
            candidate.global_verifier = self._compile_global_verifier(
                row=candidate,
                subgraph=subgraph,
                prefer_existing=False,
            )
            candidate = normalize_row_protocol(candidate)
            return candidate, "recompile_runtime_shape"

        return candidate, "no_rule"

    def _salvage_soft_rejects(
        self,
        *,
        rejected_rows: list[tuple[int, Row, str]],
        subgraph: str,
        attempt: int,
    ) -> tuple[list[tuple[int, Row]], list[tuple[int, Row, str]], list[dict[str, Any]]]:
        salvaged: list[tuple[int, Row]] = []
        unresolved: list[tuple[int, Row, str]] = []
        salvage_logs: list[dict[str, Any]] = []

        for index, row, reason in rejected_rows:
            risk = self._reject_risk_level(reason)
            if risk != "soft":
                unresolved.append((index, row, reason))
                continue

            self._salvage_attempt_count += 1
            original = normalize_row_protocol(row.model_copy(deep=True))

            # Try original row first (especially for llm_filter_drop).
            probe = self._hydrate_rows_for_global_truth([original.model_copy(deep=True)], subgraph=subgraph)[0]
            ok, probe_reason = self._qa_gate_row(probe)
            if ok and qa_gate_reason(probe, engine_hint=type(self.graph_db).__name__) is None:
                salvaged.append((index, probe))
                self._salvage_success_count += 1
                salvage_logs.append(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "attempt": attempt,
                        "candidate_index": index,
                        "reason_before": reason,
                        "rewrite_action": "none",
                        "reason_after": "accepted",
                        "row_before": original.model_dump(),
                        "row_after": probe.model_dump(),
                    }
                )
                continue

            rewritten, action = self._rewrite_row_by_reason(
                row=original,
                reason=reason if reason != "llm_filter_drop" else (probe_reason or reason),
                subgraph=subgraph,
            )
            rewritten = self._hydrate_rows_for_global_truth([rewritten], subgraph=subgraph)[0]
            ok, rewritten_reason = self._qa_gate_row(rewritten)
            final_reason = qa_gate_reason(rewritten, engine_hint=type(self.graph_db).__name__) if ok else rewritten_reason
            if ok and final_reason is None:
                salvaged.append((index, rewritten))
                self._salvage_success_count += 1
                salvage_logs.append(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "attempt": attempt,
                        "candidate_index": index,
                        "reason_before": reason,
                        "rewrite_action": action,
                        "reason_after": "accepted",
                        "row_before": original.model_dump(),
                        "row_after": rewritten.model_dump(),
                    }
                )
                continue

            unresolved.append((index, row, final_reason or reason))

        return salvaged, unresolved, salvage_logs

    def _update_quality_metrics(self, rows: list[Row]) -> None:
        for row in rows:
            row = normalize_row_protocol(row)
            self._qa_candidate_count += 1

            query_text = row.global_verifier or row.verifier or ""
            is_query = looks_like_query(query_text)
            if row.answer_scope == "global_graph":
                if is_query and row.expected_global:
                    self._qa_global_executable_count += 1
                if row.global_verifier:
                    self._qa_scope_aligned_count += 1
            else:
                self._qa_scope_aligned_count += 1

            if intent_verifier_alignment_ok(row):
                self._qa_intent_aligned_count += 1

            if row.answer_scope == "local_subgraph":
                self._qa_local_global_consistent_count += 1
            elif row.expected_global:
                self._qa_local_global_consistent_count += 1

    def _hydrate_rows_for_global_truth(self, rows: list[Row], subgraph: str) -> list[Row]:
        hydrated: list[Row] = []
        for row in rows:
            row = normalize_row_protocol(row)
            if not row.global_verifier:
                row.global_verifier = self._compile_global_verifier(row=row, subgraph=subgraph)
                row = normalize_row_protocol(row)
            row = self._repair_global_verifier_if_needed(row=row, subgraph=subgraph)
            if row.answer_scope == "global_graph" and row.global_verifier:
                executed = self._run_global_verifier(row.global_verifier)
                if executed:
                    row.expected_global = executed
            hydrated.append(row)
        return hydrated

    def extract_pairs(self, task_type: TASK_TYPES, text: str) -> list[Row]:
        """Extract TV pairs from an LLM response.

        The LLM is prompted to output a JSON list of objects. In practice, model outputs
        may be wrapped in markdown fences or include extra prose. This parser tries to be
        robust without relying on regex patterns that break on braces inside strings.
        """
        required_fields = {"level", "task", "verifier"}

        def _iter_json_blocks(raw: str) -> list[str]:
            raw = raw.strip()
            if not raw:
                return []

            blocks: list[str] = [raw]

            # Common: fenced JSON blocks.
            for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE):
                inner = m.group(1).strip()
                if inner:
                    blocks.append(inner)

            # Common: extra text around a JSON list/dict.
            lbrack = raw.find("[")
            rbrack = raw.rfind("]")
            if 0 <= lbrack < rbrack:
                blocks.append(raw[lbrack : rbrack + 1])

            lbrace = raw.find("{")
            rbrace = raw.rfind("}")
            if 0 <= lbrace < rbrace:
                blocks.append(raw[lbrace : rbrace + 1])

            # De-dup while preserving order.
            seen: set[str] = set()
            uniq: list[str] = []
            for b in blocks:
                if b not in seen:
                    uniq.append(b)
                    seen.add(b)
            return uniq

        valid_pairs: list[Row] = []

        def _to_text(value) -> str:
            if isinstance(value, str):
                return value.strip()
            if value is None:
                return ""
            if isinstance(value, (int, float, bool)):
                return str(value)
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)

        def _maybe_add_obj(obj: Dict) -> None:
            if not required_fields.issubset(set(obj.keys())):
                return
            normalized = dict(obj)
            normalized["task_type"] = task_type
            normalized["level"] = _to_text(normalized.get("level")).upper()
            task_subtype = _to_text(normalized.get("task_subtype"))
            task_subtype_id = _to_text(normalized.get("task_subtype_id"))
            if not task_subtype and task_subtype_id:
                task_subtype = task_subtype_id
            normalized["task_subtype"] = task_subtype or "unknown"
            normalized["task_subtype_id"] = task_subtype_id or None
            normalized["task"] = _to_text(normalized.get("task"))
            normalized["verifier"] = _to_text(normalized.get("verifier"))
            normalized["generation_scope"] = _to_text(
                normalized.get("generation_scope")
            ) or "local_subgraph"
            answer_scope = _to_text(normalized.get("answer_scope"))
            normalized["answer_scope"] = answer_scope if answer_scope else None
            global_verifier = _to_text(normalized.get("global_verifier"))
            normalized["global_verifier"] = global_verifier if global_verifier else None
            expected_global = _to_text(normalized.get("expected_global"))
            normalized["expected_global"] = expected_global if expected_global else None
            normalized["intent_set"] = normalize_intent_set(
                normalized.get("intent_set"),
                normalized["task_subtype"],
                subtype_id=normalized["task_subtype_id"],
                level=normalized["level"],
            )

            if not normalized["task"] or not normalized["verifier"]:
                return

            try:
                parsed_row = Row.model_validate(normalized)
                valid_pairs.append(normalize_row_protocol(parsed_row))
            except Exception as e:
                if self._debug_enabled():
                    row_preview = self._truncate(json.dumps(normalized, ensure_ascii=False))
                    self._debug_print(f"skip invalid row: {e}; row={row_preview}")
                else:
                    print(f"[SamplingDatasetGenerator][extract_pairs] skip invalid row: {e}")
                return

        for block in _iter_json_blocks(text):
            block_pairs_before = len(valid_pairs)
            try:
                parsed = json.loads(block)
            except Exception:
                # Common LLM pattern: JSON objects separated by newlines (JSONL-ish).
                for line in block.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        _maybe_add_obj(item)

                if len(valid_pairs) > block_pairs_before:
                    break
                continue

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        _maybe_add_obj(item)
            elif isinstance(parsed, dict):
                _maybe_add_obj(parsed)

            # If this block produced any pairs, treat it as the intended payload and stop.
            if len(valid_pairs) > block_pairs_before:
                break

        if len(valid_pairs) == 0:
            print(f"[Warning]generate 0 qa pair, input={text}")  
        return valid_pairs

    async def identify_strategy(self, task_desc: str) -> GENERATOR_STRATEGY:
        """Identify generation strategy (query-only).

        Query-only is a hard constraint for this refactor iteration.
        See extra_doc/refactor_plan_to_new_arch.md (Milestone A).
        """
        if self.strategy is None:
            return "query"

        if self.strategy != "query":
            print(
                f"[SamplingDatasetGenerator] strategy={self.strategy} is not supported; "
                "downgrade to 'query'"
            )
        return "query"

    async def generate_pairs(
        self,
        task_type: TASK_TYPES,
        task_types_info: GraphTaskTypesInfo,
        subgraph: str,
        task_description: str,
        nums: int,
    ) -> list[Row]:
        """Generate TV pairs from a subgraph using the LLM."""

        # prompt selection and construction
        if task_type != "query":
            raise ValueError(f"Unsupported task_type={task_type} in query-only mode")
        prompt = self._build_generate_pairs_prompt(
            task_types_info=task_types_info,
            subgraph=subgraph,
            task_description=task_description,
            nums=nums,
        )

        job_id = "generate_pairs_job"
        message = ModelMessage(
            payload=prompt,
            source_type=MessageSourceType.MODEL,
            job_id=job_id,
            step=1,
        )

        # generate response
        response = await self._llm.generate(sys_prompt="", messages=[message])
        raw_payload = response.get_payload()
        self._debug_print(
            "generate_pairs raw response:\n" + self._truncate(raw_payload)
        )

        # extract pairs from response
        qas: list[Row] = self.extract_pairs(task_type, raw_payload)
        self._debug_print(f"generate_pairs parsed rows={len(qas)}")
        return qas

    def get_task_type_from_strategy(self, strategy: GENERATOR_STRATEGY) -> TASK_TYPES:
        """Get a task type based on the generation strategy.

        In a single LLM data synthesis, a specific task type is needed.
        We will return a specific task type based on the generation strategy.
        If it is a non-mixed type, return the corresponding type directly.
        If it is a mixed type, a specific type will be randomly returned.
        """
        if strategy is None:
            raise ValueError("strategy is None")
        if strategy != "query":
            print(
                f"[SamplingDatasetGenerator] strategy={strategy} is not supported; "
                "downgrade to 'query'"
            )
        return "query"

    async def filter(
        self,
        task_type: TASK_TYPES,
        task_desc: str,
        subgraph: str,
        dataset: list[Row],
        *,
        attempt: int | None = None,
    ) -> tuple[list[Row], list[tuple[Row, str]]]:
        """Filter generated TV pairs using the LLM."""

        # prompt construction
        prompt = self._build_filter_prompt(task_desc=task_desc, subgraph=subgraph, dataset=dataset)
        job_id = "filter_job"
        message = ModelMessage(
            payload=prompt,
            source_type=MessageSourceType.MODEL,
            job_id=job_id,
            step=1,
        )

        batch_attempt = attempt if attempt is not None else -1
        raw_rows = [normalize_row_protocol(row) for row in dataset]
        # Static candidate metrics only; avoid executing raw verifier candidates here.
        self._qa_candidate_count += len(raw_rows)
        for row in raw_rows:
            if row.answer_scope == "global_graph":
                if row.global_verifier and looks_like_query(row.global_verifier):
                    self._qa_scope_aligned_count += 1
                    self._qa_global_executable_count += 1
            else:
                self._qa_scope_aligned_count += 1
            if intent_verifier_alignment_ok(row):
                self._qa_intent_aligned_count += 1
            if row.answer_scope == "local_subgraph" or row.expected_global:
                self._qa_local_global_consistent_count += 1
        raw_candidates = [
            (self._candidate_id(batch_attempt, index, row), row) for index, row in enumerate(raw_rows)
        ]
        self._append_raw_candidates(candidates=raw_candidates, attempt=batch_attempt)

        # generate response
        llm_filter_failed = False
        try:
            response = await self._llm.generate(sys_prompt="", messages=[message])
            raw_payload = response.get_payload()
            self._debug_print("filter raw response:\n" + self._truncate(raw_payload))
            # extract pairs from response
            qas: list[Row] = self.extract_pairs(task_type, raw_payload)
        except Exception as e:
            llm_filter_failed = True
            print(
                "[SamplingDatasetGenerator][filter] fallback to local qa gate due llm error, "
                f"attempt={batch_attempt}, reason={e}"
            )
            qas = [normalize_row_protocol(row.model_copy(deep=True)) for row in raw_rows]
        fp_to_indices: dict[str, list[int]] = {}
        for index, row in enumerate(raw_rows):
            fp = self._task_fingerprint(row.task)
            fp_to_indices.setdefault(fp, []).append(index)

        qas_index_map: dict[int, int] = {}
        for q_idx, row in enumerate(qas):
            fp = self._task_fingerprint(row.task)
            queue = fp_to_indices.get(fp, [])
            if queue:
                qas_index_map[q_idx] = queue.pop(0)

        llm_filter_drops: list[tuple[int, Row, str]] = []
        for left_indices in fp_to_indices.values():
            for index in left_indices:
                llm_filter_drops.append((index, raw_rows[index], "llm_filter_drop"))

        qas = self._hydrate_rows_for_global_truth(qas, subgraph=subgraph)
        accepted, _, rejected_rows = self._qa_gate_filter(qas)

        accepted_with_index: list[tuple[int, Row]] = []
        accepted_extra: list[Row] = []
        for q_idx, row in enumerate(qas):
            raw_index = qas_index_map.get(q_idx)
            if raw_index is None:
                if any(id(row) == id(acc) for acc in accepted):
                    accepted_extra.append(row)
                continue
            if any(id(row) == id(acc) for acc in accepted):
                accepted_with_index.append((raw_index, row))

        rejected_with_index: list[tuple[int, Row, str]] = []
        rejected_extra: list[tuple[Row, str]] = []
        for row, reason in rejected_rows:
            matched_index = None
            for q_idx, raw_index in qas_index_map.items():
                if id(qas[q_idx]) == id(row):
                    matched_index = raw_index
                    break
            if matched_index is not None:
                rejected_with_index.append((matched_index, row, reason))
            else:
                rejected_extra.append((row, reason))

        all_rejected = [*llm_filter_drops, *rejected_with_index]
        salvaged, unresolved, salvage_logs = self._salvage_soft_rejects(
            rejected_rows=all_rejected,
            subgraph=subgraph,
            attempt=batch_attempt,
        )

        final_accepted_rows = [row for _, row in accepted_with_index]
        final_accepted_rows.extend([row for _, row in salvaged])
        final_accepted_rows.extend(accepted_extra)
        self._qa_accept_count += len(final_accepted_rows)

        unresolved_pairs: list[tuple[Row, str]] = [(row, reason) for _, row, reason in unresolved]
        unresolved_pairs.extend(rejected_extra)
        final_reject_stats: dict[str, int] = {}
        for _, _, reason in unresolved:
            final_reject_stats[reason] = final_reject_stats.get(reason, 0) + 1
        for _, reason in rejected_extra:
            final_reject_stats[reason] = final_reject_stats.get(reason, 0) + 1
        for reason, value in final_reject_stats.items():
            self._qa_reject_stats[reason] = self._qa_reject_stats.get(reason, 0) + value
        if llm_filter_failed:
            self._qa_reject_stats["llm_filter_error"] = self._qa_reject_stats.get("llm_filter_error", 0) + 1

        decisions_by_index: dict[int, dict[str, Any]] = {
            index: {
                "status": "accepted",
                "reason": None,
                "rewritten": False,
                "final_row": row.model_dump(),
            }
            for index, row in accepted_with_index
        }
        for index, row in salvaged:
            decisions_by_index[index] = {
                "status": "salvaged",
                "reason": "soft_reject_rewritten_or_revalidated",
                "rewritten": True,
                "final_row": row.model_dump(),
            }
        for index, row, reason in unresolved:
            decisions_by_index[index] = {
                "status": "rejected",
                "reason": reason,
                "rewritten": False,
                "final_row": normalize_row_protocol(row).model_dump(),
            }

        decision_records: list[dict[str, Any]] = []
        updated_at = datetime.now().isoformat(timespec="seconds")
        for index, (candidate_id, row) in enumerate(raw_candidates):
            decision = decisions_by_index.get(
                index,
                {
                    "status": "rejected",
                    "reason": "unmapped_candidate",
                    "rewritten": False,
                    "final_row": normalize_row_protocol(row).model_dump(),
                },
            )
            decision_records.append(
                {
                    "updated_at": updated_at,
                    "attempt": batch_attempt,
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "status": decision["status"],
                    "reason": decision["reason"],
                    "rewritten": decision["rewritten"],
                    "row_before": normalize_row_protocol(row).model_dump(),
                    "row_after": decision["final_row"],
                }
            )
        self._append_decision_rows(decisions=decision_records)
        self._append_salvaged_rows(salvaged_rows=salvage_logs)

        self._debug_print(
            f"filter parsed rows={len(qas)}, qa_gate accepted={len(final_accepted_rows)}, "
            f"rejected={sum(final_reject_stats.values())}, reject_stats={final_reject_stats}, "
            f"salvaged={len(salvaged)}"
        )
        return final_accepted_rows, unresolved_pairs

    async def generate(self, task_desc: str, dataset_name: str, size: int) -> WorkflowTrainDataset:
        """Generate a dataset based on the task description and desired size."""

        # initialize
        dataset: list[Row] = []
        total = 0
        self._qa_reject_stats = {}
        self._qa_accept_count = 0
        self._qa_candidate_count = 0
        self._qa_global_executable_count = 0
        self._qa_scope_aligned_count = 0
        self._qa_intent_aligned_count = 0
        self._qa_local_global_consistent_count = 0
        self._progress_reject_stats = {}
        self._progress_reject_written_rows = 0
        self._progress_candidate_written_rows = 0
        self._progress_decision_written_rows = 0
        self._progress_salvaged_written_rows = 0
        self._salvage_attempt_count = 0
        self._salvage_success_count = 0
        self._task_fingerprint_set = set()
        if self._progress_output_dir is not None:
            for filename in (
                "progress_dataset.jsonl",
                "progress_stats.json",
                "progress_events.jsonl",
                "progress_rejections.jsonl",
                "progress_raw_candidates.jsonl",
                "progress_decisions.jsonl",
                "progress_salvaged.jsonl",
            ):
                path = self._progress_output_dir / filename
                if path.exists():
                    path.unlink()
            self._progress_written_rows = 0
        max_times = (
            size // self.nums_per_subgraph + 20
        )  # max generation attempts to avoid infinite loops
        times = 0
        subgraph_getter: SubGraphSampler = self.sampler
        strategy: GENERATOR_STRATEGY = await self.identify_strategy(task_desc)

        if strategy is None:
            raise Exception(f"Cann't indentify strategy from task description={task_desc}")

        task_types_info = GraphTaskTypesInfo(strategy=strategy)
        self._persist_progress(
            dataset=dataset,
            task_types_info=task_types_info,
            attempt=0,
            max_attempts=max_times,
            target_size=size,
            status="started",
            accepted_in_attempt=0,
        )

        # generation loop
        while total < size and times < max_times:
            intent_plan = self._build_intent_plan(
                task_desc=task_desc,
                task_types_info=task_types_info,
                remaining=size - total,
            )
            required_intents = intent_plan.get("core_intents", ["query.lookup"])
            # try to get a random subgraph from the graph database
            times += 1
            try:
                if hasattr(subgraph_getter, "get_targeted_subgraph"):
                    subgraph = subgraph_getter.get_targeted_subgraph(
                        self.graph_db,
                        max_depth=self.max_depth,
                        max_nodes=self.max_nodes,
                        max_edges=self.max_edges,
                        required_intents=required_intents,
                    )
                else:
                    subgraph = subgraph_getter.get_random_subgraph(
                        self.graph_db,
                        max_depth=self.max_depth,
                        max_nodes=self.max_nodes,
                        max_edges=self.max_edges,
                    )
                if subgraph == "":
                    raise Exception("get a empty subgraph")
            except Exception as e:
                print(
                    "[SamplingDatasetGenerator][generate] except while "
                    f"get_random_subgraph, reason={e}"
                )
                self._persist_progress(
                    dataset=dataset,
                    task_types_info=task_types_info,
                    attempt=times,
                    max_attempts=max_times,
                    target_size=size,
                    status="subgraph_error",
                    accepted_in_attempt=0,
                )
                continue

            nums = min(self.nums_per_subgraph, size - total)  # number of pairs to generate
            task_type = self.get_task_type_from_strategy(
                strategy=strategy
            )  # get a specific task type  # noqa: E501

            # try to generate pairs from the subgraph
            try:
                pairs = await self.generate_pairs(
                    task_type=task_type,
                    task_types_info=task_types_info,
                    subgraph=subgraph,
                    task_description=task_desc,
                    nums=nums,
                )
            except Exception as e:
                print(
                    f"[SamplingDatasetGenerator][generate] except while generate_pairs, reason={e}"
                )
                self._persist_progress(
                    dataset=dataset,
                    task_types_info=task_types_info,
                    attempt=times,
                    max_attempts=max_times,
                    target_size=size,
                    status="generate_pairs_error",
                    accepted_in_attempt=0,
                )
                continue

            # filter the generated pairs
            try:
                pairs, rejected_rows = await self.filter(
                    task_type=task_type,
                    task_desc=task_desc,
                    subgraph=subgraph,
                    dataset=pairs,
                    attempt=times,
                )
            except Exception as e:
                print(
                    "[SamplingDatasetGenerator][generate] except while filter, "
                    f"reason={e}"
                )
                self._persist_progress(
                    dataset=dataset,
                    task_types_info=task_types_info,
                    attempt=times,
                    max_attempts=max_times,
                    target_size=size,
                    status="filter_error",
                    accepted_in_attempt=0,
                )
                continue
            self._append_rejected_rows(
                rejected_rows=rejected_rows,
                attempt=times,
                stage="qa_gate",
            )
            if self.enable_task_dedup:
                pairs, dedup_rejected, dedup_rejected_rows = self._deduplicate_tasks(
                    pairs=pairs,
                    existing_rows=dataset,
                )
                if dedup_rejected > 0:
                    self._qa_reject_stats["near_duplicate_task"] = (
                        self._qa_reject_stats.get("near_duplicate_task", 0) + dedup_rejected
                    )
                    self._append_rejected_rows(
                        rejected_rows=dedup_rejected_rows,
                        attempt=times,
                        stage="dedup",
                    )
            pairs = self._prioritize_pairs_for_coverage(
                pairs=pairs,
                task_types_info=task_types_info,
                remaining=size - total,
            )

            if len(pairs) == 0:
                if self._debug_enabled():
                    print(
                        "[SamplingDatasetGenerator][generate] 0 valid pairs after filter, "
                        f"attempt={times}/{max_times}, total={total}/{size}, "
                        f"subgraph={self._truncate(subgraph)}"
                    )
                else:
                    print(
                        "[SamplingDatasetGenerator][generate] 0 valid pairs after filter, "
                        f"attempt={times}/{max_times}, total={total}/{size}, "
                        f"subgraph_brief={self._subgraph_brief(subgraph)}"
                    )
                if hasattr(subgraph_getter, "register_acceptance"):
                    subgraph_getter.register_acceptance(required_intents, accepted_rows_count=0)
                self._persist_progress(
                    dataset=dataset,
                    task_types_info=task_types_info,
                    attempt=times,
                    max_attempts=max_times,
                    target_size=size,
                    status="attempt_empty",
                    accepted_in_attempt=0,
                )
                continue

            remaining = size - total
            if len(pairs) > remaining:
                pairs = pairs[:remaining]

            # update task types statistics info and dataset
            task_types_info.update(pairs)
            dataset.extend(pairs)
            total += len(pairs)
            print(
                "[SamplingDatasetGenerator][generate] accepted "
                f"{len(pairs)} rows, attempt={times}/{max_times}, total={total}/{size}"
            )
            if hasattr(subgraph_getter, "register_acceptance"):
                subgraph_getter.register_acceptance(
                    required_intents,
                    accepted_rows_count=len(pairs),
                )
            self._persist_progress(
                dataset=dataset,
                task_types_info=task_types_info,
                attempt=times,
                max_attempts=max_times,
                target_size=size,
                status="attempt_accepted",
                accepted_in_attempt=len(pairs),
            )
            # time.sleep(2)  # speed control

        # create final dataset object
        candidates = max(1, self._qa_candidate_count)
        workflow_dataset = WorkflowTrainDataset(
            name=dataset_name,
            task_desc=task_desc,
            data=dataset,
            qa_gate_stats={
                "accepted_rows": self._qa_accept_count,
                "rejected_rows": sum(self._qa_reject_stats.values()),
                "global_executable_rate_pct": round(
                    100.0 * self._qa_global_executable_count / candidates, 4
                ),
                "scope_alignment_rate_pct": round(
                    100.0 * self._qa_scope_aligned_count / candidates, 4
                ),
                "intent_verifier_alignment_rate_pct": round(
                    100.0 * self._qa_intent_aligned_count / candidates, 4
                ),
                "local_global_consistency_rate_pct": round(
                    100.0 * self._qa_local_global_consistent_count / candidates, 4
                ),
                "data_retention_rate_pct": round(100.0 * self._qa_accept_count / candidates, 4),
                **{f"reject::{k}": v for k, v in self._qa_reject_stats.items()},
            },
            sampling_stats=(
                subgraph_getter.get_sampling_metrics()
                if hasattr(subgraph_getter, "get_sampling_metrics")
                else {}
            ),
        )

        print(task_types_info.get_count_info())
        self._persist_progress(
            dataset=dataset,
            task_types_info=task_types_info,
            attempt=times,
            max_attempts=max_times,
            target_size=size,
            status="completed",
            accepted_in_attempt=0,
        )
        return workflow_dataset
