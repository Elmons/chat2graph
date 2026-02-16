from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import contextlib
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from typing import TextIO

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.sdk.init_server import init_server
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.sampler import StratifiedHybridSampler
from app.core.workflow.dataset_synthesis.utils import normalize_row_protocol, qa_gate_reason

UTILS_PATH = Path(__file__).resolve().with_name("utils.py")
UTILS_SPEC = importlib.util.spec_from_file_location("workflow_generator_utils", UTILS_PATH)
assert UTILS_SPEC and UTILS_SPEC.loader
_utils_module = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(_utils_module)
register_and_get_graph_db = _utils_module.register_and_get_graph_db

init_server()

DEFAULT_TASK_DESC = "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等"
DEFAULT_DATASET_NAME = "query_real"
DEFAULT_SIZE = 15
DEFAULT_TIMEOUT_SECONDS = 0


class _Tee(TextIO):
    """Write stream output to multiple targets."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, s: str) -> int:  # noqa: D401
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:  # noqa: D401
        for stream in self._streams:
            stream.flush()

    def readable(self) -> bool:  # noqa: D401
        return False

    def writable(self) -> bool:  # noqa: D401
        return True

    def seekable(self) -> bool:  # noqa: D401
        return False


@dataclass
class RunConfig:
    task_desc: str
    dataset_name: str
    size: int
    timeout_seconds: int | None
    max_depth: int
    max_nodes: int
    max_edges: int
    nums_per_subgraph: int
    output_dir: Path
    run_tag: str | None = None
    log_to_file: bool = True


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Generate query-only synthesis dataset.")
    parser.add_argument("--task-desc", default=DEFAULT_TASK_DESC)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Set 0 to disable timeout protection.",
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=30)
    parser.add_argument("--nums-per-subgraph", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "generated_datasets"),
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--log-to-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redirect runtime logs to run.log while still printing to console.",
    )
    args = parser.parse_args()
    return RunConfig(
        task_desc=args.task_desc,
        dataset_name=args.dataset_name,
        size=max(1, int(args.size)),
        timeout_seconds=None if int(args.timeout_seconds) <= 0 else max(60, int(args.timeout_seconds)),
        max_depth=max(1, int(args.max_depth)),
        max_nodes=max(1, int(args.max_nodes)),
        max_edges=max(1, int(args.max_edges)),
        nums_per_subgraph=max(1, int(args.nums_per_subgraph)),
        output_dir=Path(args.output_dir),
        run_tag=args.run_tag,
        log_to_file=bool(args.log_to_file),
    )


def _prepare_run_dir(cfg: RunConfig) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{cfg.dataset_name}_{cfg.size}"
    if cfg.run_tag:
        suffix = f"{suffix}_{cfg.run_tag}"
    run_dir = cfg.output_dir / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_quality_audit(dataset, target_size: int) -> dict:
    rows = [normalize_row_protocol(row) for row in dataset.data]
    level_dist = Counter(row.level for row in rows)
    subtype_dist = Counter((row.task_subtype_id or row.task_subtype) for row in rows)
    intent_dist: Counter[str] = Counter()
    for row in rows:
        for intent in row.intent_set:
            intent_dist[intent] += 1

    qa_recheck_fail: Counter[str] = Counter()
    executable_ok = 0
    executable_errors: list[str] = []
    for row in rows:
        reason = qa_gate_reason(row, engine_hint=None)
        if reason is None:
            executable_ok += 1
        else:
            qa_recheck_fail[reason] += 1
            executable_errors.append(reason)

    structural_issues = {
        "non_global_scope_rows": sum(1 for row in rows if row.answer_scope != "global_graph"),
        "missing_global_verifier_rows": sum(1 for row in rows if not row.global_verifier),
        "missing_expected_global_rows": sum(1 for row in rows if not row.expected_global),
        "non_query_verifier_rows": sum(
            1
            for row in rows
            if row.global_verifier
            and not row.global_verifier.strip().lower().startswith(
                ("match", "optional match", "with", "call", "unwind", "return", "profile", "explain")
            )
        ),
        "qa_gate_recheck_fail": dict(qa_recheck_fail),
    }

    qa_stats = dataset.qa_gate_stats or {}
    checks = {
        f"size_is_{target_size}": len(rows) == target_size,
        "all_global_scope": structural_issues["non_global_scope_rows"] == 0,
        "all_have_global_verifier": structural_issues["missing_global_verifier_rows"] == 0,
        "all_have_expected_global": structural_issues["missing_expected_global_rows"] == 0,
        "all_verifier_query_like": structural_issues["non_query_verifier_rows"] == 0,
        "qa_gate_recheck_pass": len(qa_recheck_fail) == 0,
        "global_verifier_executable_rate_gte_98pct": float(qa_stats.get("global_executable_rate_pct", 0.0)) >= 98.0,
        "difficulty_level_coverage_gte_3": len(level_dist) >= 3,
        "subtype_unique_gte_8": len(subtype_dist) >= 8,
        "difficulty_not_collapsed_top_level_le_70pct": (max(level_dist.values()) / max(1, len(rows))) <= 0.7,
    }

    return {
        "target_size": target_size,
        "actual_size": len(rows),
        "qa_gate_stats": qa_stats,
        "sampling_stats": dataset.sampling_stats,
        "level_distribution": dict(level_dist),
        "subtype_distribution": dict(subtype_dist),
        "intent_distribution": dict(intent_dist),
        "structural_issues": structural_issues,
        "answerability": {
            "executable_ok": executable_ok,
            "executable_errors": executable_errors[:20],
            "executable_rate": executable_ok / max(1, len(rows)),
        },
        "quality_checks": checks,
        "quality_pass": all(checks.values()),
    }


def _build_reject_review(run_dir: Path, dataset_rows: list[dict]) -> dict:
    reject_path = run_dir / "progress_rejections.jsonl"
    rows: list[dict] = []
    if reject_path.exists():
        for line in reject_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    by_reason = Counter(item.get("reason", "unknown") for item in rows)
    by_stage = Counter(item.get("stage", "unknown") for item in rows)
    by_risk = Counter(item.get("risk_level", "unknown") for item in rows)

    soft_reject_with_global_truth = [
        item
        for item in rows
        if item.get("risk_level") == "soft"
        and isinstance(item.get("row"), dict)
        and item["row"].get("global_verifier")
        and item["row"].get("expected_global")
    ]

    def _looks_open_ended(task: str) -> bool:
        low = (task or "").lower()
        if any(token in low for token in (" all ", " every ")):
            return True
        if (" which " in f" {low} " or " who " in f" {low} ") and (
            " did " in f" {low} "
            or " are " in f" {low} "
            or " were " in f" {low} "
            or " has " in f" {low} "
            or " have " in f" {low} "
        ):
            has_open_set = any(
                token in low
                for token in (
                    "account",
                    "user",
                    "people",
                    "transaction",
                    "relationship",
                    "neighbor",
                )
            ) and any(token in low for token in (" to ", " from ", " with ", " by ", " through "))
            has_boundary = any(
                token in low
                for token in (
                    "top",
                    "limit",
                    "at most",
                    "exactly",
                    "latest",
                    "first",
                    "last",
                    "largest",
                    "highest",
                    "lowest",
                    "smallest",
                    "most",
                    "least",
                )
            )
            if has_open_set and not has_boundary:
                return True
        return False

    accepted_risk_candidates = [
        row for row in dataset_rows if _looks_open_ended(str(row.get("task", "")))
    ]

    return {
        "total_rejected": len(rows),
        "by_reason": dict(by_reason),
        "by_stage": dict(by_stage),
        "by_risk_level": dict(by_risk),
        "soft_reject_with_global_truth_count": len(soft_reject_with_global_truth),
        "soft_reject_with_global_truth_examples": soft_reject_with_global_truth[:10],
        "accepted_risk_candidates_count": len(accepted_risk_candidates),
        "accepted_risk_candidates_examples": accepted_risk_candidates[:10],
    }


def _build_decision_review(run_dir: Path) -> dict:
    raw_path = run_dir / "progress_raw_candidates.jsonl"
    decision_path = run_dir / "progress_decisions.jsonl"
    salvage_path = run_dir / "progress_salvaged.jsonl"

    raws: list[dict] = []
    decisions: list[dict] = []
    salvages: list[dict] = []
    for path, bucket in (
        (raw_path, raws),
        (decision_path, decisions),
        (salvage_path, salvages),
    ):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                bucket.append(json.loads(line))
            except Exception:
                continue

    by_status = Counter(item.get("status", "unknown") for item in decisions)
    by_reason = Counter((item.get("reason") or "none") for item in decisions)

    per_row = []
    for item in decisions:
        row_before = item.get("row_before") or {}
        row_after = item.get("row_after") or {}
        per_row.append(
            {
                "candidate_id": item.get("candidate_id"),
                "attempt": item.get("attempt"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "rewritten": bool(item.get("rewritten")),
                "task_before": row_before.get("task"),
                "task_after": row_after.get("task"),
                "subtype_before": row_before.get("task_subtype_id") or row_before.get("task_subtype"),
                "subtype_after": row_after.get("task_subtype_id") or row_after.get("task_subtype"),
            }
        )

    return {
        "raw_candidate_count": len(raws),
        "decision_count": len(decisions),
        "salvaged_count": len(salvages),
        "status_breakdown": dict(by_status),
        "reason_breakdown": dict(by_reason),
        "per_row": per_row,
        "salvage_examples": salvages[:20],
    }


async def _run(cfg: RunConfig, run_dir: Path) -> tuple[dict, dict, Path, Path]:
    db = register_and_get_graph_db()
    generator = SamplingDatasetGenerator(
        graph_db=db,
        sampler=StratifiedHybridSampler(),
        strategy="query",
        max_depth=cfg.max_depth,
        max_noeds=cfg.max_nodes,
        max_edges=cfg.max_edges,
        nums_per_subgraph=cfg.nums_per_subgraph,
        enable_task_dedup=True,
    )
    generator.set_progress_output(run_dir)

    generate_coro = generator.generate(
        task_desc=cfg.task_desc,
        dataset_name=f"{cfg.dataset_name}_{cfg.size}",
        size=cfg.size,
    )
    if cfg.timeout_seconds is None:
        dataset = await generate_coro
    else:
        dataset = await asyncio.wait_for(generate_coro, timeout=cfg.timeout_seconds)

    dataset_path = run_dir / "dataset.json"
    audit_path = run_dir / "quality_audit.json"
    reject_review_path = run_dir / "reject_review.json"
    decision_review_path = run_dir / "decision_review.json"
    meta_path = run_dir / "meta.json"

    dataset_payload = [row.model_dump() for row in dataset.data]
    dataset_path.write_text(json.dumps(dataset_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = _build_quality_audit(dataset, cfg.size)
    audit_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    reject_review = _build_reject_review(run_dir=run_dir, dataset_rows=dataset_payload)
    reject_review_path.write_text(
        json.dumps(reject_review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision_review = _build_decision_review(run_dir=run_dir)
    decision_review_path.write_text(
        json.dumps(decision_review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "dataset_name": f"{cfg.dataset_name}_{cfg.size}",
        "task_desc": cfg.task_desc,
        "target_size": cfg.size,
        "actual_size": len(dataset.data),
        "qa_gate_stats": dataset.qa_gate_stats,
        "sampling_stats": dataset.sampling_stats,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "audit_path": str(audit_path),
        "progress_dataset_path": str(run_dir / "progress_dataset.jsonl"),
        "progress_stats_path": str(run_dir / "progress_stats.json"),
        "progress_events_path": str(run_dir / "progress_events.jsonl"),
        "progress_rejections_path": str(run_dir / "progress_rejections.jsonl"),
        "progress_raw_candidates_path": str(run_dir / "progress_raw_candidates.jsonl"),
        "progress_decisions_path": str(run_dir / "progress_decisions.jsonl"),
        "progress_salvaged_path": str(run_dir / "progress_salvaged.jsonl"),
        "reject_review_path": str(reject_review_path),
        "decision_review_path": str(decision_review_path),
        "quality_pass": quality["quality_pass"],
        "run_log_path": str(run_dir / "run.log"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta, quality, dataset_path, audit_path


def main() -> None:
    cfg = _parse_args()
    run_dir = _prepare_run_dir(cfg)
    log_path = run_dir / "run.log"

    print(f"[dataset_generator_example] run_dir={run_dir}")
    print(
        "[dataset_generator_example] params="
        f"size={cfg.size}, timeout={cfg.timeout_seconds}, max_depth={cfg.max_depth}, "
        f"max_nodes={cfg.max_nodes}, max_edges={cfg.max_edges}, nums_per_subgraph={cfg.nums_per_subgraph}"
    )

    if cfg.log_to_file:
        with open(log_path, "w", encoding="utf-8") as log_file:
            tee = _Tee(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                meta, quality, dataset_path, audit_path = asyncio.run(_run(cfg, run_dir))
    else:
        meta, quality, dataset_path, audit_path = asyncio.run(_run(cfg, run_dir))

    print(f"[dataset_generator_example] dataset_path={dataset_path}")
    print(f"[dataset_generator_example] audit_path={audit_path}")
    print(f"[dataset_generator_example] actual_size={meta['actual_size']}")
    print(f"[dataset_generator_example] quality_pass={quality['quality_pass']}")
    print(f"[dataset_generator_example] qa_gate_stats={json.dumps(meta['qa_gate_stats'], ensure_ascii=False)}")
    if cfg.log_to_file:
        print(f"[dataset_generator_example] run_log={log_path}")


if __name__ == "__main__":
    main()
