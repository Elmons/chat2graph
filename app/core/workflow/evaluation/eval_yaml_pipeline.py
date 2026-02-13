from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

from app.core.common.logger import Chat2GraphLogger
from app.core.common.system_env import SystemEnv
from app.core.common.type import GraphDbType
from app.core.model.graph_db_config import GraphDbConfig
from app.core.prompt.workflow_generator import eval_prompt_template
from app.core.reasoner.model_service_factory import ModelService, ModelServiceFactory
from app.core.workflow.dataset_synthesis.model import WorkflowTrainDataset
from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import ExecuteResult
from app.core.workflow.workflow_generator.mcts_workflow_generator.runner import WorkflowRunner

logger = Chat2GraphLogger.get_logger(__name__)

ScoreMode = Literal["exact", "llm"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _score_exact(*, expected_answer: str, model_output: str) -> int:
    if _normalize_text(expected_answer) == _normalize_text(model_output):
        return 3
    return 0


async def _score_llm(
    *,
    model: ModelService,
    question: str,
    expected_answer: str,
    model_output: str,
) -> int:
    from app.core.common.type import MessageSourceType
    from app.core.common.util import parse_jsons
    from app.core.model.message import ModelMessage

    prompt = eval_prompt_template.format(
        question=question,
        expected_answer=expected_answer,
        model_output=model_output,
    )
    messages = [
        ModelMessage(
            payload=prompt,
            job_id="[eval_yaml_pipeline]",
            step=1,
            source_type=MessageSourceType.MODEL,
        )
    ]

    resp = await model.generate(sys_prompt="", messages=messages)
    parsed = parse_jsons(resp.get_payload())
    for item in parsed:
        if isinstance(item, dict) and "score" in item:
            score = item.get("score")
            if isinstance(score, int):
                return score
            if isinstance(score, float):
                return int(score)
    return 0


@dataclass(frozen=True)
class YamlEvalSummary:
    yaml_path: str
    dataset_name: str
    num_rows: int
    avg_score: float
    score_mode: ScoreMode
    results_path: str
    summary_path: str
    run_meta_path: str
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    error: str = ""


def _error_type(error: str) -> str:
    cleaned = (error or "").strip()
    if not cleaned:
        return "none"
    lowered = cleaned.lower()
    if "timeout" in lowered:
        return "timeout"
    if "load" in lowered:
        return "load_failed"
    if "tool" in lowered:
        return "tool_error"
    return "runtime_error"


def _graph_db_config_from_dict(config_dict: Dict[str, Any]) -> GraphDbConfig:
    raw_type = str(config_dict.get("type", "NEO4J")).upper()
    graph_type = GraphDbType(raw_type)
    return GraphDbConfig(
        type=graph_type,
        name=str(config_dict.get("name", "eval_graph")),
        host=str(config_dict.get("host", "localhost")),
        port=int(config_dict.get("port", 7687)),
        user=str(config_dict.get("user", "neo4j")) if config_dict.get("user") else None,
        pwd=str(config_dict.get("pwd", "")) if config_dict.get("pwd") else None,
        default_schema=(
            str(config_dict.get("default_schema"))
            if config_dict.get("default_schema") is not None
            else None
        ),
    )


def _serialize_graph_db_config(config: Optional[GraphDbConfig]) -> Dict[str, Any]:
    if config is None:
        return {}
    return config.to_dict()


async def evaluate_yaml(
    *,
    dataset: WorkflowTrainDataset,
    yaml_path: str | Path,
    out_dir: str | Path,
    graph_db_config: Optional[GraphDbConfig | Dict[str, Any]] = None,
    main_expert_name: Optional[str] = None,
    score_mode: ScoreMode = "exact",
) -> YamlEvalSummary:
    """Evaluate a single Agentic YAML against a dataset."""
    yaml_path = Path(yaml_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    summary_path = out_dir / "summary.json"
    run_meta_path = out_dir / "run_meta.json"

    started_at = time.time()
    started_at_iso = _utc_now_iso()

    normalized_graph_db: Optional[GraphDbConfig]
    if isinstance(graph_db_config, dict):
        normalized_graph_db = _graph_db_config_from_dict(graph_db_config)
    else:
        normalized_graph_db = graph_db_config

    try:
        scorer_llm: Optional[ModelService] = None
        if score_mode == "llm":
            scorer_llm = ModelServiceFactory.create(
                model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
            )

        runner = WorkflowRunner(main_expert_name=main_expert_name, batch_size=5, suppress_stdout=True)
        run_records = await runner.run_dataset(
            workflow_path=yaml_path,
            rows=dataset.data,
            graph_db_config=normalized_graph_db,
            reset_state=True,
        )

        results: List[ExecuteResult] = []
        total_score = 0.0
        total_latency_ms = 0.0
        total_tokens = 0
        success_count = 0
        error_breakdown: Dict[str, int] = {}

        for record in run_records:
            total_latency_ms += record.latency_ms
            total_tokens += int(record.tokens)
            score = 0
            error = record.error
            if not error:
                if score_mode == "llm":
                    assert scorer_llm is not None
                    score = await _score_llm(
                        model=scorer_llm,
                        question=record.task,
                        expected_answer=record.verifier,
                        model_output=record.model_output,
                    )
                else:
                    score = _score_exact(
                        expected_answer=record.verifier,
                        model_output=record.model_output,
                    )
                success_count += 1
            total_score += score

            err_type = _error_type(error)
            error_breakdown[err_type] = error_breakdown.get(err_type, 0) + (1 if error else 0)

            results.append(
                ExecuteResult(
                    task=record.task,
                    verifier=record.verifier,
                    model_output=record.model_output,
                    ori_score=-1,
                    score=score,
                    error=error,
                    succeed="unknown" if not error else "no",
                    latency_ms=record.latency_ms,
                    token_usage={"total": int(record.tokens)},
                    error_type=(None if not error else err_type),
                )
            )

        rows_cnt = max(len(dataset.data), 1)
        avg_score = total_score / rows_cnt
        avg_latency_ms = total_latency_ms / rows_cnt
        success_rate = success_count / rows_cnt

        with results_path.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump(mode="json") for r in results], f, ensure_ascii=False, indent=2)

        summary = YamlEvalSummary(
            yaml_path=str(yaml_path),
            dataset_name=dataset.name,
            num_rows=len(dataset.data),
            avg_score=avg_score,
            score_mode=score_mode,
            results_path=str(results_path),
            summary_path=str(summary_path),
            run_meta_path=str(run_meta_path),
            success_rate=success_rate,
            avg_latency_ms=avg_latency_ms,
            total_tokens=total_tokens,
            error_breakdown=error_breakdown,
            error="",
        )
    except Exception as e:
        summary = YamlEvalSummary(
            yaml_path=str(yaml_path),
            dataset_name=dataset.name,
            num_rows=len(dataset.data),
            avg_score=-1.0,
            score_mode=score_mode,
            results_path=str(results_path),
            summary_path=str(summary_path),
            run_meta_path=str(run_meta_path),
            success_rate=0.0,
            avg_latency_ms=0.0,
            total_tokens=0,
            error_breakdown={},
            error=str(e),
        )

    finished_at = time.time()
    run_meta = {
        "yaml_path": str(yaml_path),
        "dataset_name": dataset.name,
        "num_rows": len(dataset.data),
        "score_mode": score_mode,
        "main_expert_name": main_expert_name,
        "graph_db_config": _serialize_graph_db_config(normalized_graph_db),
        "started_at": started_at_iso,
        "finished_at": _utc_now_iso(),
        "duration_sec": round(finished_at - started_at, 6),
        "summary": {
            "avg_score": summary.avg_score,
            "success_rate": summary.success_rate,
            "avg_latency_ms": summary.avg_latency_ms,
            "total_tokens": summary.total_tokens,
            "error": summary.error,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    return summary


def _evaluate_yaml_subprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = WorkflowTrainDataset.model_validate(payload["dataset"])
    summary = asyncio.run(
        evaluate_yaml(
            dataset=dataset,
            yaml_path=payload["yaml_path"],
            out_dir=payload["out_dir"],
            graph_db_config=payload.get("graph_db_config"),
            main_expert_name=payload.get("main_expert_name"),
            score_mode=payload.get("score_mode", "exact"),
        )
    )
    return summary.__dict__


async def evaluate_many(
    *,
    dataset: WorkflowTrainDataset,
    yaml_paths: Sequence[str | Path],
    out_dir: str | Path,
    graph_db_config: Optional[GraphDbConfig | Dict[str, Any]] = None,
    main_expert_name: Optional[str] = None,
    score_mode: ScoreMode = "exact",
    parallelism: int = 1,
    process_isolation: bool = False,
) -> List[YamlEvalSummary]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[YamlEvalSummary] = []
    use_process_pool = process_isolation or parallelism > 1

    if not use_process_pool:
        for p in yaml_paths:
            yaml_file = Path(p)
            subdir = out_dir / yaml_file.stem
            summary = await evaluate_yaml(
                dataset=dataset,
                yaml_path=yaml_file,
                out_dir=subdir,
                graph_db_config=graph_db_config,
                main_expert_name=main_expert_name,
                score_mode=score_mode,
            )
            summaries.append(summary)
    else:
        workers = max(1, parallelism)
        payloads: List[Dict[str, Any]] = []
        graph_db_payload: Optional[Dict[str, Any]]
        if isinstance(graph_db_config, GraphDbConfig):
            graph_db_payload = graph_db_config.to_dict()
        elif isinstance(graph_db_config, dict):
            graph_db_payload = graph_db_config
        else:
            graph_db_payload = None

        dataset_payload = dataset.model_dump(mode="json")
        for p in yaml_paths:
            yaml_file = Path(p)
            payloads.append(
                {
                    "dataset": dataset_payload,
                    "yaml_path": str(yaml_file),
                    "out_dir": str(out_dir / yaml_file.stem),
                    "graph_db_config": graph_db_payload,
                    "main_expert_name": main_expert_name,
                    "score_mode": score_mode,
                }
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            rows = pool.map(_evaluate_yaml_subprocess, payloads)
        summaries = [YamlEvalSummary(**row) for row in rows]

    leaderboard = sorted(summaries, key=lambda s: s.avg_score, reverse=True)
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in leaderboard], f, ensure_ascii=False, indent=2)
    return summaries


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline evaluate Agentic YAML(s) on a dataset.")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON (list[Row]).")
    p.add_argument("--task-desc", default="offline eval", help="Dataset task description.")
    p.add_argument("--yaml", action="append", required=True, help="YAML path (repeatable).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--main-expert-name", default=None, help="Override entry expert name.")
    p.add_argument("--score", choices=["exact", "llm"], default="exact")
    p.add_argument("--ratio", type=float, default=1.0, help="Use only a ratio of dataset rows.")
    p.add_argument("--parallelism", type=int, default=1, help="Number of worker processes.")
    p.add_argument(
        "--process-isolation",
        action="store_true",
        help="Evaluate each YAML in a separate process for runtime isolation.",
    )

    p.add_argument("--graph-db-config", default=None, help="Path to graph DB config JSON.")
    p.add_argument("--graph-db-type", default=None, choices=["NEO4J", "TUGRAPH"])
    p.add_argument("--graph-db-name", default=None)
    p.add_argument("--graph-db-host", default=None)
    p.add_argument("--graph-db-port", type=int, default=None)
    p.add_argument("--graph-db-user", default=None)
    p.add_argument("--graph-db-pwd", default=None)
    p.add_argument("--graph-db-default-schema", default=None)

    return p.parse_args(list(argv) if argv is not None else None)


def _build_graph_db_config_from_args(args: argparse.Namespace) -> Optional[GraphDbConfig]:
    if args.graph_db_config:
        with Path(args.graph_db_config).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("--graph-db-config must point to a JSON object")
        return _graph_db_config_from_dict(payload)

    if args.graph_db_type and args.graph_db_name and args.graph_db_host and args.graph_db_port:
        return GraphDbConfig(
            type=GraphDbType(args.graph_db_type),
            name=args.graph_db_name,
            host=args.graph_db_host,
            port=args.graph_db_port,
            user=args.graph_db_user,
            pwd=args.graph_db_pwd,
            default_schema=args.graph_db_default_schema,
        )

    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    dataset = load_workflow_train_dataset(
        task_desc=args.task_desc,
        path=args.dataset,
        ratio=args.ratio,
    )
    graph_db_config = _build_graph_db_config_from_args(args)

    logger.info(
        "Evaluating %s YAML(s) on dataset rows=%s (score=%s, parallelism=%s, isolation=%s)",
        len(args.yaml),
        len(dataset.data),
        args.score,
        args.parallelism,
        args.process_isolation,
    )

    asyncio.run(
        evaluate_many(
            dataset=dataset,
            yaml_paths=args.yaml,
            out_dir=args.out,
            graph_db_config=graph_db_config,
            main_expert_name=args.main_expert_name,
            score_mode=args.score,
            parallelism=args.parallelism,
            process_isolation=args.process_isolation,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
