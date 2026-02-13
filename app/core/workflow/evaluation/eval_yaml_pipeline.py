from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from app.core.common.logger import Chat2GraphLogger
from app.core.common.runtime_reset import reset_runtime_state
from app.core.common.system_env import SystemEnv
from app.core.model.message import HybridMessage, TextMessage
from app.core.prompt.workflow_generator import eval_prompt_template
from app.core.reasoner.model_service_factory import ModelService, ModelServiceFactory
from app.core.sdk.agentic_service import AgenticService
from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import ExecuteResult

logger = Chat2GraphLogger.get_logger(__name__)

ScoreMode = Literal["exact", "llm"]


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
    error: str = ""


def _unwrap_payload(msg: TextMessage | HybridMessage | object) -> str:
    if isinstance(msg, HybridMessage):
        return str(msg.get_instruction_message().get_payload() or "")
    if isinstance(msg, TextMessage):
        return str(msg.get_payload() or "")
    payload = getattr(msg, "get_payload", None)
    if callable(payload):
        return str(payload() or "")
    return str(msg)


async def evaluate_yaml(
    *,
    dataset: WorkflowTrainDataset,
    yaml_path: str | Path,
    out_dir: str | Path,
    main_expert_name: Optional[str] = None,
    score_mode: ScoreMode = "exact",
) -> YamlEvalSummary:
    """Evaluate a single Agentic YAML against a dataset.

    This is a minimal, offline evaluation loop intended for:
    - Comparing hand-written YAMLs.
    - Scoring the outputs of workflow generators without running the whole MCTS loop.

    Notes:
    - The default scoring mode is `exact` so tests can run fully offline.
    - Use `score_mode="llm"` to score with an evaluator LLM (opt-in).
    """
    yaml_path = Path(yaml_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    try:
        reset_runtime_state()

        service = AgenticService.load(str(yaml_path))
        entry_expert = main_expert_name or service.entry_expert_name()
        if not entry_expert:
            raise ValueError(
                "No entry expert inferred; set main_expert_name or use single-expert YAML."
            )

        scorer_llm: Optional[ModelService] = None
        if score_mode == "llm":
            scorer_llm = ModelServiceFactory.create(
                model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
            )

        results: List[ExecuteResult] = []
        total_score = 0.0

        for row in dataset.data:
            message = TextMessage(payload=row.task, assigned_expert_name=entry_expert)
            job = service.session().submit(message)
            result_payload = ""
            try:
                model_msg = job.wait()
                result_payload = _unwrap_payload(model_msg)
                if score_mode == "llm":
                    assert scorer_llm is not None
                    score = await _score_llm(
                        model=scorer_llm,
                        question=row.task,
                        expected_answer=row.verifier,
                        model_output=result_payload,
                    )
                else:
                    score = _score_exact(expected_answer=row.verifier, model_output=result_payload)

                total_score += score
                results.append(
                    ExecuteResult(
                        task=row.task,
                        verifier=row.verifier,
                        model_output=result_payload,
                        ori_score=-1,
                        score=score,
                        error="",
                        succeed="unknown",
                    )
                )
            except Exception as e:
                results.append(
                    ExecuteResult(
                        task=row.task,
                        verifier=row.verifier,
                        model_output=result_payload,
                        ori_score=-1,
                        score=0,
                        error=str(e),
                        succeed="no",
                    )
                )

        avg_score = total_score / max(len(dataset.data), 1)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump(mode="json") for r in results], f, ensure_ascii=False, indent=2)

        summary = YamlEvalSummary(
            yaml_path=str(yaml_path),
            dataset_name=dataset.name,
            num_rows=len(dataset.data),
            avg_score=avg_score,
            score_mode=score_mode,
            results_path=str(results_path),
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
            error=str(e),
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

    return summary


async def evaluate_many(
    *,
    dataset: WorkflowTrainDataset,
    yaml_paths: Sequence[str | Path],
    out_dir: str | Path,
    main_expert_name: Optional[str] = None,
    score_mode: ScoreMode = "exact",
) -> List[YamlEvalSummary]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[YamlEvalSummary] = []
    for p in yaml_paths:
        yaml_file = Path(p)
        subdir = out_dir / yaml_file.stem
        summary = await evaluate_yaml(
            dataset=dataset,
            yaml_path=yaml_file,
            out_dir=subdir,
            main_expert_name=main_expert_name,
            score_mode=score_mode,
        )
        summaries.append(summary)

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
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    dataset = load_workflow_train_dataset(
        task_desc=args.task_desc,
        path=args.dataset,
        ratio=args.ratio,
    )

    logger.info(
        "Evaluating %s YAML(s) on dataset rows=%s (score=%s)",
        len(args.yaml),
        len(dataset.data),
        args.score,
    )

    asyncio.run(
        evaluate_many(
            dataset=dataset,
            yaml_paths=args.yaml,
            out_dir=args.out,
            main_expert_name=args.main_expert_name,
            score_mode=args.score,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
