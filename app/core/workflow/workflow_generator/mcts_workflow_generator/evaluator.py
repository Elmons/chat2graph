from abc import abstractmethod
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from app.core.common.system_env import SystemEnv
from app.core.common.type import MessageSourceType
from app.core.common.util import parse_jsons
from app.core.model.message import ModelMessage
from app.core.prompt.workflow_generator import eval_prompt_template, reflect_prompt_template
from app.core.reasoner.model_service_factory import ModelService, ModelServiceFactory
from app.core.workflow.dataset_synthesis.model import Row
from app.core.workflow.evaluation.openai_batch_file import run_batch_chat
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    ExecuteResult,
    ReflectResult,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.runner import WorkflowRunner
from app.core.workflow.workflow_generator.mcts_workflow_generator.utils import (
    JsonValue,
    generate_json,
    load_execute_result,
)


class Evaluator:
    """Evaluate workflow revisions and return quality metrics."""

    @abstractmethod
    async def evaluate_workflow(
        self,
        optimized_path: str,
        round_num: int,
        parent_round: int,
        dataset: List[Row],
        modifications: List[str],
    ) -> Tuple[float, str]:
        """Evaluate the workflow and return its score together with reflection text."""

    
class LLMEvaluator(Evaluator):
    """Leverage LLMs to score workflow executions and reflect on outcomes."""

    def __init__(self, need_reflect: bool = True, main_expert_name: Optional[str] = None):
        """Initialise the evaluator with a backing model service."""
        super().__init__()
        self._llm: ModelService = ModelServiceFactory.create(
            model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
        )
        self.job_id = "[LLMEvaluator]"
        self.need_reflect = need_reflect
        self.main_expert_name = main_expert_name
        self._scoring_batch_size = max(1, int(SystemEnv.LLM_SCORING_BATCH_SIZE))

    async def evaluate_workflow(
        self,
        optimized_path: str,
        round_num: int,
        parent_round: int,
        dataset: List[Row],
        modifications: List[str],
    ) -> Tuple[float, str]:
        """Execute a workflow against a dataset and compute an aggregate score.
        
        Core Idea: 
        1. Batch execution with agentic system. Use LLm to score each execution result.
        2. Aggregate scores and execution results to reflect on the workflow.
        
        params:
            optimized_path: The directory where the optimized workflow configurations 'yaml' files are stored.
            round_num: The current round number of the workflow generation.
            parent_round: The parent round number to compare against.
            dataset: The dataset to evaluate the workflow against.
            modifications: Any modifications made to the workflow in this round.
        
        returns:
            A tuple containing the average score across the dataset and a JSON string of the reflection result.
        """  # noqa: E501
        
        # Prepare storage for results
        save_dir = Path(optimized_path) / f"round{round_num}"
        save_dir.mkdir(parents=True, exist_ok=True)
        results_file = save_dir / "results.json"
        
        
        # Load parent round scores if applicable
        if parent_round > 0:
            parent_dir = Path(optimized_path) / f"round{parent_round}"
            parent_results_file = parent_dir / "results.json"
            parent_scores: Dict[str, int] = {}
            if parent_results_file.exists():
                try:
                    parent_results = load_execute_result(parent_results_file)
                    parent_scores = {result.task: result.score for result in parent_results}
                except Exception:
                    parent_scores = {}
        else:
            parent_scores = {}

        # initialize total score and results list
        total_score = 0.0
        results: List[ExecuteResult] = []

        try:
            runner = WorkflowRunner(
                main_expert_name=self.main_expert_name,
                batch_size=5,
                suppress_stdout=True,
            )
            run_records = await runner.run_dataset(
                workflow_path=Path(optimized_path) / f"round{round_num}" / "workflow.yml",
                rows=dataset,
                reset_state=True,
            )
        except Exception as e:
            for data in dataset:
                parent_score = parent_scores.get(data.task, -1)
                results.append(
                    ExecuteResult(
                        task=data.task,
                        verifier=data.verifier,
                        model_output="",
                        ori_score=parent_score,
                        score=-1,
                        error=(
                            "workflow execution bootstrap failed, "
                            f"reason={e}"
                        ),
                        succeed="no",
                        error_type="bootstrap_error",
                    )
                )
            avg_score = -1.0
        else:
            score_by_index: Dict[int, int] = {}
            scorable_records = [
                (idx, record) for idx, record in enumerate(run_records) if not record.error
            ]
            for start in range(0, len(scorable_records), self._scoring_batch_size):
                batch = scorable_records[start : start + self._scoring_batch_size]
                batch_payload = [
                    {
                        "question": record.task,
                        "expected_answer": record.verifier,
                        "model_output": record.model_output,
                    }
                    for _, record in batch
                ]
                batch_scores = await self._llm_batch_scoring(batch_payload)
                for i, (idx, _) in enumerate(batch):
                    score_by_index[idx] = batch_scores[i] if i < len(batch_scores) else 0

            for idx, record in enumerate(run_records):
                parent_score = parent_scores.get(record.task, -1)
                try:
                    if record.error:
                        raise RuntimeError(record.error)
                    score = score_by_index.get(idx, 0)
                    total_score += score
                    succeed: Literal["yes", "no", "unknown"]
                    if parent_score < 0:
                        succeed = "unknown"
                    elif score > parent_score:
                        succeed = "yes"
                    else:
                        succeed = "no"
                    results.append(
                        ExecuteResult(
                            task=record.task,
                            verifier=record.verifier,
                            model_output=record.model_output,
                            ori_score=parent_score,
                            score=score,
                            error="",
                            succeed=succeed,
                            latency_ms=record.latency_ms,
                            token_usage={"total": record.tokens},
                            error_type=None,
                        )
                    )
                except Exception as e:
                    results.append(
                        ExecuteResult(
                            task=record.task,
                            verifier=record.verifier,
                            model_output=record.model_output,
                            ori_score=parent_score,
                            score=0,
                            error=f"{e}",
                            succeed="no",
                            latency_ms=record.latency_ms,
                            token_usage={"total": record.tokens},
                            error_type=type(e).__name__,
                        )
                    )

            avg_score = total_score / max(len(dataset), 1)

        # Save results to disk
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump([result.model_dump() for result in results], f, ensure_ascii=False, indent=2)

        # Generate reflection if needed
        if self.need_reflect:
            reflect_result = await self._reflect(
                modifications=modifications, results=results, avg_score=avg_score
            )
        else:
            reflect_result = ReflectResult(
                failed_reason=[],
                optimize_suggestion=[]
            )

        return avg_score, reflect_result.model_dump_json(indent=2)

    async def _pack_infer_trace(
        self, modifications: List[str], results: List[Dict[str, str]], avg_score: float
    ):
        """construct the infer trace for LLM. """
        #TODO: implement this method

    async def _reflect(
        self, modifications: List[str], results: List[ExecuteResult], avg_score: float
    ) -> ReflectResult:
        """Ask the LLM to reflect on the current optimisation step."""
        prompt = reflect_prompt_template.format(
            modification=modifications,
            results=json.dumps(
                [result.model_dump() for result in results], ensure_ascii=False, indent=2
            ),
        )
        messages = [
            ModelMessage(
                payload=prompt,
                job_id=self.job_id,
                step=1,
                source_type=MessageSourceType.MODEL,
            )
        ]

        # Define a filter to validate the LLM's output
        # The output must contain all fields defined in ReflectResult
        def filter(results: List[JsonValue]) -> JsonValue:
            fields = ReflectResult.model_fields.keys()
            for result in results:
                if isinstance(result, dict):
                    for field in fields:
                        if field not in result:
                            raise Exception(f"missing {field} in {result}")
                    return result
                else:
                    raise Exception("output must be a dict")
            return None

        resp = await generate_json(
            model=self._llm,
            sys_prompt="",
            messages=messages,
            filter=filter,
        )
        return ReflectResult.model_validate(resp)

    async def _llm_scoring(self, question: str, model_output: str, expected_answer: str) -> int:
        """Score a single workflow execution result via an LLM rubric."""
        prompt = eval_prompt_template.format(
            question=question, expected_answer=expected_answer, model_output=model_output
        )
        messages = [
            ModelMessage(
                payload=prompt,
                job_id=self.job_id,
                step=1,
                source_type=MessageSourceType.MODEL,
            )
        ]

        # Define a filter to extract the score from the LLM's output
        # The output must contain a 'score' field
        def filter(results: List[JsonValue]) -> JsonValue:
            for result in results:
                if isinstance(result, dict):
                    if "score" not in result:
                        raise Exception(f"missing score field in {result}")
                    return result["score"]
                else:
                    raise Exception("output must be a json dict")

            return None

        resp = await generate_json(
            model=self._llm,
            sys_prompt="",
            messages=messages,
            filter=filter,
        )

        if isinstance(resp, int):
            return resp
        else:
            return 0

    async def _llm_batch_scoring(self, batch_items: List[Dict[str, str]]) -> List[int]:
        """Score multiple execution results in one LLM request.

        Returns one integer score (0-3) per input item, preserving input order.
        """
        if not batch_items:
            return []
        # Keep scoring prompt unchanged; each item uses eval_prompt_template.
        prompts = [
            eval_prompt_template.format(
                question=item["question"],
                expected_answer=item["expected_answer"],
                model_output=item["model_output"],
            )
            for item in batch_items
        ]

        if SystemEnv.LLM_USE_OPENAI_BATCH_FILE and len(batch_items) > 1:
            try:
                batch_results = await run_batch_chat(
                    prompts=prompts,
                    model=SystemEnv.LLM_NAME,
                    api_base=SystemEnv.LLM_ENDPOINT,
                    api_key=SystemEnv.LLM_APIKEY,
                )
                scores: List[int] = []
                for result in batch_results:
                    if result.error:
                        scores.append(0)
                    else:
                        scores.append(self._parse_score_from_payload(result.content))
                return scores
            except Exception:
                # fallback to sequential scoring to keep MCTS round progressing
                pass

        scores: List[int] = []
        for item in batch_items:
            score = await self._llm_scoring(
                question=item["question"],
                model_output=item["model_output"],
                expected_answer=item["expected_answer"],
            )
            scores.append(score)
        return scores

    def _parse_score_from_payload(self, payload: str) -> int:
        """Parse score from evaluator payload, fallback to 0 when malformed."""
        parsed = parse_jsons(payload)
        for item in parsed:
            if isinstance(item, dict) and "score" in item:
                score = item.get("score")
                if isinstance(score, int):
                    return max(0, min(3, score))
                if isinstance(score, float):
                    return max(0, min(3, int(score)))
        return 0
