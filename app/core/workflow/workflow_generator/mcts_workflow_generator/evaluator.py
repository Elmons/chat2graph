from abc import abstractmethod
import json
from pathlib import Path
import sys
from typing import Dict, List, Literal, Tuple

from app.core.common.system_env import SystemEnv
from app.core.common.type import MessageSourceType
from app.core.model.message import HybridMessage, ModelMessage, TextMessage
from app.core.prompt.workflow_generator import eval_prompt_template, reflect_prompt_template
from app.core.reasoner.model_service_factory import ModelService, ModelServiceFactory
from app.core.sdk.wrapper.job_wrapper import JobWrapper
from app.core.workflow.dataset_synthesis.model import Row
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    ExecuteResult,
    ReflectResult,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.utils import (
    JsonValue,
    generate_json,
    load_agentic_service,
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

    
class Blackhole:
    """Utility stream that discards everything written to it."""
    def write(self, *args, **kwargs):
        """Discard write calls coming from redirected stdout."""
        pass  
    
    def flush(self):
        """Accept flush calls expected by some writers."""
        pass  

class LLMEvaluator(Evaluator):
    """Leverage LLMs to score workflow executions and reflect on outcomes."""

    def __init__(self, need_reflect: bool = True):
        """Initialise the evaluator with a backing model service."""
        super().__init__()
        self._llm: ModelService = ModelServiceFactory.create(
            model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
        )
        self.job_id = "[LLMEvaluator]"
        self.need_reflect = need_reflect

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
            parent_results = load_execute_result(parent_results_file)
            parent_scores = {result.task: result.score for result in parent_results}
        else:
            parent_scores = {}

        # initialize total score and results list
        total_score = 0.0
        results: List[ExecuteResult] = []

        original_stdout = sys.stdout
        step_size = 5
        # Process dataset in batches
        for start in range(0, len(dataset), step_size):
            batch = dataset[start : start + step_size]
            
            # Load the agentic service for the current round
            try:
                agent_sys = load_agentic_service(
                    optimized_path=optimized_path,
                    round_num=round_num,
                )
            except Exception as e:
                for data in batch:
                    parent_score = parent_scores.get(data.task, -1)
                    results.append(
                        ExecuteResult(
                            task=data.task,
                            verifier=data.verifier,
                            model_output="",
                            ori_score=parent_score,
                            score=-1,
                            error=(
                                "load_agentic_service failed, the configuration file has errors, "
                                f"reason={e}"
                            ),
                            succeed="no",
                        )
                    )
                continue
            try:
                sys.stdout = Blackhole()
                jobs: List[tuple[Row, JobWrapper]] = []
                # Submit jobs for the current batch
                for data in batch:
                    message = TextMessage(payload=data.task)
                    jobs.append((data, agent_sys.session().submit(message)))

                # Collect results and score them
                for data, jobwrapper in jobs:
                    parent_score = parent_scores.get(data.task, -1)
                    result = None
                    try:
                        model_message = jobwrapper.wait()
                        if isinstance(model_message, TextMessage):
                            result = model_message.get_payload()
                        elif isinstance(model_message, HybridMessage):
                            result = model_message.get_instruction_message().get_payload()

                        # Use the LLM to score the result
                        score = await self._llm_scoring(
                            question=data.task,
                            model_output=str(result),
                            expected_answer=data.verifier,
                        )

                        # store the score and result
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
                                task=data.task,
                                verifier=data.verifier,
                                model_output=str(result),
                                ori_score=parent_score,
                                score=score,
                                error="",
                                succeed=succeed,
                            )
                        )
                    except Exception as e:
                        results.append(
                            ExecuteResult(
                                task=data.task,
                                verifier=data.verifier,
                                model_output=str(result),
                                ori_score=parent_score,
                                score=0,
                                error=f"{e}",
                                succeed="no",
                            )
                        )
            finally:
                sys.stdout = original_stdout

        avg_score = total_score / len(dataset)
        
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
        pass

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
