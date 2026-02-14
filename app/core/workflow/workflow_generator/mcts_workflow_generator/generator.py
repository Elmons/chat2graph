import json
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

from app.core.common.logger import Chat2GraphLogger
from app.core.service.graph_db_service import GraphDb
from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.workflow_generator.generator import (
    WorkflowGenerationResult,
    WorkflowGenerator,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.artifact_writer import (
    MCTSArtifactWriter,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.constraints import (
    WorkflowConstraints,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator import Evaluator
from app.core.workflow.workflow_generator.mcts_workflow_generator.expander import Expander
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    AgenticConfigSection,
    ExecuteResult,
    ReflectResult,
    WorkflowLogFormat,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.selector import Selector
from app.core.workflow.workflow_generator.mcts_workflow_generator.utils import (
    load_config_dict,
    load_execute_result,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.validator import (
    infer_single_expert_name,
    validate_candidate_config,
)


class MCTSWorkflowGenerator(WorkflowGenerator):
    """Search for high-performing workflows via a Monte-Carlo tree style process."""

    def __init__(
        self,
        db: GraphDb,
        dataset: WorkflowTrainDataset,
        selector: Selector,
        expander: Expander,
        evaluator: Evaluator,
        optimize_grain: List[AgenticConfigSection],
        main_expert_name: Optional[str] = None,
        toolset_path: str = (
            "app/core/sdk/toolsets/graph_only.yml"
        ),
        init_template_path: str = (
            "app/core/workflow/workflow_generator/mcts_workflow_generator/"
            "init_template/base_template.yml"
        ),
        workflow_constraints: Optional[WorkflowConstraints] = None,
        max_rounds: int = 30,
        optimized_path: str = "workflow_space",
        top_k: int = 5,
        max_retries: int = 5,
        no_improvement_patience: int = 0,
        resume: bool = False,
        resume_run_path: Optional[str] = None,
        train_test_split_ratio: float = 0.0,
        split_random_state: int = 42,
    ):
        """Configure generator dependencies and search hyper-parameters."""
        if optimize_grain is None:
            optimize_grain = [AgenticConfigSection.EXPERTS, AgenticConfigSection.OPERATORS]
        self.dataset = dataset
        self.db: GraphDb = db
        self.selector: Selector = selector
        self.expander: Expander = expander
        self.evaluator: Evaluator = evaluator

        self.max_rounds = max_rounds
        self.resume = resume
        self.no_improvement_patience = max(0, int(no_improvement_patience))
        self.train_test_split_ratio = max(0.0, min(1.0, float(train_test_split_ratio)))
        self.split_random_state = int(split_random_state)
        # self.validate_rounds = validate_rounds
        if self.resume:
            self.optimized_path = str(resume_run_path or optimized_path)
        else:
            self.optimized_path = (
                f"{optimized_path}/{self.dataset.name}_{str(int(time.time()))[-4:-1]}"
            )
        self.artifact_writer = MCTSArtifactWriter(self.optimized_path)
        self.top_k = top_k
        self.max_retries = max_retries
        self.logs: dict[int, WorkflowLogFormat] = {}
        self.optimize_grain = optimize_grain
        self.init_template_path = init_template_path
        self.main_expert_name = main_expert_name
        self.toolset_path = toolset_path
        self.init_config_dict: Dict[str, str] = {}
        self.max_score: float = -1
        self.optimal_round = 0
        self.workflow_constraints = workflow_constraints or WorkflowConstraints(
            main_expert_name=main_expert_name or "Main Expert",
            require_agentic_parse=True,
            require_agentic_service_dry_run=True,
        )
        self.logger = Chat2GraphLogger.get_logger(__name__)

    def init_workflow(self):
        """Seed the search space with a baseline workflow copied from a template file."""
        workflow_file = self.artifact_writer.write_round_workflow(
            round_num=1,
            base_template_path=self.init_template_path,
            toolset_path=self.toolset_path,
            candidate_sections={},
        )
        self.logger.info("Initialized default workflow at: %s", workflow_file)
        if not self.main_expert_name:
            inferred = infer_single_expert_name(workflow_file)
            if not inferred:
                raise ValueError(
                    "Cannot infer entry expert name from init_template workflow.yml; "
                    "please pass main_expert_name explicitly."
                )
            self.main_expert_name = inferred
        self.workflow_constraints.main_expert_name = self.main_expert_name
        validation = validate_candidate_config(
            workflow_file,
            constraints=self.workflow_constraints,
        )
        if not validation.ok:
            raise ValueError(f"init_template workflow.yml failed validation: {validation.errors}")

        config_dict = self.load_config_dict(round_num=1, skip_section=None)
        for section in AgenticConfigSection:
            section_name = str(section.value)
            section_context = config_dict.get(section_name)
            if section_context is None:
                self.logger.warning(
                    "[MCTSWorkflowGenerator][init_workflow] Can't find "
                    f"{section_name} in {workflow_file}"
                )
                continue
            if section not in self.optimize_grain:
                self.init_config_dict[section_name] = section_context

    def split_dataset(
        self, test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[List[Row], List[Row]]:
        """Split the dataset into training and validation sets."""
        data = list(self.dataset.data)
        if test_size <= 0:
            return data, []
        if test_size >= 1:
            return [], data
        random.seed(random_state)
        random.shuffle(data)
        split_index = int(test_size * len(data))
        train_data = data[split_index:]
        test_data = data[:split_index]
        return train_data, test_data

    def load_config_dict(
        self, round_num: int, skip_section: List[AgenticConfigSection]
    ) -> Dict[str, str]:
        """Load the workflow configuration of a given round."""
        if skip_section is None:
            skip_section = []
        workflow_path = self.optimized_path + f"/round{round_num}" + "/workflow.yml"
        return load_config_dict(workflow_path, skip_section=skip_section)

    def update_parent_feedbacks(self, parent_round: int, current_round: int):
        """Record feedback from a child round in its parent log entry."""
        self.logs[parent_round].feedbacks.append(
            {
                "child_round": f"{current_round}",
                "modification": f"{self.logs[current_round].modifications}",
                "after_score": f"{self.logs[current_round].score}",
                "reflection": self.logs[current_round].reflection,
                "succeed": f"{self.logs[current_round].score > self.logs[parent_round].score}",
            }
        )

    def log_save(self):
        """Persist the current optimization logs and summary metadata to disk."""
        workflow_platform = (
            self.workflow_constraints.workflow_platform.value
            if self.workflow_constraints.workflow_platform
            else None
        )
        return self.artifact_writer.write_logs(
            logs=self.logs,
            config={
                "max_rounds": self.max_rounds,
                "top_k": self.top_k,
                "init_template_path": self.init_template_path,
                "toolset_path": self.toolset_path,
                "max_score": self.max_score,
                "optimal_round": self.optimal_round,
                "no_improvement_patience": self.no_improvement_patience,
                "resume": self.resume,
                "train_test_split_ratio": self.train_test_split_ratio,
                "split_random_state": self.split_random_state,
                "workflow_constraints": {
                    "main_expert_name": self.workflow_constraints.main_expert_name,
                    "workflow_platform": workflow_platform,
                    "require_single_expert": self.workflow_constraints.require_single_expert,
                    "require_single_tail": self.workflow_constraints.require_single_tail,
                    "require_agentic_parse": self.workflow_constraints.require_agentic_parse,
                    "require_agentic_service_dry_run": (
                        self.workflow_constraints.require_agentic_service_dry_run
                    ),
                },
            },
        )

    @staticmethod
    def _validation_failure_suggestions(validation_errors: List[str]) -> List[str]:
        """Generate actionable suggestions from workflow validation errors."""
        suggestions: List[str] = []
        for err in validation_errors:
            low = str(err).lower()
            if "exactly 1 tail operator" in low:
                suggestions.append(
                    "Merge terminal branches so the expert.workflow ends with exactly one tail operator."
                )
            elif "cycle detected" in low:
                suggestions.append(
                    "Remove cyclic dependencies and keep expert.workflow as a DAG."
                )
            elif "operator not present" in low:
                suggestions.append(
                    "Ensure every operator referenced in expert.workflow exists in top-level operators."
                )
            else:
                suggestions.append(
                    "Fix workflow structural validation errors before running evaluation."
                )
        # Keep insertion order while removing duplicates.
        return list(dict.fromkeys(suggestions))

    def _load_parent_scores(self, parent_round: int) -> Dict[str, float]:
        """Load parent-round per-task scores for regression context."""
        if parent_round <= 0:
            return {}
        parent_results_file = Path(self.optimized_path) / f"round{parent_round}" / "results.json"
        if not parent_results_file.exists():
            return {}
        try:
            parent_results = load_execute_result(parent_results_file)
            return {result.task: result.score for result in parent_results}
        except Exception:
            return {}

    def _record_validation_failure_round(
        self,
        *,
        round_num: int,
        parent_round: int,
        dataset: List[Row],
        validation_errors: List[str],
        modifications: List[str],
        optimize_suggestions: List,
    ) -> None:
        """Persist synthetic evaluation artifacts when candidate validation fails."""
        parent_scores = self._load_parent_scores(parent_round)
        error_msg = (
            "candidate workflow.yml failed validation, "
            f"errors={validation_errors}"
        )
        results: List[ExecuteResult] = []
        for data in dataset:
            parent_score = parent_scores.get(data.task, -1)
            results.append(
                ExecuteResult(
                    task=data.task,
                    verifier=data.verifier,
                    model_output="",
                    ori_score=parent_score,
                    score=-1,
                    error=error_msg,
                    succeed="no",
                    latency_ms=None,
                    token_usage=None,
                    error_type="validation_error",
                )
            )

        self.artifact_writer.write_round_json(
            round_num=round_num,
            filename="results.json",
            payload=[result.model_dump() for result in results],
        )

        valid_parent_count = 0
        regression_count = 0
        for result in results:
            if result.ori_score >= 0:
                valid_parent_count += 1
                if result.score < result.ori_score:
                    regression_count += 1
        regression_rate = (
            regression_count / valid_parent_count if valid_parent_count > 0 else 0.0
        )
        error_rate = 1.0 if results else 0.0
        raw_avg_score = -1.0 if results else 0.0

        reflect_result = ReflectResult(
            failed_reason=[f"validation_failed: {err}" for err in validation_errors],
            optimize_suggestion=self._validation_failure_suggestions(validation_errors),
        )
        self.logs[round_num] = WorkflowLogFormat(
            round_number=round_num,
            parent_round=parent_round,
            score=raw_avg_score,
            raw_avg_score=raw_avg_score,
            regression_rate=regression_rate,
            error_rate=error_rate,
            reflection=reflect_result.model_dump_json(indent=2),
            modifications=modifications,
            feedbacks=[],
            optimize_suggestions=optimize_suggestions,
        )
        self.update_parent_feedbacks(parent_round, round_num)
        self.log_save()

    def _record_no_improvement(
        self,
        *,
        round_num: int,
        reason: str,
        no_improvement_rounds: int,
    ) -> Tuple[int, bool]:
        """Update no-improvement streak and determine whether to early-stop."""
        no_improvement_rounds += 1
        if self.no_improvement_patience <= 0:
            return no_improvement_rounds, False

        self.logger.info(
            "[run]round=%s has no improvement (%s), streak=%s/%s",
            round_num,
            reason,
            no_improvement_rounds,
            self.no_improvement_patience,
        )
        if no_improvement_rounds >= self.no_improvement_patience:
            self.logger.info(
                "[run]early stop: no improvement for %s consecutive rounds",
                self.no_improvement_patience,
            )
            return no_improvement_rounds, True
        return no_improvement_rounds, False

    def _compute_no_improvement_streak(self) -> int:
        """Compute trailing no-improvement streak from existing logs."""
        if not self.logs:
            return 0
        streak = 0
        best = float("-inf")
        for round_num in sorted(self.logs.keys()):
            score = self.logs[round_num].score
            if score > best:
                best = score
                streak = 0
            else:
                streak += 1
        return streak

    def _load_resume_state(self) -> int:
        """Restore MCTS state from disk and return the next round number."""
        log_path = Path(self.optimized_path) / "log" / "log.json"
        if not log_path.exists():
            raise ValueError(f"resume failed: log file not found at {log_path}")

        with log_path.open("r", encoding="utf-8") as f:
            raw_logs = json.load(f)
        if not isinstance(raw_logs, list) or len(raw_logs) == 0:
            raise ValueError(f"resume failed: invalid or empty log file at {log_path}")

        restored_logs: Dict[int, WorkflowLogFormat] = {}
        for item in raw_logs:
            log = WorkflowLogFormat.model_validate(item)
            restored_logs[log.round_number] = log
        self.logs = restored_logs

        if 1 not in self.logs:
            raise ValueError("resume failed: round1 not found in log history")

        self.max_score = max(log.score for log in self.logs.values())
        best_rounds = [rn for rn, log in self.logs.items() if log.score == self.max_score]
        self.optimal_round = min(best_rounds) if best_rounds else 1

        if not self.main_expert_name:
            round1_workflow = Path(self.optimized_path) / "round1" / "workflow.yml"
            inferred = infer_single_expert_name(round1_workflow)
            if inferred:
                self.main_expert_name = inferred
        if self.main_expert_name:
            self.workflow_constraints.main_expert_name = self.main_expert_name

        next_round = max(self.logs.keys()) + 1
        self.logger.info(
            "[run]resume loaded from %s, rounds=%s, next_round=%s, best_score=%s, best_round=%s",
            self.optimized_path,
            len(self.logs),
            next_round,
            self.max_score,
            self.optimal_round,
        )
        return next_round

    async def generate(self) -> WorkflowGenerationResult:
        """Run the optimization loop and return the best workflow discovered."""
        max_score, optimal_round = await self._generate_rounds()
        log_dir = Path(self.optimized_path) / "log"
        metadata = {
            "log_path": str(log_dir / "log.json"),
            "config_path": str(log_dir / "config.json"),            
        }
        return WorkflowGenerationResult(
            best_score=max_score,
            best_round=optimal_round,
            artifacts_path=Path(self.optimized_path),
            metadata=metadata,
        )

    async def _generate_rounds(self) -> Tuple[float, int]:
        """Core loop that iteratively expands, evaluates, and scores workflows."""
        train_data, _ = self.split_dataset(
            test_size=self.train_test_split_ratio,
            random_state=self.split_random_state,
        )
        no_improvement_rounds = 0

        if self.resume:
            next_round = self._load_resume_state()
            no_improvement_rounds = self._compute_no_improvement_streak()
            if (
                self.no_improvement_patience > 0
                and no_improvement_rounds >= self.no_improvement_patience
            ):
                self.logger.info(
                    "[run]resume early stop: historical no-improvement streak=%s already "
                    "reaches patience=%s",
                    no_improvement_rounds,
                    self.no_improvement_patience,
                )
                return self.max_score, self.optimal_round
        else:
            self.logger.info("[run]init_workflow...")
            self.init_workflow()
            score = 0.0
            reflection = "Round1 baseline initialized; evaluation skipped."
            self.logs[1] = WorkflowLogFormat(
                round_number=1,
                parent_round=None,
                score=score,
                raw_avg_score=score,
                regression_rate=0.0,
                error_rate=0.0,
                reflection=reflection,
                modifications=[],
                feedbacks=[],
            )
            self.max_score = score
            self.optimal_round = 1
            self.log_save()
            next_round = 2

        for round_num in range(next_round, self.max_rounds + 1):
            self.logger.info("[run]optimize, round=%s...", round_num)
            # Select a workflow
            select_retry_times = 0
            round_context = None
            select_round = None
            while select_retry_times < self.max_retries:
                select_retry_times += 1
                select_round = self.selector.select(top_k=self.top_k, logs=self.logs)
                round_context = self.logs.get(select_round.round_number, None)
                if round_context is not None:
                    break
            
            if round_context is None or select_round is None:
                self.logger.warning(
                    "[run]select workflow failed after %s retries", self.max_retries
                )
                no_improvement_rounds, should_stop = self._record_no_improvement(
                    round_num=round_num,
                    reason="selector_failed",
                    no_improvement_rounds=no_improvement_rounds,
                )
                if should_stop:
                    break
                continue
            # Load Workflow
            current_config = self.load_config_dict(select_round.round_number, skip_section=[])

            # Expand the workflow
            optimize_suggestions, optimize_resp = await self.expander.expand(
                task_tesc=self.dataset.task_desc,
                current_config=current_config,
                round_context=round_context,
            )

            if optimize_resp is None:
                self.logger.warning("[run]new flow generate failed, round=%s", round_num)
                no_improvement_rounds, should_stop = self._record_no_improvement(
                    round_num=round_num,
                    reason="expander_failed",
                    no_improvement_rounds=no_improvement_rounds,
                )
                if should_stop:
                    break
                continue

            # Save workflow
            candidate_sections = dict(optimize_resp.new_configs)
            for section in self.optimize_grain:
                section_name = str(section.value)
                if section_name in candidate_sections:
                    continue
                current_section = current_config.get(section_name)
                if current_section:
                    candidate_sections[section_name] = current_section

            try:
                new_flow_path = self.artifact_writer.write_round_workflow(
                    round_num=round_num,
                    base_template_path=self.init_template_path,
                    toolset_path=self.toolset_path,
                    candidate_sections=candidate_sections,
                )
            except Exception as e:
                self.logger.warning(
                    "[run]exception while saving workflow, round=%s, reason=%s", round_num, e
                )
                no_improvement_rounds, should_stop = self._record_no_improvement(
                    round_num=round_num,
                    reason="write_failed",
                    no_improvement_rounds=no_improvement_rounds,
                )
                if should_stop:
                    break
                continue

            validation = validate_candidate_config(
                new_flow_path,
                constraints=self.workflow_constraints,
            )
            if not validation.ok:
                self.logger.warning(
                    "[run]candidate workflow.yml failed validation, "
                    "round=%s, errors=%s",
                    round_num,
                    validation.errors,
                )
                self._record_validation_failure_round(
                    round_num=round_num,
                    parent_round=select_round.round_number,
                    dataset=train_data,
                    validation_errors=validation.errors,
                    modifications=optimize_resp.modifications,
                    optimize_suggestions=optimize_suggestions,
                )
                no_improvement_rounds, should_stop = self._record_no_improvement(
                    round_num=round_num,
                    reason="validation_failed",
                    no_improvement_rounds=no_improvement_rounds,
                )
                if should_stop:
                    break
                continue

            # Evaluate the new node
            score, reflection = await self.evaluator.evaluate_workflow(
                round_num=round_num,
                dataset=train_data,
                modifications=optimize_resp.modifications,
                optimized_path=self.optimized_path,
                parent_round=select_round.round_number,
            )
            eval_metrics = getattr(self.evaluator, "last_metrics", {}) or {}

            # save result
            self.logs[round_num] = WorkflowLogFormat(
                round_number=round_num,
                parent_round=select_round.round_number,
                score=score,
                raw_avg_score=eval_metrics.get("raw_avg_score"),
                regression_rate=eval_metrics.get("regression_rate"),
                error_rate=eval_metrics.get("error_rate"),
                reflection=reflection,
                modifications=optimize_resp.modifications,
                feedbacks=[],
                optimize_suggestions=optimize_suggestions,
            )

            # update exprience for father node
            self.update_parent_feedbacks(select_round.round_number, round_num)
            
            if self.logs[round_num].score > self.max_score:
                self.max_score = self.logs[round_num].score
                self.optimal_round = round_num
                no_improvement_rounds = 0
            else:
                no_improvement_rounds, should_stop = self._record_no_improvement(
                    round_num=round_num,
                    reason="score_not_improved",
                    no_improvement_rounds=no_improvement_rounds,
                )
                if should_stop:
                    self.log_save()
                    break
            self.log_save()

        return self.max_score, self.optimal_round
