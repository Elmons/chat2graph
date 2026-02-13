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
    WorkflowLogFormat,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.selector import Selector
from app.core.workflow.workflow_generator.mcts_workflow_generator.utils import load_config_dict
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
        # self.validate_rounds = validate_rounds
        self.optimized_path = f"{optimized_path}/{self.dataset.name}_{str(int(time.time()))[-4:-1]}"
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
        data = self.dataset.data
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
        train_data, _ = self.split_dataset()

        self.logger.info("[run]init_workflow...")
        self.init_workflow()
        score, reflection = await self.evaluator.evaluate_workflow(
            round_num=1,
            parent_round=-1,
            dataset=self.dataset.data,
            modifications=[],
            optimized_path=self.optimized_path,
        )
        self.logs[1] = WorkflowLogFormat(
            round_number=1,
            parent_round=None,
            score=score,
            reflection=reflection,
            modifications=[],
            feedbacks=[],
        )
        self.max_score = score
        self.optimal_round = 1
        self.log_save()
        for round_num in range(2, self.max_rounds + 1):
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
                continue

            # Evaluate the new node
            score, reflection = await self.evaluator.evaluate_workflow(
                round_num=round_num,
                dataset=train_data,
                modifications=optimize_resp.modifications,
                optimized_path=self.optimized_path,
                parent_round=select_round.round_number,
            )

            # save result
            self.logs[round_num] = WorkflowLogFormat(
                round_number=round_num,
                parent_round=select_round.round_number,
                score=score,
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
            self.log_save()

        return self.max_score, self.optimal_round
