import json
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

from app.core.service.graph_db_service import GraphDb
from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.workflow_generator.generator import (
    WorkflowGenerationResult,
    WorkflowGenerator,
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
    validate_workflow_yaml,
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
            "app/core/workflow/workflow_generator/mcts_workflow_generator/toolsets/default.yml"
        ),
        init_template_path: str = (
            "app/core/workflow/workflow_generator/mcts_workflow_generator/"
            "init_template/base_template.yml"
        ),
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

    def init_workflow(self):
        """Seed the search space with a baseline workflow copied from a template file."""

        # 创建保存路径
        save_path = Path(self.optimized_path) / "round1"
        save_path.mkdir(parents=True, exist_ok=True)

        # 写入 workflow.yml 文件
        workflow_file = save_path / "workflow.yml"
        base_dict = load_config_dict(self.init_template_path, skip_section=[])
        toolset_dict = load_config_dict(self.toolset_path, skip_section=[])

        def _req(section: AgenticConfigSection, src: Dict[str, str], name: str) -> str:
            key = str(section.value)
            val = src.get(key)
            if not val:
                raise ValueError(f"Missing `{key}` section in {name}")
            return val

        with open(workflow_file, "w", encoding="utf-8") as f:
            # base sections
            for section in [
                AgenticConfigSection.APP,
                AgenticConfigSection.PLUGIN,
                AgenticConfigSection.REASONER,
            ]:
                f.write(_req(section, base_dict, "base_template"))
                f.write("\n\n")

            # toolset sections (U)
            for section in [
                AgenticConfigSection.TOOLS,
                AgenticConfigSection.ACTIONS,
                AgenticConfigSection.TOOLKIT,
            ]:
                f.write(_req(section, toolset_dict, "toolset"))
                f.write("\n\n")

            # base runtime sections
            for section in [
                AgenticConfigSection.OPERATORS,
                AgenticConfigSection.EXPERTS,
                AgenticConfigSection.KNOWLEDGEBASE,
                AgenticConfigSection.MEMORY,
                AgenticConfigSection.ENV,
            ]:
                f.write(_req(section, base_dict, "base_template"))
                f.write("\n\n")

        print(f"Initialized default workflow at: {workflow_file}")
        if not self.main_expert_name:
            inferred = infer_single_expert_name(workflow_file)
            if not inferred:
                raise ValueError(
                    "Cannot infer entry expert name from init_template workflow.yml; "
                    "please pass main_expert_name explicitly."
                )
            self.main_expert_name = inferred

        validation = validate_workflow_yaml(workflow_file, main_expert_name=self.main_expert_name)
        if not validation.ok:
            raise ValueError(
                f"init_template workflow.yml failed validation: {validation.errors}"
            )

        config_dict = self.load_config_dict(round_num=1, skip_section=None)
        for section in AgenticConfigSection:
            section_name = str(section.value)
            section_context = config_dict.get(section_name)
            if section_context is None:
                print(
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
        save_dir = Path(self.optimized_path) / "log"
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / "log.json"
        edges_file = save_dir / "edges.json"
        config_file = save_dir / "config.json"
        with open(log_file, "w", encoding="utf-8") as f:
            logs = [v.model_dump(mode="json") for k, v in self.logs.items()]
            json.dump(
                logs,
                f,
                ensure_ascii=False,
                indent=2,
            )

        edges = []
        for log in self.logs.values():
            if log.parent_round is None:
                continue
            edges.append(
                {
                    "parent_round": log.parent_round,
                    "child_round": log.round_number,
                }
            )
        with open(edges_file, "w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)

        with open(config_file, "w", encoding="utf-8") as f:
            config = [
                {
                    "max_rounds": self.max_rounds,
                    "top_k": self.top_k,
                    "init_template_path": self.init_template_path,
                    "max_score": self.max_score,
                    "optimal_round": self.optimal_round,
                }
            ]
            json.dump(
                config,
                f,
                ensure_ascii=False,
                indent=2,
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

        print("[run]init_workflow...")
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
        self.log_save()
        for round_num in range(2, self.max_rounds + 1):
            print(f"[run]optimize, round={round_num}...")
            # Select a workflow
            select_retry_times = 0
            round_context = None
            while select_retry_times < self.max_retries:
                select_retry_times += 1
                select_round = self.selector.select(top_k=self.top_k, logs=self.logs)
                round_context = self.logs.get(select_round.round_number, None)
                if round_context is not None:
                    break
            
            if round_context is None or select_round is None:
                print(f"[run]select workflow failed after {self.max_retries} retries")
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
                print(f"[run]new flow generate failed, round={round_num}")
                continue

            # Save workflow
            new_flow_dir = Path(self.optimized_path + f"/round{round_num}")
            new_flow_dir.mkdir(parents=True, exist_ok=True)
            new_flow_path = new_flow_dir / "workflow.yml"
            try:
                with open(new_flow_path, "w", encoding="utf-8") as f:
                    for section in AgenticConfigSection:
                        if section not in self.optimize_grain:
                            section_name = str(section.value)
                            section_init_context = self.init_config_dict.get(section_name, None)
                            if section_init_context is None:
                                print(
                                    "[MCTSWorkflowGenerator][run] Can't find "
                                    f"{section_name} in init_config_dict"
                                )
                                continue
                            f.write(section_init_context)
                            f.write("\n\n")
                    for _, section_context in optimize_resp.new_configs.items():
                        f.write(section_context)
                        f.write("\n\n")
            except Exception:
                print("[run]exception while saving workflow")
                continue

            validation = validate_workflow_yaml(
                new_flow_path, main_expert_name=self.main_expert_name
            )
            if not validation.ok:
                print(
                    "[run]candidate workflow.yml failed validation, "
                    f"round={round_num}, errors={validation.errors}"
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
