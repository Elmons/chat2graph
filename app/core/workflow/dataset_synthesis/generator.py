from abc import ABC, abstractmethod
import json
import re
from typing import Dict, List

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
from app.core.workflow.dataset_synthesis.sampler import RandomWalkSampler, SubGraphSampler
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo


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
    ):
        super().__init__()
        self.graph_db = graph_db
        self._llm: ModelService = ModelServiceFactory.create(
            model_platform_type=SystemEnv.MODEL_PLATFORM_TYPE
        )
        if not sampler:
            sampler = RandomWalkSampler()
        self.sampler: SubGraphSampler = sampler
        self.max_depth = max_depth
        self.max_nodes = max_noeds
        self.max_edges = max_edges
        self.strategy = strategy
        self.nums_per_subgraph = nums_per_subgraph

    def extract_pairs(self, task_type: TASK_TYPES, text: str) -> list[Row]:
        """Extract TV pairs from an LLM response.

        The LLM is prompted to output a JSON list of objects. In practice, model outputs
        may be wrapped in markdown fences or include extra prose. This parser tries to be
        robust without relying on regex patterns that break on braces inside strings.
        """
        required_fields = Row.model_fields.keys()
        whitelist = ["task_type"]

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

        def _maybe_add_obj(obj: Dict) -> None:
            valid = True
            for filed in required_fields:
                if filed in whitelist:
                    continue
                if filed not in obj:
                    valid = False
                    break
            if not valid:
                return
            obj["task_type"] = task_type
            valid_pairs.append(Row.model_validate(obj))

        for block in _iter_json_blocks(text):
            block_pairs_before = len(valid_pairs)
            try:
                parsed = json.loads(block)
            except Exception:
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
        prompt_template = generate_query_tv_template
        prompt = prompt_template.format(
            task_description=task_description,
            subgraph=subgraph,
            num_pairs=nums,
            task_level_info=task_types_info.get_tasks_info(),
            task_statistic_info=task_types_info.get_count_info(),
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

        # extract pairs from response
        qas: list[Row] = self.extract_pairs(task_type, response.get_payload())
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
        self, task_type: TASK_TYPES, task_desc: str, subgraph: str, dataset: list[Row]
    ) -> list[Row]:
        """Filter generated TV pairs using the LLM."""

        # prompt construction
        prompt = filter_prompt_template.format(
            task_desc=task_desc,
            subgraph=subgraph,
            dataset=dataset,
        )
        job_id = "filter_job"
        message = ModelMessage(
            payload=prompt,
            source_type=MessageSourceType.MODEL,
            job_id=job_id,
            step=1,
        )

        # generate response
        response = await self._llm.generate(sys_prompt="", messages=[message])

        # extract pairs from response
        qas: list[Row] = self.extract_pairs(task_type, response.get_payload())
        return qas

    async def generate(self, task_desc: str, dataset_name: str, size: int) -> WorkflowTrainDataset:
        """Generate a dataset based on the task description and desired size."""

        # initialize
        dataset: list[Row] = []
        total = 0
        max_times = (
            size // self.nums_per_subgraph + 20
        )  # max generation attempts to avoid infinite loops
        times = 0
        subgraph_getter: SubGraphSampler = self.sampler
        strategy: GENERATOR_STRATEGY = await self.identify_strategy(task_desc)

        if strategy is None:
            raise Exception(f"Cann't indentify strategy from task description={task_desc}")

        task_types_info = GraphTaskTypesInfo(strategy=strategy)

        # generation loop
        while total < size and times < max_times:
            # try to get a random subgraph from the graph database
            times += 1
            try:
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
                continue

            # filter the generated pairs
            pairs = await self.filter(
                task_type=task_type, task_desc=task_desc, subgraph=subgraph, dataset=pairs
            )

            if len(pairs) == 0:
                print(
                    f"[SamplingDatasetGenerator][generate] 0 valid pairs after filter, subgraph={subgraph}"  # noqa: E501
                )
                continue

            # update task types statistics info and dataset
            task_types_info.update(pairs)
            dataset.extend(pairs)
            total += len(pairs)
            # time.sleep(2)  # speed control

        # create final dataset object
        workflow_dataset = WorkflowTrainDataset(
            name=dataset_name, task_desc=task_desc, data=dataset
        )

        print(task_types_info.get_count_info())
        return workflow_dataset
