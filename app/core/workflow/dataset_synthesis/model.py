from typing import List, Literal

from pydantic import BaseModel, Field

# Task types:
# This repo is query-only for now (see extra_doc/refactor_plan_to_new_arch.md).
TASK_TYPES = Literal["query"]

# Dataset synthesis strategy (query-only for now).
GENERATOR_STRATEGY = Literal["query", None]

# Task difficulty levels
TASK_LEVEL = Literal["L1", "L2", "L3", "L4"]

# RowV2.1 (task2 governance): where row is generated and where answer is defined.
GENERATION_SCOPE = Literal["local_subgraph"]
ANSWER_SCOPE = Literal["global_graph", "local_subgraph"]

class Row(BaseModel):
    """A single dataset sample for workflow synthesis/evaluation."""

    level: TASK_LEVEL
    task_type: TASK_TYPES
    task_subtype: str
    task_subtype_id: str | None = None
    task: str
    verifier: str
    generation_scope: GENERATION_SCOPE | None = None
    answer_scope: ANSWER_SCOPE | None = None
    intent_set: list[str] = Field(default_factory=list)
    global_verifier: str | None = None
    expected_global: str | None = None


class WorkflowTrainDataset(BaseModel):
    """Workflow training dataset model"""
    name: str
    task_desc: str
    data: list[Row]
    protocol_version: str = "row_v2_1"
    qa_gate_stats: dict[str, float | int] = Field(default_factory=dict)
    sampling_stats: dict = Field(default_factory=dict)


class SubTaskType(BaseModel):
    """Specific subtask type model"""
    level: TASK_LEVEL
    subtype_id: str
    name: str
    desc: str
    examples: list[str]
    canonical_intents: list[str] = Field(default_factory=list)
    constraint_tags: list[str] = Field(default_factory=list)
    required_query_features: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    target_ratio_min: float = 0.0
    target_ratio_max: float = 1.0


class LevelInfo(BaseModel):
    """structured information of each task level"""
    level: TASK_LEVEL
    name: str
    desc: str
    subtasks: List[SubTaskType]
