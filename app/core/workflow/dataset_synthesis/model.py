from typing import List, Literal

from pydantic import BaseModel

# Task types, "query" represents read-only tasks, "non-query" represents write-only tasks
TASK_TYPES = Literal["query", "non-query"]  # 

# The tv-pair in dataset can query-only, non-query-only, or mixed
GENERATOR_STRATEGY = Literal[
    "query", "non-query", "mixed", None
]  

# Task difficulty levels
TASK_LEVEL = Literal["L1", "L2", "L3", "L4"] 

class Row(BaseModel):
    """a data row model in the dataset, representing a single tv-pair"""
    level: TASK_LEVEL
    task_type: TASK_TYPES
    task_subtype: str
    task: str
    verifier: str


class WorkflowTrainDataset(BaseModel):
    """Workflow training dataset model"""
    name: str
    task_desc: str
    data: list[Row]


class SubTaskType(BaseModel):
    """Specific subtask type model"""
    level: TASK_LEVEL
    name: str
    desc: str
    examples: list[str]


class LevelInfo(BaseModel):
    """structured information of each task level"""
    level: TASK_LEVEL
    name: str
    desc: str
    subtasks: List[SubTaskType]
