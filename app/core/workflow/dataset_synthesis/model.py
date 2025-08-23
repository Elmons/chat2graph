from typing import List, Literal

from pydantic import BaseModel

TASK_TYPES = Literal["query", "non-query"]  # query表示查询类任务， non-query表示非查询类任务

GENERATOR_STRATEGY = Literal[
    "query", "non-query", "mixed", None
]  # 生成策略： query表示只生成查询类任务， non-query表示只生成非查询类任务， mixed表示混合生成

TASK_LEVEL = Literal["L1", "L2", "L3", "L4"] # 任务层级

class Row(BaseModel):
    """数据集中的一行数据模型"""
    level: TASK_LEVEL
    task_type: TASK_TYPES
    task_subtype: str
    task: str
    verifier: str


class WorkflowTrainDataset(BaseModel):
    """工作流训练数据集模型"""
    name: str
    task_desc: str
    data: list[Row]


class SubTaskType(BaseModel):
    """子任务类型模型"""
    level: TASK_LEVEL
    name: str
    desc: str
    examples: list[str]


class LevelInfo(BaseModel):
    """任务层级信息模型"""
    level: TASK_LEVEL
    name: str
    desc: str
    subtasks: List[SubTaskType]
