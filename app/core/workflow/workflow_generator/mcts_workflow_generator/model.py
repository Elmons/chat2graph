from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class OptimizeActionType(str, Enum):
    """Optimize action types describe the kind of modification to be made."""
    ADD = "add"
    # DELETE = "delete an operator or expert"
    MODIFY = "modify"


class OptimizeObject(str, Enum):
    """Optimize object types describe the target of the optimization action."""
    OPERATOR = "operator"
    EXPERT = "expert"


class OptimizeAction(BaseModel):
    """An optimization action suggested during workflow improvement."""
    action_type: OptimizeActionType
    optimize_object: OptimizeObject
    reason: str


class WorkflowLogFormat(BaseModel):
    """Log format for each workflow optimization round."""
    round_number: int
    parent_round: Optional[int] = None
    score: float
    raw_avg_score: Optional[float] = None
    regression_rate: Optional[float] = None
    error_rate: Optional[float] = None
    optimize_suggestions: List[OptimizeAction] = []
    modifications: List[str]  
    reflection: str
    feedbacks: List[
        Dict[str, str]
    ]  


class OptimizeResp(BaseModel):
    """Response model for optimization actions."""
    modifications: List[str]
    new_configs: Dict[str, str]


class AgenticConfigSection(Enum):
    """Configuration sections for agentic workflows."""
    APP = "app"
    PLUGIN = "plugin"
    REASONER = "reasoner"
    TOOLS = "tools"
    ACTIONS = "actions"
    TOOLKIT = "toolkit"
    OPERATORS = "operators"
    EXPERTS = "experts"
    LEADER = "leader"
    KNOWLEDGEBASE = "knowledgebase"
    MEMORY = "memory"
    ENV = "env"


class ExecuteResult(BaseModel):
    """Execution result of a optimized workflow."""
    task: str
    verifier: str
    model_output: str
    ori_score: float
    score: float
    error: str
    succeed: Literal["yes", "no", "unknown"]
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    error_type: Optional[str] = None


class ReflectResult(BaseModel):
    """Reflection result after executing an optimized workflow."""
    failed_reason: List[str]
    optimize_suggestion: List[str]
