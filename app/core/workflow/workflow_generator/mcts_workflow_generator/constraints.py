from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.common.type import WorkflowPlatformType


@dataclass
class WorkflowConstraints:
    """Unified workflow constraints shared by prompting and validation."""

    main_expert_name: str = "Main Expert"
    workflow_platform: Optional[WorkflowPlatformType] = WorkflowPlatformType.DBGPT
    require_single_expert: bool = True
    require_single_tail: bool = True
    require_agentic_parse: bool = True
    require_agentic_service_dry_run: bool = False
