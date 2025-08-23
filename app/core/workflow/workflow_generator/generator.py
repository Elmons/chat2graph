from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(slots=True)
class WorkflowGenerationResult:
    """Aggregate result produced by a workflow generator run."""

    best_score: float
    best_round: int
    artifacts_path: Optional[Path] = None
    metadata: Optional[Mapping[str, Any]] = None


class WorkflowGenerator(ABC):
    """Interface that all workflow generators must implement."""

    @abstractmethod
    async def generate(self) -> WorkflowGenerationResult:
        """Execute the generation procedure and return the best found workflow."""
