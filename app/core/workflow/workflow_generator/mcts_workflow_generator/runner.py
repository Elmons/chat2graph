from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence

from app.core.common.runtime_reset import reset_runtime_state
from app.core.model.graph_db_config import GraphDbConfig
from app.core.model.message import HybridMessage, TextMessage
from app.core.sdk.agentic_service import AgenticService
from app.core.workflow.dataset_synthesis.model import Row


@dataclass(frozen=True)
class WorkflowRunRecord:
    task: str
    verifier: str
    model_output: str
    error: str
    latency_ms: float
    tokens: int = 0


class _Blackhole:
    def write(self, *args, **kwargs):
        return 0

    def flush(self):
        return None


@contextmanager
def _suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    original_stdout = sys.stdout
    try:
        sys.stdout = _Blackhole()
        yield
    finally:
        sys.stdout = original_stdout


def _unwrap_payload(msg: TextMessage | HybridMessage | object) -> str:
    if isinstance(msg, HybridMessage):
        return str(msg.get_instruction_message().get_payload() or "")
    if isinstance(msg, TextMessage):
        return str(msg.get_payload() or "")
    payload = getattr(msg, "get_payload", None)
    if callable(payload):
        return str(payload() or "")
    return str(msg)


class WorkflowRunner:
    """Runtime runner that executes a workflow config over dataset rows."""

    def __init__(
        self,
        *,
        main_expert_name: Optional[str] = None,
        batch_size: int = 5,
        suppress_stdout: bool = True,
    ):
        self.main_expert_name = main_expert_name
        self.batch_size = max(1, batch_size)
        self.suppress_stdout = suppress_stdout

    def _resolve_entry_expert(self, service: AgenticService) -> str:
        entry_expert = self.main_expert_name or service.entry_expert_name()
        if not entry_expert:
            raise ValueError(
                "No entry expert inferred; set main_expert_name or use single-expert YAML."
            )
        return entry_expert

    def _extract_tokens(self, job_wrapper: object) -> int:
        try:
            job = getattr(job_wrapper, "job", None)
            job_id = getattr(job, "id", None)
            if not job_id:
                return 0
            from app.core.service.job_service import JobService

            return int(JobService.instance.get_job_result(job_id).tokens)
        except Exception:
            return 0

    async def run_dataset(
        self,
        *,
        workflow_path: str | Path,
        rows: Sequence[Row],
        graph_db_config: Optional[GraphDbConfig] = None,
        reset_state: bool = True,
    ) -> List[WorkflowRunRecord]:
        if reset_state:
            reset_runtime_state()
        service = AgenticService.load(str(workflow_path))
        if graph_db_config is not None:
            service.graph_db(graph_db_config)
        entry_expert = self._resolve_entry_expert(service)

        results: List[WorkflowRunRecord] = []
        with _suppress_stdout(self.suppress_stdout):
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start : start + self.batch_size]
                jobs = []
                for row in batch:
                    message = TextMessage(payload=row.task, assigned_expert_name=entry_expert)
                    submit_started = time.perf_counter()
                    job_wrapper = service.session().submit(message)
                    jobs.append((row, job_wrapper, submit_started))

                for row, job_wrapper, submit_started in jobs:
                    model_output = ""
                    error = ""
                    try:
                        model_msg = job_wrapper.wait()
                        model_output = _unwrap_payload(model_msg)
                    except Exception as e:
                        error = str(e)
                    latency_ms = (time.perf_counter() - submit_started) * 1000.0
                    results.append(
                        WorkflowRunRecord(
                            task=row.task,
                            verifier=row.verifier,
                            model_output=model_output,
                            error=error,
                            latency_ms=latency_ms,
                            tokens=self._extract_tokens(job_wrapper),
                        )
                    )
        return results
