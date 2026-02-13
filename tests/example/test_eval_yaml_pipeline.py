from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.model.message import TextMessage
from app.core.sdk.agentic_service import AgenticService
from app.core.workflow.dataset_synthesis.model import Row, WorkflowTrainDataset
from app.core.workflow.evaluation.eval_yaml_pipeline import evaluate_yaml


@pytest.mark.asyncio
async def test_evaluate_yaml_writes_results_and_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = WorkflowTrainDataset(
        name="ds",
        task_desc="desc",
        data=[
            Row(level="L1", task_type="query", task_subtype="t", task="q1", verifier="a1"),
            Row(level="L1", task_type="query", task_subtype="t", task="q2", verifier="a2"),
        ],
    )

    class _StubJob:
        def __init__(self, payload: str):
            self._payload = payload

        def wait(self):  # noqa: D401
            return TextMessage(payload=self._payload)

    class _StubService:
        def __init__(self) -> None:
            self.submitted: list[TextMessage] = []

        def entry_expert_name(self):  # noqa: D401
            return "Main Expert"

        def session(self, session_id=None):  # noqa: D401
            return self

        def submit(self, message: TextMessage):  # noqa: D401
            self.submitted.append(message)
            if message.get_payload() == "q1":
                return _StubJob("a1")
            return _StubJob("wrong")

    stub = _StubService()
    monkeypatch.setattr(AgenticService, "load", staticmethod(lambda _path: stub))

    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("app: {name: test}\nexperts: []\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    summary = await evaluate_yaml(dataset=dataset, yaml_path=yaml_path, out_dir=out_dir, score_mode="exact")

    assert summary.error == ""
    assert summary.avg_score == 1.5  # (3 + 0) / 2
    assert (out_dir / "results.json").exists()
    assert (out_dir / "summary.json").exists()

    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert [r["score"] for r in results] == [3, 0]

    assert len(stub.submitted) == 2
    assert stub.submitted[0].get_assigned_expert_name() == "Main Expert"


@pytest.mark.asyncio
async def test_evaluate_yaml_llm_scoring_uses_model_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = WorkflowTrainDataset(
        name="ds",
        task_desc="desc",
        data=[Row(level="L1", task_type="query", task_subtype="t", task="q1", verifier="a1")],
    )

    class _StubJob:
        def wait(self):  # noqa: D401
            return TextMessage(payload="anything")

    class _StubService:
        def entry_expert_name(self):  # noqa: D401
            return "Main Expert"

        def session(self, session_id=None):  # noqa: D401
            return self

        def submit(self, message: TextMessage):  # noqa: D401
            return _StubJob()

    monkeypatch.setattr(AgenticService, "load", staticmethod(lambda _path: _StubService()))

    class _StubModel:
        async def generate(self, sys_prompt: str, messages):  # noqa: D401
            from app.core.common.type import MessageSourceType
            from app.core.model.message import ModelMessage

            return ModelMessage(
                payload="```json\n{\"score\": 3}\n```",
                job_id="job",
                step=1,
                source_type=MessageSourceType.MODEL,
            )

    monkeypatch.setattr(
        "app.core.workflow.evaluation.eval_yaml_pipeline.ModelServiceFactory.create",
        lambda *args, **kwargs: _StubModel(),
    )

    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("app: {name: test}\nexperts: []\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    summary = await evaluate_yaml(
        dataset=dataset, yaml_path=yaml_path, out_dir=out_dir, score_mode="llm"
    )

    assert summary.error == ""
    assert summary.avg_score == 3.0


@pytest.mark.asyncio
async def test_evaluate_yaml_writes_summary_on_load_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = WorkflowTrainDataset(
        name="ds",
        task_desc="desc",
        data=[Row(level="L1", task_type="query", task_subtype="t", task="q1", verifier="a1")],
    )

    def _raise(_path: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(AgenticService, "load", staticmethod(_raise))

    yaml_path = tmp_path / "bad.yml"
    yaml_path.write_text(":", encoding="utf-8")
    out_dir = tmp_path / "out"

    summary = await evaluate_yaml(dataset=dataset, yaml_path=yaml_path, out_dir=out_dir, score_mode="exact")

    assert summary.avg_score == -1.0
    assert "boom" in summary.error
    assert (out_dir / "summary.json").exists()
