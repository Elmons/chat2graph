from __future__ import annotations

from pathlib import Path

from app.core.model.message import TextMessage
from app.core.sdk.agentic_service import AgenticService


def test_entry_expert_name_single_expert() -> None:
    service = AgenticService.load("app/core/sdk/minimal.yml")
    assert service.entry_expert_name() == "Q&A Expert"


def test_session_submit_autobinds_entry_expert_in_single_expert_mode(mocker) -> None:
    # avoid executing the job (no real LLM calls)
    mocker.patch("app.core.sdk.wrapper.session_wrapper.run_in_thread", lambda _fn: None)

    service = AgenticService.load("app/core/sdk/minimal.yml")
    msg = TextMessage(payload="hi")
    job = service.session().submit(msg)
    assert job.job.assigned_expert_name == "Q&A Expert"
    assert msg.get_assigned_expert_name() == "Q&A Expert"


def test_session_submit_does_not_autobind_when_multiple_experts(mocker, tmp_path: Path) -> None:
    mocker.patch("app.core.sdk.wrapper.session_wrapper.run_in_thread", lambda _fn: None)

    multi_expert_yaml = tmp_path / "multi.yml"
    multi_expert_yaml.write_text(
        """
app:
  name: "Multi Expert Minimal"
  desc: "No toolkit; two experts."
  version: "0.0.1"

plugin:
  workflow_platform: "BUILTIN"

reasoner:
  type: "DUAL"

experts:
  - profile:
      name: "Expert A"
      desc: "A"
    workflow:
      - - instruction: "You are Expert A."
          output_schema: "answer: text"
          actions: []
  - profile:
      name: "Expert B"
      desc: "B"
    workflow:
      - - instruction: "You are Expert B."
          output_schema: "answer: text"
          actions: []
""".lstrip(),
        encoding="utf-8",
    )

    service = AgenticService.load(str(multi_expert_yaml))
    assert service.entry_expert_name() is None

    msg = TextMessage(payload="hi")
    job = service.session().submit(msg)
    assert job.job.assigned_expert_name is None
    assert msg.get_assigned_expert_name() is None

