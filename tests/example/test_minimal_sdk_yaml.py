from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.common.system_env import SystemEnv
from app.core.model.message import HybridMessage, TextMessage
from app.core.sdk.agentic_service import AgenticService

MINIMAL_YAML = Path("app/core/sdk/minimal.yml")


def test_minimal_yaml_exists() -> None:
    assert MINIMAL_YAML.exists(), f"Missing minimal config: {MINIMAL_YAML}"


def test_minimal_yaml_loads_without_mcp() -> None:
    service = AgenticService.load(str(MINIMAL_YAML))
    assert service is not None
    assert service.name


def _has_llm_config() -> bool:
    return bool(SystemEnv.LLM_NAME and SystemEnv.LLM_ENDPOINT and SystemEnv.LLM_APIKEY)


@pytest.mark.real_llm
def test_minimal_submit_hi_real_llm_smoke() -> None:
    if os.getenv("CHAT2GRAPH_RUN_REAL_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_REAL_LLM_TESTS=1 to run live LLM execution")
    if not _has_llm_config():
        pytest.skip("LLM not configured (LLM_NAME/LLM_ENDPOINT/LLM_APIKEY missing)")

    service = AgenticService.load(str(MINIMAL_YAML))
    msg = service.session().submit(TextMessage(payload="hi")).wait()

    if isinstance(msg, list):
        pytest.fail("Unexpected list return from job.wait()")
    if isinstance(msg, HybridMessage):
        payload = msg.get_instruction_message().get_payload()
    else:
        payload = msg.get_payload()

    assert isinstance(payload, str)
    assert payload.strip()

