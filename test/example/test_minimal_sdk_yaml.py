from __future__ import annotations

from collections.abc import Iterator
import logging
import os
from pathlib import Path

import pytest

from app.core.common.singleton import AbcSingleton, Singleton
from app.core.common.system_env import SystemEnv
from app.core.model.message import HybridMessage, TextMessage
from app.core.sdk.agentic_service import AgenticService

MINIMAL_YAML = Path("app/core/sdk/minimal.yml")


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """AgenticService is a Singleton; clear caches to avoid test cross-talk."""
    # Silence noisy third-party loggers for this minimal example test.
    logging.getLogger("dbgpt.storage.vector_store.base").setLevel(logging.ERROR)
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
    yield
    Singleton._instances.clear()
    AbcSingleton._instances.clear()


def test_minimal_yaml_exists() -> None:
    assert MINIMAL_YAML.exists(), f"Missing minimal config: {MINIMAL_YAML}"


def test_minimal_yaml_loads_without_mcp() -> None:
    service = AgenticService.load(str(MINIMAL_YAML))
    assert service is not None
    assert service.name


def _has_llm_config() -> bool:
    # Chat2Graph's default model platform is LiteLLM; these are the common required fields.
    return bool(SystemEnv.LLM_NAME and SystemEnv.LLM_ENDPOINT and SystemEnv.LLM_APIKEY)


@pytest.mark.real_llm
def test_minimal_submit_hi_async_if_llm_configured() -> None:
    if os.getenv("CHAT2GRAPH_RUN_REAL_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_REAL_LLM_TESTS=1 to run live LLM execution")
    if not _has_llm_config():
        pytest.skip(
            "LLM not configured (LLM_NAME/LLM_ENDPOINT/LLM_APIKEY missing); skipping execute"
        )

    service = AgenticService.load(str(MINIMAL_YAML))
    # Use the existing async submission API: submit in a background thread, then wait.
    msg = service.session().submit(TextMessage(payload="hi")).wait()
    if isinstance(msg, list):
        assert msg, "No messages returned for job"
        first = msg[0]
        if isinstance(first, HybridMessage):
            payload = first.get_instruction_message().get_payload()
        else:
            payload = first.get_payload()
    elif isinstance(msg, HybridMessage):
        payload = msg.get_instruction_message().get_payload()
    else:
        payload = msg.get_payload()
    assert isinstance(payload, str)
    assert payload.strip()
