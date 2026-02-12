from __future__ import annotations

from collections.abc import Iterator
import os

import pytest

from app.core.common.singleton import AbcSingleton, Singleton
from app.core.common.system_env import SystemEnv
from app.core.model.knowledge import Knowledge
from app.core.sdk.init_server import init_server
from app.core.workflow.operator import Operator


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset Singleton caches to avoid test cross-talk.

    Also re-initialize core services/DB so migrated legacy tests that rely on
    ServiceFactory/DaoFactory don't break after resets.
    """
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
    # Ensure external services are disabled by default in tests.
    SystemEnv.ENABLE_MEMFUSE = False
    init_server()
    if os.getenv("CHAT2GRAPH_RUN_REAL_KNOWLEDGE_TESTS") not in ("1", "true", "True", "yes"):
        monkeypatch.setattr(
            Operator,
            "get_knowledge",
            lambda _self, _job: Knowledge(global_chunks=[], local_chunks=[]),
            raising=True,
        )
    yield
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
