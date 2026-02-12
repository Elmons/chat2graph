from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.common.singleton import AbcSingleton, Singleton
from app.core.sdk.init_server import init_server


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Reset Singleton caches to avoid test cross-talk.

    Also re-initialize core services/DB so migrated legacy tests that rely on
    ServiceFactory/DaoFactory don't break after resets.
    """
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
    init_server()
    yield
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
