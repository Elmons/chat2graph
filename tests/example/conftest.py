from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.common.singleton import AbcSingleton, Singleton


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Reset Singleton caches to avoid test cross-talk."""
    Singleton._instances.clear()
    AbcSingleton._instances.clear()
    yield
    Singleton._instances.clear()
    AbcSingleton._instances.clear()

