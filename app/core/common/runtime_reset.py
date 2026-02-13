"""Process-level runtime reset utilities.

These helpers are primarily intended for evaluation/benchmark loops where many
different Agentic YAML configurations are loaded in a single Python process.

Chat2Graph uses Singleton/AbcSingleton patterns for many services; without an
explicit reset, repeated loads can leak state across rounds (toolkit graph,
agent registry, cached env values, etc.) and harm reproducibility.
"""

from __future__ import annotations


def reset_runtime_state() -> None:
    """Reset in-process singleton/env caches to avoid cross-run contamination.

    Notes:
    - This does *not* delete on-disk artifacts/DB; it only clears in-memory caches.
    - Callers are expected to (re)initialize services by calling `AgenticService.load(...)`
      or `init_server()` after reset, depending on their workflow.
    """
    from app.core.common.singleton import AbcSingleton, Singleton
    from app.core.common import system_env

    Singleton._instances.clear()
    AbcSingleton._instances.clear()
    system_env._env_values.clear()
