import pytest

from app.core.common.system_env import SystemEnv
from app.core.common.type import ReasonerType
from app.core.reasoner.mono_model_reasoner import MonoModelReasoner
from app.core.service.reasoner_service import ReasonerService

EnhancedReasoner = pytest.importorskip(
    "app.core.memory.enhanced.mem_reasoner",
    reason="Enhanced memory reasoner module not present in this repo snapshot.",
).EnhancedReasoner


def test_reasoner_service_wraps_when_enabled():
    """Wrap the reasoner with EnhancedReasoner when ENABLE_MEMFUSE is True."""
    SystemEnv.ENABLE_MEMFUSE = True
    svc = ReasonerService()
    svc.init_reasoner(ReasonerType.MONO)
    r = svc.get_reasoner()
    assert isinstance(r, EnhancedReasoner)


def test_reasoner_service_no_wrap_when_disabled():
    """Do not wrap when ENABLE_MEMFUSE is False."""
    SystemEnv.ENABLE_MEMFUSE = False
    svc = ReasonerService()
    svc.init_reasoner(ReasonerType.MONO)
    r = svc.get_reasoner()
    assert not isinstance(r, EnhancedReasoner)
    assert isinstance(r, MonoModelReasoner)

