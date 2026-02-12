import pytest

from app.core.common.system_env import SystemEnv
from app.core.common.type import ModelPlatformType


def test_system_env_generally():
    assert type(SystemEnv.PATH), str
    assert SystemEnv.MODEL_PLATFORM_TYPE, ModelPlatformType.LITELLM
    assert not SystemEnv.XXX


def test_system_env_setter_updates_cache_and_casts_bool():
    original = SystemEnv.PRINT_REASONER_OUTPUT
    try:
        SystemEnv.PRINT_REASONER_OUTPUT = False
        assert isinstance(SystemEnv.PRINT_REASONER_OUTPUT, bool)
        assert SystemEnv.PRINT_REASONER_OUTPUT is False
        SystemEnv.PRINT_REASONER_OUTPUT = "true"
        assert isinstance(SystemEnv.PRINT_REASONER_OUTPUT, bool)
        assert SystemEnv.PRINT_REASONER_OUTPUT is True
    finally:
        SystemEnv.PRINT_REASONER_OUTPUT = original


def test_system_env_setter_invalid_key():
    with pytest.raises(AttributeError):
        SystemEnv.INVALID_KEY = "value"

