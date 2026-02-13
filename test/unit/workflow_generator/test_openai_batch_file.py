from __future__ import annotations

from app.core.workflow.evaluation.openai_batch_file import _normalize_batch_model


def test_normalize_batch_model_strips_openai_prefix() -> None:
    assert _normalize_batch_model("openai/qwen-max") == "qwen-max"
    assert _normalize_batch_model("openai/qwen-plus") == "qwen-plus"


def test_normalize_batch_model_keeps_qwen3_name() -> None:
    assert _normalize_batch_model("qwen3-max") == "qwen3-max"
    assert _normalize_batch_model("openai/qwen3-max") == "qwen3-max"
    assert _normalize_batch_model("openai/qwen3-plus") == "qwen3-plus"


def test_normalize_batch_model_keeps_other_names() -> None:
    assert _normalize_batch_model("gpt-4o-mini") == "gpt-4o-mini"
    assert _normalize_batch_model("anthropic/claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"
