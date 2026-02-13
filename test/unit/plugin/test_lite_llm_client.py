from __future__ import annotations

from types import SimpleNamespace

from app.plugin.lite_llm.lite_llm_client import LiteLlmClient


def test_extract_total_tokens_from_usage_dict() -> None:
    response = {"usage": {"total_tokens": 123}}
    assert LiteLlmClient._extract_total_tokens(response) == 123


def test_extract_total_tokens_from_usage_object_prompt_completion_sum() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=7),
    )
    assert LiteLlmClient._extract_total_tokens(response) == 47


def test_record_token_usage_calls_accumulate(monkeypatch) -> None:
    client = LiteLlmClient()
    calls = []

    def _fake_accumulate(*, job_id: str, inc_tokens: int) -> None:
        calls.append((job_id, inc_tokens))

    monkeypatch.setattr(client, "_accumulate_job_tokens", _fake_accumulate)

    response = {"usage": {"total_tokens": 9}}
    client._record_token_usage(job_id="job-1", model_response=response)

    assert calls == [("job-1", 9)]


def test_record_token_usage_skips_when_usage_missing(monkeypatch) -> None:
    client = LiteLlmClient()
    calls = []

    def _fake_accumulate(*, job_id: str, inc_tokens: int) -> None:
        calls.append((job_id, inc_tokens))

    monkeypatch.setattr(client, "_accumulate_job_tokens", _fake_accumulate)

    client._record_token_usage(job_id="job-1", model_response={})

    assert calls == []
