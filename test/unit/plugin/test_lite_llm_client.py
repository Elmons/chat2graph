from __future__ import annotations

from types import SimpleNamespace

from app.core.model.message import ModelMessage
from app.core.toolkit.tool import FunctionCallResult
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


def test_prepare_model_request_only_attaches_recent_function_results() -> None:
    client = LiteLlmClient()
    long_output = "X" * 12000
    fcr = FunctionCallResult(
        func_name="tool_a",
        func_args={},
        call_objective="demo",
        output=long_output,
    )
    messages = [
        ModelMessage(
            payload="m1",
            job_id="job",
            step=1,
            function_calls=[fcr],
        ),
        ModelMessage(
            payload="m2",
            job_id="job",
            step=2,
            function_calls=[fcr],
        ),
        ModelMessage(
            payload="m3",
            job_id="job",
            step=3,
            function_calls=[fcr],
        ),
    ]

    req = client._prepare_model_request(sys_prompt="sys", messages=messages)

    # req[0] is system prompt, then 3 conversation messages
    assert "<function_call_result>" in req[1]["content"]
    assert "<function_call_result>" in req[2]["content"]
    assert "<function_call_result>" in req[3]["content"]
    assert long_output in req[3]["content"]


def test_error_classifier_detects_rate_limit() -> None:
    rate_limit_error = Exception("RateLimitError: Request rate increased too quickly")

    assert LiteLlmClient._is_rate_limit_error(rate_limit_error)
