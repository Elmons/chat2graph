from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, List

from app.core.common.system_env import SystemEnv


@dataclass(frozen=True)
class BatchChatResult:
    custom_id: str
    content: str
    error: str = ""


def _normalize_batch_model(model: str) -> str:
    """Normalize model name for OpenAI-compatible batch APIs."""
    model_name = (model or "").strip()
    if not model_name:
        return model_name

    # litellm/provider prefix style: openai/qwen-max -> qwen-max
    if "/" in model_name:
        prefix, tail = model_name.split("/", 1)
        if prefix.lower() == "openai" and tail:
            model_name = tail

    return model_name


def _write_batch_jsonl(*, prompts: List[str], model: str, endpoint: str) -> Path:
    fd, file_path = tempfile.mkstemp(prefix="chat2graph_batch_", suffix=".jsonl")
    path = Path(file_path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            row = {
                "custom_id": str(idx),
                "method": "POST",
                "url": endpoint,
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _read_file_content_as_text(file_content_obj: Any) -> str:
    text = getattr(file_content_obj, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        value = text()
        if isinstance(value, str):
            return value
    read = getattr(file_content_obj, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        if isinstance(data, str):
            return data
    data = getattr(file_content_obj, "content", None)
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        return data
    raise RuntimeError("cannot decode batch output file content")


def _run_batch_chat_sync(
    *,
    prompts: List[str],
    model: str,
    api_base: str,
    api_key: str,
) -> List[BatchChatResult]:
    from openai import OpenAI

    endpoint = SystemEnv.LLM_BATCH_FILE_ENDPOINT
    completion_window = SystemEnv.LLM_BATCH_FILE_COMPLETION_WINDOW
    poll_interval = max(0.5, float(SystemEnv.LLM_BATCH_FILE_POLL_INTERVAL_SECONDS))
    timeout_raw = float(SystemEnv.LLM_BATCH_FILE_TIMEOUT_SECONDS)
    timeout_seconds = timeout_raw if timeout_raw > 0 else 0.0

    if not api_base or not api_key or not model:
        raise RuntimeError("batch scoring requires LLM_ENDPOINT, LLM_APIKEY and LLM_NAME")

    batch_model = _normalize_batch_model(model)
    jsonl_path = _write_batch_jsonl(prompts=prompts, model=batch_model, endpoint=endpoint)
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        with jsonl_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        input_file_id = _get_attr(uploaded, "id")
        if not input_file_id:
            raise RuntimeError("batch file upload failed: missing file id")

        batch_job = client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
        )
        batch_id = _get_attr(batch_job, "id")
        if not batch_id:
            raise RuntimeError("batch create failed: missing batch id")

        deadline = (time.time() + timeout_seconds) if timeout_seconds > 0 else None
        while True:
            status_obj = client.batches.retrieve(batch_id)
            status = _get_attr(status_obj, "status", "")
            if status == "completed":
                output_file_id = _get_attr(status_obj, "output_file_id")
                if not output_file_id:
                    raise RuntimeError("batch completed without output_file_id")
                output_obj = client.files.content(output_file_id)
                output_text = _read_file_content_as_text(output_obj)
                break
            if status in {"failed", "cancelled", "expired"}:
                error_info = _get_attr(status_obj, "errors", None) or _get_attr(
                    status_obj, "error", None
                )
                raise RuntimeError(f"batch failed with status={status}, errors={error_info}")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"batch timeout after {timeout_seconds}s, last_status={status}"
                )
            time.sleep(poll_interval)

        rows_by_id: dict[str, BatchChatResult] = {}
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = str(item.get("custom_id", ""))
            err = item.get("error")
            if err:
                rows_by_id[custom_id] = BatchChatResult(
                    custom_id=custom_id,
                    content="",
                    error=str(err),
                )
                continue

            resp = item.get("response", {}) if isinstance(item, dict) else {}
            body = resp.get("body", {}) if isinstance(resp, dict) else {}
            content = ""
            if isinstance(body, dict):
                choices = body.get("choices", [])
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")
            rows_by_id[custom_id] = BatchChatResult(custom_id=custom_id, content=content)

        ordered: List[BatchChatResult] = []
        for idx in range(len(prompts)):
            key = str(idx)
            ordered.append(
                rows_by_id.get(
                    key,
                    BatchChatResult(custom_id=key, content="", error="missing output row"),
                )
            )
        return ordered
    finally:
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            pass


async def run_batch_chat(
    *,
    prompts: List[str],
    model: str,
    api_base: str,
    api_key: str,
) -> List[BatchChatResult]:
    """Run OpenAI-compatible Batch File chat requests and return ordered results."""
    return await asyncio.to_thread(
        _run_batch_chat_sync,
        prompts=prompts,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
