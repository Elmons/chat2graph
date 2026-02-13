from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

import pytest

from app.core.common.system_env import SystemEnv
from app.core.workflow.evaluation.openai_batch_file import run_batch_chat


def _has_llm_config() -> bool:
    return bool(SystemEnv.LLM_NAME and SystemEnv.LLM_ENDPOINT and SystemEnv.LLM_APIKEY)


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_openai_batch_file_two_prompts_smoke() -> None:
    if os.getenv("CHAT2GRAPH_RUN_REAL_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_REAL_LLM_TESTS=1 to enable real LLM tests")
    if not _has_llm_config():
        pytest.skip("LLM not configured (LLM_NAME/LLM_ENDPOINT/LLM_APIKEY missing)")

    prompts = [
        "Reply with exactly one word: hi",
        "Reply with exactly one word: hello",
    ]
    out_path = Path(
        os.getenv(
            "CHAT2GRAPH_BATCH_TEST_OUTPUT",
            "test/example/workflow_generator/workflow_space/test_410/batch_smoke_output_from_test.json",
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_timeout = float(os.getenv("CHAT2GRAPH_BATCH_TEST_TIMEOUT_SECONDS", "0"))
    target_poll = float(os.getenv("CHAT2GRAPH_BATCH_TEST_POLL_INTERVAL_SECONDS", "2"))
    dump = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompts": prompts,
        "env": {
            "LLM_NAME": SystemEnv.LLM_NAME,
            "LLM_ENDPOINT": SystemEnv.LLM_ENDPOINT,
            "LLM_USE_OPENAI_BATCH_FILE": SystemEnv.LLM_USE_OPENAI_BATCH_FILE,
            "LLM_BATCH_FILE_TIMEOUT_SECONDS": target_timeout,
            "LLM_BATCH_FILE_POLL_INTERVAL_SECONDS": target_poll,
        },
        "status": "running",
        "results": [],
        "error": "",
    }

    old_timeout = SystemEnv.LLM_BATCH_FILE_TIMEOUT_SECONDS
    old_poll = SystemEnv.LLM_BATCH_FILE_POLL_INTERVAL_SECONDS
    SystemEnv.LLM_BATCH_FILE_TIMEOUT_SECONDS = target_timeout
    SystemEnv.LLM_BATCH_FILE_POLL_INTERVAL_SECONDS = target_poll
    try:
        results = await run_batch_chat(
            prompts=prompts,
            model=SystemEnv.LLM_NAME,
            api_base=SystemEnv.LLM_ENDPOINT,
            api_key=SystemEnv.LLM_APIKEY,
        )
        dump["status"] = "completed"
        dump["results"] = [
            {
                "custom_id": r.custom_id,
                "content": r.content,
                "error": r.error,
            }
            for r in results
        ]
        assert len(results) == 2
        assert [r.custom_id for r in results] == ["0", "1"]
        assert all(not r.error for r in results)
        assert all(isinstance(r.content, str) and r.content.strip() for r in results)
    except Exception as exc:
        dump["status"] = "failed"
        dump["error"] = repr(exc)
        raise
    finally:
        SystemEnv.LLM_BATCH_FILE_TIMEOUT_SECONDS = old_timeout
        SystemEnv.LLM_BATCH_FILE_POLL_INTERVAL_SECONDS = old_poll
        out_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"batch_result_file={out_path}")
