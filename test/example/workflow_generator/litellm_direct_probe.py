import time
import traceback

from app.core.common.system_env import SystemEnv

# Direct code config: edit here and run the file.
PROMPT = "你是千问几？"
REPEATS = 1
TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 0
MAX_TOKENS = 64
TEMPERATURE = 0.0

# By default, read model credentials from .env via SystemEnv.
MODEL = SystemEnv.LLM_NAME
API_BASE = SystemEnv.LLM_ENDPOINT
API_KEY = SystemEnv.LLM_APIKEY


def _ensure_aiohttp_compat() -> None:
    try:
        import aiohttp
    except Exception:
        return
    if not hasattr(aiohttp, "ConnectionTimeoutError"):
        aiohttp.ConnectionTimeoutError = aiohttp.ServerTimeoutError  # type: ignore[attr-defined]
    if not hasattr(aiohttp, "SocketTimeoutError"):
        aiohttp.SocketTimeoutError = aiohttp.ServerTimeoutError  # type: ignore[attr-defined]


def run_probe() -> None:
    _ensure_aiohttp_compat()
    from litellm import completion

    print(
        f"probe model={MODEL}, endpoint={API_BASE}, timeout={TIMEOUT_SECONDS}s, "
        f"max_retries={MAX_RETRIES}, repeats={REPEATS}"
    )

    for i in range(1, REPEATS + 1):
        started = time.time()
        try:
            response = completion(
                model=MODEL,
                api_base=API_BASE,
                api_key=API_KEY,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
            latency = time.time() - started
            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            print(f"[{i}] ok latency={latency:.2f}s content={content!r} usage={usage}")
        except Exception as e:
            latency = time.time() - started
            print(f"[{i}] error latency={latency:.2f}s type={type(e).__name__} err={e}")
            traceback.print_exc()


if __name__ == "__main__":
    run_probe()
