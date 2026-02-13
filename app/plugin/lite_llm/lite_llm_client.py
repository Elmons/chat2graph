import re
from typing import Any, Dict, List, Optional, Union, cast

from app.core.common.system_env import SystemEnv
from app.core.common.type import MessageSourceType
from app.core.model.message import ModelMessage
from app.core.model.task import ToolCallContext
from app.core.prompt.model_service import FUNC_CALLING_PROMPT
from app.core.reasoner.model_service import ModelService
from app.core.toolkit.tool import FunctionCallResult, Tool


class LiteLlmClient(ModelService):
    """LiteLLM Client.
    Uses LiteLLM to interact with various LLM providers.
    API keys for providers (OpenAI, Anthropic, etc.) should be set as environment variables
    (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY). LiteLLM will pick them up.
    """

    def __init__(self):
        super().__init__()
        # e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet-20240229"
        # SystemEnv.LLM_ENDPOINT can be used as api_base for custom OpenAI-compatible endpoints
        self._model_alias: str = SystemEnv.LLM_NAME
        self._api_base: str = SystemEnv.LLM_ENDPOINT
        self._api_key: str = SystemEnv.LLM_APIKEY
        self._timeout_seconds: float = SystemEnv.LLM_TIMEOUT_SECONDS
        self._max_retries: int = SystemEnv.LLM_MAX_RETRIES
        self._temperature: float = SystemEnv.TEMPERATURE

        self._max_tokens: int = SystemEnv.MAX_TOKENS
        self._max_completion_tokens: int = SystemEnv.MAX_COMPLETION_TOKENS

    @staticmethod
    def _ensure_aiohttp_compat() -> None:
        """Patch aiohttp symbols expected by newer LiteLLM transports.

        dbgpt currently pins aiohttp to 3.8.4, while LiteLLM may reference
        exceptions introduced in later aiohttp versions.
        """
        try:
            import aiohttp
        except Exception:
            return

        # liteLLM's aiohttp transport maps these exception symbols; provide
        # backward-compatible aliases for older aiohttp versions.
        if not hasattr(aiohttp, "ConnectionTimeoutError"):
            aiohttp.ConnectionTimeoutError = aiohttp.ServerTimeoutError  # type: ignore[attr-defined]
        if not hasattr(aiohttp, "SocketTimeoutError"):
            aiohttp.SocketTimeoutError = aiohttp.ServerTimeoutError  # type: ignore[attr-defined]

    async def generate(
        self,
        sys_prompt: str,
        messages: List[ModelMessage],
        tools: Optional[List[Tool]] = None,
        tool_call_ctx: Optional[ToolCallContext] = None,
    ) -> ModelMessage:
        """Generate a text given a prompt using LiteLLM."""
        # prepare model request
        litellm_messages: List[Dict[str, str]] = self._prepare_model_request(
            sys_prompt=sys_prompt, messages=messages, tools=tools
        )

        self._ensure_aiohttp_compat()
        from litellm import completion
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
        from litellm.types.utils import ModelResponse, StreamingChoices

        model_response: Union[ModelResponse, CustomStreamWrapper] = completion(
            model=self._model_alias,
            api_base=self._api_base,
            api_key=self._api_key,
            messages=litellm_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_completion_tokens=self._max_completion_tokens,
            stream=False,
            timeout=self._timeout_seconds,
            max_retries=self._max_retries,
        )
        if isinstance(model_response, CustomStreamWrapper) or isinstance(
            model_response.choices[0], StreamingChoices
        ):
            raise ValueError(
                "Streaming responses are not supported in LiteLlmClient. "
                "Please PR to add a streaming feature."
            )

        self._record_token_usage(
            job_id=messages[-1].get_job_id(),
            model_response=model_response,
        )

        # call functions based on the model output
        func_call_results: Optional[List[FunctionCallResult]] = None
        if tools:
            func_call_results = await self.call_function(
                tools=tools,
                model_response_text=cast(str, model_response.choices[0].message.content),
                tool_call_ctx=tool_call_ctx,
            )

        # filter <function_call_result>...</function_call_result> content
        # since LLM may image the function call result, which should have been provided
        # by the function's execution return values
        model_response.choices[0].message.content = (
            re.sub(
                r"<function_call_result>.*?</function_call_result>",
                "",
                cast(str, model_response.choices[0].message.content),
                flags=re.DOTALL,
            ).strip()
            + "\n"
        )

        # parse model response to agent message
        response: ModelMessage = self._parse_model_response(
            model_response=model_response,
            messages=messages,
            func_call_results=func_call_results,
        )

        return response

    @staticmethod
    def _extract_total_tokens(model_response: Any) -> int:
        usage = getattr(model_response, "usage", None)
        if usage is None and isinstance(model_response, dict):
            usage = model_response.get("usage")
        if usage is None:
            return 0

        total_tokens = None
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int | float) and isinstance(
                    completion_tokens, int | float
                ):
                    total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if isinstance(prompt_tokens, int | float) and isinstance(
                    completion_tokens, int | float
                ):
                    total_tokens = prompt_tokens + completion_tokens

        if not isinstance(total_tokens, int | float):
            return 0
        return max(0, int(total_tokens))

    @staticmethod
    def _accumulate_job_tokens(job_id: str, inc_tokens: int) -> None:
        if not job_id or inc_tokens <= 0:
            return
        try:
            from app.core.service.job_service import JobService

            job_result = JobService.instance.get_job_result(job_id)
            job_result.tokens = int(job_result.tokens) + int(inc_tokens)
            JobService.instance.save_job_result(job_result)
        except Exception:
            # This path is best-effort: some model calls use synthetic job ids
            # (e.g., evaluator prompts) that do not exist in JobService.
            return

    def _record_token_usage(self, *, job_id: str, model_response: Any) -> None:
        total_tokens = self._extract_total_tokens(model_response)
        if total_tokens <= 0:
            return
        self._accumulate_job_tokens(job_id=job_id, inc_tokens=total_tokens)

    def _prepare_model_request(
        self,
        sys_prompt: str,
        messages: List[ModelMessage],
        tools: Optional[List[Tool]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare base messages for the LLM client."""
        if len(messages) == 0:
            raise ValueError("No messages provided.")

        # convert system prompt to system message
        if tools:
            sys_message = sys_prompt + FUNC_CALLING_PROMPT.strip()
        else:
            sys_message = sys_prompt.strip()
        base_messages: List[Dict[str, str]] = [{"role": "system", "content": sys_message}]

        # convert the conversation messages for LiteLLM
        for i, message in enumerate(messages):
            # handle the func call information in the agent message
            base_message_content = message.get_payload()
            func_call_results = message.get_function_calls()
            if func_call_results:
                base_message_content += (
                    "<function_call_result>\n"
                    + "\n".join(
                        [
                            f"{i + 1}. {result.status.value} called function "
                            f"{result.func_name}:\n"
                            f"Call objective: {result.call_objective}\n"
                            f"Function Output: {result.output}"
                            for j, result in enumerate(func_call_results)
                        ]
                    )
                    + "\n</function_call_result>"
                )

            # Chat2Graph <-> LiteLLM's last message role should be "user"
            if (len(messages) + i) % 2 == 1:
                base_messages.append({"role": "user", "content": base_message_content.strip()})
            else:
                base_messages.append({"role": "assistant", "content": base_message_content.strip()})

        return base_messages

    def _parse_model_response(
        self,
        model_response: Any,
        messages: List[ModelMessage],
        func_call_results: Optional[List[FunctionCallResult]] = None,
    ) -> ModelMessage:
        """Parse model response to agent message."""

        # determine the source type of the response
        if messages[-1].get_source_type() == MessageSourceType.MODEL:
            source_type = MessageSourceType.MODEL
        elif messages[-1].get_source_type() == MessageSourceType.ACTOR:
            source_type = MessageSourceType.THINKER
        else:
            source_type = MessageSourceType.ACTOR

        response = ModelMessage(
            payload=cast(
                str,
                (
                    model_response.choices[0].message.content or "The LLM response was missing."
                ).strip(),
            ),
            job_id=messages[-1].get_job_id(),
            step=messages[-1].get_step() + 1,
            source_type=source_type,
            function_calls=func_call_results,
        )

        return response
