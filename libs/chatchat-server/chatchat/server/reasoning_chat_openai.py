from typing import Any, AsyncIterator, List, Optional

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_delta_to_message_chunk


class ReasoningChatOpenAI(ChatOpenAI):
    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                       run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class = AIMessageChunk
        async for raw_chunk in await self.async_client.create(messages=message_dicts, **params):
            raw_chunk = raw_chunk.dict() if not isinstance(raw_chunk, dict) else raw_chunk
            if not raw_chunk["choices"]:
                continue
            choice = raw_chunk["choices"][0]
            message = _convert_delta_to_message_chunk(choice["delta"], default_chunk_class)
            reasoning = choice["delta"].get("reasoning_content")
            if reasoning:
                message.additional_kwargs["reasoning_content"] = reasoning
            generation_info = {k: choice[k] for k in ("finish_reason", "logprobs") if choice.get(k)}
            default_chunk_class = message.__class__
            chunk = ChatGenerationChunk(message=message, generation_info=generation_info or None)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.text, chunk=chunk, logprobs=choice.get("logprobs"))
            yield chunk
