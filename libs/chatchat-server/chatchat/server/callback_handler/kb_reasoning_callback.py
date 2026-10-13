import asyncio
from typing import AsyncIterator, Dict, Optional

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.outputs import ChatGenerationChunk


class KBReasoningCallbackHandler(AsyncIteratorCallbackHandler):
    queue: asyncio.Queue

    async def on_llm_new_token(
        self, token: str, *, chunk: Optional[ChatGenerationChunk] = None, **kwargs
    ) -> None:
        reasoning_content = ""
        if chunk is not None:
            reasoning_content = chunk.message.additional_kwargs.get(
                "reasoning_content", ""
            )
        if token or reasoning_content:
            self.queue.put_nowait(
                {"content": token, "reasoning_content": reasoning_content}
            )

    async def aiter(self) -> AsyncIterator[Dict[str, str]]:
        """Drain queued chunks before treating completion as the end of the stream."""
        while True:
            if not self.queue.empty():
                yield self.queue.get_nowait()
                continue
            if self.done.is_set():
                break

            get_task = asyncio.create_task(self.queue.get())
            done_task = asyncio.create_task(self.done.wait())
            try:
                await asyncio.wait(
                    (get_task, done_task), return_when=asyncio.FIRST_COMPLETED
                )
                if get_task.done():
                    yield get_task.result()
            finally:
                for task in (get_task, done_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(get_task, done_task, return_exceptions=True)
