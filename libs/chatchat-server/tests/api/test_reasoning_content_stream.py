import asyncio
import importlib
import importlib.metadata
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.callback_handler.kb_reasoning_callback import KBReasoningCallbackHandler
from chatchat.server.reasoning_chat_openai import ReasoningChatOpenAI
from chatchat.server import utils


def test_chunk_serializes_reasoning_inside_delta_only():
    payload = OpenAIChatOutput(content="", reasoning_content="think").model_dump()
    assert payload["choices"][0]["delta"]["reasoning_content"] == "think"
    assert "reasoning_content" not in payload
    assert "reasoning_content" not in OpenAIChatOutput(
        object="chat.completion", content="answer", reasoning_content="think"
    ).model_dump()


def test_content_only_and_docs_events_do_not_gain_reasoning_content():
    content_only = OpenAIChatOutput(content="answer").model_dump()
    docs_event = OpenAIChatOutput(content="", docs=["source"]).model_dump()
    assert "reasoning_content" not in content_only["choices"][0]["delta"]
    assert "reasoning_content" not in docs_event["choices"][0]["delta"]
    assert docs_event["docs"] == ["source"]


@pytest.mark.asyncio
async def test_callback_yields_reasoning_only_and_skips_empty_chunks():
    callback = KBReasoningCallbackHandler()
    received = []
    received_event = asyncio.Event()

    async def consume():
        async for piece in callback.aiter():
            received.append(piece)
            received_event.set()

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await callback.on_llm_new_token(
        "", chunk=ChatGenerationChunk(message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": "think"}))
    )
    await callback.on_llm_new_token("", chunk=ChatGenerationChunk(message=AIMessageChunk(content="")))
    await received_event.wait()
    await callback.on_llm_end({})
    await consumer
    assert received == [{"content": "", "reasoning_content": "think"}]


@pytest.mark.asyncio
async def test_callback_drains_queued_pieces_after_generation_ends():
    callback = KBReasoningCallbackHandler()
    for content in ("first", "second", "third"):
        await callback.on_llm_new_token(content)
    callback.done.set()
    received = [piece async for piece in callback.aiter()]
    assert [piece["content"] for piece in received] == ["first", "second", "third"]


def test_langchain_openai_version_is_locked():
    assert importlib.metadata.version("langchain-openai") == "0.0.6"


def test_real_openai_chunk_preserves_reasoning_content_through_dict():
    delta = ChoiceDelta(content="", reasoning_content="think")
    chunk = ChatCompletionChunk(
        id="test", choices=[Choice(delta=delta, index=0, finish_reason=None)],
        created=0, model="test", object="chat.completion.chunk",
    )
    assert delta.model_extra["reasoning_content"] == "think"
    assert chunk.dict()["choices"][0]["delta"]["reasoning_content"] == "think"


@pytest.mark.asyncio
async def test_astream_preserves_reasoning_from_real_openai_chunk():
    class Stream:
        def __init__(self, item): self.item = item
        def __aiter__(self): return self
        async def __anext__(self):
            if self.item is None: raise StopAsyncIteration
            item, self.item = self.item, None
            return item

    raw = ChatCompletionChunk(
        id="test", choices=[Choice(delta=ChoiceDelta(content="", reasoning_content="think"), index=0, finish_reason=None)],
        created=0, model="test", object="chat.completion.chunk",
    )
    model = ReasoningChatOpenAI.construct(async_client=SimpleNamespace(create=AsyncMock(return_value=Stream(raw))))
    object.__setattr__(model, "_create_message_dicts", Mock(return_value=([], {})))
    chunks = [chunk async for chunk in model._astream([])]
    assert chunks[0].text == ""
    assert chunks[0].message.additional_kwargs["reasoning_content"] == "think"


@pytest.mark.asyncio
async def test_astream_preserves_content_and_reasoning_in_one_chunk():
    class Stream:
        def __init__(self, item): self.item = item
        def __aiter__(self): return self
        async def __anext__(self):
            if self.item is None: raise StopAsyncIteration
            item, self.item = self.item, None
            return item

    raw = ChatCompletionChunk(
        id="test", choices=[Choice(delta=ChoiceDelta(content="answer", reasoning_content="think"), index=0, finish_reason=None)],
        created=0, model="test", object="chat.completion.chunk",
    )
    model = ReasoningChatOpenAI.construct(async_client=SimpleNamespace(create=AsyncMock(return_value=Stream(raw))))
    object.__setattr__(model, "_create_message_dicts", Mock(return_value=([], {})))
    chunks = [chunk async for chunk in model._astream([])]
    assert chunks[0].text == "answer"
    assert chunks[0].message.additional_kwargs["reasoning_content"] == "think"


@pytest.mark.asyncio
async def test_astream_keeps_finish_metadata_and_notifies_empty_token():
    class Stream:
        def __aiter__(self):
            self.items = iter([
                {"choices": []},
                {"choices": [{"delta": {"role": "assistant", "reasoning_content": "think"},
                              "finish_reason": "stop", "logprobs": {"tokens": [1]}}]},
            ])
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    model = ReasoningChatOpenAI.construct(
        async_client=SimpleNamespace(create=AsyncMock(return_value=Stream()))
    )
    object.__setattr__(model, "_create_message_dicts", Mock(return_value=([], {})))
    run_manager = SimpleNamespace(on_llm_new_token=AsyncMock())
    chunks = [chunk async for chunk in model._astream([], run_manager=run_manager)]
    assert len(chunks) == 1
    assert chunks[0].generation_info == {
        "finish_reason": "stop", "logprobs": {"tokens": [1]}
    }
    run_manager.on_llm_new_token.assert_awaited_once_with(
        token="", chunk=chunks[0], logprobs={"tokens": [1]}
    )


@pytest.mark.parametrize("platform_type, expected", [("custom openai", ReasoningChatOpenAI), ("openai", utils.ChatOpenAI), (None, utils.ChatOpenAI)])
def test_model_factory_selects_reasoning_adapter_only_for_custom_openai(monkeypatch, platform_type, expected):
    monkeypatch.setattr(utils, "get_model_info", lambda _: {"platform_type": platform_type})
    monkeypatch.setattr(expected, "__init__", lambda self, **kwargs: None)
    model = utils.get_ChatOpenAI(model_name="test", callbacks=[])
    assert type(model) is expected


@pytest.fixture
def kb_chat_module(monkeypatch):
    # tools_factory imports the live knowledge-base DB at module import time.
    # The KB chat path under test does not call this search-engine dependency.
    module_name = "chatchat.server.chat.kb_chat"
    chat_package = importlib.import_module("chatchat.server.chat")
    previous_module = sys.modules.pop(module_name, None)
    previous_attribute = getattr(chat_package, "kb_chat", None)
    tools_package = ModuleType("chatchat.server.agent.tools_factory")
    tools_package.__path__ = []
    search_module = ModuleType("chatchat.server.agent.tools_factory.search_internet")
    search_module.search_engine = lambda *args: None
    monkeypatch.setitem(sys.modules, tools_package.__name__, tools_package)
    monkeypatch.setitem(sys.modules, search_module.__name__, search_module)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_attribute is None:
            if hasattr(chat_package, "kb_chat"):
                delattr(chat_package, "kb_chat")
        else:
            chat_package.kb_chat = previous_attribute


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_kb_chat_keeps_reasoning_separate_from_answer(monkeypatch, stream, kb_chat_module):
    pieces = [
        ("", "think"),
        ("answer", "more thought"),
        (" done", ""),
    ]
    monkeypatch.setattr(
        kb_chat_module.KBServiceFactory, "get_service_by_name", lambda _: object()
    )

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return []

    monkeypatch.setattr(kb_chat_module, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(kb_chat_module, "format_reference", lambda *args: ["source"])
    monkeypatch.setattr(kb_chat_module, "get_prompt_template", lambda *args: "{context} {question}")

    class FakeChain:
        async def ainvoke(self, values):
            for content, reasoning in pieces:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs={"reasoning_content": reasoning},
                    )
                )
                await callback.on_llm_new_token(content, chunk=chunk)

    class FakePrompt:
        def __or__(self, llm):
            return FakeChain()

    monkeypatch.setattr(
        kb_chat_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: FakePrompt(),
    )

    def fake_model_factory(**kwargs):
        nonlocal callback
        callback = kwargs["callbacks"][0]
        return object()

    callback = None
    monkeypatch.setattr(kb_chat_module, "get_ChatOpenAI", fake_model_factory)
    response = await kb_chat_module.kb_chat(
        query="question",
        mode="local_kb",
        kb_name="test",
        top_k=3,
        score_threshold=1.0,
        history=[],
        stream=stream,
        model="test",
        temperature=0,
        max_tokens=32,
        prompt_name="default",
        return_direct=False,
        request=SimpleNamespace(base_url="http://test/"),
    )
    if stream:
        events = [json.loads(item) async for item in response.body_iterator]
        assert events[0]["docs"] == ["source"]
        assert "reasoning_content" not in events[0]["choices"][0]["delta"]
        assert len(events) == 4
        assert [item["choices"][0]["delta"].get("reasoning_content") for item in events[1:]] == [
            "think", "more thought", None
        ]
        assert [item["choices"][0]["delta"]["content"] for item in events[1:]] == [
            "", "answer", " done"
        ]
    else:
        event = json.loads(response)
        assert event["choices"][0]["message"]["content"] == "answer done"
        assert "reasoning_content" not in event
