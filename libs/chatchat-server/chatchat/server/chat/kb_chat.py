from __future__ import annotations

import asyncio, json
from contextlib import nullcontext

from anyio import CancelScope
from time import perf_counter

from chatchat.server.chat.payload_logging import log_payload
from chatchat.server.rag_performance import new_trace, stage, ModelTimingCallback
from urllib.parse import urlencode
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional, Literal

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from chatchat.server.callback_handler.kb_reasoning_callback import KBReasoningCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate


from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template,
                                )


class KBEventSourceResponse(EventSourceResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # sse-starlette may be cancelled in send() while the generator is
            # suspended at yield. Explicitly close it before returning so the
            # model task and request trace do not outlive a disconnected client.
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


async def _close_request_resources(iterator, callback_iterator, model_task):
    # Attempt every cleanup even if an earlier generator's finally raises.
    # Shield against the AnyIO cancellation scope used for SSE disconnects.
    with CancelScope(shield=True):
        try:
            await iterator.aclose()
        finally:
            try:
                if callback_iterator is not None:
                    await callback_iterator.aclose()
            finally:
                if model_task is not None:
                    if not model_task.done():
                        model_task.cancel()
                    await asyncio.gather(model_task, return_exceptions=True)


async def kb_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
                kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
                top_k: Optional[int] = Body(None, description="匹配向量数；未提供时读取 kb_chat_settings.yaml"),
                score_threshold: Optional[float] = Body(
                    None,
                    description="知识库匹配相关度阈值；未提供时读取 kb_chat_settings.yaml。取值范围0-2，SCORE越小，相关度越高",
                    ge=0,
                    le=2,
                ),
                history: List[History] = Body(
                    [],
                    description="历史对话",
                    examples=[[
                        {"role": "user",
                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                        {"role": "assistant",
                        "content": "虎头虎脑"}]]
                ),
                stream: bool = Body(True, description="流式输出"),
                model: str = Body(get_default_llm(), description="LLM 模型名称。"),
                temperature: Optional[float] = Body(None, description="LLM 采样温度；未提供时读取 kb_chat_settings.yaml", ge=0.0, le=2.0),
                top_p: Optional[float] = Body(None, description="核采样概率；未提供时读取 kb_chat_settings.yaml", gt=0.0, le=1.0),
                max_tokens: Optional[int] = Body(
                    None,
                    description="限制LLM生成Token数量；未提供或为null时，按思考模式读取 kb_chat_settings.yaml"
                ),
                prompt_name: str = Body(
                    "default",
                    description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                ),
                return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                request: Request = None,
                chat_template_kwargs: Optional[Dict[str, Any]] = Body(
                    None,
                    description="模型聊天模板参数，例如 {\"enable_thinking\": false}；不提供时沿用模型服务默认值",
                ),
                repetition_penalty: Optional[float] = Body(
                    None, gt=0.0,
                    description="重复惩罚；未提供或为null时，按思考模式读取 kb_chat_settings.yaml",
                ),
                ):
    defaults = Settings.kb_chat_settings
    if top_k is None:
        top_k = defaults.TOP_K
    if score_threshold is None:
        score_threshold = defaults.SCORE_THRESHOLD
    if temperature is None:
        temperature = defaults.TEMPERATURE
    if top_p is None:
        top_p = defaults.TOP_P
    thinking = not (isinstance(chat_template_kwargs, dict)
                    and chat_template_kwargs.get("enable_thinking") is False)
    if max_tokens is None:
        max_tokens = defaults.THINKING_MAX_TOKENS if thinking else defaults.NON_THINKING_MAX_TOKENS
    if repetition_penalty is None:
        repetition_penalty = (defaults.THINKING_REPETITION_PENALTY if thinking
                              else defaults.NON_THINKING_REPETITION_PENALTY)
    trace = new_trace(request)
    request_id = trace.request_id if trace else uuid.uuid4().hex
    log_payload('/chat/kb_chat', 'request', request_id, dict(
        query=query, mode=mode, kb_name=kb_name, top_k=top_k,
        score_threshold=score_threshold,
        history=[History.from_data(h).model_dump() for h in history],
        stream=stream, model=model, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, repetition_penalty=repetition_penalty, prompt_name=prompt_name,
        return_direct=return_direct, chat_template_kwargs=chat_template_kwargs,
    ))
    if trace:
        trace.event("request.start", mode=mode, model=model)
    model_task = None
    callback_iterator = None
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            if trace:
                trace.event("request.end", status="not_found", duration_ms=round((perf_counter() - trace.started) * 1000, 3))
            response = BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
            log_payload('/chat/kb_chat', 'response', request_id, response.model_dump())
            return response
    
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        nonlocal history, prompt_name, model_task, callback_iterator

        history = [History.from_data(h) for h in history]

        with stage("retrieval"):
            if mode == "local_kb":
                docs = await run_in_threadpool(search_docs,
                                                query=query,
                                                knowledge_base_name=kb_name,
                                                top_k=top_k,
                                                score_threshold=score_threshold,
                                                file_name="",
                                                metadata={})
                source_documents = format_reference(kb_name, docs, request.base_url)
            elif mode == "temp_kb":
                docs = await run_in_threadpool(search_temp_docs,
                                                kb_name,
                                                query=query,
                                                top_k=top_k,
                                                score_threshold=score_threshold)
                source_documents = format_reference(kb_name, docs, request.base_url)
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                docs = [x.dict() for x in result.get("docs", [])]
                source_documents = [f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n""" for i,d in enumerate(docs)]
            else:
                docs = []
                source_documents = []
        # import rich
        # rich.print(dict(
        #     mode=mode,
        #     query=query,
        #     knowledge_base_name=kb_name,
        #     top_k=top_k,
        #     score_threshold=score_threshold,
        # ))
        # rich.print(docs)
        if return_direct:
            yield OpenAIChatOutput(
                id=f"chat{uuid.uuid4()}",
                model=None,
                object="chat.completion",
                content="",
                role="assistant",
                finish_reason="stop",
                docs=source_documents,
            ) .model_dump_json()
            return

        with stage("generation.prepare"):
            callback = KBReasoningCallbackHandler()
            callbacks = [callback]
            if trace:
                callbacks.append(ModelTimingCallback(trace))

            # Enable langchain-chatchat to support langfuse
            import os
            langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
            langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
            langfuse_host = os.environ.get('LANGFUSE_HOST')
            if langfuse_secret_key and langfuse_public_key and langfuse_host :
                from langfuse import Langfuse
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)

            model_kwargs = {}
            if top_p is not None:
                model_kwargs["top_p"] = top_p
            # The OpenAI SDK merges extra_body into the model service's JSON body.
            extra_body = {}
            if chat_template_kwargs is not None:
                extra_body["chat_template_kwargs"] = chat_template_kwargs
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
            if extra_body:
                model_kwargs["extra_body"] = extra_body

            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
                **({"model_kwargs": model_kwargs} if model_kwargs else {}),
            )
            # TODO： 视情况使用 API
            # # 加入reranker
            # if Settings.kb_settings.USE_RERANKER:
            #     reranker_model_path = get_model_path(Settings.kb_settings.RERANKER_MODEL)
            #     reranker_model = LangchainReranker(top_n=top_k,
            #                                     device=embedding_device(),
            #                                     max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,
            #                                     model_name_or_path=reranker_model_path
            #                                     )
            #     print("-------------before rerank-----------------")
            #     print(docs)
            #     docs = reranker_model.compress_documents(documents=docs,
            #                                              query=query)
            #     print("------------after rerank------------------")
            #     print(docs)
            context = "\n\n".join([doc["page_content"] for doc in docs])

            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_name = "empty"
            prompt_template = get_prompt_template("rag", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])

            chain = chat_prompt | llm

        # Begin a task that runs in the background.
        async def invoke_model():
            with stage("model.invoke"):
                await chain.ainvoke({"context": context, "question": query})

        model_task = asyncio.create_task(wrap_done(invoke_model(), callback.done))

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        callback_iterator = callback.aiter()
        if stream:
            # yield documents first
            ret = OpenAIChatOutput(
                id=f"chat{uuid.uuid4()}",
                object="chat.completion.chunk",
                content="",
                role="assistant",
                model=model,
                docs=source_documents,
            )
            yield ret.model_dump_json()

            first_content = True
            async for piece in callback_iterator:
                if piece["content"] and first_content:
                    if trace:
                        trace.event("api.first_content")
                    first_content = False
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content=piece["content"],
                    reasoning_content=piece["reasoning_content"],
                    role="assistant",
                    model=model,
                )
                yield ret.model_dump_json()
        else:
            answer = ""
            first_content = True
            async for piece in callback_iterator:
                if piece["content"] and first_content:
                    if trace:
                        trace.event("api.first_content")
                    first_content = False
                answer += piece["content"]
            ret = OpenAIChatOutput(
                id=f"chat{uuid.uuid4()}",
                object="chat.completion",
                content=answer,
                role="assistant",
                model=model,
            )
            yield ret.model_dump_json()
        await model_task

    async def timed_iterator():
        completed = False
        iterator = knowledge_base_chat_iterator()
        response_payload = {}
        contents, reasoning = [], []
        try:
            while True:
                # Do not retain a ContextVar token across a yield to the SSE
                # consumer: closing/disconnect may happen in another context.
                with trace.activate() if trace else nullcontext():
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        completed = True
                        break
                chunk = json.loads(item)
                response_payload.update({key: value for key, value in chunk.items()
                                         if key != 'docs'})
                if 'docs' in chunk:
                    response_payload['docs'] = chunk['docs']
                choice = chunk['choices'][0]
                message = choice.get('delta', choice.get('message', {}))
                contents.append(message.get('content') or '')
                reasoning.append(message.get('reasoning_content') or '')
                yield item
            choice = response_payload['choices'][0]
            message = choice.get('delta', choice.get('message', {}))
            message['content'] = ''.join(contents)
            if any(reasoning):
                message['reasoning_content'] = ''.join(reasoning)
            log_payload('/chat/kb_chat', 'response', request_id, response_payload)
        except BaseException as exc:
            if trace:
                trace.status = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "error"
                trace.event("request.error", error_type=type(exc).__name__)
            raise
        finally:
            with trace.activate() if trace else nullcontext():
                try:
                    await _close_request_resources(iterator, callback_iterator, model_task)
                except BaseException as exc:
                    if trace:
                        trace.status = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "error"
                        trace.event("request.error", phase="cleanup", error_type=type(exc).__name__)
                    raise
                finally:
                    if trace:
                        status = trace.status if completed or trace.status != "ok" else "cancelled"
                        trace.event("request.end", status=status,
                                    duration_ms=round((perf_counter() - trace.started) * 1000, 3))

    if stream:
        headers = {"X-Request-ID": trace.request_id} if trace else None
        return KBEventSourceResponse(timed_iterator(), headers=headers)
    else:
        # Exhaust the iterator so completion and context cleanup precede return.
        async for item in timed_iterator():
            result = item
        return result
