"""Opt-in RAG timing, without logging prompts, retrieved text or model output."""
import asyncio
import os
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from time import perf_counter
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from loguru import logger


_trace: ContextVar[Optional["RequestTrace"]] = ContextVar("rag_performance", default=None)


def current_trace():
    return _trace.get()


def new_trace(request):
    if os.environ.get("RAG_PERF_LOG", "").lower() not in ("1", "true", "yes"):
        return None
    headers = getattr(request, "headers", {})
    request_id = headers.get("x-request-id", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", request_id):
        request_id = uuid.uuid4().hex
    return RequestTrace(request_id)


class RequestTrace:
    def __init__(self, request_id):
        self.request_id = request_id
        self.started = perf_counter()
        self.status = "ok"

    @contextmanager
    def activate(self):
        token = _trace.set(self)
        try:
            yield self
        finally:
            _trace.reset(token)

    def event(self, name, **fields):
        payload = dict(
            event=name, request_id=self.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_ms=round((perf_counter() - self.started) * 1000, 3),
            **fields,
        )
        logger.info("RAG_PERF {}", json.dumps(payload, ensure_ascii=False))


@contextmanager
def stage(name):
    trace = current_trace()
    if trace is None:
        yield
        return
    started = perf_counter()
    trace.event(name + ".start")
    status, error = "ok", {}
    try:
        yield
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "error"
        trace.status = status
        error = {"error_type": type(exc).__name__}
        raise
    finally:
        trace.event(name + ".end", duration_ms=round((perf_counter() - started) * 1000, 3),
                    status=status, **error)


class ModelTimingCallback(BaseCallbackHandler):
    """Runs inline so timestamps reflect callback arrival rather than executor delay."""
    run_inline = True

    def __init__(self, trace):
        self.trace = trace
        self.first_content = False
        self.started = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.started = perf_counter()
        self.trace.event("model.start")

    def on_llm_new_token(self, token, **kwargs):
        if token and not self.first_content:
            self.first_content = True
            self.trace.event("model.first_content")

    def on_llm_end(self, response, **kwargs):
        self._end("ok")

    def on_llm_error(self, error, **kwargs):
        self.trace.status = "error"
        self._end("error", error_type=type(error).__name__)

    def _end(self, status, **fields):
        duration = {} if self.started is None else {
            "duration_ms": round((perf_counter() - self.started) * 1000, 3)
        }
        self.trace.event("model.end", status=status, **duration, **fields)


class RetrievalTimingCallback(BaseCallbackHandler):
    run_inline = True

    def __init__(self, trace):
        self.trace = trace
        self.runs = {}

    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        name = (serialized or {}).get("id", [""])[-1]
        stage_name = {"BM25Retriever": "bm25.search", "VectorStoreRetriever": "vector.search"}.get(name)
        if stage_name:
            self.runs[run_id] = (stage_name, perf_counter())
            self.trace.event(stage_name + ".start")

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id, "ok", document_count=len(documents))

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self.trace.status = "error"
        self._end(run_id, "error", error_type=type(error).__name__)

    def _end(self, run_id, status, **fields):
        run = self.runs.pop(run_id, None)
        if run:
            name, started = run
            self.trace.event(name + ".end", status=status,
                             duration_ms=round((perf_counter() - started) * 1000, 3), **fields)
