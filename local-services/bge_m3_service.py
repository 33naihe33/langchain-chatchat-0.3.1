"""OpenAI-compatible, CPU-only embedding service for BAAI/bge-m3."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_ID = "bge-m3"
MAX_INPUT_TOKENS = 8192


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


def _load_components() -> tuple[Any, Any]:
    """Load the model only when the service receives its first request."""
    from FlagEmbedding import BGEM3FlagModel
    from transformers import AutoTokenizer

    model_path = os.environ.get("BGE_M3_MODEL_PATH", "models/bge-m3")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BGEM3FlagModel(model_path, use_fp16=False, device="cpu")
    return model, tokenizer


def create_app(
    model_loader: Callable[[], tuple[Any, Any]] = _load_components,
) -> FastAPI:
    app = FastAPI(title="Local BGE-M3 Embedding API")
    components: tuple[Any, Any] | None = None
    components_lock = threading.Lock()
    inference_gate = threading.BoundedSemaphore(1)

    def get_components() -> tuple[Any, Any]:
        nonlocal components
        with components_lock:
            if components is None:
                components = model_loader()
            return components

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/embeddings")
    def create_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        if request.model != MODEL_ID:
            raise HTTPException(
                status_code=400,
                detail={"message": f"unsupported model: {request.model}", "model": MODEL_ID},
            )

        texts = [request.input] if isinstance(request.input, str) else request.input
        model, tokenizer = get_components()
        token_counts = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
        actual_tokens = max(token_counts, default=0)
        if actual_tokens > MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "input exceeds BGE-M3 limit; lower CHUNK_SIZE",
                    "actual_tokens": actual_tokens,
                    "limit_tokens": MAX_INPUT_TOKENS,
                },
            )

        with inference_gate:
            result = model.encode(texts, return_dense=True)
        vectors = result["dense_vecs"]
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [float(value) for value in vector],
                    "index": index,
                }
                for index, vector in enumerate(vectors)
            ],
            "model": MODEL_ID,
            "usage": {"prompt_tokens": sum(token_counts), "total_tokens": sum(token_counts)},
        }

    return app
