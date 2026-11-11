"""Process-local BM25 indexes owned by the loaded vector store.

Callers must hold the vector store's acquire lock for reads and mutations, and
invalidate before mutating documents. Indexes are never written to disk.
"""
from threading import Lock

from langchain_community.retrievers import BM25Retriever

from chatchat.server.rag_performance import current_trace, stage


_attach_lock = Lock()
_CACHE_ATTRIBUTE = "_chatchat_bm25_cache"


class _BM25Cache:
    def __init__(self):
        self.lock = Lock()
        self.retriever = None


def get_bm25_retriever(vectorstore, top_k):
    # Only initialization is global; different knowledge bases build independently.
    with _attach_lock:
        cache = getattr(vectorstore, _CACHE_ATTRIBUTE, None)
        if cache is None:
            cache = _BM25Cache()
            setattr(vectorstore, _CACHE_ATTRIBUTE, cache)

    with cache.lock:
        hit = cache.retriever is not None
        trace = current_trace()
        if trace:
            trace.event("bm25.cache", hit=hit)
        if not hit:
            with stage("bm25.build"):
                import jieba

                # Keep the same Document references as the uncached retriever:
                # FAISS docstore.search attaches IDs to their metadata for fusion.
                # Add/delete replace entries, leaving existing request lists intact.
                docs = list(vectorstore.docstore._dict.values())
                cache.retriever = BM25Retriever.from_documents(
                    docs, preprocess_func=jieba.lcut_for_search,
                )
        # Share expensive index data, never mutable request parameters such as k.
        return cache.retriever.copy(update={"k": top_k}, deep=False)


def invalidate_bm25_cache(vectorstore):
    cache = getattr(vectorstore, _CACHE_ATTRIBUTE, None)
    if cache is not None:
        with cache.lock:
            cache.retriever = None
