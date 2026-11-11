from chatchat.server.rag_performance import current_trace, RetrievalTimingCallback

from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService
from chatchat.server.file_rag.retrievers.bm25_cache import get_bm25_retriever


class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int or float,
    ):
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        bm25_retriever = get_bm25_retriever(vectorstore, top_k)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return EnsembleRetrieverService(retriever=ensemble_retriever)

    def get_relevant_documents(self, query: str):
        trace = current_trace()
        kwargs = {"callbacks": [RetrievalTimingCallback(trace)]} if trace else {}
        return self.retriever.get_relevant_documents(query, **kwargs)[: self.top_k]
