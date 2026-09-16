import re
from typing import List, Optional, Tuple

from langchain.docstore.document import Document

from .chinese_recursive_text_splitter import ChineseRecursiveTextSplitter


DEFAULT_QA_SEPARATORS = [
    "\n\n",
    "\n",
    "。|！|？",
    r"\.\s|\!\s|\?\s",
    r"；|;\s",
    r"，|,\s",
    "",
]
QUESTION_MARKER = re.compile(r"^[ \t]*(?:Q|q|问)[ \t]*[:：][ \t]*", re.MULTILINE)
ANSWER_MARKER = re.compile(r"^[ \t]*(?:A|a|答)[ \t]*[:：][ \t]*", re.MULTILINE)


class QATextSplitter(ChineseRecursiveTextSplitter):
    """Keep each line-marked question and answer together when possible."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("pipeline", None)
        separators = list(kwargs.pop("separators", DEFAULT_QA_SEPARATORS))
        if "" not in separators:
            separators.append("")
        super().__init__(*args, separators=separators, **kwargs)

    def _parse_question_and_answer(
        self, record: str
    ) -> Tuple[str, Optional[str], str]:
        match = ANSWER_MARKER.search(record)
        if match is None:
            return record, None, ""
        return record[: match.start()].rstrip(), match.group(), record[match.end() :]

    def _split_record(self, record: str) -> List[str]:
        record = record.strip()
        if not record:
            return []
        if len(record) <= self._chunk_size:
            return [record]

        question, answer_marker, answer_body = self._parse_question_and_answer(record)
        if answer_marker is None:
            return super().split_text(record)

        first_prefix = f"{question}\n{answer_marker}\n"
        if len(first_prefix) + self._chunk_overlap >= self._chunk_size:
            raise ValueError(
                "increase chunk_size or lower chunk_overlap for the QA question prefix"
            )

        answer_chunks = ChineseRecursiveTextSplitter(
            chunk_size=self._chunk_size - len(first_prefix),
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
        ).split_text(answer_body.strip())
        if not answer_chunks:
            return [record]

        return [f"{first_prefix}{answer_chunks[0]}"] + [
            f"{question}\n{chunk}" for chunk in answer_chunks[1:]
        ]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        matches = list(QUESTION_MARKER.finditer(text))
        if not matches:
            return super().split_text(text)

        chunks = super().split_text(text[: matches[0].start()])
        starts = [match.start() for match in matches] + [len(text)]
        for start, end in zip(starts, starts[1:]):
            chunks.extend(self._split_record(text[start:end]))
        return chunks

    def _split_source_group(self, documents: List[Document]) -> List[Document]:
        metadata = documents[0].metadata.copy()
        text = "\n".join(document.page_content for document in documents)
        return [
            Document(page_content=chunk, metadata=metadata.copy())
            for chunk in self.split_text(text)
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        group: List[Document] = []
        for document in documents:
            source = document.metadata.get("source")
            if group and (
                source is None or source != group[0].metadata.get("source")
            ):
                chunks.extend(self._split_source_group(group))
                group = []
            group.append(document)
        if group:
            chunks.extend(self._split_source_group(group))
        return chunks
