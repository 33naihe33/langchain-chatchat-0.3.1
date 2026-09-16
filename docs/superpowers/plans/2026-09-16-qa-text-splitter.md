# QA Text Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable splitter that makes each structured QA pair one vector-store chunk.

**Architecture:** `QATextSplitter` extends `ChineseRecursiveTextSplitter`. It groups sequential loader documents by source, joins their content with one newline, then splits preamble and QA records. Overlong answers are split by a new inner recursive splitter whose budget includes the repeated question prefix.

**Tech Stack:** Python, LangChain `Document`/`TextSplitter`, pytest.

## Global Constraints

- Do not modify the source DOCX or change the default `TEXT_SPLITTER_NAME`.
- Support `Q/q/问` and `A/a/答` at line start, with `:` or `：` and optional whitespace.
- Split preamble and non-QA text through `ChineseRecursiveTextSplitter`; continuation lines remain in the current QA until the next question marker.
- Trim records before `len()` checks. For overlong QA, retain the question in every child and its recognized answer marker only in the first child.
- Use one inner `ChineseRecursiveTextSplitter(chunk_size=chunk_size-len(first_prefix), chunk_overlap=chunk_overlap)`; never mutate `self` sizing fields.
- Only on the overlong path, raise `ValueError` when `len(first_prefix) + chunk_overlap >= chunk_size` and state that the operator must increase `chunk_size` or lower `chunk_overlap`.
- Group only consecutive documents having the same `metadata["source"]`; use one newline as joiner and copy the first document metadata to all resulting chunks.
- Output preamble chunks first, then QA chunks in source order. After opt-in, restart the service and rebuild the vector store.

---

### Task 1: Establish the QA-splitter behavior contract

**Files:**
- Create: `libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py`

**Interfaces:**
- Consumes: `QATextSplitter` from `chatchat.server.file_rag.text_splitter`.
- Produces: regression tests that initially fail because the splitter is absent.

- [ ] **Step 1: Write failing marker and boundary tests**

```python
from chatchat.server.file_rag.text_splitter import QATextSplitter


def test_splits_supported_marker_variants():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_text(
        "Q: one\nA: answer one\nq ： two\na：answer two\n问：三\n答：答案三"
    )
    assert chunks == [
        "Q: one\nA: answer one",
        "q ： two\na：answer two",
        "问：三\n答：答案三",
    ]


def test_preamble_precedes_qa_and_answer_continuation_is_retained():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    assert splitter.split_text("intro\nQ: one\nA: first\nsecond") == [
        "intro", "Q: one\nA: first\nsecond"
    ]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `cd libs/chatchat-server && pytest tests/custom_splitter/test_qa_text_splitter.py -v`

Expected: collection fails because `QATextSplitter` does not yet exist.

- [ ] **Step 3: Add failing grouping, metadata, fallback, budget, and error-path tests**

```python
import pytest
from langchain.docstore.document import Document


def test_groups_same_source_and_uses_first_metadata():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    docs = [
        Document(page_content="Q: one", metadata={"source": "one.docx", "page": 1}),
        Document(page_content="A: answer", metadata={"source": "one.docx", "page": 2}),
        Document(page_content="Q: two\nA: other", metadata={"source": "two.docx", "page": 1}),
    ]
    chunks = splitter.split_documents(docs)
    assert [chunk.page_content for chunk in chunks] == ["Q: one\nA: answer", "Q: two\nA: other"]
    assert chunks[0].metadata == {"source": "one.docx", "page": 1}


def test_long_answer_preserves_question_and_stays_within_limit():
    splitter = QATextSplitter(chunk_size=40, chunk_overlap=5)
    chunks = splitter.split_text("Q: why?\n答：" + "词" * 100)
    assert len(chunks) > 1
    assert all(chunk.startswith("Q: why?\n") and len(chunk) <= 40 for chunk in chunks)
    assert chunks[0].startswith("Q: why?\n答：\n")
    assert all("答：" not in chunk for chunk in chunks[1:])


def test_overlong_record_with_prefix_plus_overlap_exhausting_budget_raises():
    splitter = QATextSplitter(chunk_size=20, chunk_overlap=5)
    with pytest.raises(ValueError, match="increase chunk_size or lower chunk_overlap"):
        splitter.split_text("Q: " + "问" * 12 + "\nA: " + "答" * 30)


def test_complete_short_record_with_same_prefix_does_not_raise():
    splitter = QATextSplitter(chunk_size=20, chunk_overlap=5)
    assert splitter.split_text("Q: " + "问" * 12 + "\nA: 答")


def test_non_qa_text_uses_recursive_fallback():
    splitter = QATextSplitter(chunk_size=10, chunk_overlap=0)
    assert splitter.split_text("甲乙丙丁戊己庚辛壬癸子丑") == ["甲乙丙丁戊己庚辛壬癸", "子丑"]
```

- [ ] **Step 4: Run the module again and verify the new behavior is RED**

Run: `cd libs/chatchat-server && pytest tests/custom_splitter/test_qa_text_splitter.py -v`

Expected: FAIL until production implementation exists.

### Task 2: Implement and export the splitter

**Files:**
- Create: `libs/chatchat-server/chatchat/server/file_rag/text_splitter/qa_text_splitter.py`
- Modify: `libs/chatchat-server/chatchat/server/file_rag/text_splitter/__init__.py`
- Test: `libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py`

**Interfaces:**
- Produces: `QATextSplitter(ChineseRecursiveTextSplitter)` with `split_text(text: str) -> List[str]` and `split_documents(documents: List[Document]) -> List[Document]`.

- [ ] **Step 1: Implement constructor and marker parsing**

```python
QUESTION_MARKER = re.compile(r"^\s*(?:Q|q|问)\s*[:：]\s*", re.MULTILINE)
ANSWER_MARKER = re.compile(r"^\s*(?:A|a|答)\s*[:：]\s*", re.MULTILINE)


class QATextSplitter(ChineseRecursiveTextSplitter):
    def __init__(self, *args, **kwargs):
        kwargs.pop("pipeline", None)
        super().__init__(*args, **kwargs)
```

- [ ] **Step 2: Implement `split_text` and overlong-record split**

```python
def _split_record(self, record: str) -> List[str]:
    record = record.strip()
    if not record:
        return []
    if len(record) <= self._chunk_size:
        return [record]
    question, answer_marker, answer_body = self._parse_question_and_answer(record)
    first_prefix = f"{question}\n{answer_marker}\n"
    if len(first_prefix) + self._chunk_overlap >= self._chunk_size:
        raise ValueError("increase chunk_size or lower chunk_overlap")
    answer_chunks = ChineseRecursiveTextSplitter(
        chunk_size=self._chunk_size - len(first_prefix),
        chunk_overlap=self._chunk_overlap,
    ).split_text(answer_body)
    return [f"{first_prefix}{answer_chunks[0]}"] + [f"{question}\n{chunk}" for chunk in answer_chunks[1:]]
```

Use `super().split_text(text.strip())` for no-QA and preamble text, and only invoke `_split_record` for question-started records.

- [ ] **Step 3: Implement source grouping and module export**

```python
def _split_source_group(self, docs: List[Document]) -> List[Document]:
    metadata = docs[0].metadata.copy()
    text = "\n".join(doc.page_content for doc in docs)
    return [Document(page_content=chunk, metadata=metadata.copy()) for chunk in self.split_text(text)]
```

`split_documents` must flush a group whenever the next `source` differs; add `from .qa_text_splitter import QATextSplitter` to `__init__.py`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `cd libs/chatchat-server && pytest tests/custom_splitter/test_qa_text_splitter.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the implementation and regression tests**

```bash
git add libs/chatchat-server/chatchat/server/file_rag/text_splitter/qa_text_splitter.py libs/chatchat-server/chatchat/server/file_rag/text_splitter/__init__.py libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py
git commit -m "feat: add QA text splitter"
```

### Task 3: Register, regress, and inspect the actual DOCX

**Files:**
- Modify: `libs/chatchat-server/chatchat/settings.py:220-247`
- Test: `libs/chatchat-server/tests/custom_splitter/test_different_splitter.py`
- Test: `/Users/caomengdi/chatchat-data/data/knowledge_base/司库test/content/test.docx`

**Interfaces:**
- Produces: `make_text_splitter("QATextSplitter", ...)` compatibility while preserving the existing default.

- [ ] **Step 1: Write a failing settings-factory test**

```python
from chatchat.server.knowledge_base.utils import make_text_splitter


def test_factory_creates_qa_text_splitter():
    assert make_text_splitter("QATextSplitter", 200, 0).__class__.__name__ == "QATextSplitter"
```

- [ ] **Step 2: Verify RED, then register the settings entry**

Run: `cd libs/chatchat-server && pytest tests/custom_splitter/test_qa_text_splitter.py::test_factory_creates_qa_text_splitter -v`

Expected before edit: FAIL with missing `QATextSplitter` configuration key.

Add beside the other local entries in `text_splitter_dict`:

```python
"QATextSplitter": {"source": "", "tokenizer_name_or_path": ""},
```

- [ ] **Step 3: Run complete regression and source inspection**

Run: `cd libs/chatchat-server && pytest tests/custom_splitter/test_qa_text_splitter.py tests/custom_splitter/test_different_splitter.py -v`

Expected: PASS.

Run: `cd libs/chatchat-server && python -c 'from chatchat.server.knowledge_base.utils import KnowledgeFile; from chatchat.settings import Settings; Settings.kb_settings.TEXT_SPLITTER_NAME="QATextSplitter"; f=KnowledgeFile("test.docx", "司库test"); chunks=f.file2text(refresh=True, chunk_size=750, chunk_overlap=150); print(len(chunks))'`

Expected: prints `7` for the user's DOCX; it does not edit the DOCX or write vectors.

- [ ] **Step 4: Commit settings and final tests**

```bash
git add libs/chatchat-server/chatchat/settings.py libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py
git commit -m "test: cover QA splitter boundaries"
```
