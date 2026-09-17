# QA Text Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable splitter that makes each structured QA pair one vector-store chunk.

**Architecture:** `QATextSplitter` extends `ChineseRecursiveTextSplitter`. It groups sequential loader documents by source, joins their content with one newline, then splits preamble and QA records. Overlong answers are split by a new inner recursive splitter whose budget includes the repeated question prefix.

**Tech Stack:** Python, LangChain `Document`/`TextSplitter`, pytest.

## Global Constraints

- Do not modify the source DOCX or change the default `TEXT_SPLITTER_NAME`.
- Support `Q/q/问` and `A/a/答` at line start, with `:` or `：` and optional whitespace.
- Initialize `QATextSplitter` with its normal separator list plus `""`, so `super().split_text` handles preamble and non-QA text even without punctuation. Continuation lines remain in the current QA until the next question marker. Treat all content before the first recognized answer marker as the question prefix, including loader-inserted line breaks.
- Trim records before `len()` checks. For overlong QA, retain the question in every child and its recognized answer marker only in the first child.
- Use one inner `ChineseRecursiveTextSplitter(chunk_size=chunk_size-len(first_prefix), chunk_overlap=chunk_overlap, separators=self._separators)`; never mutate `self` sizing fields.
- Only on the overlong path, raise `ValueError` when `len(first_prefix) + chunk_overlap >= chunk_size` and state that the operator must increase `chunk_size` or lower `chunk_overlap`.
- Group only consecutive documents having the same `metadata["source"]`; use one newline as joiner and copy the first document metadata to all resulting chunks. Read the grouping key with `metadata.get("source")`; documents whose `source` is missing are never concatenated with others.
- Implement `_parse_question_and_answer(record)` returning `(record[:match.start()].rstrip(), match.group(), record[match.end():])` from the first answer-marker match, or `(record, None, "")` when the record has no answer marker. The `rstrip` keeps interior line breaks in the question but drops its trailing whitespace so the emitted prefix is exactly `question + "\n" + marker + "\n"`. A `None` marker routes the record to the strict recursive splitter.
- An overlong record without an answer marker is delegated whole to the strict recursive splitter like the preamble; it never raises.
- Output preamble chunks first, then QA chunks in source order. After opt-in, restart the service and rebuild the vector store.
- Run every test and inspection with `/Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest` or `/Users/caomengdi/miniforge3/envs/chatchat-v031/bin/python`; never use the system interpreter. When testing inside the `.worktrees/qa-text-splitter` checkout, invoke `/Users/caomengdi/miniforge3/envs/chatchat-v031/bin/python -m pytest -o addopts=""` instead: the environment's editable `chatchat` install resolves to the main checkout (bare `pytest` imports the main repo's package), and the worktree `pyproject.toml` sets `--strict-config` with `asyncio_mode`, which aborts the run because `pytest-asyncio` is not installed.

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


def test_multiline_question_prefix_is_preserved_until_answer_marker():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    assert splitter.split_text("Q: 配置中心\n\n岗位权限\nA: 说明") == [
        "Q: 配置中心\n\n岗位权限\nA: 说明"
    ]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `cd libs/chatchat-server && /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest tests/custom_splitter/test_qa_text_splitter.py -v`

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


def test_empty_answer_body_does_not_index_an_empty_chunk_list():
    splitter = QATextSplitter(chunk_size=20, chunk_overlap=0)
    assert splitter.split_text("Q: x\nA:") == ["Q: x\nA:"]


def test_overlong_record_without_answer_marker_uses_recursive_fallback():
    splitter = QATextSplitter(chunk_size=10, chunk_overlap=0)
    chunks = splitter.split_text("Q: " + "问" * 20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == "Q: " + "问" * 20


def test_trailing_whitespace_does_not_change_length_decision():
    splitter = QATextSplitter(chunk_size=9, chunk_overlap=0)
    assert splitter.split_text("Q: x\nA: 答 \n") == ["Q: x\nA: 答"]


def test_overlong_record_keeps_multiline_question_prefix_in_every_child():
    splitter = QATextSplitter(chunk_size=30, chunk_overlap=0)
    chunks = splitter.split_text("Q: 配置中心\n\n岗位权限\nA: " + "词" * 40)
    assert len(chunks) > 1
    assert chunks[0].startswith("Q: 配置中心\n\n岗位权限\nA: \n")
    assert all(chunk.startswith("Q: 配置中心\n\n岗位权限\n") for chunk in chunks[1:])
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_documents_without_source_are_not_concatenated():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    docs = [
        Document(page_content="Q: one", metadata={}),
        Document(page_content="A: answer", metadata={}),
    ]
    chunks = splitter.split_documents(docs)
    assert [chunk.page_content for chunk in chunks] == ["Q: one", "A: answer"]
```

- [ ] **Step 4: Run the module again and verify the new behavior is RED**

Run: `cd libs/chatchat-server && /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest tests/custom_splitter/test_qa_text_splitter.py -v`

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
DEFAULT_QA_SEPARATORS = ["\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", r"；|;\s", r"，|,\s", ""]
QUESTION_MARKER = re.compile(r"^[ \t]*(?:Q|q|问)[ \t]*[:：][ \t]*", re.MULTILINE)
ANSWER_MARKER = re.compile(r"^[ \t]*(?:A|a|答)[ \t]*[:：][ \t]*", re.MULTILINE)


class QATextSplitter(ChineseRecursiveTextSplitter):
    def __init__(self, *args, **kwargs):
        kwargs.pop("pipeline", None)
        separators = list(kwargs.pop("separators", DEFAULT_QA_SEPARATORS))
        if "" not in separators:
            separators.append("")
        super().__init__(*args, separators=separators, **kwargs)
```

Write the regex separator patterns as raw strings (`r"\.\s|\!\s|\?\s"`, `r"；|;\s"`, `r"，|,\s"`) so Python emits no invalid-escape-sequence warning; keep `"\n\n"` and `"\n"` as normal strings because they must be real newline characters, not regex escapes.

- [ ] **Step 2: Implement `split_text`, record parsing, and overlong-record split**

```python
def _parse_question_and_answer(self, record: str):
    match = ANSWER_MARKER.search(record)
    if match is None:
        return record, None, ""
    return record[: match.start()].rstrip(), match.group(), record[match.end():]


def split_text(self, text: str) -> List[str]:
    text = text.strip()
    matches = list(QUESTION_MARKER.finditer(text))
    if not matches:
        return super().split_text(text)
    chunks = super().split_text(text[: matches[0].start()])
    starts = [m.start() for m in matches] + [len(text)]
    for start, end in zip(starts, starts[1:]):
        chunks.extend(self._split_record(text[start:end]))
    return chunks


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
        raise ValueError("increase chunk_size or lower chunk_overlap")
    answer_chunks = ChineseRecursiveTextSplitter(
        chunk_size=self._chunk_size - len(first_prefix),
        chunk_overlap=self._chunk_overlap,
        separators=self._separators,
    ).split_text(answer_body)
    if not answer_chunks:
        return [record]
    return [f"{first_prefix}{answer_chunks[0]}"] + [f"{question}\n{chunk}" for chunk in answer_chunks[1:]]
```

`split_text` slices records from each question-marker match start to the next match start, keeping the marker text inside the record; the preamble is everything before the first match. The constructor has already installed the strict separator list, so `super().split_text` covers no-QA text, the preamble, answer-marker-less overlong records, and empty slices (which return `[]`).

- [ ] **Step 3: Implement source grouping and module export**

```python
def _split_source_group(self, docs: List[Document]) -> List[Document]:
    metadata = docs[0].metadata.copy()
    text = "\n".join(doc.page_content for doc in docs)
    return [Document(page_content=chunk, metadata=metadata.copy()) for chunk in self.split_text(text)]


def split_documents(self, documents: List[Document]) -> List[Document]:
    chunks: List[Document] = []
    group: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source")
        if group and (source is None or source != group[0].metadata.get("source")):
            chunks.extend(self._split_source_group(group))
            group = []
        group.append(doc)
    if group:
        chunks.extend(self._split_source_group(group))
    return chunks
```

`split_documents` flushes a group whenever the next `source` differs or is missing, and flushes the trailing group after the loop; a missing `source` never concatenates documents. Add `from .qa_text_splitter import QATextSplitter` to `__init__.py`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `cd libs/chatchat-server && /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest tests/custom_splitter/test_qa_text_splitter.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the implementation and regression tests**

```bash
git add libs/chatchat-server/chatchat/server/file_rag/text_splitter/qa_text_splitter.py libs/chatchat-server/chatchat/server/file_rag/text_splitter/__init__.py libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py
git commit -m "feat: add QA text splitter"
```

### Task 3: Register, regress, and inspect the actual DOCX

**Files:**
- Modify: `libs/chatchat-server/chatchat/settings.py:220-247`
- Modify: `libs/chatchat-server/chatchat/server/knowledge_base/utils.py` (settings-entry lookup falls back to `{"source": "", "tokenizer_name_or_path": ""}`)
- Modify: `libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py` (append the factory test)
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

Run: `cd libs/chatchat-server && /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest tests/custom_splitter/test_qa_text_splitter.py::test_factory_creates_qa_text_splitter -v`

Expected before edit: FAIL because `make_text_splitter` catches the missing configuration key and silently returns `RecursiveCharacterTextSplitter`, so the class-name assertion fails.

Add beside the other local entries in `text_splitter_dict`:

```python
"QATextSplitter": {"source": "", "tokenizer_name_or_path": ""},
```

The delivered implementation also changed `make_text_splitter` to read the entry with `text_splitter_dict.get(splitter_name, {"source": "", "tokenizer_name_or_path": ""})`, so a local splitter is constructed even without a settings entry instead of silently degrading to `RecursiveCharacterTextSplitter`; `test_factory_creates_qa_text_splitter_without_a_settings_entry` pins this.

- [ ] **Step 3: Run complete regression and source inspection**

Run: `cd libs/chatchat-server && /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest tests/custom_splitter/test_qa_text_splitter.py -v`

Expected: PASS. Do not include `test_different_splitter.py`: its module-level `transformers` import is unavailable in the `chatchat-v031` environment and would fail test collection independently of this feature.

Run every Python and pytest command in this plan with `/Users/caomengdi/miniforge3/envs/chatchat-v031/bin/python` and `/Users/caomengdi/miniforge3/envs/chatchat-v031/bin/pytest`, respectively. For the source inspection run:

Run: `cd libs/chatchat-server && CHATCHAT_ROOT=/Users/caomengdi/chatchat-data /Users/caomengdi/miniforge3/envs/chatchat-v031/bin/python -c 'from chatchat.server.knowledge_base.utils import KnowledgeFile; from chatchat.server.file_rag.text_splitter import QATextSplitter; f=KnowledgeFile("test.docx", "司库test"); chunks=f.file2text(refresh=True, chunk_size=750, chunk_overlap=150, text_splitter=QATextSplitter(chunk_size=750, chunk_overlap=150)); print(len(chunks))'`

Expected: prints the observed chunk count without editing the DOCX or writing vectors. Pass the splitter explicitly via the `text_splitter` parameter: `Settings.kb_settings` has `auto_reload = True`, so a runtime assignment like `Settings.kb_settings.TEXT_SPLITTER_NAME = "QATextSplitter"` is wiped on the next property access (the cached instance re-runs `__init__()` from `kb_settings.yaml`), and `KnowledgeFile.__init__` would still read the yaml's `ChineseRecursiveTextSplitter`. The inspection has observed seven QA records, one chunk per question with its question and answer intact; verify that result after implementation and report any discrepancy.

- [ ] **Step 4: Commit settings and final tests**

```bash
git add libs/chatchat-server/chatchat/settings.py libs/chatchat-server/tests/custom_splitter/test_qa_text_splitter.py
git commit -m "test: cover QA splitter boundaries"
```
