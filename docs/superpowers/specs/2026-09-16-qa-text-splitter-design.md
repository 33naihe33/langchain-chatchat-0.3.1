# QA Text Splitter Design

## Purpose

Preserve each question and its answer from structured QA documents as one vector-store document. This prevents the default length-based splitter from separating a question from its answer or merging unrelated QA pairs.

## Scope

Add a configurable `QATextSplitter` to the knowledge-base text splitters. The source DOCX remains unchanged. The feature takes effect when the knowledge base is rebuilt with `TEXT_SPLITTER_NAME` set to `QATextSplitter`.

`RapidOCRDocLoader` can produce several LangChain `Document` objects for one source DOCX. To preserve QA pairs that span those loader objects, `QATextSplitter.split_documents` concatenates the consecutive documents it receives for one source file with `"\n"` separators before applying QA parsing. It never combines documents from different sources.

## Input Format

The splitter recognizes question and answer markers at the start of a line, with optional surrounding whitespace:

- Questions: `Q:`, `Q：`, `Q :`, `Q ：`, lowercase `q` variants, and `问` variants with either colon and optional spaces
- Answers: `A:`, `A：`, `A :`, `A ：`, lowercase `a` variants, and `答` variants with either colon and optional spaces

The question marker starts a new QA record. Its record ends immediately before the next question marker. Answer markers are retained as part of the record content. Every non-marker line after a question marker, including answer continuation lines and blank lines, belongs to the current QA record until the next question marker. For a record with an answer marker, the question prefix is all content before that answer marker, including any loader-inserted line breaks.

## Splitting Behavior

1. Each record is trimmed with `strip()` before any emptiness or length check. A QA record whose complete trimmed text has `len(text) <= chunk_size` becomes exactly one chunk containing its question and answer.
2. For an overlong QA record, the question prefix is retained in every child chunk. The answer body is divided once by a new `ChineseRecursiveTextSplitter` instance with the configured overlap and a uniform budget of `chunk_size - len(first_prefix)`, where `first_prefix` is `question + "\n" + answer_marker + "\n"`. This conservative budget ensures both the first child and the later, shorter-prefixed children remain within `chunk_size`. The recognized answer marker is retained only in the first child chunk.
3. Only for a record that actually requires answer-body splitting under rule 2, the splitter raises `ValueError` with an instruction to increase `chunk_size` or lower `chunk_overlap` if `len(first_prefix) + chunk_overlap >= chunk_size`. Records emitted intact under rule 1 are exempt. This avoids silently emitting an over-limit chunk, dropping the question, or constructing an inner splitter whose overlap is not smaller than its chunk budget.
4. The leading text before the first recognized question marker is a preamble. It is independently delegated to a `ChineseRecursiveTextSplitter` constructed with its normal separators plus a final empty-string separator, and is never attached to the first QA record.
5. A mixed document is processed by segment: each QA record uses QA behavior, while the preamble uses that strict recursive splitter. A document without any recognized question marker wholly uses that strict recursive splitter. The same explicit empty-string separator fallback is used by the inner answer-body splitter, guaranteeing no emitted chunk exceeds `chunk_size`.
6. Empty or whitespace-only records are not emitted.
7. Output order is the preamble chunks first, followed by QA chunks in source-document order.
8. Output chunks retain the metadata of their source document. When multiple loader-produced documents from one source are concatenated, output chunks copy the first document's metadata; later conflicting values (such as page number) are not merged.

## Implementation Constraints

`QATextSplitter` subclasses `ChineseRecursiveTextSplitter`, and therefore indirectly subclasses LangChain `TextSplitter`. It accepts arbitrary keyword arguments, including the `pipeline="zh_core_web_sm"` passed by `make_text_splitter`. It overrides `split_text` for QA-aware text splitting and `split_documents` solely to concatenate loader-produced documents with the same `source` before calling `split_text` and restoring their metadata. It constructs a new strict recursive splitter for fallback text and answer bodies with `separators=[*self._separators, ""]`, and never mutates `self._chunk_size` or `self._chunk_overlap`. An empty parsed answer body returns the stripped record intact rather than indexing an empty answer-chunk list.

## Configuration

Register `QATextSplitter` in `text_splitter_dict` with the existing local splitter configuration style. Operators enable it by setting:

```python
TEXT_SPLITTER_NAME = "QATextSplitter"
```

The default splitter remains unchanged to avoid altering existing knowledge bases unexpectedly.

## Tests

Add focused tests that verify:

- Chinese and English question/answer colons, including optional spaces, form separate QA chunks.
- Lowercase `q:/a:` and Chinese `问：/答：` markers form QA chunks.
- A short QA is emitted intact as one chunk.
- A long answer is split into multiple chunks and each includes the original question.
- A source document without QA markers falls back to standard character splitting.
- Preamble text in a mixed document is split separately and answer continuation lines remain in their preceding QA record.
- Consecutive loader-produced documents from the same source are combined so a question and answer that were loaded separately become one QA chunk; different sources remain isolated.
- Preamble chunks precede QA chunks; a long answer retains its question in every child and its answer marker only in the first child.
- Overlong-answer budgets deduct their complete output prefixes, and trailing whitespace does not affect length decisions.
- Punctuation-free fallback text and answer bodies split at character boundaries through the explicit empty-string separator.
- A QA record with an empty answer body does not raise `IndexError`.
- An overlong QA record whose question prefix plus overlap exhausts the chunk budget raises `ValueError` with a clear recovery message; a complete, short QA with the same prefix does not.
- Source metadata is copied to emitted chunks.
- Concatenated documents with conflicting metadata use the first document's metadata.

## Operational Notes

Existing vectors are not retroactively rewritten. After enabling the splitter, restart the service so `make_text_splitter` clears its `lru_cache`, then recreate the vector store for the target knowledge base to replace the existing chunks.
