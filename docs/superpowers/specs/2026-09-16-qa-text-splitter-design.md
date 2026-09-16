# QA Text Splitter Design

## Purpose

Preserve each question and its answer from structured QA documents as one vector-store document. This prevents the default length-based splitter from separating a question from its answer or merging unrelated QA pairs.

## Scope

Add a configurable `QATextSplitter` to the knowledge-base text splitters. The source DOCX remains unchanged. The feature takes effect when the knowledge base is rebuilt with `TEXT_SPLITTER_NAME` set to `QATextSplitter`.

## Input Format

The splitter recognizes question and answer markers at the start of a line, with optional surrounding whitespace:

- Questions: `Q:`, `Q：`, `Q :`, and `Q ：`
- Answers: `A:`, `A：`, `A :`, and `A ：`

The question marker starts a new QA record. Its record ends immediately before the next question marker. Answer markers are retained as part of the record content. Every non-marker line after a question marker, including answer continuation lines and blank lines, belongs to the current QA record until the next question marker.

## Splitting Behavior

1. A QA record whose complete text fits within `chunk_size` becomes exactly one chunk containing its question and answer.
2. For an overlong QA record, the question prefix is retained in every child chunk. Only the answer body is divided using the existing length and overlap settings.
3. The leading text before the first recognized question marker is a preamble. It is independently delegated to the existing character-based splitter and is never attached to the first QA record.
4. A mixed document is processed by segment: each QA record uses QA behavior, while the preamble uses the existing character-based splitter. A document without any recognized question marker wholly uses the existing splitter.
5. Empty or whitespace-only records are not emitted.
6. `split_documents` preserves the metadata of the source document for every output chunk.

## Configuration

Register `QATextSplitter` in `text_splitter_dict` with the existing local splitter configuration style. Operators enable it by setting:

```python
TEXT_SPLITTER_NAME = "QATextSplitter"
```

The default splitter remains unchanged to avoid altering existing knowledge bases unexpectedly.

## Tests

Add focused tests that verify:

- Chinese and English question/answer colons, including optional spaces, form separate QA chunks.
- A short QA is emitted intact as one chunk.
- A long answer is split into multiple chunks and each includes the original question.
- A source document without QA markers falls back to standard character splitting.
- Preamble text in a mixed document is split separately and answer continuation lines remain in their preceding QA record.
- Source metadata is copied to emitted chunks.

## Operational Notes

Existing vectors are not retroactively rewritten. After enabling the splitter, recreate the vector store for the target knowledge base to replace the existing chunks.
