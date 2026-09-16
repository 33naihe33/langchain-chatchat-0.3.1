import pytest
from langchain.docstore.document import Document

from chatchat.server.file_rag.text_splitter import QATextSplitter
from chatchat.server.knowledge_base.utils import make_text_splitter


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
        "intro",
        "Q: one\nA: first\nsecond",
    ]


def test_multiline_question_prefix_is_preserved_until_answer_marker():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)

    assert splitter.split_text("Q: 配置中心\n\n岗位权限\nA: 说明") == [
        "Q: 配置中心\n\n岗位权限\nA: 说明"
    ]


def test_groups_same_source_and_uses_first_metadata():
    splitter = QATextSplitter(chunk_size=200, chunk_overlap=0)
    docs = [
        Document(page_content="Q: one", metadata={"source": "one.docx", "page": 1}),
        Document(page_content="A: answer", metadata={"source": "one.docx", "page": 2}),
        Document(page_content="Q: two\nA: other", metadata={"source": "two.docx", "page": 1}),
    ]

    chunks = splitter.split_documents(docs)

    assert [chunk.page_content for chunk in chunks] == [
        "Q: one\nA: answer",
        "Q: two\nA: other",
    ]
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

    assert splitter.split_text("甲乙丙丁戊己庚辛壬癸子丑") == [
        "甲乙丙丁戊己庚辛壬癸",
        "子丑",
    ]


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


def test_factory_creates_qa_text_splitter():
    assert make_text_splitter("QATextSplitter", 200, 0).__class__.__name__ == "QATextSplitter"


def test_factory_creates_qa_text_splitter_without_a_settings_entry(monkeypatch):
    from chatchat.settings import Settings

    make_text_splitter.cache_clear()
    monkeypatch.delitem(
        Settings.kb_settings.text_splitter_dict, "QATextSplitter", raising=False
    )

    assert make_text_splitter("QATextSplitter", 200, 0).__class__.__name__ == "QATextSplitter"
