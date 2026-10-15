import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.datastructures import URL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chatchat.server.knowledge_base.utils import format_reference


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:7861",
    "http://127.0.0.1:7861/",
    URL("http://127.0.0.1:7861/"),
])
def test_reference_download_url_has_one_slash_and_encoded_parameters(base_url):
    filename = "目录/测试 & 文档.docx"
    reference = format_reference(
        "司库test", [{"metadata": {"source": filename}, "page_content": "正文"}], base_url
    )[0]
    url = reference.split("](", 1)[1].split(")", 1)[0]
    parsed = urlsplit(url)
    assert parsed.path == "/knowledge_base/download_doc"
    assert parse_qs(parsed.query) == {
        "knowledge_base_name": ["司库test"], "file_name": [filename]
    }
    assert "正文" in reference
