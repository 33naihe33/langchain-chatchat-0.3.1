# SSE TTFT / TPOT 基准脚本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `/chat/kb_chat` 的 SSE 回答提供可复测、可落盘的 TTFT 与 TPOT 命令行基准工具。

**Architecture:** `scripts/benchmark_sse.py` 以 `requests` 串行调用 SSE，并使用 `time.perf_counter()` 记录首个有效文本和正常 EOF。纯函数处理 SSE 分帧、token 计数与统计；网络层注入这些函数，测试通过 mock 响应覆盖。

**Tech Stack:** Python 3.8+、标准库、项目既有 `requests`、pytest。

## Global Constraints

- 默认地址：聊天 `http://127.0.0.1:7861/chat/kb_chat`；计数 `http://127.0.0.1:8080/tokenize`。
- 默认预热 2 次、测量 5 次，所有请求严格串行。
- 请求体固定使用 `mode=local_kb`、`stream=true`、`max_tokens=512`、`top_k=3`、`score_threshold=1.0`、`temperature=0`。
- TTFT 是请求前单调时钟到首个非空 `delta.content`；TPOT 是 `(EOF - first_content)/max(tokens-1, 1)`。
- `/tokenize` 请求固定为 `{"content": answer, "add_special": false}`；失败只影响 `tpot_success`。
- 默认输出文件为 `data/benchmarks/<timestamp>.jsonl`，保存 `output_tokens`、`output_chars`、SHA-256，不保存正文；`--include-output` 才保存全文。

---

## File Structure

- Create: `scripts/__init__.py` — 使根目录 `scripts` 成为显式 Python 包。
- Create: `scripts/benchmark_sse.py` — CLI、SSE/统计函数、HTTP 调用和 JSONL 写入。
- Create: `tests/test_benchmark_sse.py` — 无网络单元测试。测试命令必须从仓库根目录以 `python -m pytest` 执行，保证根目录位于 `sys.path`。

### Task 1: 实现纯度量逻辑

**Files:**
- Create: `scripts/benchmark_sse.py`
- Create: `tests/test_benchmark_sse.py`

**Interfaces:**
- Produces: `iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]`
- Produces: `calculate_tpot(first_content_at: float, eof_at: float, token_count: int) -> float`
- Produces: `summarize(records: Sequence[dict]) -> dict`

- [ ] **Step 1: 写入失败测试**

```python
import pytest
from scripts.benchmark_sse import calculate_tpot, iter_sse_data, summarize

def test_parser_ignores_ping_and_handles_chunk_boundary():
    chunks = [b": ping\n\ndata: {\"a\":", b"1}\n\nnoise\n\ndata: {\"a\":2}\n\n"]
    assert list(iter_sse_data(chunks)) == ['{"a":1}', '{"a":2}']

def test_tpot_uses_generated_intervals():
    assert calculate_tpot(1.0, 1.3, 4) == pytest.approx(0.1)

def test_summary_has_independent_populations():
    rows = [{"chat_success": True, "ttft_ms": 10, "tpot_success": True, "tpot_ms": 5}, {"chat_success": True, "ttft_ms": 30, "tpot_success": False}]
    assert summarize(rows)["ttft"]["sample_count"] == 2
    assert summarize(rows)["tpot"]["sample_count"] == 1
```

- [ ] **Step 2: 运行失败测试**

Run: `conda run -n chatchat-v031 python -m pytest tests/test_benchmark_sse.py -v`

Expected: FAIL，模块 `scripts.benchmark_sse` 尚不存在。

- [ ] **Step 3: 编写最小实现**

```python
def iter_sse_data(chunks):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    for chunk in chunks:
        buffer += decoder.decode(chunk)
        events = buffer.replace("\r\n", "\n").split("\n\n")
        buffer = events.pop()
        for event in events:
            data = [line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")]
            if data:
                yield "\n".join(data)
    buffer += decoder.decode(b"", final=True)
    if buffer:
        data = [line[5:].lstrip() for line in buffer.replace("\r\n", "\n").split("\n") if line.startswith("data:")]
        if data:
            yield "\n".join(data)

def calculate_tpot(first_content_at, eof_at, token_count):
    return (eof_at - first_content_at) / max(token_count - 1, 1)

def summarize(records):
    def metric(key, flag):
        values = sorted(row[key] for row in records if row.get(flag))
        return {"sample_count": len(values), "mean": sum(values) / len(values) if values else None, "p50": values[(len(values)-1)//2] if values else None, "min": values[0] if values else None, "max": values[-1] if values else None}
    return {"record_type": "summary", "ttft": metric("ttft_ms", "chat_success"), "tpot": metric("tpot_ms", "tpot_success")}
```

- [ ] **Step 4: 验证通过并提交**

Run: `conda run -n chatchat-v031 python -m pytest tests/test_benchmark_sse.py -v`

Expected: PASS。

```bash
git add scripts/benchmark_sse.py tests/test_benchmark_sse.py
git commit -m "feat: add SSE benchmark metric helpers"
```

### Task 2: 实现 HTTP 测量和 CLI

**Files:**
- Modify: `scripts/benchmark_sse.py`
- Modify: `tests/test_benchmark_sse.py`

**Interfaces:**
- Produces: `run_once(args: argparse.Namespace, is_warmup: bool, run_number: int) -> dict`
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int`

- [ ] **Step 1: 写入失败测试**

```python
def test_tokenize_failure_keeps_chat_measurement(monkeypatch, args):
    monkeypatch.setattr("scripts.benchmark_sse.requests.post", fake_streaming_post)
    monkeypatch.setattr("scripts.benchmark_sse.count_tokens", raise_tokenize_error)
    row = run_once(args, False, 1)
    assert row["chat_success"] is True and row["tpot_success"] is False
    assert row["ttft_ms"] is not None and "output_text" not in row

def test_malformed_sse_fails_chat(monkeypatch, args):
    monkeypatch.setattr("scripts.benchmark_sse.requests.post", malformed_sse_post)
    row = run_once(args, False, 1)
    assert row["chat_success"] is False and row["tpot_success"] is False
```

- [ ] **Step 2: 运行失败测试**

Run: `conda run -n chatchat-v031 python -m pytest tests/test_benchmark_sse.py -v`

Expected: FAIL，`run_once` 和 `count_tokens` 未定义。

- [ ] **Step 3: 编写最小实现**

实现所有 spec 列出的 argparse 参数。聊天调用使用 `requests.post(..., stream=True, timeout=(connect_timeout, read_timeout))`；在调用前立即取 `started`，从 `iter_sse_data(response.iter_content(chunk_size=None))` 解析 JSON，在首个非空内容取 `first_content_at`，响应体结束时取 `eof_at`。实现 `count_tokens`，向 `/tokenize` 发送规定请求体。聊天失败写 `chat_error`；仅计数失败写 `tpot_error`。每一轮同时向 stdout 与 JSONL 写 `json.dumps(..., ensure_ascii=False)`；用 `Path(...).parent.mkdir(parents=True, exist_ok=True)` 建目录，正文以 SHA-256 摘要代替，除非 `--include-output`。

- [ ] **Step 4: 验证通过并提交**

Run: `conda run -n chatchat-v031 python -m pytest tests/test_benchmark_sse.py -v`

Expected: PASS，覆盖心跳、协议错误、计数失败保留 TTFT、独立统计和超时。

```bash
git add scripts/benchmark_sse.py tests/test_benchmark_sse.py
git commit -m "feat: add SSE TTFT TPOT benchmark CLI"
```

### Task 3: 端到端验收

**Files:**
- Modify: `scripts/benchmark_sse.py`（仅验收发现缺陷时）

- [ ] **Step 1: 检查本机服务**

Run: `curl --fail http://127.0.0.1:8080/v1/models && curl --fail http://127.0.0.1:8081/v1/models`

Expected: 返回 `MiniCPM5-2B` 与 `bge-m3`。

- [ ] **Step 2: 运行轻量基准**

Run: `curl --fail http://127.0.0.1:7861/knowledge_base/list_knowledge_bases && conda run -n chatchat-v031 python scripts/benchmark_sse.py --query '请简述资金头寸管理。' --kb-name '司库test' --runs 1 --warmups 1 --max-tokens 64 --output /tmp/minicpm5-benchmark.jsonl`

Expected: 列表包含 `司库test`；stdout 和 JSONL 都含一条 `chat_success=true`、`tpot_success=true` 的测量记录及一条汇总记录。

- [ ] **Step 3: 核对汇总**

Run: `conda run -n chatchat-v031 python -c "import json; rows=[json.loads(x) for x in open('/tmp/minicpm5-benchmark.jsonl')]; print(rows[-1])"`

Expected: `ttft` 与 `tpot` 均有独立样本数；成功时指标为非负。

- [ ] **Step 4: 如有验收修复则提交**

```bash
git add scripts/benchmark_sse.py tests/test_benchmark_sse.py
git commit -m "fix: harden SSE benchmark runtime handling"
```

## Review Corrections

The following corrections supersede the earlier snippets where they conflict.

- Create `scripts/__init__.py`; tests run from repo root via `conda run -n chatchat-v031 python -m pytest tests/test_benchmark_sse.py -v`.
- `iter_sse_data` uses `codecs.getincrementaldecoder("utf-8")()` and never calls `chunk.decode("utf-8")`; add a test splitting the UTF-8 bytes for “资金” across chunks. After decoding the final bytes, it must also parse a non-empty trailing buffer as one final event.
- Import pytest and assert TPOT with `pytest.approx(0.1)`.
- `summarize` first excludes `is_warmup=true`; test this. Define P50 as the lower median and emit `p50_method: "lower_median"`.
- Define all test fixtures/helpers (`args`, response double, fake streaming post, malformed stream, tokenizer error) in the test module.
- Add separate `requests.ConnectTimeout` and `requests.ReadTimeout` tests; both mark chat and TPOT unsuccessful with distinct `chat_error` values.
- `count_tokens` calls `raise_for_status()`, uses the same `(connect_timeout, read_timeout)` pair as the chat request, and requires list-valued `tokens`; all token endpoint failures only set `tpot_success=false` and `tpot_error`.
- Every record includes UTC `started_at`, `generation_duration_ms`, `request_params`, `output_tokens`, `output_chars`, and `output_sha256`; add `output_text` only with `--include-output`.

