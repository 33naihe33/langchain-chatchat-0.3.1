import json
import threading

import pytest
import requests

from scripts import benchmark_sse as benchmark

from scripts.benchmark_sse import calculate_tpot, iter_sse_data, summarize


def test_parser_ignores_ping_and_handles_split_utf8_character():
    chunks = [b": ping\n\ndata: {\"text\":\"\xe8", b"\xb5\x84\xe9\x87\x91\"}\n\n"]

    assert list(iter_sse_data(chunks)) == ['{"text":"资金"}']


def test_tpot_uses_generated_token_intervals():
    assert calculate_tpot(1.0, 1.3, 4) == pytest.approx(0.1)


def test_summary_excludes_warmups_and_uses_independent_populations():
    records = [
        {"is_warmup": True, "chat_success": True, "ttft_ms": 1, "tpot_success": True, "tpot_ms": 1},
        {"is_warmup": False, "chat_success": True, "ttft_ms": 10, "tpot_success": True, "tpot_ms": 5},
        {"is_warmup": False, "chat_success": True, "ttft_ms": 30, "tpot_success": False},
    ]

    summary = summarize(records)

    assert summary["p50_method"] == "lower_median"
    assert summary["ttft"]["sample_count"] == 2
    assert summary["tpot"]["sample_count"] == 1


class Response:
    def __init__(self, chunks=(), payload=None, error=None):
        self.chunks = chunks
        self.payload = payload
        self.error = error
        self.closed = False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_content(self, chunk_size=None):
        yield from self.chunks

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


def sse(event):
    return ("data: " + json.dumps(event) + "\n\n").encode()


@pytest.fixture
def args():
    return benchmark.prepare_args(["--query", "资金", "--kb-name", "test"])


@pytest.mark.parametrize("event", [[], None, {"choices": []}, {"choices": [None]},
                                         {"choices": [{"delta": None}]},
                                         {"choices": [{"delta": {"content": 42}}]}])
def test_invalid_sse_schema_records_failure(monkeypatch, args, event):
    response = Response([sse(event)])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    row = benchmark.run_once(args, False, 1)
    assert row["chat_success"] is False
    assert row["tpot_success"] is False
    assert row["chat_error"]


@pytest.mark.parametrize("payload", [[], None, {"tokens": "bad"}])
def test_invalid_tokenizer_schema_is_value_error(monkeypatch, payload):
    response = Response(payload=payload)
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    with pytest.raises(ValueError):
        benchmark.count_tokens("http://tokenize", "资金", (10, 120))
    assert response.closed


@pytest.mark.parametrize("token_error", [None, requests.HTTPError("tokenizer unavailable")])
def test_timing_skips_docs_and_reasoning_and_closes_responses(monkeypatch, args, token_error):
    chat = Response([sse({"choices": [{"delta": {"content": ""}}], "docs": ["reference"]}),
                     sse({"choices": [{"delta": {"reasoning_content": "思考", "content": ""}}]}),
                     sse({"choices": [{"delta": {"content": "资金"}}]})])
    tokenizer = Response(payload={"tokens": [1, 2, 3, 4]}, error=token_error)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return chat if url == args.url else tokenizer

    monkeypatch.setattr(benchmark.requests, "post", post)
    times = iter([1.0, 1.1, 1.2, 1.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    row = benchmark.run_once(args, False, 1)
    assert row["chat_success"] is True
    assert row["concurrency"] == 1
    assert row["first_token_ttft_ms"] == pytest.approx(100)
    assert row["ttft_ms"] == pytest.approx(200)
    assert row["generation_duration_ms"] == pytest.approx(300)
    assert row["tpot_success"] is (token_error is None)
    if token_error is None:
        assert row["tpot_ms"] == pytest.approx(100)
        assert row["output_tokens"] == 4
    else:
        assert row["tpot_error"]
        assert row["tpot_ms"] is None
    assert calls[1][1]["json"] == {"content": "资金", "add_special": False}
    assert calls[0][1]["timeout"] == calls[1][1]["timeout"] == (10, 120)
    assert "output_text" not in row
    assert chat.closed and tokenizer.closed


@pytest.mark.parametrize("error", [requests.ConnectTimeout("connect"), requests.ReadTimeout("read")])
def test_timeout_records_failure(monkeypatch, args, error):
    def post(*a, **k):
        raise error

    monkeypatch.setattr(benchmark.requests, "post", post)
    row = benchmark.run_once(args, False, 1)
    assert not row["chat_success"] and not row["tpot_success"]
    assert type(error).__name__ in row["chat_error"]


def test_http_error_closes_response(monkeypatch, args):
    response = Response(error=requests.HTTPError("404"))
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    assert benchmark.run_once(args, False, 1)["chat_error"]
    assert response.closed


def test_cli_continues_after_invalid_schema_and_writes_summary(monkeypatch, tmp_path, capsys):
    response = Response([sse({"choices": []})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    output = tmp_path / "results.jsonl"
    benchmark.main(["--query", "资金", "--kb-name", "test", "--warmups", "0",
                    "--runs", "2", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[-1]["ttft"]["sample_count"] == 0
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows


@pytest.mark.parametrize("option,value", [("--runs", "0"), ("--runs", "-1"),
                                         ("--warmups", "-1"), ("--connect-timeout", "0"),
                                         ("--read-timeout", "nan"), ("--read-timeout", "inf"),
                                         ("--max-tokens", "0"), ("--concurrency", "0"),
                                         ("--concurrency", "-1")])
def test_cli_rejects_invalid_measurement_parameters(monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(benchmark, "run_once", lambda *a: {"chat_success": False, "tpot_success": False})
    output = tmp_path / "results.jsonl"
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--query", "资金", "--kb-name", "test", option, value,
                        "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("chunks", [[b"data: invalid\n\n"],
                                  [sse({"choices": [{"delta": {"content": ""}}]})],
                                  [sse({"code": 500, "msg": "failed"})]])
def test_protocol_error_or_empty_answer_records_failure(monkeypatch, args, chunks):
    response = Response(chunks)
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    row = benchmark.run_once(args, False, 1)
    assert not row["chat_success"] and not row["tpot_success"]
    assert response.closed


def test_stream_interruption_after_content_is_not_success(monkeypatch, args):
    def chunks():
        yield sse({"choices": [{"delta": {"content": "资金"}}]})
        raise requests.ConnectionError("stream interrupted")

    response = Response(chunks())
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    row = benchmark.run_once(args, False, 1)
    assert not row["chat_success"] and not row["tpot_success"]
    assert "ConnectionError" in row["chat_error"]
    assert row["ttft_ms"] is None
    assert response.closed


def test_include_output_is_explicit(monkeypatch, args):
    args.include_output = True
    response = Response([sse({"choices": [{"delta": {"content": "资金"}}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a, **k: 2)
    assert benchmark.run_once(args, False, 1)["output_text"] == "资金"


def test_parser_handles_split_crlf_multiline_and_trailing_event():
    chunks = [b": ping\r", b"\n\r\ndata: first\r\ndata:  second\r", b"\n\r\n",
              b"event: message\ndata: last"]
    assert list(iter_sse_data(chunks)) == ["first\nsecond", "last"]


def test_summary_statistics_and_empty_population():
    summary = summarize([{"chat_success": True, "ttft_ms": 30},
                         {"chat_success": True, "ttft_ms": 10}])
    assert summary["ttft"] == {"sample_count": 2, "mean": 20, "p50": 10, "p90": None, "p95": None, "min": 10, "max": 30}
    assert summary["tpot"] == {"sample_count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "min": None, "max": None}


@pytest.mark.parametrize("delta", [{"content": "资金"},
                                 {"content": "资金", "reasoning_content": "思考"},
                                 {"content": "资金", "reasoning_content": None},
                                 {"content": "资金", "reasoning_content": ""}])
def test_first_token_and_content_share_timestamp(monkeypatch, args, delta):
    chat = Response([b": ping\n\n", sse({"choices": [{"delta": {"content": None}}]}),
                     sse({"choices": [{"delta": delta}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: chat)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 4)
    times = iter([1.0, 1.2, 1.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    args.include_output = True
    row = benchmark.run_once(args, False, 1)
    assert row["chat_success"] is True
    assert row["first_token_ttft_ms"] == row["ttft_ms"] == pytest.approx(200)
    assert row["tpot_ms"] == pytest.approx(100)
    assert row["output_text"] == "资金"


def test_first_token_uses_first_of_multiple_reasoning_events(monkeypatch, args):
    chat = Response([sse({"choices": [{"delta": {"reasoning_content": "思考"}}]}),
                     sse({"choices": [{"delta": {"reasoning_content": "继续思考"}}]}),
                     sse({"choices": [{"delta": {"content": "资金"}}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: chat)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 4)
    times = iter([1.0, 1.1, 1.3, 1.6])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    row = benchmark.run_once(args, False, 1)
    assert row["first_token_ttft_ms"] == pytest.approx(100)
    assert row["ttft_ms"] == pytest.approx(300)
    assert row["tpot_ms"] == pytest.approx(100)


@pytest.mark.parametrize("reasoning", ["只有思考", 42, ["bad"]])
def test_reasoning_without_valid_answer_is_not_success(monkeypatch, args, reasoning):
    chat = Response([sse({"choices": [{"delta": {"reasoning_content": reasoning}}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: chat)
    row = benchmark.run_once(args, False, 1)
    assert not row["chat_success"] and not row["tpot_success"]
    assert row["first_token_ttft_ms"] is None
    assert row["ttft_ms"] is None
    assert row["chat_error"]
    assert chat.closed


def test_summary_first_token_excludes_warmups_failures_and_legacy_records():
    records = [{"is_warmup": True, "chat_success": True, "ttft_ms": 1, "first_token_ttft_ms": 1},
               {"chat_success": True, "ttft_ms": 30, "first_token_ttft_ms": 20},
               {"chat_success": True, "ttft_ms": 10, "first_token_ttft_ms": 0},
               {"chat_success": False, "first_token_ttft_ms": 999},
               {"chat_success": True, "ttft_ms": 50},
               {"chat_success": True, "ttft_ms": 60, "first_token_ttft_ms": None}]
    summary = summarize(records)
    assert summary["first_token_ttft"] == {"sample_count": 2, "mean": 10, "p50": 0, "p90": None, "p95": None, "min": 0, "max": 20}
    assert summary["ttft"]["sample_count"] == 4
    assert summarize([])["first_token_ttft"]["sample_count"] == 0


def test_cli_writes_both_ttft_metrics_and_excludes_warmup(monkeypatch, tmp_path, capsys):
    responses = iter([
        Response([sse({"choices": [{"delta": {"reasoning_content": "思考"}}]}),
                  sse({"choices": [{"delta": {"content": "资金"}}]})]),
        Response([sse({"choices": [{"delta": {"content": "资金"}}]})]),
        Response(payload={"tokens": [1, 2, 3, 4]}),
        Response(payload={"tokens": [1, 2, 3, 4]}),
    ])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: next(responses))
    times = iter([1.0, 1.1, 1.2, 1.5, 2.0, 2.2, 2.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    output = tmp_path / "results.jsonl"
    assert benchmark.main(["--query", "资金", "--kb-name", "test", "--warmups", "1",
                           "--runs", "1", "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["is_warmup"] is True
    assert rows[0]["first_token_ttft_ms"] == pytest.approx(100)
    assert rows[0]["ttft_ms"] == pytest.approx(200)
    assert rows[1]["first_token_ttft_ms"] == rows[1]["ttft_ms"] == pytest.approx(200)
    for metric in ("first_token_ttft", "ttft"):
        assert rows[2][metric]["sample_count"] == 1
        assert rows[2][metric]["mean"] == pytest.approx(200)
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows


@pytest.mark.parametrize("with_failure", [False, True])
def test_cli_bounds_concurrency_and_finishes_warmups_first(monkeypatch, tmp_path, capsys, with_failure):
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    active = 0
    peak = 0
    completed_warmups = 0

    def run(args, is_warmup, run_number):
        nonlocal active, peak, completed_warmups
        with lock:
            if not is_warmup:
                assert completed_warmups == 2
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            success = not (with_failure and run_number == 3)
            return {"record_type": "run", "run_number": run_number, "is_warmup": is_warmup,
                    "concurrency": args.concurrency, "chat_success": success, "tpot_success": success,
                    "first_token_ttft_ms": 10 if success else None,
                    "ttft_ms": 20 if success else None, "tpot_ms": 5 if success else None}
        finally:
            with lock:
                active -= 1
                completed_warmups += int(is_warmup)

    monkeypatch.setattr(benchmark, "run_once", run)
    output = tmp_path / "concurrent.jsonl"
    assert benchmark.main(["--query", "资金", "--kb-name", "test", "--warmups", "2",
                           "--runs", "4", "--concurrency", "2", "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert peak == 2 and active == 0
    assert [row["run_number"] for row in rows[:-1]] == list(range(1, 7))
    assert [row["is_warmup"] for row in rows[:-1]] == [True, True, False, False, False, False]
    assert all(row["concurrency"] == 2 for row in rows)
    for metric in ("first_token_ttft", "ttft", "tpot"):
        assert rows[-1][metric]["sample_count"] == 4 - int(with_failure)
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows


def test_cli_default_concurrency_keeps_serial_order(monkeypatch, tmp_path, capsys):
    calls = []

    def run(args, is_warmup, run_number):
        calls.append((args.concurrency, is_warmup, run_number))
        return {"run_number": run_number, "is_warmup": is_warmup, "chat_success": False,
                "tpot_success": False}

    monkeypatch.setattr(benchmark, "run_once", run)
    output = tmp_path / "serial.jsonl"
    benchmark.main(["--query", "资金", "--kb-name", "test", "--warmups", "1",
                    "--runs", "2", "--output", str(output)])
    assert calls == [(1, True, 1), (1, False, 2), (1, False, 3)]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["concurrency"] == 1


@pytest.mark.parametrize("mode,runs,temperature,top_p,url", [
    ("model", 30, 0, 1, "http://127.0.0.1:8080/v1/chat/completions"),
    ("rag", 30, 0.7, 0.95, "http://127.0.0.1:7861/chat/kb_chat"),
])
def test_prepare_args_resolves_distinct_profiles(mode, runs, temperature, top_p, url):
    argv = ["--benchmark-mode", mode, "--query", "资金"]
    if mode == "rag":
        argv += ["--kb-name", "test"]
    resolved = benchmark.prepare_args(argv)
    assert (resolved.runs, resolved.temperature, resolved.top_p, resolved.url) == (runs, temperature, top_p, url)


@pytest.mark.parametrize("argv", [
    ["--benchmark-mode", "model", "--concurrency", "2"],
    ["--benchmark-mode", "rag"],
    ["--benchmark-mode", "model", "--top-p", "0"],
    ["--benchmark-mode", "model", "--temperature", "nan"],
    ["--benchmark-mode", "model", "--min-duration", "-1"],
    ["--benchmark-mode", "model", "--max-tokens-list", "128", "0"],
])
def test_profile_validation_rejects_invalid_combinations(argv):
    with pytest.raises(SystemExit) as exc:
        benchmark.prepare_args(["--query", "资金"] + argv)
    assert exc.value.code == 2


def test_model_stream_supports_done_usage_and_native_timings(monkeypatch):
    args = benchmark.prepare_args(["--benchmark-mode", "model", "--query", "资金"])
    events = [sse({"choices": [{"delta": {"reasoning_content": "思考"}}]}),
              sse({"choices": [{"delta": {"content": "资金"}, "finish_reason": "stop"}]}),
              sse({"choices": [], "usage": {"completion_tokens": 5},
                   "timings": {"prompt_n": 10, "prompt_ms": 20, "predicted_n": 5,
                               "predicted_ms": 100, "predicted_per_second": 50}}), b"data: [DONE]\n\n"]
    response = Response(events)
    calls = []
    monkeypatch.setattr(benchmark.requests, "post", lambda url, **kw: calls.append((url, kw)) or response)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 4)
    times = iter([1.0, 1.1, 1.2, 1.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    row = benchmark.run_once(args, False, 1)
    body = calls[0][1]["json"]
    assert body["messages"] == [{"role": "user", "content": "资金"}]
    assert "kb_name" not in body and "top_k" not in body
    assert row["chat_success"] and row["tpot_success"]
    assert row["finish_reason"] == "stop" and row["truncated"] is False
    assert row["server_decode_tokens_per_second"] == 50
    assert row["server_generated_tokens"] == 5
    assert row["usage_completion_tokens"] == 5
    assert row["total_latency_ms"] == pytest.approx(500)
    assert response.closed


def test_model_only_reasoning_can_record_budget_exhaustion(monkeypatch):
    args = benchmark.prepare_args(["--benchmark-mode", "model", "--query", "资金"])
    response = Response([sse({"choices": [{"delta": {"reasoning_content": "思考"}, "finish_reason": "length"}]}),
                         b"data: [DONE]\n\n"])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: pytest.fail("No正文 to tokenize"))
    row = benchmark.run_once(args, False, 1)
    assert row["chat_success"] and row["truncated"] is True
    assert row["first_token_ttft_ms"] is not None
    assert row["ttft_ms"] is None and not row["tpot_success"]
    assert row["reasoning_chars"] == 2


def test_model_normal_eof_without_completion_evidence_is_failure(monkeypatch):
    args = benchmark.prepare_args(["--benchmark-mode", "model", "--query", "资金"])
    response = Response([sse({"choices": [{"delta": {"content": "资金"}}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    row = benchmark.run_once(args, False, 1)
    assert not row["chat_success"] and row["chat_error"]


def test_rag_sends_sampling_and_records_retrieval_evidence(monkeypatch):
    args = benchmark.prepare_args(["--query", "资金", "--kb-name", "test"])
    response = Response([sse({"choices": [{"delta": {"content": ""}}], "docs": ["出处 [1] 文档"]}),
                         sse({"choices": [{"delta": {"content": "资金"}}]})])
    calls = []
    monkeypatch.setattr(benchmark.requests, "post", lambda url, **kw: calls.append(kw) or response)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 2)
    row = benchmark.run_once(args, False, 1)
    assert calls[0]["json"]["top_p"] == 0.95
    assert calls[0]["json"]["temperature"] == 0.7
    assert row["rag_retrieval_status"] == "hit"
    assert row["reference_count"] == 1
    assert row["truncated"] is None


def test_summary_percentiles_failures_and_truncation_are_explicit():
    records = [{"chat_success": True, "ttft_ms": value, "truncated": value == 100} for value in range(1, 101)]
    records += [{"chat_success": False, "truncated": None}]
    summary = summarize(records)
    assert summary["ttft"]["p90"] == 90 and summary["ttft"]["p95"] == 95
    assert summary["request_count"] == 101 and summary["chat_failure_count"] == 1
    assert summary["truncation"] == {"known_count": 100, "truncated_count": 1, "unknown_count": 1}
    assert summarize(records[:29])["ttft"]["p90"] is None
    assert summarize(records[:99])["ttft"]["p95"] is None


def test_model_matrix_separates_prompts_and_output_budgets(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "queries.json"
    dataset.write_text(json.dumps([{"id": "short", "query": "短问题", "group": "short"},
                                  {"id": "long", "query": "较长的输入问题", "group": "long"}]))
    calls = []
    def run(args, warmup, number):
        calls.append((args.query, args.max_tokens, warmup))
        return {"run_number": number, "is_warmup": warmup, "chat_success": True, "ttft_ms": 1, "tpot_success": False}
    monkeypatch.setattr(benchmark, "run_once", run)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 3)
    output = tmp_path / "matrix.jsonl"
    benchmark.main(["--benchmark-mode", "model", "--queries-file", str(dataset), "--runs", "2",
                    "--warmups", "1", "--max-tokens-list", "128", "256", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    summaries = [row for row in rows if row["record_type"] == "summary"]
    assert len(calls) == 12 and len(summaries) == 4
    assert all(row["request_count"] == 2 for row in summaries)
    assert len({row["configuration_id"] for row in summaries}) == 4
    report = output.with_suffix(".md").read_text()
    assert "128" in report and "256" in report and "样本" in report
    assert "最大支持" not in report
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows


def test_rag_round_robin_and_deferred_tokenize(monkeypatch, tmp_path):
    dataset = tmp_path / "queries.json"
    dataset.write_text(json.dumps([{"id": "a", "query": "问题甲"}, {"id": "b", "query": "问题乙"}]))
    finished = []
    def post(url, **kw):
        if url.endswith("/tokenize"):
            assert len(finished) == 4, "Tokenize must follow the load phase"
            return Response(payload={"tokens": [1, 2]})
        finished.append(kw["json"]["query"])
        return Response([sse({"choices": [{"delta": {"content": "资金"}}]})])
    monkeypatch.setattr(benchmark.requests, "post", post)
    output = tmp_path / "rag.jsonl"
    benchmark.main(["--queries-file", str(dataset), "--kb-name", "test", "--runs", "4", "--warmups", "0",
                    "--concurrency", "2", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["query_id"] for row in rows[:-1]] == ["a", "b", "a", "b"]
    assert all(row["tpot_success"] for row in rows[:-1])
    assert rows[-1]["throughput"]["successful_requests_per_second"] > 0
    assert set(rows[-1]["by_query"]) == {"a", "b"}
    assert not any(key.startswith("_") for row in rows for key in row)


@pytest.mark.parametrize("dataset", [[], [{"id": "a", "query": ""}],
                                    [{"id": "a", "query": "甲"}, {"id": "a", "query": "乙"}],
                                    [{"id": "a", "query": "甲", "group": 42}]])
def test_invalid_dataset_is_rejected_before_file_creation(tmp_path, dataset):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(dataset))
    output = tmp_path / "result.jsonl"
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--benchmark-mode", "model", "--queries-file", str(path), "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()


def test_minimum_duration_adds_requests_and_keeps_query_rotation(monkeypatch, tmp_path):
    elapsed = 0
    calls = []
    def run(args, warmup, number):
        nonlocal elapsed
        elapsed += 1
        calls.append((warmup, number))
        return {"run_number": number, "is_warmup": warmup, "chat_success": False, "tpot_success": False}
    monkeypatch.setattr(benchmark, "run_once", run)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: elapsed)
    output = tmp_path / "duration.jsonl"
    benchmark.main(["--query", "资金", "--kb-name", "test", "--warmups", "0", "--runs", "2",
                    "--min-duration", "5", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 5
    assert rows[-1]["request_count"] == 5
    assert rows[-1]["throughput"]["load_duration_seconds"] == 5


def test_rag_concurrency_stages_have_separate_summaries(monkeypatch, tmp_path):
    def run(args, warmup, number):
        return {"run_number": number, "is_warmup": warmup, "chat_success": True, "ttft_ms": 10,
                "tpot_success": False}
    monkeypatch.setattr(benchmark, "run_once", run)
    output = tmp_path / "concurrencies.jsonl"
    benchmark.main(["--query", "资金", "--kb-name", "test", "--runs", "2", "--warmups", "1",
                    "--concurrency-list", "1", "2", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    summaries = [row for row in rows if row["record_type"] == "summary"]
    assert [row["concurrency"] for row in summaries] == [1, 2]
    assert [row["request_count"] for row in summaries] == [2, 2]


def test_rag_miss_warning_is_not_counted_as_reference(monkeypatch):
    args = benchmark.prepare_args(["--query", "资金", "--kb-name", "test"])
    response = Response([sse({"choices": [{"delta": {"content": ""}}], "docs": ["未找到相关文档,该回答为大模型自身能力解答！"]}),
                         sse({"choices": [{"delta": {"content": "资金"}}]})])
    monkeypatch.setattr(benchmark.requests, "post", lambda *a, **k: response)
    monkeypatch.setattr(benchmark, "count_tokens", lambda *a: 2)
    row = benchmark.run_once(args, False, 1)
    assert row["rag_retrieval_status"] == "miss" and row["reference_count"] == 0


def test_report_contains_reproduction_command_and_native_token_counts(monkeypatch, tmp_path):
    def run(args, warmup, number):
        return {"run_number": number, "is_warmup": warmup, "chat_success": True, "tpot_success": False,
                "server_generated_tokens": 123, "server_prompt_tokens": 456}
    monkeypatch.setattr(benchmark, "run_once", run)
    output = tmp_path / "report.jsonl"
    benchmark.main(["--benchmark-mode", "model", "--query", "资金", "--runs", "1", "--warmups", "0", "--output", str(output)])
    report = output.with_suffix(".md").read_text()
    assert "复测命令" in report and "123.000" in report and "456.000" in report


@pytest.mark.parametrize("mode", ["model", "rag"])
def test_dataset_history_reaches_actual_request(monkeypatch, tmp_path, mode):
    dataset = tmp_path / "history.json"
    history = [{"role": "user", "content": "前一个问题"}, {"role": "assistant", "content": "前一个回答"}]
    dataset.write_text(json.dumps([{"id": "history", "query": "继续说明", "history": history}]))
    requests_seen = []
    def post(url, **kw):
        if url.endswith("/tokenize"):
            return Response(payload={"tokens": [1, 2]})
        requests_seen.append(kw["json"])
        return Response([sse({"choices": [{"delta": {"content": "资金"}, "finish_reason": "stop"}]})])
    monkeypatch.setattr(benchmark.requests, "post", post)
    argv = ["--benchmark-mode", mode, "--queries-file", str(dataset), "--runs", "1", "--warmups", "0",
            "--output", str(tmp_path / "history.jsonl")]
    if mode == "rag":
        argv += ["--kb-name", "test", "--prompt-name", "business"]
    benchmark.main(argv)
    if mode == "model":
        assert requests_seen[0]["messages"] == history + [{"role": "user", "content": "继续说明"}]
        assert requests_seen[0]["cache_prompt"] is False
    else:
        assert requests_seen[0]["history"] == history and requests_seen[0]["prompt_name"] == "business"


def test_no_content_is_not_a_tokenizer_failure():
    summary = summarize([{"chat_success": True, "first_token_ttft_ms": 1,
                          "ttft_ms": None, "tpot_success": False, "truncated": True}])
    assert summary["chat_failure_count"] == 0
    assert summary["no_content_count"] == 1
    assert summary["tpot_failure_count"] == 0
    assert summary["tpot_unavailable_count"] == 1


def test_duration_extension_keeps_round_robin_cursor(monkeypatch, tmp_path):
    elapsed, calls = 0, []
    lock = threading.Lock()
    def run(args, warmup, number):
        nonlocal elapsed
        with lock:
            elapsed += 1
            calls.append((number, args.query))
        return {"run_number": number, "is_warmup": warmup, "chat_success": False, "tpot_success": False}
    monkeypatch.setattr(benchmark, "run_once", run)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: elapsed)
    dataset = tmp_path / "queries.json"
    dataset.write_text(json.dumps([{"id": name, "query": name} for name in ("a", "b", "c")]))
    benchmark.main(["--queries-file", str(dataset), "--kb-name", "test", "--runs", "4", "--warmups", "0",
                    "--concurrency", "4", "--min-duration", "5", "--output", str(tmp_path / "rotation.jsonl")])
    assert len(calls) >= 5
    assert [query for _, query in sorted(calls)] == [("a", "b", "c")[index % 3] for index in range(len(calls))]


def test_duration_load_replenishes_fast_worker_before_slow_one_finishes(monkeypatch, tmp_path):
    slow_started, replacement_started = threading.Event(), threading.Event()
    elapsed = 0
    def run(args, warmup, number):
        nonlocal elapsed
        if number == 1:
            slow_started.set()
            assert replacement_started.wait(5), "fast worker must be refilled while slow request is running"
        elif number == 2:
            assert slow_started.wait(5)
        elif number == 3:
            elapsed = 6
            replacement_started.set()
        return {"run_number": number, "is_warmup": warmup, "chat_success": False, "tpot_success": False}
    monkeypatch.setattr(benchmark, "run_once", run)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: elapsed)
    benchmark.main(["--query", "资金", "--kb-name", "test", "--runs", "2", "--warmups", "0",
                    "--concurrency", "2", "--min-duration", "5", "--output", str(tmp_path / "refill.jsonl")])
    assert replacement_started.is_set()


@pytest.mark.parametrize("suffix", [".md", ".MD"])
def test_output_cannot_collide_with_markdown_report(monkeypatch, tmp_path, suffix):
    monkeypatch.setattr(benchmark, "run_once", lambda args, warmup, number:
                        {"run_number": number, "is_warmup": warmup, "chat_success": False, "tpot_success": False})
    output = tmp_path / ("result" + suffix)
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--benchmark-mode", "model", "--query", "资金", "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()


def test_report_references_actual_raw_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "run_once", lambda args, warmup, number:
                        {"run_number": number, "is_warmup": warmup, "chat_success": False, "tpot_success": False})
    output = tmp_path / "custom.txt"
    benchmark.main(["--benchmark-mode", "model", "--query", "资金", "--runs", "1", "--warmups", "0", "--output", str(output)])
    report = output.with_suffix(".md").read_text()
    assert '原始记录：`custom.txt`' in report


def test_model_baseline_rejects_duration_extension():
    with pytest.raises(SystemExit) as exc:
        benchmark.prepare_args(["--benchmark-mode", "model", "--query", "资金", "--min-duration", "1"])
    assert exc.value.code == 2
    args = benchmark.prepare_args(["--benchmark-mode", "model", "--query", "资金", "--runs", "7", "--min-duration", "0"])
    assert args.runs == 7 and args.min_duration == 0


def test_help_identifies_llama_specific_default_field():
    help_text = benchmark.build_parser().format_help()
    assert "cache_prompt=false" in help_text
    assert "service-default" in help_text


def test_report_labels_sources_and_rejects_cross_profile_subtraction(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "run_once", lambda args, warmup, number:
                        {"run_number": number, "is_warmup": warmup, "chat_success": True, "tpot_success": True,
                         "ttft_ms": 2, "tpot_ms": 3, "output_tokens": 4, "server_decode_tokens_per_second": 5})
    output = tmp_path / "sources.jsonl"
    benchmark.main(["--benchmark-mode", "model", "--query", "资金", "--runs", "1", "--warmups", "0", "--output", str(output)])
    report = output.with_suffix(".md").read_text()
    assert "| 指标 | 来源 |" in report
    assert "| 正文 ms/token | 客户端计时＋正文重分词 |" in report
    assert "| 服务端 decode tokens/s | 模型服务原生 timings |" in report
    assert "不能用 rag 延迟减 model 延迟" in report
    assert "摸底" in report and "暂无验收标准" in report
