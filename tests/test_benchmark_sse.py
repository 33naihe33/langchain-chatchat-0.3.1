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
    return benchmark.build_parser().parse_args(["--query", "资金", "--kb-name", "test"])


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
    assert summary["ttft"] == {"sample_count": 2, "mean": 20, "p50": 10, "min": 10, "max": 30}
    assert summary["tpot"] == {"sample_count": 0, "mean": None, "p50": None, "min": None, "max": None}


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
    assert summary["first_token_ttft"] == {"sample_count": 2, "mean": 10, "p50": 0, "min": 0, "max": 20}
    assert summary["ttft"]["sample_count"] == 4
    assert summarize([])["first_token_ttft"]["sample_count"] == 0


def test_cli_writes_both_ttft_metrics_and_excludes_warmup(monkeypatch, tmp_path, capsys):
    responses = iter([
        Response([sse({"choices": [{"delta": {"reasoning_content": "思考"}}]}),
                  sse({"choices": [{"delta": {"content": "资金"}}]})]),
        Response(payload={"tokens": [1, 2, 3, 4]}),
        Response([sse({"choices": [{"delta": {"content": "资金"}}]})]),
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
