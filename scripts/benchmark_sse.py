import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
from itertools import repeat
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import requests


def iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    def consume():
        nonlocal buffer
        events = buffer.replace("\r\n", "\n").split("\n\n")
        buffer = events.pop()
        for event in events:
            data = [line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")]
            if data:
                yield "\n".join(data)

    for chunk in chunks:
        buffer += decoder.decode(chunk)
        yield from consume()
    buffer += decoder.decode(b"", final=True)
    if buffer:
        data = [line[5:].lstrip() for line in buffer.replace("\r\n", "\n").split("\n") if line.startswith("data:")]
        if data:
            yield "\n".join(data)


def calculate_tpot(first_content_at: float, eof_at: float, token_count: int) -> float:
    return (eof_at - first_content_at) / max(token_count - 1, 1)


def summarize(records: Sequence[dict]) -> dict:
    records = [row for row in records if not row.get("is_warmup")]

    def metric(key, flag):
        values = sorted(row[key] for row in records if row.get(flag) and row.get(key) is not None)
        return {"sample_count": len(values), "mean": statistics.fmean(values) if values else None,
                "p50": values[(len(values) - 1) // 2] if values else None,
                "min": values[0] if values else None, "max": values[-1] if values else None}

    return {"record_type": "summary", "p50_method": "lower_median",
            "first_token_ttft": metric("first_token_ttft_ms", "chat_success"),
            "ttft": metric("ttft_ms", "chat_success"), "tpot": metric("tpot_ms", "tpot_success")}


def count_tokens(url: str, content: str, timeout: Tuple[float, float]) -> int:
    response = requests.post(url, json={"content": content, "add_special": False}, timeout=timeout)
    try:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
            raise ValueError("/tokenize 响应缺少 tokens 数组")
        return len(payload["tokens"])
    finally:
        response.close()


def run_once(args: argparse.Namespace, is_warmup: bool, run_number: int) -> dict:
    params = {"query": args.query, "mode": "local_kb", "kb_name": args.kb_name, "stream": True,
              "model": args.model, "max_tokens": args.max_tokens, "top_k": args.top_k,
              "score_threshold": args.score_threshold, "temperature": args.temperature}
    row = {"record_type": "run", "run_number": run_number, "is_warmup": is_warmup,
           "concurrency": args.concurrency,
           "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "request_params": params,
           "chat_success": False, "chat_error": None, "tpot_success": False, "tpot_error": None,
           "first_token_ttft_ms": None, "ttft_ms": None,
           "generation_duration_ms": None, "output_tokens": None, "tpot_ms": None}
    answer = ""
    first_token_at = None
    first_content_at = None
    response = None
    try:
        started = time.perf_counter()
        response = requests.post(args.url, json=params, stream=True,
                                 timeout=(args.connect_timeout, args.read_timeout))
        response.raise_for_status()
        for data in iter_sse_data(response.iter_content(chunk_size=None)):
            event = json.loads(data)
            choices = event.get("choices") if isinstance(event, dict) else None
            if (not isinstance(choices, list) or not choices
                    or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get("delta"), dict)):
                raise ValueError("SSE 响应缺少有效的 choices[0].delta")
            content = choices[0]["delta"].get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise ValueError("SSE delta.content 必须是字符串或 null")
            reasoning = choices[0]["delta"].get("reasoning_content")
            if reasoning is None:
                reasoning = ""
            if not isinstance(reasoning, str):
                raise ValueError("SSE delta.reasoning_content 必须是字符串或 null")
            if first_token_at is None and (reasoning or content):
                first_token_at = time.perf_counter()
                if content:
                    first_content_at = first_token_at
            if content:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                answer += content
        eof_at = time.perf_counter()
        if first_content_at is None:
            raise ValueError("SSE 响应未包含非空生成内容")
        row.update(chat_success=True, first_token_ttft_ms=(first_token_at - started) * 1000,
                   ttft_ms=(first_content_at - started) * 1000,
                   generation_duration_ms=(eof_at - first_content_at) * 1000)
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        row["chat_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
    row["output_chars"] = len(answer)
    row["output_sha256"] = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    if args.include_output:
        row["output_text"] = answer
    if row["chat_success"]:
        try:
            tokens = count_tokens(args.tokenize_url, answer, (args.connect_timeout, args.read_timeout))
            row.update(tpot_success=True, output_tokens=tokens,
                       tpot_ms=calculate_tpot(first_content_at, eof_at, tokens) * 1000)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            row["tpot_error"] = f"{type(exc).__name__}: {exc}"
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测量 Chatchat SSE TTFT 与 TPOT")
    parser.add_argument("--url", default="http://127.0.0.1:7861/chat/kb_chat")
    parser.add_argument("--tokenize-url", default="http://127.0.0.1:8080/tokenize")
    parser.add_argument("--query", required=True)
    parser.add_argument("--kb-name", required=True)
    parser.add_argument("--model", default="MiniCPM5-2B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1, help="最多同时执行的轮次，默认 1（串行）")
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=120)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-output", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmups < 0 or args.max_tokens < 1:
        parser.error("--runs 和 --max-tokens 必须大于 0，--warmups 必须不小于 0")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.connect_timeout, args.read_timeout)):
        parser.error("--connect-timeout 和 --read-timeout 必须是有限正数")
    output = Path(args.output or f"data/benchmarks/{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for is_warmup, start, count in ((True, 1, args.warmups), (False, args.warmups + 1, args.runs)):
            indexes = range(start, start + count)
            for row in executor.map(run_once, repeat(args), repeat(is_warmup), indexes):
                records.append(row)
                line = json.dumps(row, ensure_ascii=False)
                print(line)
                handle.write(line + "\n")
        summary = summarize(records)
        summary["concurrency"] = args.concurrency
        line = json.dumps(summary, ensure_ascii=False)
        print(line)
        handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
