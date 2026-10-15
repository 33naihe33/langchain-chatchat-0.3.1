import argparse
import codecs
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
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
                "p90": values[math.ceil(len(values) * 0.90) - 1] if len(values) >= 30 else None,
                "p95": values[math.ceil(len(values) * 0.95) - 1] if len(values) >= 100 else None,
                "min": values[0] if values else None, "max": values[-1] if values else None}

    failures = sum(not row.get("chat_success") for row in records)
    known = [row for row in records if row.get("truncated") is not None]
    return {"record_type": "summary", "p50_method": "lower_median", "tail_percentile_method": "nearest_rank",
            "request_count": len(records), "chat_failure_count": failures,
            "chat_failure_rate": failures / len(records) if records else None,
            "no_content_count": sum(bool(row.get("chat_success")) and row.get("ttft_ms") is None for row in records),
            "tpot_unavailable_count": sum(not row.get("tpot_success") for row in records),
            "tpot_failure_count": sum(bool(row.get("chat_success")) and row.get("ttft_ms") is not None
                                      and not row.get("tpot_success") for row in records),
            "truncation": {"known_count": len(known), "truncated_count": sum(bool(row["truncated"]) for row in known),
                           "unknown_count": len(records) - len(known)},
            "output_variant_count": len({row["output_sha256"] for row in records if row.get("chat_success")
                                         and row.get("output_chars", 0) > 0 and row.get("output_sha256")}),
            "first_token_ttft": metric("first_token_ttft_ms", "chat_success"),
            "ttft": metric("ttft_ms", "chat_success"), "tpot": metric("tpot_ms", "tpot_success"),
            "total_latency": metric("total_latency_ms", "chat_success"),
            "generation_duration": metric("generation_duration_ms", "chat_success"),
            "output_tokens": metric("output_tokens", "tpot_success"),
            "usage_prompt_tokens": metric("usage_prompt_tokens", "chat_success"),
            "server_prompt_tokens": metric("server_prompt_tokens", "chat_success"),
            "server_generated_tokens": metric("server_generated_tokens", "chat_success"),
            "server_generation_ms": metric("server_generation_ms", "chat_success"),
            "server_decode_tokens_per_second": metric("server_decode_tokens_per_second", "chat_success")}


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
    mode = args.benchmark_mode
    params = {"stream": True, "model": args.model, "max_tokens": args.max_tokens,
              "temperature": args.temperature if args.temperature is not None else (0 if mode == "model" else 0.7),
              "top_p": args.top_p if args.top_p is not None else (1 if mode == "model" else 0.95)}
    if mode == "model":
        params["messages"] = getattr(args, "history", []) + [{"role": "user", "content": args.query}]
        params["stream_options"] = {"include_usage": True}
        if args.prompt_cache != "service-default":
            params["cache_prompt"] = args.prompt_cache == "on"
    else:
        params.update(query=args.query, mode="local_kb", kb_name=args.kb_name,
                      top_k=args.top_k, score_threshold=args.score_threshold,
                      history=getattr(args, "history", []), prompt_name=args.prompt_name)
    row = {"record_type": "run", "run_number": run_number, "is_warmup": is_warmup,
           "concurrency": args.concurrency,
           "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "request_params": params,
           "chat_success": False, "chat_error": None, "tpot_success": False, "tpot_error": None,
           "first_token_ttft_ms": None, "ttft_ms": None,
           "generation_duration_ms": None, "output_tokens": None, "tpot_ms": None,
           "benchmark_mode": mode, "total_latency_ms": None, "completed_at": None,
           "finish_reason": None, "truncated": None, "reasoning_chars": 0,
           "rag_retrieval_status": None, "reference_count": None,
           "usage_completion_tokens": None, "usage_prompt_tokens": None, "server_generated_tokens": None,
           "generated_tokens_source": None,
           "server_generation_ms": None, "server_decode_tokens_per_second": None,
           "server_prompt_tokens": None, "server_prompt_ms": None}
    answer = ""
    first_token_at = None
    first_content_at = None
    response = None
    saw_done = False
    eof_at = None
    try:
        started = time.perf_counter()
        response = requests.post(args.url, json=params, stream=True,
                                 timeout=(args.connect_timeout, args.read_timeout))
        response.raise_for_status()
        for data in iter_sse_data(response.iter_content(chunk_size=None)):
            if mode == "model" and data.strip() == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            if mode == "model" and isinstance(event, dict):
                usage = event.get("usage")
                if isinstance(usage, dict):
                    for source, target in (("completion_tokens", "usage_completion_tokens"), ("prompt_tokens", "usage_prompt_tokens")):
                        value = usage.get(source)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            row[target] = value
                timings = event.get("timings")
                if isinstance(timings, dict):
                    for source, target in (("predicted_n", "server_generated_tokens"),
                                           ("predicted_ms", "server_generation_ms"),
                                           ("predicted_per_second", "server_decode_tokens_per_second"),
                                           ("prompt_n", "server_prompt_tokens"), ("prompt_ms", "server_prompt_ms")):
                        value = timings.get(source)
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                            row[target] = value
                if event.get("choices") == [] and (isinstance(usage, dict) or isinstance(timings, dict)):
                    continue
            if mode == "rag" and isinstance(event, dict) and isinstance(event.get("docs"), list):
                refs = [doc for doc in event["docs"] if not (isinstance(doc, str) and "未找到相关文档" in doc)]
                row.update(reference_count=len(refs), rag_retrieval_status="hit" if refs else "miss")
            choices = event.get("choices") if isinstance(event, dict) else None
            if (not isinstance(choices, list) or not choices
                    or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get("delta"), dict)):
                raise ValueError("SSE 响应缺少有效的 choices[0].delta")
            finish = choices[0].get("finish_reason")
            if finish is not None:
                if not isinstance(finish, str):
                    raise ValueError("finish_reason 必须是字符串或 null")
                row.update(finish_reason=finish, truncated=finish == "length")
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
            row["reasoning_chars"] += len(reasoning)
            if first_token_at is None and (reasoning or content):
                first_token_at = time.perf_counter()
                if content:
                    first_content_at = first_token_at
            if content:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                answer += content
        eof_at = time.perf_counter()
        if mode == "model" and not (saw_done or row["finish_reason"]):
            raise ValueError("模型流未提供 DONE 或 finish_reason，无法确认完成")
        if first_token_at is None or (mode == "rag" and first_content_at is None):
            raise ValueError("SSE 响应未包含非空生成内容")
        if row["server_generated_tokens"] is not None:
            row["generated_tokens_source"] = "timings.predicted_n"
        elif row["usage_completion_tokens"] is not None:
            row.update(server_generated_tokens=row["usage_completion_tokens"], generated_tokens_source="usage.completion_tokens")
        row.update(chat_success=True, first_token_ttft_ms=(first_token_at - started) * 1000,
                   ttft_ms=(first_content_at - started) * 1000 if first_content_at is not None else None,
                   generation_duration_ms=(eof_at - first_content_at) * 1000 if first_content_at is not None else None,
                   total_latency_ms=(eof_at - started) * 1000,
                   completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        row["chat_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
    row["output_chars"] = len(answer)
    row["output_sha256"] = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    if args.include_output:
        row["output_text"] = answer
    if getattr(args, "defer_tokenize", False):
        row.update(_answer=answer, _first_content_at=first_content_at, _eof_at=eof_at)
    else:
        finish_token_count(args, row, answer, first_content_at, eof_at)
    return row


def finish_token_count(args, row, answer, first_content_at, eof_at):
    if row["chat_success"] and first_content_at is not None:
        try:
            tokens = count_tokens(args.tokenize_url, answer, (args.connect_timeout, args.read_timeout))
            row.update(tpot_success=True, output_tokens=tokens,
                       tpot_ms=calculate_tpot(first_content_at, eof_at, tokens) * 1000)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            row["tpot_error"] = f"{type(exc).__name__}: {exc}"
    elif row["chat_success"]:
        row["tpot_error"] = "仅思考、无正文，正文 TTFT/TPOT 不适用"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型服务基线与 RAG 业务性能摸底（无验收结论）")
    parser.add_argument("--benchmark-mode", choices=("model", "rag"), default="rag")
    parser.add_argument("--url", default=None)
    parser.add_argument("--tokenize-url", default="http://127.0.0.1:8080/tokenize")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query")
    inputs.add_argument("--queries-file", help="UTF-8 JSON 数组：id、query、可选 group")
    parser.add_argument("--kb-name")
    parser.add_argument("--prompt-name", default="default", help="RAG 业务提示词模板名")
    parser.add_argument("--model", default="MiniCPM5-2B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-tokens-list", nargs="+", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--runs", type=int, default=None, help="model 每题/预算默认 30；rag 每预算/并发默认 30")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1, help="最多同时执行的轮次，默认 1（串行）")
    parser.add_argument("--concurrency-list", nargs="+", type=int)
    parser.add_argument("--min-duration", type=float, default=0, help="rag 每配置正式负载最短秒数，需同时满足 runs；model 仅允许 0；默认 0")
    parser.add_argument("--environment-label", default="未提供硬件/模型服务详细配置")
    parser.add_argument("--prompt-cache", choices=("off", "on", "service-default"), default=None,
                        help="model 默认 off，发送 llama.cpp 扩展 cache_prompt=false；通用 OpenAI 兼容接口可能拒绝，请选 service-default 省略该字段")
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=120)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-output", action="store_true")
    return parser


def prepare_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.invocation = list(argv) if argv is not None else sys.argv[1:]
    if args.output and Path(args.output).suffix.lower() == ".md":
        parser.error("--output 不可使用 .md 后缀，避免与自动报告覆盖；建议使用 .jsonl")
    model_mode = args.benchmark_mode == "model"
    args.runs = args.runs if args.runs is not None else 30
    args.temperature = args.temperature if args.temperature is not None else (0 if model_mode else 0.7)
    args.top_p = args.top_p if args.top_p is not None else (1 if model_mode else 0.95)
    args.prompt_cache = args.prompt_cache or ("off" if model_mode else "service-default")
    args.url = args.url or ("http://127.0.0.1:8080/v1/chat/completions" if model_mode else "http://127.0.0.1:7861/chat/kb_chat")
    args.budgets = args.max_tokens_list or [args.max_tokens]
    args.concurrencies = args.concurrency_list or [args.concurrency]
    if args.runs < 1 or args.warmups < 0 or args.max_tokens < 1:
        parser.error("--runs 和 --max-tokens 必须大于 0，--warmups 必须不小于 0")
    if any(value < 1 for value in args.concurrencies):
        parser.error("--concurrency 必须大于 0")
    if any(value < 1 for value in args.budgets):
        parser.error("所有输出预算必须大于 0")
    if model_mode and any(value != 1 for value in args.concurrencies):
        parser.error("model 基线模式并发固定为 1；业务并发请使用 rag 模式")
    if not model_mode and not args.kb_name:
        parser.error("rag 模式需要 --kb-name")
    if not model_mode and args.prompt_cache != "service-default":
        parser.error("rag 模式沿用业务缓存配置，不支持 --prompt-cache on/off")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("temperature 必须在 0–2 之间")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("top_p 必须满足 0 < top_p <= 1")
    if not math.isfinite(args.min_duration) or args.min_duration < 0:
        parser.error("min-duration 必须是有限非负数")
    if model_mode and args.min_duration != 0:
        parser.error("model 基线按固定 runs 测量，--min-duration 仅允许 0；持续负载请使用 rag 模式")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.connect_timeout, args.read_timeout)):
        parser.error("--connect-timeout 和 --read-timeout 必须是有限正数")
    try:
        queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8-sig")) if args.queries_file else [{"id": "single", "query": args.query}]
        if not isinstance(queries, list) or not queries:
            raise ValueError("题集必须是非空 JSON 数组")
        ids = set()
        for query in queries:
            if (not isinstance(query, dict) or not isinstance(query.get("id"), str) or not query["id"].strip()
                    or not isinstance(query.get("query"), str) or not query["query"].strip()
                    or not isinstance(query.get("group", query["id"]), str) or query["id"] in ids):
                raise ValueError("每题需要唯一非空字符串 id、非空 query 及可选字符串 group")
            ids.add(query["id"])
            history = query.get("history", [])
            if not isinstance(history, list) or any(not isinstance(item, dict) or item.get("role") not in ("user", "assistant", "system")
                                                   or not isinstance(item.get("content"), str) for item in history):
                raise ValueError("history 必须是含 role 与字符串 content 的消息数组")
        args.queries = queries
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(f"题集错误：{exc}")
    return args


def configurations(args):
    groups = [[query] for query in args.queries] if args.benchmark_mode == "model" else [args.queries]
    for index, queries in enumerate(groups, 1):
        for budget in dict.fromkeys(args.budgets):
            for concurrency in dict.fromkeys(args.concurrencies):
                config = argparse.Namespace(**vars(args))
                config.queries, config.max_tokens, config.concurrency = queries, budget, concurrency
                config.configuration_id = f"{args.benchmark_mode}-input{index}-tokens{budget}-c{concurrency}"
                config.defer_tokenize = True
                yield config


def run_job(config, is_warmup, index, offset):
    query = config.queries[offset % len(config.queries)]
    params = argparse.Namespace(**vars(config))
    params.query = query["query"]
    params.history = query.get("history", [])
    row = run_once(params, is_warmup, index)
    row.update(record_type="run", benchmark_mode=config.benchmark_mode, configuration_id=config.configuration_id,
               concurrency=config.concurrency, query_id=query["id"], input_group=query.get("group", query["id"]),
               query_chars=len(query["query"]))
    return row


def execute_phase(config, executor, is_warmup, start, count):
    return list(executor.map(lambda index: run_job(config, is_warmup, index, index - start), range(start, start + count)))


def execute_load(config, executor, start):
    deadline = time.monotonic() + config.min_duration
    submitted, pending, records = 0, set(), []

    def submit():
        nonlocal submitted
        pending.add(executor.submit(run_job, config, False, start + submitted, submitted))
        submitted += 1

    initial = config.concurrency if config.min_duration > 0 else min(config.concurrency, config.runs)
    for _ in range(initial):
        submit()
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.remove(future)
            records.append(future.result())
            if submitted < config.runs or time.monotonic() < deadline:
                submit()
    return sorted(records, key=lambda row: row["run_number"])


def write_report(path, args, summaries):
    lines = ["# 模型基线与 RAG 性能摸底测试说明", "", f"- 模式：`{args.benchmark_mode}`；接口：`{args.url}`。",
             f"- 客户端：{platform.platform()}；Python {platform.python_version()}；requests {requests.__version__}。",
             f"- 环境备注：{args.environment_label}。", f"- 采样参数：temperature={args.temperature}，top_p={args.top_p}。",
             f"- prompt 缓存：{args.prompt_cache}（off/on 为 llama.cpp 扩展，仅 model 模式发送）。",
             "- 暂无验收标准，仅描述本批实测表现。成功不代表答案正确或具备生产容量。",
             "- 模型服务槽位、量化、思考开关与线程数未自动探测，需由环境备注补充。", "",
             "## 指标口径", "",
             "首 token 是首个非空思考或正文事件；首正文 TTFT 包含此前思考。总客户端延迟到 EOF/DONE 为止，含网络及服务排队。",
             "正文 TPOT 仍以正文重分词数和首正文到结束时间计算；它不是原生模型解码速度。",
             "服务端 decode tokens/s 仅取模型返回的 timings.predicted_per_second；缺失则留空，不能用正文速度替代。",
             "完整输入长度取 usage.prompt_tokens；timings.prompt_n 是实际 prefill 处理 token 数，缓存启用时不一定等于完整输入长度。",
             "预热不进入正式统计，首轮预热不等于已验证的冷启动。RAG 截断/检索字段缺失时标为未知。",
             "负载采用固定并发闭环：完成一个请求后补充下一个；吞吐量为成功问答数/负载墙钟秒数。辅助 tokenize 在负载结束后执行。",
             "P50 使用 lower_median；P90/P95 使用 nearest_rank，至少分别需要 30/100 个有效样本；这是报告规则，不是可靠性保证。", ""]
    for summary in summaries:
        config = summary["config"]
        lines += [f"## {summary['configuration_id']}", "",
                  f"- 输出预算：{config['max_tokens']}；并发上限：{summary['concurrency']}；预热 {summary['warmup_count']} 次。",
                  f"- 正式请求数：{summary['request_count']}；失败数：{summary['chat_failure_count']}；失败率：{summary['chat_failure_rate']:.2%}。",
                  f"- 负载持续 {summary['throughput']['load_duration_seconds']:.3f} 秒；成功问答吞吐量 {summary['throughput']['successful_requests_per_second']:.3f} 请求/秒。",
                  f"- 截断：已知 {summary['truncation']['known_count']}，其中截断 {summary['truncation']['truncated_count']}；未知 {summary['truncation']['unknown_count']}。",
                  f"- 无正文但生成完成：{summary['no_content_count']}；有正文但计数失败：{summary['tpot_failure_count']}；TPOT 不可用总数：{summary['tpot_unavailable_count']}。",
                  f"- 输出摘要变体数：{summary['output_variant_count']}；有正文但计数失败的轮次会缺少 TPOT。", "",
                  "| 指标 | 来源 | 有效样本 | 均值 | P50 | P90 | P95 | 最小 | 最大 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, label, source in (("usage_prompt_tokens", "模型实际输入 tokens", "模型服务 usage.prompt_tokens"),
                           ("server_prompt_tokens", "服务端 prefill tokens", "模型服务原生 timings"),
                           ("server_generated_tokens", "服务端实际生成 tokens", "模型服务 timings.predicted_n，缺失时取 usage.completion_tokens"),
                           ("server_generation_ms", "服务端解码 ms", "模型服务原生 timings"),
                           ("first_token_ttft", "首 token ms", "客户端计时"), ("ttft", "首正文 ms", "客户端计时"),
                           ("total_latency", "总客户端延迟 ms", "客户端计时"),
                           ("output_tokens", "正文 token 数", "客户端调用 /tokenize 对正文重分词"),
                           ("tpot", "正文 ms/token", "客户端计时＋正文重分词"),
                           ("server_decode_tokens_per_second", "服务端 decode tokens/s", "模型服务原生 timings")):
            metric = summary[key]
            values = ["未提供/样本不足" if metric[name] is None else f"{metric[name]:.3f}" for name in ("mean", "p50", "p90", "p95", "min", "max")]
            lines.append(f"| {label} | {source} | {metric['sample_count']} | " + " | ".join(values) + " |")
        lines += ["", "### 分题样本", "", "| 题目 ID | 输入组 | 查询字符数 | 正式请求 | 聊天失败 | 首正文均值 ms |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for key, group in summary["by_query"].items():
            value = group["ttft"]["mean"]
            lines.append(f"| {key.replace('|', '/') } | {group['input_group'].replace('|', '/')} | {group['query_chars']} | {group['request_count']} | {group['chat_failure_count']} | {value:.3f} |" if value is not None else
                         f"| {key.replace('|', '/')} | {group['input_group'].replace('|', '/')} | {group['query_chars']} | {group['request_count']} | {group['chat_failure_count']} | 无正文 |")
        lines += ["", f"检索证据：{json.dumps(summary['retrieval'], ensure_ascii=False)}。", ""]
        lines += [f"- 注意：{warning}" for warning in summary["warnings"]]
        lines.append("")
    raw_path = Path(args.output) if args.output else path.with_suffix(".jsonl")
    lines += ["## 数据与限制", "", f"原始记录：`{raw_path.name}`。慢请求全部保留，各配置分别统计，未混合成一个整体结论。",
              "不能用 rag 延迟减 model 延迟来分离检索开销：两模式的采样、缓存和实际输入不同。分离各阶段耗时需业务链路计时或另做控制变量测试。",
              "max_tokens 是思考与正文共享的上限，不保证实际生成到此长度；模型模式仅思考后 length 结束仍可记录推理性能，正文指标不适用。",
              "temperature=0 不保证输出完全相同；查询字符数不等于模型实际输入长度，RAG 模式该长度通常不可观测。",
              "题集应覆盖真实问题、不同输入/上下文与回答长度，并在不同时间重复运行。未自动评价回答质量，生产采样与业务参数仍需确认。",
              "正文不足 2 个 token 时没有实际 token 间隔，现有 TPOT 分母取 1 仅为边界处理。", "",
              "## 复测命令", "", "在项目根目录执行；复测应更换 --output 文件名，题集文件和服务配置也需保持一致。", "",
              "```bash", "LC_ALL=en_US.UTF-8 " + shlex.join([sys.executable, "-u", "scripts/benchmark_sse.py", *args.invocation]), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = prepare_args(argv)
    output = Path(args.output or f"data/benchmarks/{args.benchmark_mode}-{dt.datetime.now():%Y%m%d-%H%M%S-%f}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries, start = [], 1
    with output.open("w", encoding="utf-8") as handle:
        for config in configurations(args):
            with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
                warmups = execute_phase(config, executor, True, start, config.warmups)
                start += len(warmups)
                began = time.monotonic()
                measured = execute_load(config, executor, start)
                start += len(measured)
                duration = time.monotonic() - began
            records = warmups + measured
            for row in records:
                if "_answer" in row:
                    answer, first, end = row.pop("_answer"), row.pop("_first_content_at"), row.pop("_eof_at")
                    finish_token_count(config, row, answer, first, end)
                line = json.dumps(row, ensure_ascii=False)
                print(line)
                handle.write(line + "\n")
                handle.flush()
            summary = summarize(records)
            config_keys = ["model", "url", "tokenize_url", "max_tokens", "temperature", "top_p", "prompt_cache", "min_duration"]
            if config.benchmark_mode == "rag":
                config_keys += ["top_k", "score_threshold", "kb_name", "prompt_name"]
            summary.update(configuration_id=config.configuration_id, benchmark_mode=config.benchmark_mode,
                           concurrency=config.concurrency, warmup_count=len(warmups),
                           environment={"label": args.environment_label, "client_platform": platform.platform(),
                                        "python": platform.python_version(), "requests": requests.__version__},
                           config={key: getattr(config, key) for key in config_keys},
                           throughput={"load_duration_seconds": duration,
                                       "successful_requests_per_second": sum(bool(row.get("chat_success")) for row in measured) / duration if duration > 0 else 0})
            summary["by_query"] = {}
            for query in config.queries:
                subset = [row for row in measured if row["query_id"] == query["id"]]
                group = summarize(subset)
                group.update(input_group=query.get("group", query["id"]), query_chars=len(query["query"]))
                summary["by_query"][query["id"]] = group
            summary["retrieval"] = {name: sum(row.get("rag_retrieval_status") == name for row in measured) for name in ("hit", "miss")}
            summary["retrieval"]["unknown"] = sum(row.get("rag_retrieval_status") is None for row in measured)
            target = 30
            summary["warnings"] = []
            if summary["request_count"] < target:
                summary["warnings"].append(f"样本不足：此配置仅 {summary['request_count']} 次，摸底起步建议 {target} 次。")
            if len(config.queries) == 1:
                summary["warnings"].append("仅覆盖一个问题，不能推断整体业务表现。")
            if config.benchmark_mode == "rag" and config.min_duration == 0:
                summary["warnings"].append("未设置持续时长下限，仅按请求数取样；持续负载可设置 --min-duration。")
            if summary["truncation"]["unknown_count"]:
                summary["warnings"].append("部分响应未提供结束原因，无法核验是否因输出预算截断。")
            if summary["server_decode_tokens_per_second"]["sample_count"] == 0 and config.benchmark_mode == "model":
                summary["warnings"].append("模型未提供原生推理 timings，客户端延迟不能直接等同硬件原生解码速度。")
            summaries.append(summary)
            line = json.dumps(summary, ensure_ascii=False)
            print(line)
            handle.write(line + "\n")
            handle.flush()
    write_report(output.with_suffix(".md"), args, summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
