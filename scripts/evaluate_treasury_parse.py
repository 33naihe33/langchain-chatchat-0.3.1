"""Evaluate synthetic, business-reviewed JSONL labels against the parsing API.

No model service is started and no business interface is called by this CLI.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GROUPS = {"account": 40, "transactions": 40, "balance": 40,
          "single_only": 40, "unrelated": 20, "historical": 20}
SINGLES = {"account", "transactions", "balance"}
CODES = {"C010100324", "C010100001", "C010100026"}
FIELDS = {
    "C010100324": {"acctNo", "acctNoNme", "bankCd", "bankNme", "dirModeFlg", "ccyCds", "pageSize", "currentPage"},
    "C010100001": {"startDate", "endDate", "bankCd", "acctNo", "acctNoNme", "debitCdtFlg", "summary", "amtSort", "pageSize", "currentPage"},
    "C010100026": {"acctNo", "acctNoNme", "bankCd", "txnMnmcSymbl", "pageSize", "currentPage"},
}
METADATA_KEYS = {"model", "quantization", "backend_version", "hardware", "context_length", "mode", "max_tokens", "call_timeout", "total_timeout"}


def business_date():
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


def bind_date(sample, today):
    result = dict(sample)
    result["expected"] = json.loads(json.dumps(sample["expected"], ensure_ascii=False).replace("{{TODAY}}", today))
    return result


def protocol_valid(value, today=None):
    if not _structural_protocol_valid(value):
        return False
    library = str(Path(__file__).resolve().parents[1] / "libs/chatchat-server")
    if library not in sys.path:
        sys.path.insert(0, library)
    from chatchat.server.chat.treasury_parse import OutputError, validate_result
    try:
        return validate_result(value, date.fromisoformat(today or business_date())) == value
    except (OutputError, TypeError, ValueError):
        return False


def _structural_protocol_valid(value):
    if not isinstance(value, dict) or set(value) != {"status", "trCode", "params", "candidates", "clarification"}:
        return False
    status, code = value["status"], value["trCode"]
    params, candidates, question = value["params"], value["candidates"], value["clarification"]
    if not isinstance(params, dict) or not isinstance(candidates, list) or candidates:
        return False
    if code is not None and (not isinstance(code, str) or code not in CODES or not set(params) <= FIELDS[code]):
        return False
    if code is None and params:
        return False
    if status == "ready":
        return code in CODES and not candidates and question is None
    if not isinstance(question, str) or not question.strip():
        return False
    if status == "unsupported":
        return code is None and not params and not candidates
    return False


def _canonical(value):
    """List filters have set semantics; code/name strings remain exact."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        encoded = {json.dumps(_canonical(v), sort_keys=True, ensure_ascii=False) for v in value}
        return [json.loads(v) for v in sorted(encoded)]
    return value


def score(expected, actual, today=None):
    protocol = protocol_valid(actual, today)
    if not protocol:
        return dict(protocol=False, intent=False, exact=False)
    def signature(v):
        return v["status"], v["trCode"]
    intent = signature(actual) == signature(expected)
    exact = intent and _canonical(actual["params"]) == _canonical(expected["params"])
    return dict(protocol=protocol, intent=intent, exact=exact)


def load_samples(path, smoke=False, today=None):
    samples = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise ValueError("样例集不能为空")
    ids = set()
    bound = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("每条样例必须是JSON对象")
        if not isinstance(sample.get("id"), str) or not sample["id"].strip() or sample["id"] in ids:
            raise ValueError("每条样例需要唯一字符串id")
        ids.add(sample["id"])
        if not isinstance(sample.get("group"), str) or sample["group"] not in GROUPS or not isinstance(sample.get("query"), str) or not sample["query"].strip():
            raise ValueError("样例group或query非法")
        if type(sample.get("critical", False)) is not bool:
            raise ValueError("critical必须是布尔值")
        if "expected" not in sample:
            raise ValueError("缺少expected")
        sample = bind_date(sample, today or business_date())
        if not protocol_valid(sample["expected"], today):
            raise ValueError("期望响应不符合协议: " + sample["id"])
        bound.append(sample)
    if not smoke and Counter(s["group"] for s in samples) != Counter(GROUPS):
        raise ValueError("正式验收必须200条，分组为40/40/40/40/20/20；调试可用--smoke")
    return bound


def summarize(rows):
    count = len(rows)
    singles = [r for r in rows if r["group"] in SINGLES]
    rate = lambda name, data: sum(bool(r[name]) for r in data) / len(data) if data else 0.0
    times = sorted(r["latency_ms"] for r in rows)
    percentile = lambda p: times[max(0, math.ceil(len(times) * p) - 1)] if times else None
    summary = dict(count=count, protocol_rate=rate("protocol", rows), intent_rate=rate("intent", rows),
                   single_exact_rate=rate("exact", singles), p50_ms=percentile(.5), p95_ms=percentile(.95))
    summary["critical_passed"] = all(r["exact"] for r in rows if r.get("critical"))
    known = [r for r in rows if type(r.get("first_pass")) is bool]
    summary["known_first_pass_rate"] = sum(r["first_pass"] for r in known) / len(known) if known else None
    summary["first_pass_unknown"] = count - len(known)
    summary["first_pass_rate"] = summary["known_first_pass_rate"] if len(known) == count else None
    summary["telemetry_missing"] = sum(r.get("first_pass") is None or not r.get("call_ms")
        or r.get("attempts") is None or r.get("server_elapsed_ms") is None for r in rows)
    summary["metric_thresholds_met"] = (count == 200 and summary["protocol_rate"] >= .99
        and summary["intent_rate"] >= .95 and len(singles) == 120
        and summary["single_exact_rate"] >= .90 and summary["critical_passed"])
    # Human review of clarification semantics, frozen labels and timing is still required.
    summary["accepted"] = False
    summary["acceptance_note"] = "自动指标仅供核验；正式验收还需业务标签/单轮置空语义审核、关键边界覆盖及并发/超时检查"
    return summary


def evaluate_one(sample, url, model_name, timeout, today=None):
    start = time.monotonic()
    payload = {"query": sample["query"]}
    if model_name:
        payload["model_name"] = model_name
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    actual, status, attempts = None, None, None
    first_pass, call_ms, server_elapsed = None, None, None
    error = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            attempts = response.headers.get("X-Parser-Attempts")
            raw_first = response.headers.get("X-Parser-First-Pass")
            first_pass = {"true": True, "false": False}.get(raw_first)
            try:
                call_ms = [float(v) for v in response.headers.get("X-Parser-Call-Ms", "").split(",") if v]
                server_elapsed = float(response.headers["X-Parser-Elapsed-Ms"])
            except (ValueError, TypeError, KeyError):
                call_ms, server_elapsed = None, None
            actual = json.load(response)
    except HTTPError as exc:
        status = exc.code
        error = "HTTP_ERROR"
    except (URLError, TimeoutError, OSError):
        error = "TRANSPORT_ERROR"
    except (ValueError, UnicodeError):
        error = "INVALID_RESPONSE_JSON"
    flags = score(sample["expected"], actual, today) if status == 200 else dict(protocol=False, intent=False, exact=False)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    return dict(id=sample["id"], group=sample["group"], critical=sample.get("critical", False),
                expected=sample["expected"], actual=actual, http_status=status, attempts=attempts,
                error=error, latency_ms=elapsed, first_pass=first_pass, call_ms=call_ms,
                server_elapsed_ms=server_elapsed, transport_and_routing_ms=(round(elapsed - server_elapsed, 2) if server_elapsed is not None else None), **flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:7861/chat/treasury_parse")
    parser.add_argument("--model-name")
    parser.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    parser.add_argument("--timeout", type=float, default=95)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="JSON文件，记录模型/后端版本、硬件、量化及解析配置")
    parser.add_argument("--smoke", action="store_true", help="允许小样例集，不作为正式验收")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout必须是有限正数")
    try:
        today = business_date()
        samples = load_samples(args.samples, args.smoke, today)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
        if not args.smoke and (not isinstance(metadata, dict) or not METADATA_KEYS <= set(metadata) or any(metadata[k] in (None, "") for k in METADATA_KEYS)):
            raise ValueError("正式评估需要metadata字段: " + ",".join(sorted(METADATA_KEYS)))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda s: evaluate_one(s, args.url, args.model_name, args.timeout, today), samples))
    report = dict(summary=summarize(rows), rows=rows, metadata=metadata, business_date=today,
                  date_changed_during_run=datetime.now(timezone(timedelta(hours=8))).date().isoformat() != today,
                  samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  concurrency=args.concurrency, smoke=args.smoke)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    passed = all(r["exact"] for r in rows) if args.smoke else report["summary"]["metric_thresholds_met"]
    return 0 if passed and not report["date_changed_during_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
