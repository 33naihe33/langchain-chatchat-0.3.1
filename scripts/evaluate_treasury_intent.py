"""Evaluate raw first-stage classification, before protocol repair.

Only synthetic versioned samples belong here. Reports contain sample IDs and
classification signatures, never questions, parameter values or model text.
This CLI does not start services or execute treasury business operations.
"""
import argparse
import asyncio
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'libs/chatchat-server'))
from langchain_core.messages import HumanMessage, SystemMessage
from chatchat.server.chat import treasury_parse as tp

CRITICAL_TAGS = {'month_detail', 'named_bank_balance', 'unfiltered_detail', 'multi_intent', 'historical_balance'}


def load_samples(path):
    samples = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(samples) < 33:
        raise ValueError('golden set requires at least 33 samples')
    ids, queries = set(), set()
    tags, codes = Counter(), Counter()
    for sample in samples:
        if (not isinstance(sample.get('id'), str) or not sample['id'].strip() or sample['id'] in ids
                or not isinstance(sample.get('query'), str) or not sample['query'].strip() or sample['query'] in queries
                or not isinstance(sample.get('tags'), list) or not all(isinstance(t, str) for t in sample['tags'])):
            raise ValueError('invalid or duplicate sample')
        tp.ParseRequest(query=sample['query'])
        tp.validate_intent(sample['expected'])
        ids.add(sample['id'])
        queries.add(sample['query'])
        tags.update(set(sample['tags']))
        if sample['expected']['status'] == 'ready':
            codes[sample['expected']['trCode']] += 1
    if any(tags[tag] < 3 for tag in CRITICAL_TAGS) or any(codes[code] < 6 for code in tp.NAMES):
        raise ValueError('insufficient critical or single-intent coverage')
    return samples


def signature(value):
    return {key: value[key] for key in ('status', 'trCode')}


async def evaluate_one(sample, model, name, today, timeout):
    started = time.monotonic()
    prompt = tp.render_prompt(today)
    previous, first_correct, final_correct = None, False, False
    errors, call_ms, predictions = [], [], []
    technical_error = False
    attempts = 0
    options = tp._model_options(name)
    context = int(os.getenv('TREASURY_CONTEXT_LENGTH', '8192'))
    output_tokens = int(os.getenv('TREASURY_MAX_TOKENS', '2048'))
    single = float(os.getenv('TREASURY_CALL_TIMEOUT', '45'))
    expected = signature(sample['expected'])
    for attempt in (1, 2):
        messages = [SystemMessage(content=prompt), HumanMessage(content=sample['query'] + tp._retry_suffix(previous, 'intent'))]
        remaining = timeout - (time.monotonic() - started)
        try:
            if remaining <= 0:
                raise asyncio.TimeoutError()
            if options.get('context_check', 'conservative') == 'llama_cpp':
                count = await asyncio.wait_for(tp._llama_token_count(model, messages, remaining), remaining)
            else:
                count = len((prompt + sample['query'] + tp._max_retry_suffix('intent')).encode('utf-8')) + 128
            if count + output_tokens > context:
                errors.append('context_budget')
                technical_error = True
                break
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise asyncio.TimeoutError()
            call_start = time.monotonic()
            attempts += 1
            result = await asyncio.wait_for(model.agenerate([messages], **tp._options(name, 'intent')), min(single, remaining))
            call_ms.append(round((time.monotonic() - call_start) * 1000, 2))
            try:
                generation = result.generations[0][0]
                finish = (generation.generation_info or {}).get('finish_reason')
                content = generation.message.content
            except (AttributeError, IndexError, TypeError):
                raise tp.OutputError('generation')
            if finish != 'stop':
                raise tp.OutputError('finish_reason')
            value = tp.validate_intent(tp.parse_json(content))
            predicted = signature(value)
            predictions.append(predicted)
            # Score raw classification before any protocol repair.
            if attempt == 1:
                first_correct = predicted == expected
            final_correct = predicted == expected
            break
        except tp.OutputError as exc:
            previous = str(exc)
            errors.append(previous)
            if attempt == 2:
                technical_error = True
        except Exception as exc:
            errors.append(tp._classify(exc)[0])
            technical_error = True
            break
    if time.monotonic() - started >= timeout:
        technical_error = True
        errors.append('MODEL_TIMEOUT')
    if technical_error:
        # A technical failure is a failure for acceptance even after a correct
        # raw classification; raw signatures remain available for diagnosis.
        first_correct = final_correct = False
    return dict(id=sample['id'], tags=sample['tags'], expected=expected, predictions=predictions,
                first_correct=first_correct, final_correct=final_correct, technical_error=technical_error,
                errors=errors, attempts=attempts, call_ms=call_ms,
                elapsed_ms=round((time.monotonic() - started) * 1000, 2))


def summarize(rows):
    count = len(rows)
    critical = {tag: [row for row in rows if tag in row['tags']] for tag in CRITICAL_TAGS}
    critical_rates = {tag: sum(row['first_correct'] for row in group) / len(group) if group else 0
                      for tag, group in critical.items()}
    first = sum(row['first_correct'] for row in rows) / count if count else 0
    final = sum(row['final_correct'] for row in rows) / count if count else 0
    technical = sum(row['technical_error'] for row in rows) / count if count else 0
    return dict(count=count, first_intent_rate=first, repaired_intent_rate=final,
                technical_error_rate=technical, critical_first_rates=critical_rates,
                critical_counts={tag: len(group) for tag, group in critical.items()},
                passed=count >= 33 and first >= .95 and all(len(group) >= 3 for group in critical.values())
                and all(rate == 1 for rate in critical_rates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--model-name')
    parser.add_argument('--base-url', help='Optional direct model endpoint; otherwise use configured project model')
    parser.add_argument('--mode', choices=['prompt', 'json_object', 'json_schema'], default='json_object')
    parser.add_argument('--context-check', choices=['conservative', 'llama_cpp'], default='llama_cpp')
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--metadata', type=Path, required=True, help='JSON: model, quantization, backend_version, hardware')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        samples = load_samples(args.samples)
        metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict) or any(not metadata.get(key) for key in ('model', 'quantization', 'backend_version', 'hardware')):
            raise ValueError('metadata requires model, quantization, backend_version, hardware')
        name = args.model_name
        if not name:
            from chatchat.server import utils
            name = utils.get_default_llm()
        settings = json.loads(os.getenv('TREASURY_MODEL_OPTIONS', '{}'))
        settings[name] = dict(mode=args.mode, context_check=args.context_check, disable_thinking=args.disable_thinking)
        os.environ['TREASURY_MODEL_OPTIONS'] = json.dumps(settings)
        if args.base_url:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=name, base_url=args.base_url, api_key=os.getenv('TREASURY_EVAL_API_KEY', 'local'),
                               temperature=0, streaming=False, max_tokens=int(os.getenv('TREASURY_MAX_TOKENS', '2048')),
                               max_retries=0, request_timeout=float(os.getenv('TREASURY_CALL_TIMEOUT', '45')),
                               model_kwargs=tp._options(name))
        else:
            model, name = tp._model(name)
            if model is None:
                raise ValueError('model unavailable')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    today = datetime.now(timezone(timedelta(hours=8))).date()
    async def run():
        rows = []
        for sample in samples:
            row = await evaluate_one(sample, model, name, today, args.timeout)
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('id', 'first_correct', 'final_correct', 'errors', 'elapsed_ms')}), flush=True)
        return rows
    rows = asyncio.run(run())
    changed = datetime.now(timezone(timedelta(hours=8))).date() != today
    report = dict(summary=summarize(rows), rows=rows, metadata=metadata, business_date=today.isoformat(),
                  date_changed_during_run=changed, model_name=name,
                  options=tp._model_options(name), context_length=int(os.getenv('TREASURY_CONTEXT_LENGTH', '8192')),
                  max_tokens=int(os.getenv('TREASURY_MAX_TOKENS', '2048')), timeout=args.timeout,
                  call_timeout=float(os.getenv('TREASURY_CALL_TIMEOUT', '45')),
                  langchain_openai=importlib.metadata.version('langchain-openai'),
                  samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  prompt_sha256=hashlib.sha256(tp.render_prompt(today).encode()).hexdigest(),
                  note='Synthetic regression gate; not a production accuracy guarantee or full parameter acceptance.')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    return 0 if report['summary']['passed'] and not changed else 1


if __name__ == '__main__':
    raise SystemExit(main())
