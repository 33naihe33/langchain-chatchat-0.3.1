"""Stateless treasury intent parsing; no banking transactions are executed here."""
from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import re
import time
import threading
import uuid
from datetime import date, datetime, timezone, timedelta
from typing import Optional

import httpx
import openai
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, field_validator

from chatchat.settings import Settings

from .payload_logging import log_payload
from .treasury_prompts import INTENT_PROMPT, PARAMETER_PROMPTS

log = logging.getLogger(__name__)
NAMES = {'C010100324': '账户信息查询', 'C010100001': '当日明细查询', 'C010100026': '账户实时余额查询'}
UNSUPPORTED_REASON = '未识别到唯一且受支持的查询意图。'
FIELDS = {
    'C010100324': {'acctNo': 40, 'acctNoNme': 255, 'bankCd': None, 'bankNme': 255, 'dirModeFlg': 10, 'ccyCds': None, 'pageSize': None, 'currentPage': None},
    'C010100001': {'startDate': 10, 'endDate': 10, 'bankCd': 4, 'acctNo': 40, 'acctNoNme': 255, 'debitCdtFlg': 1, 'summary': None, 'amtSort': 2, 'pageSize': None, 'currentPage': None},
    'C010100026': {'acctNo': 40, 'acctNoNme': 255, 'bankCd': 4, 'txnMnmcSymbl': 200, 'pageSize': None, 'currentPage': None},
}
ENUMS = {'dirModeFlg': {'0', '1'}, 'debitCdtFlg': {'C', 'D'}, 'amtSort': {'DZ', 'DD', 'CZ', 'CD', '0'}}
OUTPUT_ERROR_CODES = frozenset({'candidate', 'content_type', 'duplicate_key', 'enum',
    'fields', 'finish_reason', 'generation', 'json', 'length', 'schema', 'think', 'type', 'unicode'})
# Protect the process-wide counter across threads and independent event loops.
_inflight = 0
_inflight_lock = threading.Lock()

class OutputError(ValueError):
    def __init__(self, code, position=None):
        super().__init__(code)
        self.position = position

class ParseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, protected_namespaces=())
    query: str
    model_name: Optional[str] = None

    @field_validator('query')
    @classmethod
    def query_valid(cls, value):
        try:
            value.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError('invalid query') from None
        if len(value) > 2000 or not value.strip():
            raise ValueError('invalid query')
        return value.strip()

    @field_validator('model_name')
    @classmethod
    def model_valid(cls, value):
        if value is not None:
            try:
                value.encode('utf-8')
            except UnicodeEncodeError:
                raise ValueError('invalid model') from None
            if not value.strip():
                raise ValueError('invalid model')
        return value.strip() if value else value

def error(code, message, status, headers=None, request_id=None):
    return JSONResponse(status_code=status, content={'error': {'code': code, 'message': message, 'request_id': request_id or uuid.uuid4().hex}}, headers=headers)

class ParserRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def wrapped(request: Request):
            request.state.treasury_request_id = uuid.uuid4().hex
            try:
                payload = await request.json()
            except ValueError:
                payload = {"invalid_json": True}
            log_payload('/chat/treasury_parse', 'request', request.state.treasury_request_id, payload)
            try:
                response = await handler(request)
            except RequestValidationError as exc:
                invalid_model = any(e.get('loc', ())[-1:] == ('model_name',) for e in exc.errors())
                response = error('INVALID_MODEL' if invalid_model else 'INVALID_REQUEST', '模型名称无效' if invalid_model else '请求参数无效', 422, request_id=request.state.treasury_request_id)
            except StarletteHTTPException as exc:
                if exc.status_code == 400:
                    response = error('INVALID_REQUEST', '请求参数无效', 422, request_id=request.state.treasury_request_id)
                else:
                    raise
            except Exception as exc:
                log.error('treasury error request_id=%s classification=INTERNAL_ERROR exception_type=%s', request.state.treasury_request_id, type(exc).__name__)
                response = error('INTERNAL_ERROR', '服务内部错误', 500, request_id=request.state.treasury_request_id)
            log_payload('/chat/treasury_parse', 'response', request.state.treasury_request_id, json.loads(response.body))
            return response
        return wrapped

router = APIRouter(route_class=ParserRoute)

def _validate_unicode(value):
    if isinstance(value, str):
        try:
            value.encode('utf-8')
        except UnicodeEncodeError:
            raise OutputError('unicode') from None
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode(key)
            _validate_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _validate_unicode(item)

def parse_json(content):
    if not isinstance(content, str):
        raise OutputError('content_type')
    s = content.strip()
    if s.startswith('<think>'):
        end = s.find('</think>')
        if end < 0:
            raise OutputError('think')
        s = s[end + len('</think>'):].strip()
    match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', s, re.S | re.I)
    if match:
        s = match.group(1).strip()
    try:
        value = json.loads(s, object_pairs_hook=_unique_object)
        _validate_unicode(value)
        return value
    except OutputError:
        raise
    except json.JSONDecodeError as exc:
        raise OutputError('json', position=exc.pos) from exc
    except (ValueError, TypeError) as exc:
        raise OutputError('json') from exc

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputError('duplicate_key')
        result[key] = value
    return result

def _params(code, params, today):
    """Keep trusted optional filters; uncertain business values become null."""
    if not isinstance(params, dict) or set(params) - set(FIELDS[code]):
        raise OutputError('fields')
    result = dict.fromkeys(FIELDS[code])
    for key, value in params.items():
        if value is None or value == '':
            continue
        if key in ('pageSize', 'currentPage'):
            if type(value) is not int:
                raise OutputError('type')
            result[key] = value if value > 0 else None
            continue
        if code == 'C010100324' and key in ('bankCd', 'ccyCds'):
            if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                raise OutputError('type')
            result[key] = value or None
            continue
        if not isinstance(value, str):
            raise OutputError('type')
        limit = FIELDS[code][key]
        if (limit and len(value) > limit) or (key in ENUMS and value not in ENUMS[key]):
            continue
        result[key] = value
    if code == 'C010100001':
        invalid = False
        for key in ('startDate', 'endDate'):
            value = result[key]
            if value is None:
                invalid |= params.get(key) not in (None, '')
                continue
            try:
                parsed = date.fromisoformat(value)
                invalid |= parsed.isoformat() != value
            except ValueError:
                invalid = True
        if invalid or (result['startDate'] and result['endDate'] and result['startDate'] > result['endDate']):
            result['startDate'] = result['endDate'] = None
    return result, None


def validate_result(value, today):
    _validate_unicode(value)
    if not isinstance(value, dict) or set(value) != {'status', 'trCode', 'params', 'candidates', 'clarification'}:
        raise OutputError('schema')
    status, code, params, candidates, clarification = (value[k] for k in ('status', 'trCode', 'params', 'candidates', 'clarification'))
    if not isinstance(status, str) or status not in ('ready', 'unsupported') or not isinstance(candidates, list) or candidates:
        raise OutputError('schema')
    if status == 'unsupported':
        if code is not None or params != {} or not isinstance(clarification, str) or not clarification.strip():
            raise OutputError('schema')
        return value
    if not isinstance(code, str) or code not in NAMES or clarification is not None:
        raise OutputError('schema')
    normalized, _ = _params(code, params, today)
    return {'status': 'ready', 'trCode': code, 'params': normalized, 'candidates': [], 'clarification': None}

def validate_intent(value):
    _validate_unicode(value)
    if not isinstance(value, dict) or set(value) != {'status', 'trCode', 'clarification'}:
        raise OutputError('schema')
    status, code, reason = (value[k] for k in ('status', 'trCode', 'clarification'))
    if status == 'ready':
        if not isinstance(code, str) or code not in NAMES or reason is not None:
            raise OutputError('schema')
    elif status == 'unsupported':
        if code is not None or not isinstance(reason, str) or not re.search(r'[\u4e00-\u9fff]', reason):
            raise OutputError('schema')
    else:
        raise OutputError('schema')
    return value


def validate_parameters(value, code, today):
    _validate_unicode(value)
    if not isinstance(value, dict) or set(value) != {'status', 'params', 'clarification'}:
        raise OutputError('schema')
    if value['status'] != 'ready' or value['clarification'] is not None:
        raise OutputError('schema')
    normalized, _ = _params(code, value['params'], today)
    return {**value, 'params': normalized}


def stage_schema(stage='intent', code=None):
    properties = {'status': {'type': 'string', 'enum': ['ready', 'unsupported'] if stage == 'intent' else ['ready']},
                  'clarification': {'type': ['string', 'null']} if stage == 'intent' else {'type': 'null'}}
    if stage == 'intent':
        properties['trCode'] = {'type': ['string', 'null'], 'enum': [*NAMES, None]}
        branches = [
            {'properties': {'status': {'const': 'ready'}, 'trCode': {'type': 'string'}, 'clarification': {'type': 'null'}}},
            {'properties': {'status': {'const': 'unsupported'}, 'trCode': {'type': 'null'}, 'clarification': {'type': 'string', 'minLength': 1, 'pattern': r'[\u4e00-\u9fff]'}}},
        ]
    elif stage == 'params' and code in FIELDS:
        fields = {}
        for key in FIELDS[code]:
            field = {'type': ['string', 'null']}
            if key in ('pageSize', 'currentPage'):
                field = {'type': ['integer', 'null']}
            elif code == 'C010100324' and key in ('bankCd', 'ccyCds'):
                field = {'type': ['array', 'null'], 'items': {'type': 'string', 'minLength': 1}}
            fields[key] = field
        properties['params'] = {'type': 'object', 'properties': fields, 'additionalProperties': False}
        branches = [{'properties': {'status': {'const': 'ready'}, 'clarification': {'type': 'null'}}}]
    else:
        raise ValueError('invalid stage')
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False, 'anyOf': branches}


def render_prompt(today, stage='intent', code=None):
    template = INTENT_PROMPT if stage == 'intent' else PARAMETER_PROMPTS[code]
    prompt = template.replace('{today}', today.isoformat()).replace('{year}', str(today.year))
    if stage == 'params' and code == 'C010100001':
        # Query-independent calendar context, never a rewrite of model output.
        last_month_end = today.replace(day=1) - timedelta(days=1)
        prompt = prompt.replace('{last_month_start}', last_month_end.replace(day=1).isoformat())
        prompt = prompt.replace('{last_month_end}', last_month_end.isoformat())
    def example(match):
        value = json.loads(match.group(1))
        normalized = validate_intent(value) if stage == 'intent' else validate_parameters(value, code, today)
        return json.dumps(normalized, ensure_ascii=False, separators=(',', ':'))
    return re.sub(r'(?<= => )(\{[^\n]*\})(?=。)', example, prompt)

def _model_options(name):
    override = os.getenv('TREASURY_MODEL_OPTIONS')
    configured = json.loads(override) if override else Settings.model_settings.TREASURY_MODEL_OPTIONS
    options = configured.get(name, {})
    if options.get('context_check', 'conservative') not in ('conservative', 'llama_cpp'):
        raise ValueError('invalid context check mode')
    return options

def _options(name, stage='intent', code=None):
    options = _model_options(name)
    mode = options.get('mode', 'prompt')
    kwargs = {}
    if mode == 'json_object':
        kwargs['response_format'] = {'type': 'json_object'}
    elif mode == 'json_schema':
        kwargs['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'treasury_' + stage, 'schema': stage_schema(stage, code)}}
    elif mode != 'prompt':
        raise ValueError('invalid mode')
    if options.get('disable_thinking', False):
        kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    return kwargs

def _model(name):
    from chatchat.server import utils
    names = utils.get_config_models(model_type='llm')
    if name is not None and name not in names:
        return None, 'INVALID_MODEL'
    if not names:
        return None, 'MODEL_UNAVAILABLE'
    name = name or utils.get_default_llm()
    kwargs = _options(name)
    max_tokens = int(os.getenv('TREASURY_MAX_TOKENS', '2048'))
    if max_tokens <= 0:
        raise ValueError('invalid token budget')
    model = utils.get_ChatOpenAI(model_name=name, streaming=False, temperature=0, max_tokens=max_tokens, request_timeout=float(os.getenv('TREASURY_CALL_TIMEOUT', '45')), max_retries=0, local_wrap=False, model_kwargs=kwargs)
    return model, name

def _retry_suffix(code, stage='intent'):
    if code is None:
        return ''
    repair = '每个键只能出现一次；在原位置更新参数值，禁止追加同名键。' if code == 'duplicate_key' else ''
    if stage == 'intent':
        rules = ('只输出status、trCode、clarification三个键，不输出参数。只支持单轮，不追问。'
                 'ready必须有唯一编号且clarification=null；无法确定唯一意图或不支持时为unsupported、trCode=null，原因用中文陈述句。'
                 '银行、账号、日期等筛选条件不影响明确业务分类。明确月份流水只有一个流水意图，应分类为ready和流水编号。')
    else:
        rules = ('只输出status、params、clarification三个键，禁止trCode和candidates。status必须ready，clarification必须null。'
                 '只允许当前接口字段，返回全部字段；缺失、无效或无法确定的参数为null，不补日期、代码或分页，不追问。')
    return f'\n上次输出错误代码：{code}。{repair}{rules}请重新按当前阶段格式输出。'


def _max_retry_suffix(stage='intent'):
    return max((_retry_suffix(code, stage) for code in OUTPUT_ERROR_CODES),
               key=lambda suffix: len(suffix.encode('utf-8')))

async def _llama_token_count(model, messages, timeout):
    base = str(model.openai_api_base or '').rstrip('/')
    if not base.startswith(('http://', 'https://')) or not base.endswith('/v1'):
        raise ValueError('invalid model base URL')
    base = base[:-3]
    body = {'messages': [{'role': 'system' if isinstance(m, SystemMessage) else 'user', 'content': m.content} for m in messages], 'add_generation_prompt': True}
    template_kwargs = (model.model_kwargs.get('extra_body') or {}).get('chat_template_kwargs')
    if template_kwargs is not None:
        if not isinstance(template_kwargs, dict):
            raise ValueError('invalid template options')
        body['chat_template_kwargs'] = template_kwargs
    api_key = getattr(model, 'openai_api_key', None)
    if hasattr(api_key, 'get_secret_value'):
        api_key = api_key.get_secret_value()
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError('invalid model credentials')
    headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=min(timeout, 5.0), trust_env=False, headers=headers) as client:
        rendered = await client.post(base + '/apply-template', json=body)
        rendered.raise_for_status()
        prompt = rendered.json()['prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('invalid rendered prompt')
        tokenized = await client.post(base + '/tokenize', json={'content': prompt, 'add_special': False, 'parse_special': True})
        tokenized.raise_for_status()
        tokens = tokenized.json()['tokens']
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('invalid tokens')
        return len(tokens)

def _classify(exc):
    if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError)):
        return 'MODEL_TIMEOUT', '模型解析超时', 504
    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)) or isinstance(exc, openai.APIStatusError) and exc.status_code in (503, 504):
        return 'MODEL_UNAVAILABLE', '模型服务不可用', 503
    if isinstance(exc, openai.APIError):
        return 'MODEL_UPSTREAM_ERROR', '模型服务请求失败', 502
    return 'INTERNAL_ERROR', '服务内部错误', 500

@router.post('/treasury_parse')
async def treasury_parse(body: ParseRequest, request: Request = None):
    global _inflight
    request_id = request.state.treasury_request_id if request is not None else uuid.uuid4().hex
    def fail(code, message, status, headers=None):
        log.warning('treasury error request_id=%s classification=%s', request_id, code)
        return error(code, message, status, headers, request_id=request_id)
    started = time.monotonic()
    with _inflight_lock:
        if _inflight >= 2:
            return fail('PARSER_BUSY', '解析服务繁忙', 503, {'Retry-After': '1'})
        _inflight += 1
    try:
        try:
            model, name = _model(body.model_name)
        except Exception as exc:
            log.warning('treasury initialization request_id=%s exception_type=%s', request_id, type(exc).__name__)
            return fail('MODEL_UNAVAILABLE', '模型服务不可用', 503)
        if model is None:
            if name == 'INVALID_MODEL':
                return fail('INVALID_MODEL', '模型名称无效', 422)
            return fail('MODEL_UNAVAILABLE', '模型服务不可用', 503)
        today = datetime.now(timezone(timedelta(hours=8))).date()
        try:
            context_length = int(os.getenv('TREASURY_CONTEXT_LENGTH', '8192'))
            max_tokens = int(os.getenv('TREASURY_MAX_TOKENS', '2048'))
            if context_length <= 0 or max_tokens <= 0:
                raise ValueError('invalid context')
        except ValueError:
            return fail('MODEL_UNAVAILABLE', '模型服务配置无效', 503)
        try:
            context_mode = _model_options(name).get('context_check', 'conservative')
            total = float(os.getenv('TREASURY_TOTAL_TIMEOUT', '90'))
            single = float(os.getenv('TREASURY_CALL_TIMEOUT', '45'))
            if not math.isfinite(total) or total <= 0 or not math.isfinite(single) or single <= 0:
                raise ValueError('invalid timeout')
            # Validate every template before consuming a generation call.
            prompts = {'intent': render_prompt(today)}
            prompts.update({code: render_prompt(today, 'params', code) for code in NAMES})
        except (ValueError, TypeError, AttributeError, KeyError):
            return fail('MODEL_UNAVAILABLE', '模型服务配置无效', 503)
        call_ms = []
        first_pass = True

        async def run_stage(stage, code=None):
            nonlocal first_pass
            prompt = prompts['intent' if stage == 'intent' else code]
            previous = None
            # Protocol errors share the single repair opportunity for this stage.
            for attempt in (1, 2):
                remaining = total - (time.monotonic() - started)
                if remaining <= 0:
                    return fail('MODEL_TIMEOUT', '模型解析超时', 504)
                messages = [SystemMessage(content=prompt), HumanMessage(content=body.query + _retry_suffix(previous, stage))]
                if context_mode == 'conservative':
                    bound = len((prompt + body.query + _max_retry_suffix(stage)).encode('utf-8')) + 128
                    if bound + max_tokens > context_length:
                        return fail('INVALID_REQUEST', '输入超出模型上下文预算，请缩短问题', 422)
                else:
                    try:
                        count = await asyncio.wait_for(_llama_token_count(model, messages, remaining), timeout=remaining)
                    except asyncio.TimeoutError:
                        return fail('MODEL_TIMEOUT', '模型解析超时', 504)
                    except Exception as exc:
                        log.warning('treasury tokenizer request_id=%s stage=%s trCode=%s exception_type=%s', request_id, stage, code, type(exc).__name__)
                        return fail('MODEL_UNAVAILABLE', '模型服务不可用', 503)
                    if count + max_tokens > context_length:
                        return fail('INVALID_REQUEST', '输入超出模型上下文预算，请缩短问题', 422)
                remaining = total - (time.monotonic() - started)
                if remaining <= 0:
                    return fail('MODEL_TIMEOUT', '模型解析超时', 504)
                content = None
                finish = None
                try:
                    call_started = time.monotonic()
                    # Per-call options override the initial intent schema without
                    # mutating the model shared by the two stages or other requests.
                    result = await asyncio.wait_for(model.agenerate([messages], **_options(name, stage, code)), timeout=min(single, remaining))
                    call_ms.append(round((time.monotonic() - call_started) * 1000, 2))
                    try:
                        generation = result.generations[0][0]
                        finish = (generation.generation_info or {}).get('finish_reason')
                        content = generation.message.content
                    except (AttributeError, IndexError, TypeError):
                        raise OutputError('generation')
                    if finish != 'stop':
                        raise OutputError('finish_reason')
                    value = parse_json(content)
                    if stage == 'intent':
                        return validate_intent(value)
                    value = validate_parameters(value, code, today)
                    return validate_result({**value, 'trCode': code, 'candidates': []}, today)
                except OutputError as exc:
                    first_pass = False
                    previous = str(exc)
                    safe_finish = finish if finish in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call', None) else 'other'
                    log.warning('treasury output invalid request_id=%s model=%s stage=%s trCode=%s attempt=%s error=%s length=%s json_offset=%s finish_reason=%s elapsed=%.3f', request_id, name, stage, code, attempt, previous, len(content) if isinstance(content, str) else None, exc.position, safe_finish, time.monotonic() - started)
                except Exception as exc:
                    classification = _classify(exc)
                    log.warning('treasury model request_id=%s stage=%s trCode=%s classification=%s exception_type=%s', request_id, stage, code, classification[0], type(exc).__name__)
                    return fail(*classification)
            return fail('INVALID_MODEL_OUTPUT', '模型输出格式无效', 502)

        selected = await run_stage('intent')
        if isinstance(selected, JSONResponse):
            return selected
        if selected['status'] == 'ready':
            output = await run_stage('params', selected['trCode'])
            if isinstance(output, JSONResponse):
                return output
        else:
            # Do not expose a model-generated follow-up even if it arrived in
            # the unsupported reason field of an otherwise valid response.
            output = validate_result({**selected, 'params': {}, 'candidates': [], 'clarification': UNSUPPORTED_REASON}, today)
        if total - (time.monotonic() - started) <= 0:
            return fail('MODEL_TIMEOUT', '模型解析超时', 504)
        return JSONResponse(content=output, headers={
            'X-Parser-Attempts': str(len(call_ms)),
            'X-Parser-First-Pass': str(first_pass).lower(),
            'X-Parser-Call-Ms': ','.join(map(str, call_ms)),
            'X-Parser-Elapsed-Ms': str(round((time.monotonic() - started) * 1000, 2)),
        })
    finally:
        with _inflight_lock:
            _inflight -= 1
