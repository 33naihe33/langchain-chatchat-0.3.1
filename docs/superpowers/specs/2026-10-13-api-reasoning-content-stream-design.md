# API reasoning_content 流式透传设计

## 目标

让 `POST /chat/kb_chat` 的 SSE 响应分别透传模型的 `reasoning_content` 与 `content`，解决 MiniCPM5 经 llama.cpp 流式生成时思考字段被丢弃的问题。

## 范围

- 仅修改服务端 API 后端；不修改 WebUI、前端或 llama.cpp。
- 前提：llama.cpp（或其它上游模型服务）已在原始流式 delta 中输出 `reasoning_content`；本设计只透传该字段，不生成、不推断该字段，也不修改 llama.cpp 配置。
- SSE 的 `choices[0].delta` 可同时包含 `reasoning_content` 和 `content`，两者不合并。`reasoning_content` 必须是 `delta` 的显式字段，绝不能作为事件顶层的 `model_extra`（`docs` 仍保持事件顶层字段）。
- 只产生 `content` 的既有模型保持现有响应语义。

## 设计

当前锁定的 `langchain-openai 0.0.6` 会在 `ChatOpenAI._astream` 中将原始 OpenAI delta 转为 `AIMessageChunk`；其转换函数只保留 `function_call` 与 `tool_calls`，会丢弃 `reasoning_content`。因此不能仅替换 callback。

新增本项目专用的 `ChatOpenAI` 子类，覆写 `_astream`：以 `langchain-openai 0.0.6` 的原始实现为模板，只在 raw `choice["delta"]` 转换前读取 `reasoning_content` 并写入生成 chunk 的 `message.additional_kwargs`。必须保留 `chunk.dict()`、空 choices 跳过、`finish_reason` / `logprobs` 的 generation_info、default chunk class 更新，以及对 `on_llm_new_token` 的无条件调用（reasoning-only chunk 的普通 token 为空）。升级 `langchain-openai` 时必须重新核对该内部方法。

`get_ChatOpenAI` 根据 `get_model_info(model_name).get("platform_type") == "custom openai"` 选择该子类；不得以 `local_wrap` 作为条件，因为 `/chat/kb_chat` 不传 `local_wrap`。其它平台类型保持原始 `ChatOpenAI`。

知识库问答使用专用 callback，reasoning-only chunk 即使普通 token 为空也必须产出 `{ "content": "", "reasoning_content": ... }`。仅当 `content` 与 `reasoning_content` 至少一个非空时才产出片段，纯 role/finish 空块必须跳过。callback 覆写 `on_llm_new_token` 将该字典入队，保留父类的 `on_llm_end`/`on_llm_error` 终止信号。父类 `aiter` 在完成信号与已排队片段同时就绪时可能先退出并丢失片段，因此专用 callback 必须覆写 `aiter`：先排空队列，再判断完成信号；等待新片段时同时监听队列与完成信号，退出前再次排空队列，并取消未完成的等待任务。`queue` 使用兼容 Python 3.8 的 `asyncio.Queue` 注解；`aiter` 的返回值类型为 `AsyncIterator[Dict[str, str]]`。`/chat/kb_chat` 为每个片段输出一条 SSE；同一原始 chunk 同时带两字段时，输出同一条 SSE 事件。输出模型保持既有 `content` 与 `tool_calls` 序列化语义，只在 reasoning 非空时额外写入 `delta.reasoning_content`。首个携带 `docs` 的引用事件继续保持 `content: ""`，不包含 `reasoning_content`。

非流式 `/chat/kb_chat` 从 callback 片段显式读取并只拼接 `piece["content"]`，不自动将 reasoning 拼入最终回答；模型只有 reasoning 而没有 content 时，非流式回答可为空。`object == "chat.completion"` 的非流式响应不输出 `reasoning_content`，这一限制须在接口文档说明。

## 文档契约

同步更新 `docs/api/kb-chat接口文档.md`：说明 `reasoning_content` 是可选的 `choices[0].delta` 字段，与 `content` 独立，且不是事件顶层 `model_extra`；客户端应按需单独展示或保存，不能假设每个事件均有任一字段。非流式 `chat.completion` 不输出该字段。

## 验证

- 单元测试：输出模型分别序列化 reasoning-only、content-only、双字段事件；docs 首事件不含 reasoning。
- 单元测试：用真实 `openai.types.chat.ChatCompletionChunk`（其 `ChoiceDelta.model_extra` 含 `reasoning_content`）验证 `.dict()` 后字段仍可读；再以该真实 delta 驱动自定义 `_astream`，验证 reasoning 保留。知识库专用 callback 分别保留 reasoning 与 content。
- 单元测试：纯空 role/finish chunk 不产事件，reasoning-only（`token == ""`）仍产事件；生成结束时队列内的所有片段仍被消费；content-only 模型不回归，非流式分支不拼也不输出 reasoning，同一 chunk 双字段只产生一条事件。
- 单元测试：`custom openai` 平台选择子类，其它平台选择原 `ChatOpenAI`。
- 单元测试：断言 `importlib.metadata.version("langchain-openai") == "0.0.6"`，版本漂移即失败。
- 端到端前置：先直连 llama.cpp 的 `/v1/chat/completions`，确认原始 SSE delta 已含 `reasoning_content`；缺失时应排查/配置上游而非归因 API 层。此前置通过后，MiniCPM5 调用 `/chat/kb_chat` 时收到 reasoning 事件；普通模型 content 响应不回归。

上游前置核验命令（当前 `local-services/start-minicpm5.sh` 未设置 `--reasoning-format`，故不得跳过）：

```bash
curl --no-buffer -X POST 'http://127.0.0.1:8080/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  --data '{"model":"MiniCPM5-2B","messages":[{"role":"user","content":"请简要说明资金头寸管理。"}],"stream":true}'
```

只在原始 SSE 的 `choices[0].delta` 可见 `reasoning_content` 后，才执行 API 透传验收。
