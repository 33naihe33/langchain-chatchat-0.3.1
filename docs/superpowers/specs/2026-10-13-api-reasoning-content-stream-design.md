# API reasoning_content 流式透传设计

## 目标

让 `POST /chat/kb_chat` 的 SSE 响应分别透传模型的 `reasoning_content` 与 `content`，解决 MiniCPM5 经 llama.cpp 流式生成时思考字段被丢弃的问题。

## 范围

- 仅修改服务端 API 后端；不修改 WebUI、前端或 llama.cpp。
- SSE 的 `choices[0].delta` 可同时包含 `reasoning_content` 和 `content`，两者不合并。
- 只产生 `content` 的既有模型保持现有响应语义。

## 设计

当前锁定的 `langchain-openai 0.0.6` 会在 `ChatOpenAI._astream` 中将原始 OpenAI delta 转为 `AIMessageChunk`；其转换函数只保留 `function_call` 与 `tool_calls`，会丢弃 `reasoning_content`。因此不能仅替换 callback。

新增本项目专用的 `ChatOpenAI` 子类，覆写 `_astream`：在 raw `choice["delta"]` 转换前读取 `content` 与 `reasoning_content`，将后者写入生成 chunk 的 `message.additional_kwargs`，再按既有 LangChain 回调流程传递。该子类仅由本地 OpenAI 兼容模型工厂选用，不升级依赖，也不改变其它平台模型。

知识库问答使用专用 callback，读取普通 token 及 chunk 元数据，并产出 `{ "content": ..., "reasoning_content": ... }` 片段。`/chat/kb_chat` 为每个片段输出一条 SSE；同一原始 chunk 同时带两字段时，输出同一条 SSE 事件。输出模型保持既有 `content` 与 `tool_calls` 序列化语义，只在 reasoning 非空时额外写入 `delta.reasoning_content`。首个携带 `docs` 的引用事件继续保持 `content: ""`，不包含 `reasoning_content`。

非流式 `/chat/kb_chat` 继续只拼接并返回 `content`，不自动将 reasoning 拼入最终回答；模型只有 reasoning 而没有 content 时，非流式回答可为空，这一限制须在接口文档说明。

## 文档契约

同步更新 `docs/api/kb-chat接口文档.md`：说明 `reasoning_content` 是可选的 SSE delta 字段，与 `content` 独立；客户端应按需单独展示或保存，不能假设每个事件均有任一字段。

## 验证

- 单元测试：输出模型分别序列化 reasoning-only、content-only、双字段事件；docs 首事件不含 reasoning。
- 单元测试：自定义 `_astream` 从 raw delta 保留 reasoning；知识库专用 callback 分别保留 reasoning 与 content。
- 单元测试：content-only 模型不回归，非流式分支不拼 reasoning，同一 chunk 双字段只产生一条事件。
- 端到端：MiniCPM5 调用 `/chat/kb_chat` 时收到 reasoning 事件；普通模型 content 响应不回归。
