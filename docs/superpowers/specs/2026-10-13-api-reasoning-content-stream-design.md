# API reasoning_content 流式透传设计

## 目标

让 `POST /chat/kb_chat` 的 SSE 响应分别透传模型的 `reasoning_content` 与 `content`，解决 MiniCPM5 经 llama.cpp 流式生成时思考字段被丢弃的问题。

## 范围

- 仅修改服务端 API 后端；不修改 WebUI、前端或 llama.cpp。
- SSE 的 `choices[0].delta` 可同时包含 `reasoning_content` 和 `content`，两者不合并。
- 只产生 `content` 的既有模型保持现有响应语义。

## 设计

为知识库问答使用的流式回调读取 LangChain `on_llm_new_token` 回调中的 chunk 元数据。普通 token 写入 `content`；模型 chunk 的 `message.additional_kwargs["reasoning_content"]` 写入 `reasoning_content`。输出模型在序列化 chunk 时仅包含非空字段，避免为旧客户端额外发出空字段。

每个 SSE 事件最多携带当前的一段 reasoning 与当前的一段 content。首个携带 `docs` 的引用事件维持现有的空 `content` 结构。非流式响应只保存/返回最终 `content`，不将 reasoning 自动拼接进回答。

## 验证

- 单元测试：输出模型分别序列化 reasoning-only、content-only、双字段事件。
- 单元测试：流式回调从 chunk 元数据提取 reasoning，并保留普通 content token。
- 端到端：MiniCPM5 调用 `/chat/kb_chat` 时收到 reasoning 事件；普通模型 content 响应不回归。
