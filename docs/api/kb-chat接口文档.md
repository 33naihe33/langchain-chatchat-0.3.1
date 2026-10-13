# 知识库问答接口（`POST /chat/kb_chat`）

## 1. 用途与地址

该接口以指定知识库检索到的内容作为上下文，调用已配置的大语言模型生成回答。设置 `stream: true` 时，接口以 **Server-Sent Events（SSE）** 分块返回回答，适合定制化 Web 前端逐字展示。

| 项目 | 值 |
| --- | --- |
| 方法 | `POST` |
| 路径 | `/chat/kb_chat` |
| 默认服务端口 | `7861` |
| 请求类型 | `application/json` |
| 流式响应类型 | `text/event-stream` |
| 非流式响应类型 | `application/json` |

生产环境应通过 HTTPS 域名访问，例如 `https://rag.example.com/chat/kb_chat`。不要向浏览器暴露 bge-m3、MiniCPM5-2B 或向量库的内部服务端口。

## 2. 请求参数

### 最小流式请求

```json
{
  "query": "请介绍司库资金头寸管理流程",
  "mode": "local_kb",
  "kb_name": "treasury",
  "stream": true
}
```

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | — | 用户本轮问题。 |
| `mode` | string | 否 | `local_kb` | 知识来源：`local_kb`、`temp_kb` 或 `search_engine`。生产知识库问答使用 `local_kb`。 |
| `kb_name` | string | `mode=local_kb` 时是 | `""` | 知识库名称，必须已创建；例如 `treasury`。 |
| `top_k` | integer | 否 | 服务端 `VECTOR_SEARCH_TOP_K` 配置 | 召回的候选文档数。建议从 `5` 开始调优。 |
| `score_threshold` | number | 否 | 服务端 `SCORE_THRESHOLD` 配置 | 相关度筛选阈值，允许范围 `0`–`2`；值越小表示要求越相关。建议从 `0.5` 开始调优。 |
| `history` | array | 否 | `[]` | 历史消息数组；每项格式为 `{ "role": "user" | "assistant", "content": "..." }`。前端应自行截断过长历史。 |
| `stream` | boolean | 否 | `true` | `true` 返回 SSE；`false` 返回单个 JSON 回答。 |
| `model` | string | 否 | 服务端默认 LLM | 模型名称，必须与 `model_settings.yaml` 中的名称完全一致（当前为 `MiniCPM5-2B`，区分大小写）。 |
| `temperature` | number | 否 | 服务端 `TEMPERATURE` 配置 | 采样温度，范围 `0`–`2`。知识库问答建议 `0.1`–`0.3`。 |
| `max_tokens` | integer 或 `null` | 否 | 服务端 `MAX_TOKENS` 配置 | 单次最大生成 Token 数；`null` 表示使用模型最大值。 |
| `prompt_name` | string | 否 | `default` | `prompt_settings.yaml` 中的 RAG 提示词模板名。 |
| `return_direct` | boolean | 否 | `false` | `true` 时只返回检索来源，不调用大语言模型。 |

> 注意：知识库名称字段是 **`kb_name`**，不是 `knowledge_base_name`。后者来自旧测试代码，不适用于当前路由。

## 3. 流式响应协议

请求 `stream: true` 后，响应由若干 SSE `data:` 事件组成。每个事件的 `data` 是一段 JSON。

### 事件顺序

1. 第一个事件：`choices[0].delta.content` 为空，`docs` 包含检索到的来源。
2. 中间事件：每个事件可包含一小段 `choices[0].delta.content`，也可在模型产生非空推理文本时包含独立的 `choices[0].delta.reasoning_content`；两者可同时出现，也可单独出现。
3. 服务端在生成完成后关闭连接。当前实现**不会**发送 OpenAI 风格的 `[DONE]` 事件；前端应以流读取结束作为完成信号。

`reasoning_content` 只会作为 `choices[0].delta` 内的可选显式字段出现，不能放在事件顶层（顶层 `docs` 仍只表示引用）。客户端应将它与 `content` 分开呈现或保存，不能将它并入最终回答，也不能假定每个事件都有其中任一字段。`stream: false` 返回的 `object: "chat.completion"` 不输出 `reasoning_content`。

示例：

```text
data: {"id":"chat...","object":"chat.completion.chunk","model":"MiniCPM5-2B","choices":[{"delta":{"content":"","tool_calls":[]},"role":"assistant"}],"docs":["出处 [1] ..."]}

data: {"id":"chat...","object":"chat.completion.chunk","model":"MiniCPM5-2B","choices":[{"delta":{"content":"","reasoning_content":"先判断当前资金缺口。","tool_calls":[]},"role":"assistant"}]}

data: {"id":"chat...","object":"chat.completion.chunk","model":"MiniCPM5-2B","choices":[{"delta":{"content":"资金头寸管理","tool_calls":[]},"role":"assistant"}]}
```

来源字段 `docs` 是格式化后的 Markdown/HTML 文本，可能包含文档名、链接和文档片段。定制前端应将其作为引用信息展示；渲染 HTML 前须进行可信内容控制或净化。

## 4. 浏览器调用示例

浏览器原生 `EventSource` 只支持 GET，因此本接口应使用 `fetch` 读取 POST 的响应流。

```js
async function streamKnowledgeBaseAnswer({ query, kbName, history = [] }, onEvent) {
  const response = await fetch('/chat/kb_chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      mode: 'local_kb',
      kb_name: kbName,
      history,
      stream: true,
      model: 'MiniCPM5-2B',
      top_k: 5,
      score_threshold: 0.5,
      temperature: 0.2,
      max_tokens: 1024,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`知识库问答请求失败：HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop();
    for (const event of events) {
      const line = event.split(/\r?\n/).find((item) => item.startsWith('data:'));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(5).trimStart()));
    }
  }
}

let answer = '';
let references = [];
await streamKnowledgeBaseAnswer(
  { query: '请介绍司库资金头寸管理流程', kbName: 'treasury' },
  (event) => {
    if (event.docs) references = event.docs;
    const delta = event.choices?.[0]?.delta ?? {};
    answer += delta.content ?? '';
    // 如需展示推理过程，单独保存/渲染 delta.reasoning_content；不要并入 answer。
    renderAnswer(answer, references);
  },
);
```

## 5. 非流式调用

将 `stream` 改为 `false`，接口返回单个 `chat.completion` JSON 对象；完整答案位于 `choices[0].message.content`。

```json
{
  "query": "请介绍司库资金头寸管理流程",
  "mode": "local_kb",
  "kb_name": "treasury",
  "stream": false
}
```

## 6. 错误与联调处理

| 场景 | HTTP 状态 / 表现 | 前端处理建议 |
| --- | --- | --- |
| 知识库不存在 | `404`，JSON 中含 `code` 与 `msg` | 提示管理员创建或检查 `kb_name`。 |
| 请求字段校验失败 | `422` | 记录响应详情并检查参数类型、`query` 与 `kb_name`。 |
| 模型服务不可用 | `5xx` 或流中断 | 显示“模型服务暂不可用”，允许用户重试；不要将未完成答案写入正式会话记录。 |
| 没有命中相关文档 | 正常 `200`；首事件的 `docs` 含“未找到相关文档”提示 | 前端提示该回答可能来自模型通用知识。 |
| 用户取消生成 | 浏览器中止 `fetch` | 使用 `AbortController` 取消读取；后端会随客户端断开结束本次连接。 |

## 7. Nginx 流式转发

若 API 经 Nginx 对外提供，必须禁用该接口的代理缓冲；否则内容可能在生成完成后才一次性返回。

```nginx
location = /chat/kb_chat {
    proxy_pass http://127.0.0.1:7861;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    add_header X-Accel-Buffering no;
}
```

同时应在网关层配置 HTTPS、前端允许来源（CORS）、鉴权和限流；不要将 ChatChat 的 `7861` 端口直接暴露到公网。

## 8. 验收命令

用以下命令确认首个引用事件与后续文本增量能实时输出；上游模型产生推理文本时，必须还能观察到独立的 `choices[0].delta.reasoning_content` 事件，才算完成本次透传验收。

```bash
curl --no-buffer -X POST 'http://127.0.0.1:7861/chat/kb_chat' \
  -H 'Content-Type: application/json' \
  --data '{
    "query":"请介绍司库资金头寸管理流程",
    "mode":"local_kb",
    "kb_name":"treasury",
    "stream":true,
    "model":"MiniCPM5-2B"
  }'
```
