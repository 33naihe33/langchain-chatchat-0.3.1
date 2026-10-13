# SSE TTFT / TPOT 基准测试脚本设计

## 目标

为现有 `POST /chat/kb_chat` SSE 接口提供一个命令行基准脚本，在预热后测量 MiniCPM5-2B 的**用户可见**响应延迟（TTFT 与 TPOT）。

## 范围

- 先发 2 次预热请求（不计入指标），再串行发送 5 次测量请求。
- 首个 `choices[0].delta.content` 非空的事件即首个生成文本。
- 每次运行记录：TTFT、生成阶段时长、输出文本、输出 token 数、TPOT，并分别用 `chat_success` 与 `tpot_success` 标记两类成功（二者相互独立）。
- 汇总只输出均值、P50、最小值、最大值；**不输出 P95**（样本量为 5 时无统计意义）。
- 串行执行，匹配本地平台的 `api_concurrencies: 1`，避免并发排队污染 TTFT。

## 计数与时间口径

### 输出 token 计数

- 唯一计数来源：调用正在服务该 GGUF 的 llama.cpp `POST /tokenize`（默认 `http://127.0.0.1:8080/tokenize`），以响应中 `tokens` 数组的长度作为输出 token 数。
- 请求体必须显式传 `{"content": <输出全文>, "add_special": false}`；`add_special=false` 用于避免 BOS 使计数偏大。
- **不使用** HF tokenizer，**不估算**，**不**以 SSE 分块数推算。
- `/tokenize` 不可用或返回非 2xx：仅将 `tpot_success` 置为 false，不输出 token 数与 TPOT，也不降级为估算值；该轮已有的 `chat_success`、TTFT 与生成阶段时长保留，不因计数失败而丢弃。
- 已知限制：`/tokenize` 对拼接后的全文计数，与逐块 delta 数可能不一致；按本设计，TPOT 分母一律以 `/tokenize` 为准。

### 时间口径

- **TTFT**：在发出 HTTP 请求前取 `time.perf_counter()`，到收到首个非空 `delta.content` 的单调时钟差。包含检索、排队、建连与 prefill。
- **生成结束**：以 SSE 响应体正常 EOF 为准。
- **TPOT**：`(EOF - first_content) / max(output_tokens - 1, 1)`，其中 `EOF` 为响应体结束时刻，`first_content` 为首次非空内容时刻。

## 可复现参数与请求体

- 同一 `query`、`temperature=0`，并固定下表参数；预热、测量与重复运行必须使用同一组值，否则结果不可比：

| 参数 | 默认值 | 来源与理由 |
| --- | --- | --- |
| `max_tokens` | `512` | 服务端 `MAX_TOKENS` 当前为空（模型最大值），不可复现，故脚本必须显式给出上限；512 足以覆盖固定 query 的自然结束并限制单轮 CPU 耗时，可用 CLI 覆盖。 |
| `top_k` | `3` | 与 `kb_settings.yaml` 的 `VECTOR_SEARCH_TOP_K` 一致，反映当前检索行为。 |
| `score_threshold` | `1.0` | 与 `kb_settings.yaml` 的 `SCORE_THRESHOLD` 一致。 |
| `temperature` | `0` | 去除采样波动，保证输出长度与耗时稳定。 |

- 2 次预热与 5 次测量使用**完全相同**的请求体（`mode=local_kb`、`stream=true`、同一 `model`/`kb_name`）。
- 若 `kb_settings.yaml` 或服务端配置变化，必须同步更新上表默认值，并在结果记录中写入实际使用的参数。

## 接口

命令行参数（含默认值）：

- `--url`：聊天接口地址，默认 `http://127.0.0.1:7861/chat/kb_chat`。
- `--tokenize-url`：llama.cpp 计数端点，默认 `http://127.0.0.1:8080/tokenize`。
- `--query`：问题文本（必填）。
- `--kb-name`：知识库名称（必填）。
- `--model`：模型名，默认 `MiniCPM5-2B`。
- `--max-tokens`：默认 `512`。
- `--top-k`：默认 `3`。
- `--score-threshold`：默认 `1.0`。
- `--temperature`：默认 `0`。
- `--runs`：测量次数，默认 `5`。
- `--warmups`：预热次数，默认 `2`。
- `--connect-timeout`：建立 TCP 连接的超时（秒），默认 `10`。
- `--read-timeout`：流式读取的单次等待超时（秒），默认 `120`；指相邻两次 `reader.read()` 之间无任何数据的最长等待，而非整轮总时长上限，CPU 生成慢但数据持续到达时不触发。
- `--output`：结果文件路径，默认 `data/benchmarks/<timestamp>.jsonl`。
- `--include-output`：默认关闭；开启后在单轮结果中额外写入 `output_text`。

## 统计与结果

- stdout 每轮输出一行 JSON。
- 默认把每条记录写入 `data/benchmarks/` 下的 JSONL 文件，一行一条。
- 单轮记录字段：运行序号、是否预热、`chat_success`、`chat_error`、`tpot_success`、`tpot_error`、TTFT、生成阶段时长、输出 token 数、TPOT、`output_chars`、输出文本 SHA-256 摘要、实际使用的请求参数、时间戳。默认不落盘输出全文；仅在显式指定 `--include-output` 时写入 `output_text`。
- 两个成功标记相互独立，且 `tpot_success=true` 以 `chat_success=true` 为前提：
  - `chat_success=true`：SSE 正常 EOF 且至少有一个非空 `delta.content`，此时 TTFT 与生成阶段时长有效。
  - `tpot_success=true`：`chat_success=true` 且 `/tokenize` 成功，此时输出 token 数与 TPOT 有效。
- 最后追加一条汇总记录：TTFT 的均值、P50、最小值、最大值只基于 `chat_success=true` 的轮次；TPOT 的均值、P50、最小值、最大值只基于 `tpot_success=true` 的轮次。两组统计分别记录其样本数。

## SSE 解析

- 空行仅用作事件边界，不作为数据事件解析；跳过注释心跳（以 `:` 开头）与非 `data:` 行。
- 以空行分隔事件，逐事件解析 `data:` 后的 JSON。

## 错误处理

- 以下情况置 `chat_success=false` 并记录 `chat_error`，写入结果后继续后续运行：非 2xx；SSE JSON 解析失败；响应结束但无任何非空生成内容；连接超时；流读取超时；SSE 中断。
- 仅 `/tokenize` 不可用或返回非 2xx 时置 `tpot_success=false` 并记录 `tpot_error`，不改变 `chat_success`、TTFT 与生成阶段时长（见计数口径）。
- `tpot_success=true` 只在 `chat_success=true` 时成立。
- 统计分组：TTFT 只使用 `chat_success=true` 的轮次，TPOT 只使用 `tpot_success=true` 的轮次；两类失败轮次均不进入各自统计。

## 验证

- 单元测试覆盖：SSE 事件解析（含注释心跳、空行、非 `data:` 行）、首个非空内容选取、`/tokenize` 计数（含 `add_special=false`）、TPOT 计算、P50 与均值/最值汇总。
- 单元测试覆盖成功标记分支：`/tokenize` 失败时 `chat_success=true` 且 `tpot_success=false`，该轮 TTFT 仍进入 TTFT 统计；`chat_error` 时两个标记均为 false。
- 单元测试覆盖连接超时与流读取超时分别触发对应的 `chat_error`。
- 端到端：在服务与知识库可用时运行脚本，确认每轮输出 JSON 并落盘 JSONL。
