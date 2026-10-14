# SSE TTFT / TPOT 基准测试脚本设计

## 目标

为现有 `POST /chat/kb_chat` SSE 接口提供一个命令行基准脚本，在预热后测量 MiniCPM5-2B 的**用户可见**响应延迟（TTFT 与 TPOT）。

## 范围

- 默认先发 2 次预热请求（不计入指标），再发送 5 次测量请求。`--concurrency` 默认 1（串行），可配置同时执行的轮次上限。
- 首个 `choices[0].delta.content` 非空的事件即首个生成文本。
- 同时测量首 token 延迟：首个非空 `delta.reasoning_content` 或 `delta.content` 事件即首 token，不等待完整思考阶段结束。
- 每次运行记录：TTFT、生成阶段时长、输出文本、输出 token 数、TPOT，并分别用 `chat_success` 与 `tpot_success` 标记两类成功（二者相互独立）。
- 汇总只输出均值、P50、最小值、最大值；**不输出 P95**（样本量为 5 时无统计意义）。
- 使用标准库线程池限制并发，预热阶段全部完成后才开始正式测量；每轮包含聊天和 tokenize，按运行序号由主线程输出，不交错写入 JSONL。
- 客户端并发数与模型服务槽位数分别配置。当前 `/chat/kb_chat` 直接访问模型服务，不经过 Chatchat `/v1` 的 `api_concurrencies` 信号量。脚本不会更改模型服务的 `--parallel`；比较不同并发数时应固定服务槽位、其它参数与知识库状态。

## 计数与时间口径

### 输出 token 计数

- 唯一计数来源：调用正在服务该 GGUF 的 llama.cpp `POST /tokenize`（默认 `http://127.0.0.1:8080/tokenize`），以响应中 `tokens` 数组的长度作为输出 token 数。
- 请求体必须显式传 `{"content": <输出全文>, "add_special": false}`；`add_special=false` 用于避免 BOS 使计数偏大。
- **不使用** HF tokenizer，**不估算**，**不**以 SSE 分块数推算。
- `/tokenize` 不可用或返回非 2xx：仅将 `tpot_success` 置为 false，不输出 token 数与 TPOT，也不降级为估算值；该轮已有的 `chat_success`、TTFT 与生成阶段时长保留，不因计数失败而丢弃。
- 已知限制：`/tokenize` 对拼接后的全文计数，与逐块 delta 数可能不一致；按本设计，TPOT 分母一律以 `/tokenize` 为准。

### 时间口径

- **TTFT**：在发出 HTTP 请求前取 `time.perf_counter()`，到收到首个非空 `delta.content` 的单调时钟差。包含检索、排队、建连与 prefill。
- **首 token TTFT（`first_token_ttft_ms`）**：从同一次请求前的时刻，到首个非空 `delta.reasoning_content` 或 `delta.content` 事件。包含检索、排队、建连与 prefill，以及生成首个片段所需的时间，但不等待后续完整思考过程。这不是关闭思考后的首正文延迟；若需后者，必须关闭模型思考另做一组实验。
- **首正文 TTFT（`ttft_ms`）**：保留原定义，包含正文之前的思考阶段。两项均以客户端 SSE 事件到达/解析时刻为准。没有独立思考事件时，两项相等；同一事件首次同时携带思考与正文时，复用同一时间戳，两项也相等。若思考内容被上游合并进 `content`，脚本无法单独区分该阶段。
- 并发模式仍在工作线程实际发送该轮请求前开始计时；不包含该轮在客户端线程池等待空位的时间，包含请求发出后的服务端排队及资源竞争。每阶段末尾可能少于指定并发数，不保证整阶段一直满负载。
- **生成结束**：以 SSE 响应体正常 EOF 为准。
- **TPOT**：`(EOF - first_content) / max(output_tokens - 1, 1)`，其中 `EOF` 为响应体结束时刻，`first_content` 为首次非空内容时刻。
- 上述指标是客户端正文的延迟口径：`docs` 与纯 `reasoning_content` 事件不触发首正文计时，正文之前的思考耗时会进入 TTFT。TPOT 包含最后一段正文到 EOF 的尾部耗时，且 SSE 块可能包含多个 token，因此不可直接当作模型原生逐 token 解码耗时。输出不超过 1 个 token 时，分母取 1 只是边界处理，没有实际 token 间隔可测。
- 当前 `/chat/kb_chat` 未发送明确的生成完成标记，正常 EOF 是现有完成判据；若上游静默提前结束且 HTTP 层没有报错，脚本无法识别回答是否完整。

## 可复现参数与请求体

- 同一 `query`、`temperature=0`，并固定下表参数；预热、测量与重复运行必须使用同一组值，否则结果不可比。还须保持知识库内容、检索/提示词配置与模型 GGUF 相同，并确保 `/tokenize` 对应实际生成模型；脚本不会自动核验这些外部条件：

| 参数 | 默认值 | 来源与理由 |
| --- | --- | --- |
| `max_tokens` | `512` | 服务端 `MAX_TOKENS` 当前为空（模型最大值），故脚本显式给出上限，可用 CLI 覆盖。思考与正文共享该预算，512 不保证自然结束或出现正文；2026-10-14 直连实测出现 512 个思考事件、无正文且 `finish_reason=length`，真实基准改用 2048。 |
| `top_k` | `3` | 与 `kb_settings.yaml` 的 `VECTOR_SEARCH_TOP_K` 一致，反映当前检索行为。 |
| `score_threshold` | `1.0` | 与 `kb_settings.yaml` 的 `SCORE_THRESHOLD` 一致。 |
| `temperature` | `0` | 减少采样波动，但不能保证输出长度或耗时完全相同。 |

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
- `--concurrency`：最多同时执行的轮次，默认 `1`。`runs` 与 `warmups` 均为阶段总轮数，而非每个并发用户的轮数；实际并发上限是 concurrency 与该阶段总轮数的较小值。例如 `--runs 4 --warmups 2 --concurrency 2` 表示先完成 2 轮预热，再以最多 2 个请求同时执行 4 轮正式测量。
- `--connect-timeout`：建立 TCP 连接的超时（秒），默认 `10`。
- `--read-timeout`：流式读取的单次等待超时（秒），默认 `120`；指相邻两次 `reader.read()` 之间无任何数据的最长等待，而非整轮总时长上限，CPU 生成慢但数据持续到达时不触发。
- `--output`：结果文件路径，默认 `data/benchmarks/<timestamp>.jsonl`。
- `--include-output`：默认关闭；开启后在单轮结果中额外写入 `output_text`。
- 参数校验：`runs`、`max_tokens` 与 `concurrency` 必须大于 0，`warmups` 必须不小于 0，两个 timeout 必须为有限正数；校验失败时退出码为 2，且不发送请求或创建结果文件。

## 统计与结果

- stdout 每轮输出一行 JSON。
- 默认把每条记录写入 `data/benchmarks/` 下的 JSONL 文件，一行一条。
- 单轮记录字段：运行序号、是否预热、`chat_success`、`chat_error`、`tpot_success`、`tpot_error`、TTFT、生成阶段时长、输出 token 数、TPOT、`output_chars`、输出文本 SHA-256 摘要、实际使用的请求参数、时间戳。默认不落盘输出全文；仅在显式指定 `--include-output` 时写入 `output_text`。
- 单轮新增 `first_token_ttft_ms`，原 `ttft_ms` 保持首正文含思考口径。思考内容不拼入正文，不传给 `/tokenize`，也不计入正文输出或 TPOT 分母。
- 单轮及汇总记录 `concurrency`，表示指定的客户端并发上限，便于区分不同负载测试。
- 两个成功标记相互独立，且 `tpot_success=true` 以 `chat_success=true` 为前提：
  - `chat_success=true`：SSE 正常 EOF 且至少有一个非空 `delta.content`，此时 TTFT 与生成阶段时长有效。
  - `tpot_success=true`：`chat_success=true` 且 `/tokenize` 成功，此时输出 token 数与 TPOT 有效。
- 最后追加一条汇总记录：TTFT 的均值、P50、最小值、最大值只基于 `chat_success=true` 的轮次；TPOT 的均值、P50、最小值、最大值只基于 `tpot_success=true` 的轮次。两组统计分别记录其样本数。
- 汇总新增 `first_token_ttft`（毫秒），与 `ttft` 使用相同的 `chat_success=true` 且非预热轮次，分别输出样本数、均值、P50、最值；旧记录缺失新字段或该字段为 null 时不进入新指标统计。仅思考而无正文仍为聊天失败，失败轮次的两项 TTFT 均为 null；计数失败则保留两项 TTFT。

## SSE 解析

- 空行仅用作事件边界，不作为数据事件解析；跳过注释心跳（以 `:` 开头）与非 `data:` 行。
- 以空行分隔事件，逐事件解析 `data:` 后的 JSON。
- 针对现有 `/chat/kb_chat` 校验 `choices[0].delta`，`content` 允许缺省、null 或字符串；空 choices、非对象 JSON、错误类型等协议异常记录为该轮 `chat_error`，继续后续轮次。
- `reasoning_content` 同样只允许缺省、null 或字符串；空字符串不触发首 token 计时，引用事件与注释心跳也不触发计时。

## 错误处理

- 以下情况置 `chat_success=false` 并记录 `chat_error`，写入结果后继续后续运行：非 2xx；SSE JSON 解析失败；响应结束但无任何非空生成内容；连接超时；流读取超时；SSE 中断。
- 仅 `/tokenize` 不可用或返回非 2xx 时置 `tpot_success=false` 并记录 `tpot_error`，不改变 `chat_success`、TTFT 与生成阶段时长（见计数口径）。
- `tpot_success=true` 只在 `chat_success=true` 时成立。
- 聊天和计数 HTTP 响应在成功或失败路径均显式关闭，避免提前退出解析时遗留连接。
- 统计分组：TTFT 只使用 `chat_success=true` 的轮次，TPOT 只使用 `tpot_success=true` 的轮次；两类失败轮次均不进入各自统计。

## 验证

- 单元测试覆盖：SSE 事件解析（含注释心跳、空行、非 `data:` 行）、首个非空内容选取、`/tokenize` 计数（含 `add_special=false`）、TPOT 计算、P50 与均值/最值汇总。
- 单元测试覆盖成功标记分支：`/tokenize` 失败时 `chat_success=true` 且 `tpot_success=false`，该轮 TTFT 仍进入 TTFT 统计；`chat_error` 时两个标记均为 false。
- 单元测试覆盖连接超时与流读取超时分别触发对应的 `chat_error`。
- 确定时钟测试覆盖思考先到、同事件双字段、仅正文、连续思考片段、思考-only 失败、思考字段异常类型、新指标旧记录兼容与汇总预热排除。
- 使用 Barrier/Lock 的同步测试验证确实同时执行两轮且不超过并发上限，预热与正式阶段不重叠；覆盖默认串行、并发非法值、并发失败轮次独立统计及 JSONL 顺序与完整性。
- 端到端：在服务与知识库可用时运行脚本，确认每轮输出 JSON 并落盘 JSONL。
