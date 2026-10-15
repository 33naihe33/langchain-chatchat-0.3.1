# 模型性能基线与 RAG 业务性能摸底

这两部分使用同一脚本，但分别记录和统计。暂无验收标准，结果仅描述指定环境与负载下的表现，不判断通过、硬件合格或最大支持并发。

## 两种模式

| 项目 | model：模型服务基线 | rag：业务端到端摸底 |
| --- | --- | --- |
| 默认入口 | `http://127.0.0.1:8080/v1/chat/completions` | `http://127.0.0.1:7861/chat/kb_chat` |
| 调用内容 | 直接发送 messages，无知识库/检索参数 | 知识库检索、业务提示词、历史及生成 |
| 默认正式轮数 | 每个输入题目、输出预算组合 30 次 | 每个输出预算、并发档位 30 次，总数而非每题 30 次 |
| 并发 | 固定 1；用于控制变量 | 可配置单档或档位列表，各档串行执行 |
| 默认采样 | temperature=0、top_p=1 | temperature=0.7、top_p=0.95，用户确认的摸底值，尚非生产定值 |
| 缓存 | llama.cpp 扩展 cache_prompt=false，避免相同输入绕过 prefill | 沿用业务服务缓存配置 |
| 主要指标 | 原生输入/生成 tokens、prefill/解码计时、decode tokens/s、首正文与总客户端延迟 | 延迟分位数、失败率、闭环请求吞吐量、输出长度、检索证据 |

**行为变化：** 旧脚本不指定 runs 时仅测 5 次、temperature 默认为 0。现行默认模式为 rag，默认正式轮数改为 30，temperature 改为 0.7，并显式发送 top_p=0.95。可通过显式参数覆盖；先做功能验证时请设置较小的 `--runs`。

## 环境固定与记录

比较不同硬件时，固定模型/GGUF/量化、上下文上限、思考模式、线程数/CPU-GPU 分配、服务槽位数及输入。比较并发时，仅改变客户端并发，模型服务槽位和其他业务参数保持一致。并发数无需等于服务槽位数。

用 `--environment-label` 记录模型运行环境；报告自动记录的 platform/Python/requests 信息仅来自客户端，不能证明远端模型硬件。`--url` 和 `--tokenize-url` 可以覆盖默认地址；tokenize 必须对应实际使用的模型，脚本不自动核验远端模型文件。

预热默认每配置 2 次，全部完成后才开始正式负载。首轮预热不是已验证的冷启动；如果要研究冷启动，应另行记录进程/模型加载状态，不能把首轮较慢直接归因于冷启动。

## 题集

`--query` 与 `--queries-file` 二选一。题集是 UTF-8 JSON 数组，每题包含唯一非空 id、非空 query，以及可选字符串 group、history 消息数组。例如：

```json
[
  {"id": "config-intro", "group": "配置介绍", "query": "介绍司库配置中心"},
  {
    "id": "cash-position", "group": "资金头寸",
    "query": "请结合上一轮背景说明资金头寸管理的重点。",
    "history": [
      {"role": "user", "content": "集团需要汇总各成员企业的账户余额、预计收支及融资计划。"},
      {"role": "assistant", "content": "可以统一收集基础数据，再按时间和账户维度分析资金需求。"}
    ]
  }
]
```

这是格式示例，不是完整代表性题集。模型基线应准备不同输入长度的受控 messages，例如约 128、512、2048 tokens 的输入，按服务端 usage.prompt_tokens 确认实际长度，不将字符数当作 token 数。RAG 可先准备 10–20 个真实业务问题，覆盖不同检索上下文、历史和预期回答长度，再按实际业务补充。输入和输出总量必须在模型服务上下文上限内。

model 对题集中的每一题单独执行整个输出预算矩阵；rag 在每个档位内均匀轮换题集。这是均匀题集负载，不自动模拟真实问题频率。每题分组统计在 JSONL 汇总的 by_query 中；当一档 30 次分给 10 题时每题仅约 3 次，不能据此声称分题 P95 可靠。

## 第一部分：模型性能基线

在项目根目录、现有 chatchat-v031 环境中执行：

```bash
LC_ALL=en_US.UTF-8 python -u scripts/benchmark_sse.py \
  --benchmark-mode model \
  --queries-file /path/to/model-inputs.json \
  --max-tokens-list 128 256 512 1024 2048 \
  --runs 30 --warmups 2 \
  --environment-label '客户硬件、GGUF量化、上下文、线程/GPU和服务槽位配置' \
  --output data/benchmarks/model-baseline.jsonl
```

model 不允许并发大于 1，不需要 kb-name。每题/预算单独预热、测量和汇总，正式请求数固定为 runs（默认 30）。`--min-duration` 仅允许为 0，传入正数会报错，避免时长扩展改变基线样本数。temperature=0 用于控制采样，不保证每次输出完全一致。

**接口兼容性：model 默认发送非标准字段 `cache_prompt=false`（llama.cpp 扩展），通用 OpenAI 兼容服务可能返回 400。连接此类服务时请显式指定 `--prompt-cache service-default` 省略该字段。** 报告会记录此设置；也可设为 on 单独研究缓存影响，不能与 off 的 prefill 数据混为一谈。

max_tokens 是思考和正文共享的上限，不能保证生成足量正文。模型流正常结束且有思考或正文事件时计为生成请求成功；仅思考后 finish_reason=length 的轮次也保留原生推理指标，并标记 truncated=true，但正文 TTFT/TPOT 为 null。根据实际生成 tokens、截断情况、首正文等待和总耗时观察合理预算，尚不判断用户是否接受。

服务端原生指标优先读取 llama.cpp timings：predicted_n、predicted_ms、predicted_per_second、prompt_n、prompt_ms。缺少 predicted_n 时可保留 usage.completion_tokens，并记录来源。完整输入长度使用 usage.prompt_tokens；prompt_n 是实际 prefill 处理量，缓存开启时两者可能不同。若服务未提供原生 timings，报告留空，不用客户端正文重分词速率冒充硬件原生解码速度。

## 第二部分：RAG 业务性能摸底

```bash
LC_ALL=en_US.UTF-8 python -u scripts/benchmark_sse.py \
  --benchmark-mode rag \
  --queries-file /path/to/business-queries.json \
  --kb-name test --prompt-name default \
  --temperature 0.7 --top-p 0.95 \
  --top-k 3 --score-threshold 1.0 \
  --max-tokens 2048 \
  --concurrency-list 1 2 4 \
  --runs 30 --warmups 2 \
  --environment-label '业务知识库/提示词版本及模型服务配置' \
  --output data/benchmarks/rag-load.jsonl
```

本例检索配置沿用当前脚本默认值，仍应核对真实业务配置。top_p 已增加到 kb_chat 可选参数中；提供时传给模型，未提供时保持服务原有行为。部署修改后需重启问答 API 才能生效，不需要为脚本修改模型服务槽位。

runs 是每个档位的最低总请求数；min-duration 是正式负载最短秒数，两个条件同时满足才结束，因此实际轮数可能超过 runs。其默认值为 0（仅按轮数执行），需要持续负载时显式设置。各档位分别预热和测量，不混合汇总。

并发采用固定工作线程闭环，完成请求后补充下一个；阶段尾部不保证满并发。它不是固定到达率测试。辅助 tokenize 延后到该配置负载结束后执行，预热和 tokenize 都不进入正式吞吐量分母。容量测试前应确认真实文档命中；报告记录 hit/miss/unknown，但并不评价答案质量，也不从这些标记推断重排效果。

## 统计和报告

每次运行保存 JSONL，并自动生成同名 `.md` 测试说明。指定相同 output 会覆盖原文件，复测请更换文件名。output 不接受 .md 后缀，避免原始数据与报告互相覆盖。每配置负载结束并完成辅助计数后，输出完整逐轮记录和一条 summary；长时间测试期间该配置的最终记录尚未写出。不再把不同输入、预算或并发混成一个总平均值。

报告指标表逐项标注来源：正文 ms/token 是客户端计时与正文重分词计算值；服务端 decode tokens/s 来自模型原生 timings，两者不能视作同一种解码速度。

不能用 rag 延迟减 model 延迟来分离检索开销：默认采样、缓存及实际输入不同，延迟差并非单独的检索耗时。需要分离各阶段时，应增加业务链路计时，或另做保持其他条件一致的控制变量测试。

- 原有 first_token_ttft_ms：首个非空思考或正文事件的客户端延迟；不是关闭思考后的首正文延迟。
- 原有 ttft_ms：首正文客户端延迟，包含此前检索、排队及思考。
- 新增 total_latency_ms：请求到正常 EOF/DONE 的客户端总时间，不含后续辅助计数。
- 原有正文 TPOT：正文首段到结束时间 / max(正文重分词 token 数 - 1, 1)，含尾部开销。只有不足 2 个正文 token 时，该公式仅为边界处理，不能解释为真实 token 间隔。
- P50 保持 lower_median；P90/P95 采用 nearest_rank，按每个指标的有效样本数决定是否输出：至少 30 个才输出 P90，至少 100 个才输出 P95。不输出 P99。这是本脚本的样本报告规则，不是统计可靠性的保证。
- 记录请求失败率、计数可用性、实际输出长度、摘要变化、截断已知/未知数、有效样本数和分题结果；慢请求全部保留。
- model 需要 DONE 或 finish_reason 才确认完成；当前 RAG 接口未返回明确 finish_reason，正常 EOF 为现有判据，其截断状态会显示未知。无法识别 HTTP 层未报错的上游静默提前结束。

当前两模式均以 30 次作为摸底起步规模。波动大时应增加样本，并在不同时间重复测试；新问题、新上下文或新服务配置需要另存结果。没有验收标准时，报告只能表述已测档位的实测延迟、成功率与吞吐量，不能直接给出普遍性能或线上容量结论。

## 开发验证与正式测试

开发时先用 runs=1/2、少量预算做协议与功能验证，报告会提示样本不足。完整模型矩阵与 RAG 30 次/档位可能在 CPU 上运行很久，开发 smoke 的成功不能代替正式摸底。

现有相关测试命令：

```bash
LC_ALL=en_US.UTF-8 python -m pytest tests/test_benchmark_sse.py libs/chatchat-server/tests/api/test_reasoning_content_stream.py -q -o addopts=''
```

用户本轮确认 RAG 也先测 30 次/档位，默认按轮数执行。后续若研究持续负载，可额外指定 --min-duration 300（可能增加轮数）；若需 P95，应把 runs 提高到至少 100，并检查该指标有效样本数是否达到 100。
