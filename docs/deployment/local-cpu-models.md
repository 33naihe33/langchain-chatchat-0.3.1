# 本机 CPU 模型运行

本项目的本地模型文件位于项目根目录：

- `models/bge-m3/`：BAAI/bge-m3。
- `models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf`：官方 Q4_K_M GGUF。
- `runtime/llama.cpp/llama-server`：Apple Silicon 的 llama.cpp 运行时。

## 启动顺序

在项目根目录分别打开两个终端：

```bash
BGE_M3_ENV=bge-m3 bash local-services/start-bge-m3.sh
bash local-services/start-minicpm5.sh
```

两个服务只监听回环地址：BGE-M3 为 `127.0.0.1:8081`，MiniCPM5-2B 为 `127.0.0.1:8080`。

随后用原有 Chatchat 环境启动服务，并固定数据根目录：

```bash
CHATCHAT_ROOT=/Users/caomengdi/chatchat-data conda run -n chatchat-v031 chatchat start -a
```

`/Users/caomengdi/chatchat-data/model_settings.yaml` 已将默认 LLM 配为 `MiniCPM5-2B`、默认 embedding 配为 `bge-m3`。该文件已有备份 `model_settings.yaml.bak-20260916-local-models`。

## 健康检查

```bash
curl http://127.0.0.1:8081/v1/models
curl http://127.0.0.1:8080/v1/models
```

## 运行约束

- BGE 服务在进程内将推理串行化；知识库批量导入请求会排队。
- 单段输入不得超过 8192 tokens；服务返回 422 时应降低 Chatchat 的 `CHUNK_SIZE`。
- 更换 embedding 后必须重建既有知识库的向量索引，不能混用旧向量。
- MiniCPM5 响应可能含 `reasoning_content`；Chatchat 展示时以标准 `content` 字段为准，并用业务样本验收是否需要关闭 reasoning 保留。
