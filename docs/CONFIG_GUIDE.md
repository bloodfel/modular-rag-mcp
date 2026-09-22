# 配置指南（settings.yaml 逐项说明）

所有可插拔组件都由 `config/settings.yaml` 驱动：**改配置 → 重启服务**，不需要改任何代码。
本文逐节解释每个配置块的含义、可选值和常见搭配。

> 凭证永远不写进 yaml：配置里用 `${ENV_VAR}` 占位符，真实值放仓库根目录的 `.env`
> （已被 gitignore）。服务启动时自动加载 `.env` 并展开占位符，缺失变量会直接报错。

---

## 1. llm —— 主 LLM（查询改写、元数据增强、重写等）

```yaml
llm:
  provider: "glm"            # openai | azure | ollama | deepseek | glm
  model: "glm-4-flash"
  api_key: "${GLM_API_KEY}"
  temperature: 0.0
  max_tokens: 4096
```

| 字段 | 说明 |
|---|---|
| `provider` | 后端实现，由 `LLMFactory` 按该值实例化 |
| `model` | 模型名，跟随 provider 的命名（如 `gpt-4o` / `deepseek-chat` / `glm-4-flash` / `qwen2.5:7b`） |
| `api_key` | 支持 `${VAR}` 占位符；Ollama 不需要 |
| `temperature` | 重排/打分类任务建议 0.0 保证稳定 |

- **Azure OpenAI**：额外配 `azure_endpoint`、`deployment_name`、`api_version`
- **Ollama**：额外配 `base_url`（默认 `http://localhost:11434`），`api_key` 留空
- **DeepSeek**：`provider: "deepseek"`，key 走 `DEEPSEEK_API_KEY`

## 2. embedding —— 向量编码（决定检索的语义质量与成本）

```yaml
embedding:
  provider: "ollama"          # openai | azure | ollama
  model: "nomic-embed-text"
  dimensions: 768
```

- **Ollama 本地**：免费、离线、数据不出域（默认推荐）。先 `ollama pull nomic-embed-text`
- **OpenAI / Azure**：质量好但按 token 计费；`dimensions` 必须与模型输出一致
- ⚠️ 换 embedding 模型后**必须重新灌库**（新旧向量维度/空间不同，不能混存）

## 3. vision_llm —— 图片描述（多模态摄取专用，可独立于主 LLM）

```yaml
vision_llm:
  enabled: true
  provider: "glm"             # 与 llm 同一套 provider 选项
  model: "glm-4v-flash"
  api_key: "${GLM_API_KEY}"
  max_image_size: 2048        # 送入模型前的最大边长（省 token）
```

不处理含图 PDF 时可 `enabled: false`，摄取更快更省。

## 4. rerank —— 精排（质量提升最大的一档，基准见 reports/）

```yaml
rerank:
  enabled: true
  provider: "jev"             # none | cross_encoder | llm | jev
  model: "jev-1.13.0"
  min_score: 1.0              # 仅 jev：评分阈值过滤
  top_k: 5
```

| provider | 是什么 | 什么时候用 |
|---|---|---|
| `none` | 不精排，直接用 RRF 融合排名 | 极致低延迟 / 基线 |
| `cross_encoder` | 本地交叉编码器（如 `BAAI/bge-reranker-base`） | 有本地算力，要免费+最快 |
| `llm` | 聊天模型逐字输出 JSON 打分 | 无本地算力、能接受 ~40s 延迟 |
| `jev` | TypeSafe System One API，typed 评分无文本生成 | 要质量+速度+无解析失败，~$0.17/千次 |

`min_score` 是"质量门"：低于阈值的候选被过滤。基准中 4.0 过于激进（10 候选只剩 0-2 条），
做纯排序对比用 1.0，做证据充分性过滤再调高。

## 5. vector_store / retrieval —— 存储与召回参数

```yaml
vector_store:
  provider: "chroma"           # 当前实现 chroma；接口已预留 qdrant 等
  persist_directory: "./data/db/chroma"

retrieval:
  dense_top_k: 20              # 向量召回条数
  sparse_top_k: 20             # BM25 召回条数
  fusion_top_k: 10             # 融合后保留条数（精排的输入深度）
  rrf_k: 60                    # RRF 平滑常数，越大越尊重原始排名
```

## 6. ingestion —— 摄取流水线

```yaml
ingestion:
  chunk_size: 1000             # 块大小（字符）
  chunk_overlap: 200           # 相邻块重叠，防语义截断
  splitter: "recursive"        # recursive | semantic | fixed_length
  batch_size: 100
  chunk_refiner:
    use_llm: true              # LLM 二次精修块（失败自动回退规则版）
  metadata_enricher:
    use_llm: true              # LLM 生成 title/summary/tags（同上回退）
```

省钱组合：两个 `use_llm` 都设 `false` → 纯规则处理，零 LLM 消耗（质量略降）。

## 7. evaluation / observability —— 评估与追踪

```yaml
evaluation:
  enabled: false
  provider: "custom"           # ragas | deepeval | custom

observability:
  trace_enabled: true
  trace_file: "./logs/traces.jsonl"   # Dashboard 和 Langfuse 导出器都读这里
```

---

## 常见配方

| 目标 | 配置组合 |
|---|---|
| **全本地零成本** | llm: ollama(qwen2.5) + embedding: ollama + rerank: cross_encoder(bge) |
| **免费云端** | llm/vision: glm-4-flash(-v) + embedding: ollama + rerank: jev 或 none |
| **最高质量** | llm: deepseek-chat + rerank: jev + refiner/enricher 开 LLM |
| **最快响应** | rerank: none + refiner/enricher 关 LLM + embedding: ollama |

## 验证配置是否生效

```bash
python scripts/query.py --query "测试"          # 跑通即配置可加载
streamlit run src/observability/dashboard/app.py   # 系统总览页显示当前组件
```
