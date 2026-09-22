# modular-rag-mcp

![quality-gate](https://github.com/bloodfel/modular-rag-mcp/actions/workflows/quality-gate.yml/badge.svg)

基于 MCP（Model Context Protocol）的模块化 RAG 检索服务：将私有文档构建为可检索知识库，供 Claude Desktop / Cursor / ZCode 等任意 MCP 客户端通过工具调用完成检索与引用回答。本地运行，数据不出机器。

## 功能

- **混合检索**：BM25（jieba 分词）+ dense embedding 双路召回，RRF（k=60）融合
- **可插拔架构**：LLM / Embedding / Reranker / VectorStore / Splitter / Evaluator 均为抽象接口 + 工厂 + `settings.yaml` 配置驱动，切换后端零代码修改
- **重排**：`none` / cross-encoder（本地 BGE）/ Jev（类型化相关性评分 API）/ LLM 四种实现，附 BEIR 基准
- **多模态**：PDF 图片提取（PyMuPDF）→ Vision LLM 生成描述 → 描述入索引，"搜文字、出图片"
- **可观测**：Ingestion / Query 双链路全阶段 trace（JSONL），自动经 OTLP 上报 Langfuse；按阶段归因延迟、token、成本；降级路径打标可统计
- **评测**：离线金标（hit_rate / MRR / nDCG@10）+ Ragas LLM-as-Judge 四指标（faithfulness / answer_relevancy / context_precision / context_recall），评测分数自动回写 Langfuse trace
- **CI 质量门禁**：每次变更自动重跑检索评测，指标低于阈值阻断合并
- **MCP 双传输**：stdio（本地子进程）+ Streamable HTTP（stateless，`--http`）

## 基准与评测数据

### 重排选型（BEIR test split，nDCG@10 ×100，越高越好）

| dataset/retriever | none  | bge   | jev      | llm   |
| ----------------- | ----- | ----- | -------- | ----- |
| scifact/bm25      | 64.99 | 65.27 | **73.29** | 66.16 |
| scifact/dense     | 67.90 | 69.47 | **70.61** | 68.16 |
| scifact/hybrid    | 67.40 | 67.19 | **72.02** | 69.68 |
| nfcorpus/bm25     | 35.98 | 36.07 | **37.14** | 35.92 |
| nfcorpus/dense    | 36.94 | 38.07 | **39.69** | 38.00 |
| nfcorpus/hybrid   | 38.22 | 37.55 | **38.23** | 38.51 |

Jev 在 6 格中取 5 个第一（scifact/bm25 +8.3，相对 +12.8%），成本 ~$0.17/千次查询，p95 < 5.1s。
完整四维报告（质量 / 延迟 / 成本 / 可靠性）见 [`reports/BENCHMARK_REPORT.md`](reports/BENCHMARK_REPORT.md)。

### 评测基线（语料为本仓库文档，231 chunks；hybrid 检索，rerank 关闭，top-10）

| 指标 | 数值 |
| --- | --- |
| hit@10 | 1.0000 |
| MRR | 0.5061 |
| nDCG@10 | 0.5405 |
| Ragas faithfulness（60 例标注集） | 0.9611 |
| Ragas answer_relevancy | 0.7602 |
| Ragas context_precision | 0.7800 |
| Ragas context_recall | 0.9167 |

全部数字的来源、复现命令与已知缺口见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 快速开始

前置条件：Python ≥ 3.10；[Ollama](https://ollama.com)（`ollama pull nomic-embed-text`，本地 embedding）；GLM API key（免费档，可选 Jev key 启用重排）。

```bash
git clone https://github.com/bloodfel/modular-rag-mcp.git
cd modular-rag-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # 填入 GLM_API_KEY；embedding 默认走本地 Ollama

python scripts/ingest.py data/sample_docs                    # 离线摄取
python scripts/query.py --query "Modular RAG 是什么？"       # 命令行检索
```

### 接入 MCP 客户端

个人本地（stdio）——在 ZCode / Claude Desktop / Cursor 的 MCP 配置中加入（替换 `/绝对路径` 为克隆路径）：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "/绝对路径/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/绝对路径/modular-rag-mcp"
    }
  }
}
```

团队 / 远程（Streamable HTTP，stateless）：

```bash
python -m src.mcp_server.server --http --host 0.0.0.0 --port 8000
# MCP 端点: http://<host>:8000/mcp
```

可用工具：`query_knowledge_hub`（混合检索 + 重排 + 引用）、`ingest_document`（文本 / Markdown 上传）、`list_collections`、`get_document_summary`。公网部署请置于反向代理后并自行叠加鉴权（当前版本未内置）。

## 评测与 CI 门禁

```bash
# 构建确定性评测语料（无 LLM 调用，可离线复现）
for f in docs/CONFIG_GUIDE.md README.md AGENTS.md docs/specs DEV_SPEC.md; do
  python scripts/ingest.py --config config/settings.eval.yaml \
    --path "$f" --collection project_docs_ci --extensions .md --force
done

# 离线检索评测 + 质量门禁（阈值：hit_rate ≥ 1.0 / MRR ≥ 0.45 / nDCG ≥ 0.48）
python scripts/quality_gate.py --collection project_docs_ci

# Ragas 四指标评测，分数回写 Langfuse trace
python scripts/evaluate_ragas.py --test-set tests/fixtures/golden_ragas_set.json
```

金标以 `(仓库相对路径, chunk_index)` 寻址，与机器路径和入库配置解耦（详见 [`docs/RESULTS.md`](docs/RESULTS.md) 的寻址方案说明）。

## 架构

```
AI 助手（Copilot / Claude / ZCode 等 MCP Client）
        │  JSON-RPC（stdio 或 Streamable HTTP）
        ▼
MCP Server 层：query_knowledge_hub / ingest_document / list_collections / ...
        ▼
检索流水线：QueryProcessor → Dense + BM25 并行召回 → RRF 融合 → Rerank → 带引用响应
        ▼
可插拔层（Factory + settings.yaml）
  LLM: openai / azure / ollama / deepseek / glm      Embedding: openai / azure / ollama / glm
  Reranker: none / cross_encoder / jev / llm         VectorStore: chroma        Splitter: recursive
```

| 目录 | 内容 |
| --- | --- |
| `src/libs/` | 可插拔 provider（LLM / Embedding / Reranker / VectorStore / Loader / Evaluator） |
| `src/core/` | 摄取与检索流水线、查询编排、trace、配置 |
| `src/mcp_server/` | MCP Server 与工具层（stdio / Streamable HTTP） |
| `src/observability/` | trace、Langfuse OTLP 导出、指标库、评估器、Dashboard |
| `scripts/` | ingest / query / quality_gate / evaluate_ragas / 基准跑批 |
| `reports/` | 基准与评测报告 |
| `docs/` | 配置指南、产出数据总账、子功能设计文档 |

## 换后端 = 改一行配置

```yaml
# config/settings.yaml —— 把重排切换为 Jev：
rerank:
  enabled: true
  provider: "jev"          # none | cross_encoder | llm | jev
  model: "jev-1.13.0"
  min_score: 1.0
```

凭证一律走 `.env` + `${ENV_VAR}` 占位符，加载时 fail-fast 校验，不进代码库。

## 测试与复现

```bash
pytest -q                                                    # 1300+ 测试（不访问网络）
python scripts/prepare_beir.py --datasets scifact nfcorpus   # BEIR 语料灌库
python scripts/benchmark_rerank.py --datasets scifact nfcorpus --num-queries 50
```

## 文档

- [`DEV_SPEC.md`](DEV_SPEC.md) — 架构设计与阶段排期
- [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md) — settings.yaml 逐项配置说明
- [`docs/RESULTS.md`](docs/RESULTS.md) — 产出数据总账（可观测 / 检索基线 / Ragas / 工程实验）
- [`reports/BENCHMARK_REPORT.md`](reports/BENCHMARK_REPORT.md) — 重排基准详细分析

## Roadmap

- [ ] fiqa 数据集（5.7 万篇）全量对比
- [ ] BGE + Jev 双信号融合重排
- [ ] HTTP 传输叠加鉴权（Bearer token）
- [ ] `ingest_document` 支持 PDF / 图片上传
- [ ] 发布 PyPI 包
- [ ] BM25 索引按文档清理（消除路径变更后的残留词项）
- [ ] MCP 查询链路实时上报

---

基于 Python 3.10+ / MCP 官方 SDK / Chroma / FastEmbed-free 本地 embedding（Ollama）/ Streamlit 构建。
