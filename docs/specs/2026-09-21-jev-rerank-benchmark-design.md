# Jev Reranker 接入与 BEIR 重排基准测试 — 设计文档

日期：2026-09-21 · 分支：dev · 状态：待评审

## 1. 背景与目标

TypeSafe 的 Jev（System One 系列）是输出类型化判断（Choice/Noul/Score）+ 校准概率、
不产文本的模型。本项目（Modular RAG MCP Server）要在检索链路中接入 Jev 作为重排
（rerank）后端，并在公开 BEIR 数据集（scifact、nfcorpus）上量化其相对收益。

**产出目标：一张对标 TypeSafe 官方评测格式的对比表**

```
            Original   BGE    Jev   GLM-LLM
scifact/bm25   70.35  73.07  76.57    …
scifact/dense  69.12  73.87  79.46    …
scifact/hybrid  …      …      …       …
nfcorpus/…      …      …      …       …
```

主指标 nDCG@10×100，辅指标 MRR@10、Hit Rate@10、rerank 延迟（p50/p95）、
估算成本（$/千查询）、失败回退率。

## 2. 非目标（本期不做）

- fiqa 数据集（5.7 万段落，灌库耗时长，第二期后台跑）
- Fusion 列（BGE+Jev 信号叠加，回答的是另一问题）
- Langfuse 接入（现有 Streamlit dashboard + traces.jsonl 够用；后续可做 exporter）
- Ragas（它评生成质量，本基准评检索质量，不相干）
- MCP tools / Dashboard 页面结构变更（rerank 结果经现有 trace 通道自动呈现）

## 3. 总体架构

```
BEIR 数据集(HF)                settings.yaml
   │ corpus/queries/qrels         │ rerank.provider: jev / cross_encoder / llm / none
   ▼                              ▼
scripts/prepare_beir.py    RerankerFactory ──┬─ NoneReranker（现状）
   │ 灌库：chroma collection +               ├─ CrossEncoderReranker（已有，验证并配置 BGE 模型）
   │     BM25 索引 + qrels 落盘              ├─ LLMReranker（已有）
   ▼                                        └─ JevReranker（新建）
scripts/benchmark_rerank.py
   │ 对每个 dataset × retriever(bm25/dense/hybrid) × reranker：
   │   召回 top-10 → 重排 → nDCG@10 / MRR@10 / Hit@10
   │   记录延迟、token 用量、失败回退
   ▼
reports/rerank_benchmark_<date>.md (+ .json 原始数据)
```

## 4. 组件设计

### 4.1 指标模块 `src/observability/evaluation/metrics.py`（新建）

纯函数，无 I/O，全部带单元测试：

- `ndcg_at_k(ranked_ids, qrels, k) -> float`：二分增益 `2^rel - 1`，IDCG 按该 query
  qrels 的理想排序计算；qrels 为空时返回 0.0
- `mrr_at_k(ranked_ids, qrels, k) -> float`：第一个 rel>0 的倒数排名
- `hit_at_k(ranked_ids, qrels, k) -> float`：top-k 内存在 rel>0 即 1.0

### 4.2 JevReranker `src/libs/reranker/jev_reranker.py`（新建）

- 继承 `BaseReranker`，结构对照 `llm_reranker.py`
- 调用方式：**httpx 直连 REST** `POST https://api.typesafe.ai/v1/systemone`
  （httpx 已是既有传递依赖；不引入官方 SDK，减少依赖面）
  - `state` = query 文本；`questions` = 每个候选一条 Score 问题（有序档位 1-10
    相关性）；单次调用并行评估全部候选（Jev 要求单次 ≤10 问题，正好匹配
    rerank depth=10）
  - 响应：每条的档位分数 + 概率分布 + confidence
- 过滤与排序：分数低于 `min_score`（配置，默认 4）的候选**剔除**（不参与最终
  排序），其余按分数降序；通过者写入 `rerank_score`
- 失败语义：网络/超时/HTTP 错误抛 `JevRerankError`，由 CoreReranker 既有 fallback
  兜底——**无格式解析失败路径**（对比 LLMReranker 的核心卖点）
- 凭证：`TYPESAFE_API_KEY` 从环境变量读取（.env 已就绪）；不进 settings.yaml
- 观测：在返回前通过 `trace.record_stage` 已由 CoreReranker 处理；JevReranker 额外
  把 `tokens_in`、`est_cost_usd`（$0.042/MTok，output 免费）塞进返回候选的
  metadata，供 trace 与基准脚本取用
- 注册：`RerankerFactory` 增加 `"jev"` 分支；`settings.yaml` 注释更新

### 4.3 CrossEncoderReranker 验证（已有代码，本票验证 + 修）

- `cross_encoder_reranker.py` 已存在但 README 标注"未充分测试"
- 本票：安装 `sentence-transformers`（仅 dev/optional 依赖，加入
  `[project.optional-dependencies] bench`），模型配置为 `BAAI/bge-reranker-base`
  （本地推理，免费）；跑通单测；如与 BGE 模型输出不兼容（如 sigmoid/原始 logit）
  则修正归一化

### 4.4 数据准备 `scripts/prepare_beir.py`（新建）

- 数据源：HuggingFace `BeIR/scifact`、`BeIR/nfcorpus`（corpus / queries / qrels
  三个 split；`datasets` 库已是项目依赖）；若 HF 布局与预期不符，fallback 用
  `ir_datasets`（实施计划中钉死具体加载代码）
- 每个数据集：语料灌入独立 chroma collection（命名 `beir-scifact` /
  `beir-nfcorpus`）+ BM25 索引目录 `data/db/bm25/beir-<name>/`；qrels/queries
  落盘 `data/beir/<name>/`
- 灌库复用现有 ingestion embedding 链路（Ollama nomic-embed-text）；scifact
  约 5 分钟、nfcorpus 约 4 分钟（一次性，可后台）
- 支持断点续跑：已存在的 collection 跳过（`--force` 重灌）

### 4.5 基准运行器 `scripts/benchmark_rerank.py`（新建）

- 参数：`--datasets scifact nfcorpus`、`--retrievers bm25 dense hybrid`、
  `--rerankers none bge jev llm`、`--num-queries 50`、`--seed 42`（固定抽样）
- 每个 dataset：从有 qrels 的 test queries 中固定种子抽 50 条
- 每个 retriever：用项目现有 `HybridSearch` 组件（bm25 = 仅 sparse 路径，dense =
  仅 dense 路径，hybrid = RRF 融合），取 top-10
- 每个 reranker：对 top-10 重排（depth=10，公平对比且符合 Jev 单次 ≤10 约束；
  none 即原序）
- 评分：nDCG@10（主）/ MRR@10 / Hit@10；每组合记录 rerank 耗时列表、token 用量、
  失败回退次数
- 输出：`reports/rerank_benchmark_<date>.md`（用户已确认的四段表格式：质量/速度/
  成本/可靠性 + 样本量警告）+ 同名 `.json`（原始逐 query 数据，供复算）
- LLMReranker（GLM）列保留可选：`--rerankers` 不含 llm 则跳过（它慢，~3s/查询）

## 5. 配置与凭证汇总

| 项 | 位置 | 值 |
|---|---|---|
| `TYPESAFE_API_KEY` | `.env`（gitignored） | 用户向 typesafe.ai 申请（**待办**） |
| `rerank.provider` | settings.yaml | `jev` / `cross_encoder` / `llm` / `none` |
| `rerank.min_score` | settings.yaml | 新增字段，默认 4 |
| `sentence-transformers` | pyproject `[project.optional-dependencies].bench` | BGE 依赖 |
| BGE 模型 | 首次运行自动下载 ~1GB | `BAAI/bge-reranker-base` |

## 6. 测试策略

- **单元**（mock，不联网）：metrics 三函数边界用例；JevReranker 用 mock httpx
  验证请求构造/响应映射/阈值过滤/异常路径；CrossEncoder 归一化
- **集成**（真实环境）：`prepare_beir.py --datasets scifact` 小规模冒烟；
  `benchmark_rerank.py --datasets scifact --num-queries 5 --rerankers none jev`
  端到端冒烟
- **全量**：两数据集 × 3 检索器 × 4 重排器 × 50 查询，产出最终报告

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| HF BeIR 数据集布局与预期不符 | prepare 脚本封装加载函数，fallback 到 ir_datasets（计划期钉死） |
| Jev 单次 >10 候选触发 context rot | depth 固定 10，代码硬限制 |
| BGE 首次下载模型 ~1GB、torch 安装大 | 文档标注；一次性成本 |
| TYPESAFE_API_KEY 未就绪阻塞真实调用 | 单测全 mock；key 到位前可跑 none/bge/llm 三列 |
| LLM 列拖慢全量运行 | 默认可从 `--rerankers` 摘除 |

## 8. 阶段总览与任务拆分

> 对齐原 DEV_SPEC 第 6 节"项目排期"的格式。原项目阶段编号用到 I，本工作接续为
> **阶段 J**；后续每个子功能的 spec 均放 `docs/specs/`，沿用此结构。

**阶段 J：Jev 重排接入与 BEIR 重排基准**
- 目的：把类型化判断模型（Jev）接入 Reranker 可插拔插槽，并在公开 BEIR 数据集
  （scifact、nfcorpus）上用统一基准量化其相对 BGE / GLM-LLM / 不重排的收益，
  产出可复现的对比报告。

任务（实施计划阶段展开为带验证点的详细步骤）：

1. **J1 metrics 模块**：`metrics.py` 三函数（nDCG@10 / MRR@10 / Hit@10）+ 单测
2. **J2 JevReranker**：httpx client + reranker + factory 注册 + settings 字段 + 单测
3. **J3 CrossEncoder×BGE**：验证/修复 + `bench` optional 依赖 + 单测
4. **J4 prepare_beir**：数据准备脚本 + scifact 冒烟
5. **J5 benchmark runner**：运行器 + 表格输出 + 两数据集全量跑 + 报告落盘

依赖关系：J1 → J5；J2/J3 → J5；J4 → J5。J1/J2/J3 可并行。

---

## 📊 进度跟踪表 (Progress Tracking)

> **状态说明**：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成
>
> **更新时间**：每完成一个子任务后更新对应状态

### 阶段 J：Jev 重排接入与 BEIR 重排基准

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| J1 | 评测指标模块（nDCG@10 / MRR@10 / Hit@10） | [x] | 2026-09-21 | 纯函数 + 单元测试 |
| J2 | JevReranker 实现（httpx REST + factory 注册） | [x] | 2026-09-21 | 单测用 mock，无解析失败路径 |
| J3 | CrossEncoderReranker × BGE 验证 | [x] | 2026-09-21 | Fake BGE 模型注入，5 候选降序/负 logit 通过；bench extra 登记 |
| J4 | BEIR 数据准备脚本（scifact / nfcorpus） | [x] | 2026-09-21 | 灌库 + qrels 落盘，支持断点续跑；scifact 5183 docs / nfcorpus 3633 docs 全量入 chroma+BM25 |
| J5 | 基准运行器与报告输出 | [x] | 2026-09-21 | 运行器+四段表已落地并冒烟通过；待全量跑（bge/llm/jev）后转 [x] |

## 9. 验收标准

- [ ] `pytest tests/unit` 无新增失败（现有 29 个失败为存量，与本工作无关）
- [x] scifact + nfcorpus 全量基准跑完，`reports/` 下产出 .md + .json
- [x] 表格含 none/bge/jev/llm 四列 × 6 行（2 数据集 × 3 检索器），Jev 列为真实 API 数据
- [x] 延迟/成本/回退率三个辅表齐备
- [~] JevReranker 走 MCP 链路时（factory 契约与既有渲染通道推断，未单独演示） Dashboard Query Traces 正常显示（现有页面零改动
      或仅加渲染字段）
