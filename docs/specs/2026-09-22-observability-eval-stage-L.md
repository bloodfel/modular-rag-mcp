# Langfuse 可观测闭环（原「阶段 L」缩减版）

> **本文档随时更新**（每完成一项就改进度表）。
> 关联：DEV_SPEC 阶段 F（trace）/ G（Dashboard）/ K（Langfuse MVP）
> 简历素材见 [docs/RESUME_NOTES.md](../RESUME_NOTES.md)

## 1. 这一件事是什么

一次查询 / 摄取，程序会在 `logs/traces.jsonl` 里留一张"小票"（每个阶段多久、调了哪个模型、用了多少 token）。
把这张小票再发一份到 Langfuse 云端，于是能看到四样东西：

- **延迟** —— 每个阶段各花多久
- **token** —— 每次 LLM 调用用了多少
- **成本** —— Langfuse 按模型价格表自己算，我们不自造价格表
- **降级** —— 哪个环节失败后回退到了哪条路

**就这四样。四件事：打点 → 发上去 → 看到数 → 降级率可统计。**

## 2. 现状：4 件全部做完

| # | 任务 | 状态 | 说明 |
|---|---|---|---|
| 1 | **打点**：每次调用记下阶段 / 模型 / token | [x] | `BaseLLM.chat()` 单一入口，任何 provider 白拿；真实摄取已验证 24 次调用 |
| 2 | **发到云上** | [x] | `scripts/export_langfuse.py`（OTLP 格式）；span id 由 trace 推导，重复导出不产生重复数据 |
| 3 | **云端按类型显示 + 自动算成本** | [x] | UI 已确认：`glm-4-flash`、`242 prompt → 70 completion`、`4.62s`、`$0.00`（GLM 免费额度，符合预期） |
| 4 | **降级率可统计** | [x] | `scripts/degradation_report.py`；真实验证：断掉 Ollama → 查询降级到 BM25-only → 报告记为 1/6（dense 路径挂掉） |

**每次 LLM 调用怎么认出是哪一次**（UI 上看到的层级）：

```
b9c99eb1…  ingestion 整条 trace
├── stage:load                    读 PDF
├── stage:split                   切成 12 块
├── stage:chunk_refiner           这一阶段的 12 次调用全挂它下面
│   ├── chunk_refiner#1    4.52s  366 tokens   ← 名字前缀 = 哪个阶段
│   ├── chunk_refiner#2    4.64s               ← #N = 该阶段的第几个（**完成顺序**，并行跑，不一定是第几块）
│   └── … 共 12 个                              ← label = chunk id，这才是确定身份
├── stage:metadata_enricher       另外 12 次挂这里
│   ├── metadata_enricher#1 …
│   └── … 共 12 个
└── stage:embed / upsert …
```

每次调用都带：`label`（chunk id）、`Input`（发给模型的完整提示词）、`Output`（模型的回复）。

**已知小缺口**（不影响结论，按需再补）：

- `stage:llm_enrich` 在 UI 上显示 0.00s —— chunk_refiner / metadata_enricher 记账时没传 `elapsed_ms`，属于显示噪音。

## 3. 为什么不照官方博客那样写（它确实更短）

官方那篇 RAG observability 博客能那么短，靠的是两个我们这里不成立的前提：

1. **它用 LangChain** —— `CallbackHandler()` 自动抓走每次 LLM 调用的模型、token、耗时，一行插桩都不用写。
   本项目是自研可插拔架构，LLM 调用直接用 httpx，没有框架帮忙抓，只能自己插桩（已做，集中在 `BaseLLM` 一处）。
2. **它的代码能包成 `with start_as_current_observation(...) as span:`** —— 起止时间由上下文管理器实时测量。

而本项目的阶段是**事后记账**：26 处 `record_stage(...)` 都在阶段结束后才调用，`stage_timer` 这类上下文管理器只写在 docstring 里、从未实现。

官方 SDK 的 `start_observation` **没有 `start_time` 参数**（只有 `.end(end_time=...)`），也拿不到底层 OTel span 走后门（已验证）。
所以把已经跑完的历史记录回放给 SDK，所有 span 都会被盖成"现在"，瀑布图就废了。
**这就是我们手写 OTLP 的唯一原因：OTLP 支持显式时间戳，能忠实还原已经跑完的调用。**

## 4. 另外三件（不属于上面这个闭环）

这三件是"简历 / 工程能力"项目，跟"看到可观测数据"不是一件事。之前把它们塞进同一个阶段，是文档臃肿的根源。**另开阶段或按需排期。**

| 主题 | 内容 |
|---|---|
| 质量门 | 金标测试集补 `expected_chunk_ids`；CI 里指标低于阈值即拦截合并 |
| 安全 | MCP HTTP Bearer 鉴权；工具级读写权限（只读场景禁 `ingest_document`） |
| ADR | `docs/adr/` 5 篇选型决策记录（重排选型 / min_score / bge 串行 / RRF / stateless HTTP） |

## 5. 验收标准（就这四条）

- [x] Langfuse 云端能看到一条真实查询的：阶段瀑布 + token + 成本
- [x] 降级率能从 `traces.jsonl` 统计出来
- [x] 重复导出同一 trace 不产生重复数据
- [x] `pytest tests/unit` 零新增失败（基线 29 failed / 1222 passed，当前 29 failed / 1263 passed）

## 6. 延期项：fiqa 5.7 万语料基准

需下载并用本地 Ollama 嵌入 5.7 万篇文档，再跑 3 条检索路径 × 4 种重排，耗时数小时；
结论增量低（只是验证"scifact/nfcorpus 上的结论在更大规模仍成立"）。

```bash
.venv/bin/python scripts/prepare_beir.py --datasets fiqa
.venv/bin/python scripts/benchmark_rerank.py --datasets fiqa --path hybrid --rerankers none jev
# 完整矩阵：--path 依次 dense / sparse / hybrid，--rerankers 加 bge llm
```
