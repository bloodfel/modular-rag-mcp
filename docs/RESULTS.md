# 产出数据总账

> 本文档汇总截至目前所有已验证的实测数字，每节附复现入口。
> 更新约定：数字必须来自真实跑批，注明日期与来源文件。

更新：2026-09-22

---

## 1. 可观测性（tracing → Langfuse）

| 项 | 数字 / 状态 | 来源 |
|---|---|---|
| 闭环四步 | 打点 → 自动上报 → 云端瀑布（generation/embedding/retriever 分型、`purpose#N` 命名）→ score 回写，全部验证 | Langfuse UI / `v3/scores` API |
| 单文档入库成本 | **$0.000985**（2 chunks、4 次 LLM 调用、23.84s） | trace `61e00bdb` |
| LLM 速度对照 | chunk_refiner 单次：GLM-4-Flash 36.37s → DeepSeek **1.23s**（≈30×），质量更干净（保留 IMAGE 占位符、无 Markdown 残渣） | Langfuse trace 对照 |
| 图片链路 | pymupdf 提取 2 图 → vision caption 入 chunk 文本；静默降级已加监控（`image_extraction` 标记 + 诊断页警告） | commit aea5272 |
| score 回写 | `ragas.*` 四分数挂到每条 query trace；60 例共 **238 个**分数已上云 | `GET /api/public/v3/scores?traceId=` |

## 2. 检索质量（离线金标，10 条人工核对，hybrid、rerank 关、top-10）

| 入库配置 | hit@10 | MRR | nDCG@10 |
|---|---|---|---|
| LLM 精炼（生产默认） | 1.0000 | 0.4767 | 0.5046 |
| 规则清洗（确定性，离线） | 1.0000 | 0.5061 | 0.5405 |

- 语料：仓库自身文档 231 chunks（`project_docs` / `project_docs_ci`）。
- 寻址方案：金标以 `expected_locations`（仓库相对路径 + chunk 序号） portable 寻址，
  门禁运行时翻译为当前集合的 chunk id（id 本身绑定机器路径与 transform 配置，不可直接用）。
- 复现：`scripts/evaluate.py --collection project_docs_ci --test-set tests/fixtures/golden_test_set.json`。

## 3. 生成质量（Ragas LLM-as-Judge，60 条标注集，评委 DeepSeek）

| 指标 | 平均分 |
|---|---|
| faithfulness 忠实度 | **0.9611** |
| answer_relevancy 答案相关性 | 0.7602 |
| context_precision 上下文精度 | 0.7800 |
| context_recall 上下文召回 | **0.9167** |

- 60/60 成功、0 失败；238 个分数回写 Langfuse。
- 标注集为 LLM 起草（每条带 provenance chunk），**尚未人工复核**，当"人工金标"引用前需抽查。
- 复现：`scripts/evaluate_ragas.py --test-set tests/fixtures/golden_ragas_set.json --collection project_docs_ci`。

## 4. 工程决策实验

- **min_score 三档**（BEIR scifact+nfcorpus，jev 重排）：生产默认 4.0 → **24% 空结果率**、nDCG 61.51；
  1.0 → 0% 空结果、70.28；0 → 72.05。结论：默认改 1.0，高阈值必须配"全滤空回退"。
- **重排选型**（`reports/rerank_benchmark_2026-09-22.md`）：Jev 较无重排 **+8.3 nDCG@10（相对 +12.8%）**，
  $0.17/千次查询，p95 < 5.1s，失败回退 0；本地 bge 免费但几乎无增益；LLM 重排延迟 37–45s 不可交互。

## 5. 评测资产

| 资产 | 说明 |
|---|---|
| `tests/fixtures/golden_test_set.json` v2 | 10 条 query，21 个 ground-truth chunk（人工核对） |
| `tests/fixtures/golden_ragas_set.json` | 60 条 QA（LLM 起草，带 provenance，待人工复核） |
| `config/settings.eval.yaml` | 确定性入库配置（无 LLM 调用、可离线、可复现） |
| `scripts/evaluate_ragas.py` | 检索→生成→四指标→回写 Langfuse |
| `scripts/generate_golden_set.py` | 从语料起草标注集（可扩到 200 条） |

## 6. 已知缺口 / 未做

- CI 门禁已建并**云端验证通过**：GitHub Actions run 35721344680（4m44s），
  hit 1.0000 / MRR 0.5050 / nDCG 0.5398——与本地基线几乎一致（跨机器稳定）。
- mcp 2.x 迁移（当前钉 <2.0，功能正常但是旧 API）。
- chunk id 可移植性 → CI 门禁的寻址方案待定（方案 A：source+index；方案 B：内容寻址 id）。
- 60 条标注集未人工复核；Ragas 评委与生成器同为 DeepSeek，可能有同源偏好。
- CI 门禁未建；fiqa 大语料基准延期；安全（L4）明确不做。
