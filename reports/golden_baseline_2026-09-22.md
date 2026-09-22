# 金标评测基线（2026-09-22）

## 被测对象

- **语料**：本仓库自身文档（README、DEV_SPEC、AGENTS、docs/CONFIG_GUIDE.md、docs/specs/*.md），
  共 **231 个 chunk**。MD/文本入库由 `TextLoader` 支持（本次新增，见 `scripts/ingest.py --extensions`）。
- **金标**：`tests/fixtures/golden_test_set.json` v2.0，**10 条 query**（5 英文 + 5 中文），
  **21 个 ground-truth chunk**，全部按"读候选内容"标注（未抄检索排名）。
- **检索栈**：hybrid（dense nomic-embed-text + BM25，RRF 融合），**rerank 关闭**（生产默认），top-10。
- **评测器**：CustomEvaluator（hit_rate / mrr / ndcg）。

## 基线数字

| 入库配置 | hit_rate | MRR | nDCG@10 | 集合 |
|---|---|---|---|---|
| LLM 精炼+富集（生产默认） | **1.0000** | 0.4767 | 0.5046 | `project_docs` |
| 规则清洗（确定性，离线可跑） | **1.0000** | **0.5061** | **0.5405** | `project_docs_ci` |

两个配置的 chunk 边界一致（同为 231 块），差异只在文本清洗方式。10 条 query 的样本量偏小，
两条数字的差距（nDCG ±0.04）不宜解读为"LLM 精炼更差"，只能说**没有证据表明 LLM 精炼提升了检索质量**。

## 复现命令

```bash
# 1) 建确定性索引（不上 LLM、不花钱、可离线）
for f in docs/CONFIG_GUIDE.md README.md AGENTS.md docs/specs DEV_SPEC.md; do
  .venv/bin/python scripts/ingest.py --config config/settings.eval.yaml \
    --path "$f" --collection project_docs_ci --extensions .md --force
done

# 2) 跑金标
.venv/bin/python scripts/evaluate.py --collection project_docs_ci \
  --test-set tests/fixtures/golden_test_set.json --top-k 10
```

（`project_docs` 用默认 `config/settings.yaml` 同样命令，去掉 `--config`。）

## 跑通过程中修掉的三个阻断项

1. `evaluation.enabled: false`（提交版配置）→ 工厂返回 `NoneEvaluator`，**评测静默不产出任何指标**。
   已改为默认开启，指标列表改为 custom 评估器真实支持的 `hit_rate/mrr/ndcg`。
2. `CustomEvaluator` 对**对象**只认 `.id` 字段，而检索器返回的 `RetrievalResult` 只有 `chunk_id`
   → 每条 query 直接抛错、零指标。已支持全部 id 字段（含单测）。
3. `.md` 无法入库：管线只有 `PdfLoader`（而 dashboard 上传器一直宣称支持 md/txt）。已补 `TextLoader`。

## 已知边界（写 CI 门禁前必须处理）

**chunk id 不可移植，不能直接当 CI 里的 ground truth。** 落库 id 由 `VectorUpserter._generate_chunk_id`
生成，格式 `{source_hash}_{index:04d}_{content_hash}`，其中：

- `content_hash` 哈希的是 **transform 之后**的文本 → 换清洗方式（LLM↔规则）id 全变；
- `source_hash` 哈希的是 **绝对路径** → 换机器/checkout 目录 id 全变。

实测证据：同一段内容（同一 source、同一个 chunk_index=146）在两个集合里的 id 分别是
`52330cc8_0146_055a806f` 与 `52330cc8_0146_8748dc3b`（内容哈希不同），而两种模式下 21 个标注点
按 `(仓库相对路径, chunk_index)` 全部一一对应。

**两个可选修法**（属 CI 门禁设计决策）：

- **A（推荐，改动最小）**：金标改用 `(仓库相对 source_path, chunk_index)` 寻址，评估器支持这种
  ground-truth 形式。它对 transform 模式与机器路径都稳定，只在 chunk 边界变化（切分算法/参数改动）时失效
  —— 而这正是门禁应该拦截的变化。
- **B**：把 `_generate_chunk_id` 改为内容寻址（如用 `metadata["doc_hash"]` 替代 source_path），
  并把金标绑定到一份固定的入库配置（本次已提供 `config/settings.eval.yaml`）。代价是生产 id 方案变更、
  存量集合 id 失效。

**阈值建议**（以确定性基线为准，留 10–15% 余量避免波动误报）：
`hit_rate >= 1.0`（当前满值，掉任何一条都是真回归）、`mrr >= 0.45`、`ndcg >= 0.48`。

## 不覆盖的范围

- 生成质量（reference_answer 已写但未评，Ragas 依赖冲突待修，见 DEV_SPEC 阶段 L 已知缺口）；
- 重排质量（rerank 默认关闭）—— 重排选型有 BEIR 基准覆盖，见 `reports/rerank_benchmark_2026-09-22.md`；
- 图片/多模态链路（本语料为纯文本 markdown）。
