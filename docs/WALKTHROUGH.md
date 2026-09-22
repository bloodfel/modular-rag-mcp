# 完整流程走查：产出什么、怎么读、怎么用

> 配套：[Langfuse 闭环 spec](specs/2026-09-22-observability-eval-stage-L.md) · [简历素材](RESUME_NOTES.md)
> 本文回答三个问题：**两个界面分别给什么数据？换掉可插拔零件后还灵吗？怎么把数据变成决策？**

## 1. 链路上有哪些环节

**摄取**（一个 PDF 进来）：

```
load      读 PDF 抽文本（markitdown）
split     切成 chunk（recursive）
transform ├── chunk_refiner      每个 chunk 调一次 LLM 精炼      ┐ 都是 LLM
          └── metadata_enricher  每个 chunk 调一次 LLM 生成元数据 ┘
embed     向量化（Ollama）+ BM25 词频
upsert    写进 Chroma + BM25 索引
```

**查询**（一个问题进来）：

```
query_processing   分词、抽关键词、解析过滤条件
dense_retrieval    语义检索（向量）      ┐ 两路并行
sparse_retrieval   关键词检索（BM25）    ┘
fusion             RRF 融合两路结果
rerank             重排（可选，none/bge/jev/llm）
```

每个环节都会往 `logs/traces.jsonl` 写一行记录：**阶段名 + 耗时 + 它自己的明细**。这是唯一的事实源，下面两个界面都读它。

## 2. Streamlit 和 Langfuse 分别给什么

| 你想知道 | Streamlit | Langfuse |
|---|---|---|
| 一次查询每个阶段花多久 | ✅ Query Traces 页柱状图 | ✅ 瀑布图（更细，含 LLM 子节点） |
| **检索回来的 chunk 原文** | ✅ 能点开逐个看全文、分数、来源 | ❌ 只记了数量 |
| **每次 LLM 调用的 token 和钱** | ❌ 不显示 | ✅ 模型名 + token + 自动算成本 |
| **一周的查询量 / 成本趋势** | ❌ | ✅ 聚合图表 |
| **摄取时 chunk 精炼前 vs 后** | ✅ Ingestion Traces 页 | ❌ |
| **哪条查询发生了降级** | ✅ trace 里有标记 | ✅ `retrieval_fallback` 节点 |
| **数据资产全貌**（几篇文档、几个 chunk） | ✅ Data Browser | ❌ |
| 集合的增删、文档上传 | ✅ Ingestion Manager | ❌ |

一句话：**Streamlit 是白盒放大镜**（看清每个 chunk、每段文本、数据资产），**Langfuse 是趋势和钱的账本**（token、成本、跨时间的聚合）。两个都读同一份 `traces.jsonl`，只是展示目的不同。

> 需要频繁看趋势或算钱用 Langfuse；需要"这条结果为什么不对"用 Streamlit。

## 3. 「降级」是什么

检索有两路：**语义**（dense，靠 Ollama 算向量）和**关键词**（sparse，BM25）。
正常时两路都跑，融合后给你结果。

**降级 = 依赖坏了，系统用备选方案继续服务，而不是直接报错。**

真实例子（我们刚造的）：把 `OLLAMA_BASE_URL` 指到一个死端口 →

```
Dense retrieval error: Ollama API request failed with status 502
Dense retrieval failed, using sparse only          ← 降级发生了：只用关键词那路
RESULTS (returned=2)                                ← 照样给出了答案，没崩
```

用户感受是"还能用，但变差了一点"。**问题在于：如果没人统计，你永远不知道线上正在悄悄变差。**

代码里实现的三级降级：

| 出问题的东西 | 系统怎么办 |
|---|---|
| 重排后端（Jev / LLM 超时） | 回退用融合排序，不重排 |
| 一路检索（嵌入服务挂了） | 用另一路的结果 |
| LLM 增强（GLM 失败） | 回退规则增强（正则、词频那一套） |
| **两路都挂** | **报错**（这个不该降级，必须让人知道） |

度量它：`.venv/bin/python scripts/degradation_report.py` →
`检索降级 1 次 16.7%（dense 路径挂掉）`。线上真实值应该接近 0，一旦超过阈值就告警。

## 4. 换了可插拔零件，追踪还灵吗

**灵，而且一行代码都不用改。** 原因：**打点写在"指挥层"，不在"零件"里。**

证据（源码）：

```
src/ingestion/pipeline.py:275   record_stage("load",   method: "markitdown")
src/ingestion/pipeline.py:303   record_stage("split",  method: "recursive")
src/ingestion/pipeline.py:354   record_stage("transform", …)
src/ingestion/pipeline.py:421   record_stage("embed",  …)
src/ingestion/pipeline.py:499   record_stage("upsert", backend: "ChromaDB")
```

这五个阶段全部由 `pipeline.py`（指挥层）记录；零件只负责在 `data` 里报上自己的名字。
所以你换成别的 loader / splitter / 向量库 / 重排器，阶段照记，只是明细里的名字变了。

**LLM 更彻底**：打点收在 `BaseLLM.chat()` 一处，任何新 provider 只要实现 `_chat`，
自动就有 token / 延迟 / 成本，不需要碰观测代码。

**只有两种情况需要动代码**（诚实说清）：

1. **新增一个阶段**（不是换零件）→ 要在 pipeline 里加一行 `record_stage`；Streamlit 的 tab 也要加一行（它按阶段名分派渲染）。
2. 已知小缺口：`dense_retrieval` 明细里的 `provider` 显示 `unknown`（dense retriever 没暴露 `provider_name`）；`embed` 阶段没记 embedding provider 名字。**换了嵌入后端，从 trace 里看不出换成了哪个。**

所以对"别人换 config 里的 provider / 换 API key 就能用"这个问题：**是的**，
符合宪法的「配置驱动 + 接口隔离」。上面两点是需要补的边角，不影响主链路。

## 5. 真实数据长什么样

**摄取**（12 chunk 的 PDF，`b9c99eb1-c7b8-4487-9d19-dfa59d1850bf`）：

```
总耗时 35.95s
  transform            33.88s   ← 占 94%
  … 24 次 LLM 调用分摊其中，单次 3.4–11.7s
  embed                 1.84s
  load                0.136s
  upsert              0.175s
token 合计: 8213 输入 / 2615 输出
```

**查询**（`2a01d620-78e0-4d22-abb7-c38e3b7f834e`）：

```
总耗时 567ms
  query_processing    392ms   ← 看起来是大头，但见下节
  dense_retrieval     174ms
  sparse_retrieval    7.6ms
  fusion              0.03ms   ← RRF 几乎免费
```

## 6. 怎么从数据变成决策

### 例 1：摄取慢的瓶颈在哪（性能决策）

数据：transform 33.88s / 总 35.95s = **94%**，而解析只有 0.136s、嵌入 1.84s。
决策：要提速就别优化 PDF 解析，去调 LLM 那一步 —— 降并发数、关掉某个增强阶段、或换更快的模型。
**这条就是 BEIR 基准的同一手法**：先用数据定位瓶颈，再动手。

### 例 2：差点下错结论（方法论，很值钱）

第一眼数据说"查询最大开销是 query_processing 392ms"。
但同一个进程里再跑两次：

```
第 1 次: query_processing=408.7ms   dense=160.7ms
第 2 次: query_processing=  0.09ms   dense= 32.7ms
```

392ms 是 **jieba 词典首次加载**，不是真实查询开销；dense 那 160ms 也含首次建连接。
lesson：**别拿单次 CLI 的冷启动数据当稳态性能。** 长驻进程（Playground / MCP）只付一次。

### 例 3：成本（成本决策）

数据：一次摄取 24 次 LLM 调用、8213 输入 + 2615 输出 token。
决策依据：乘以模型单价就知道每篇文档多少钱。GLM 免费额度是 0；换 DeepSeek 就是真金白银 ——
于是"要不要开 LLM 增强"从感觉变成算术。

### 例 4：可靠性（阈值决策）

数据：`scripts/degradation_report.py` 给出降级率，并按路径归因。
决策：给降级率设告警阈值（如 >1% 就查 Ollama），因为降级不报错、只会静默变差。

## 7. 简历上能写什么

遵循「动词 + 系统 + 量化结果」：

- 接入 **Langfuse** 实现摄取/查询双链路实时可观测：**延迟与 token 成本按阶段归因**，
  定位到 LLM 增强占摄取耗时 **94%（33.9s/36.0s）**，据此优化并发路径
- 设计并**度量三级降级**（重排失败→融合序、嵌入失败→BM25-only、LLM 失败→规则增强），
  故障注入验证 + 降级率可统计、可告警
- 用 Streamlit + Langfuse 双视角做**链路白盒**：Streamlit 看 chunk 级细节，Langfuse 看成本与趋势
- （已有）BEIR 四维基准选型：Jev 重排 +8.3 nDCG@10，$0.17/千次查询

面试讲法（每个模块通用）：
**"我们发现 X 问题 → 建了 Y 度量 → 数据显示 Z → 据此做了 W 决策 → 效果 Q"**
