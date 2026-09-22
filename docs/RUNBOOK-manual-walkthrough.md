# 手把手第一次跑通（你自己动手版）

> 我（AI）不代跑。你按顺序执行，每步记下数字。全程约 15 分钟。
> 配套读物：[完整流程走查](WALKTHROUGH.md)

## 总体目标

亲手跑一遍「一次摄取 + 一次查询」，然后在 **Streamlit** 和 **Langfuse** 上各看一遍同一份数据，
再亲手**制造一次降级**、亲手**换一次可插拔插件**，把关键数字记下来。

跑完你应该能回答这四个问题：

1. 一次摄取里，时间和钱花在哪个环节？
2. 一次查询里，延迟花在哪？
3. 降级发生时，我在哪里能看到它？
4. 换掉一个可插拔组件，需要改代码吗？追踪还灵吗？

**贯穿全程的一句话**：所有数据只有一份来源 —— `logs/traces.jsonl`。
Streamlit 和 Langfuse 只是读它的两个不同视角。

---

## 记录表（边跑边填，跑完发我）

| 步骤 | 要记的数字 | 你填 |
|---|---|---|
| 1 摄取 | chunk 数 | |
| 2 摄取 | transform 占总耗时比例 | |
| 4 摄取 | LLM 调用次数 | |
| 4 摄取 | token 合计（输入/输出） | |
| 4 摄取 | 单次 LLM 调用最慢多少秒 | |
| 4 摄取 | 单次调用成本（$） | |
| 5 查询 | 返回条数 / 总耗时 | |
| 6 查询 | 第 1 次 vs 第 2 次 `query_processing` | |
| 7 查询 | 最慢的阶段是哪个、占比 | |
| 8 降级 | 降级次数 / 降级率 | |
| 9 降级 | 哪一路挂了？系统还能用吗？ | |
| 10 换插件 | trace 里多了什么阶段？有没有 token？ | |

---

## Step 0 · 环境检查（3 条命令）

```bash
cd /Users/hong/projects/active/MODULAR-RAG-PLAYGROUND

# 1) Ollama 在跑吗（嵌入要用它）
curl -s -m 3 http://localhost:11434/api/tags >/dev/null && echo "Ollama UP" || echo "Ollama DOWN"

# 2) Langfuse 凭证在不在（只看前 8 位，别打印完整 key）
grep -o 'LANGFUSE_[A-Z_]*=.\{0,8\}' .env

# 3) Streamlit 起没起（返回 200 就是起着）
curl -s -o /dev/null -w "%{http_code}\n" -m 3 http://localhost:8501
```

**期望**：`Ollama UP`、能看到 `LANGFUSE_PUBLIC_KEY=pk-lf-...` 和 `LANGFUSE_SECRET_KEY=sk-lf-...`、`200`。

如果 Streamlit 没起：`.venv/bin/python scripts/start_dashboard.py`，浏览器开 http://localhost:8501

**记录**：三项是否正常。

---

## Step 1 · 摄取一篇文档

```bash
.venv/bin/python scripts/ingest.py \
  --path tests/fixtures/sample_documents/complex_technical_doc.pdf \
  --collection demo-run
```

**这一步在做什么**：读 PDF → 切块 → 每块调 2 次 LLM（精炼 + 生成元数据）→ 向量化 → 写进 Chroma 和 BM25 索引。

**期望**：最后一行 `Total chunks generated: 12`。约 40 秒（大部分时间在等 LLM）。

**记录**：chunk 数。

---

## Step 2 · 在 Streamlit 看摄取过程

浏览器打开 http://localhost:8501 → 左侧选 **Ingestion Traces** → 展开最新那条（`demo-run`）。

**期望看到**：五个阶段的耗时条 —— `load` / `split` / `transform` / `embed` / `upsert`，
以及每个 chunk 精炼前 vs 后的文本对比（这是 Streamlit 独有的，Langfuse 看不到）。

**记录**：transform 占总耗时的大概比例。

**这里要体会的**：Streamlit 的价值是"白盒" —— 你能点开看每一块文本到底被改成了什么。

---

## Step 3 · 确认它已经自动发到云上了

**这一步不用敲命令** —— 摄取结束时，trace 已经自动写入 `logs/traces.jsonl` **并且自动上报了 Langfuse**。
（打点在 `TraceCollector.collect()` 里：先写本地文件，再发云端；发送失败只记日志，绝不影响摄取/查询。）

直接跳到 Step 4 去网页上看。如果那里是空的，说明发送失败或没配凭证，用这条命令回填：

```bash
.venv/bin/python scripts/export_langfuse.py --last 1 --type ingestion
```

**期望**：`[ok] exported 1/1 trace(s) to https://jp.cloud.langfuse.com`

> 没配 Langfuse 凭证时，程序**不会报错**：trace 只写本地文件，Streamlit 照常能看，只是云端没有。

---

## Step 4 · 在 Langfuse 看同一份数据

浏览器打开 https://jp.cloud.langfuse.com → **Traces** → 点最上面那条（名字是 `ingestion`）。

**期望看到**：

- 一棵树：`load` / `split` / `transform` / `embed` / `upsert`
- 展开 `transform`，下面挂着 **24 个** LLM 调用，名字是 `chunk_refiner#1` … `#12` 和 `metadata_enricher#1` … `#12`
- 点开任意一个 `chunk_refiner#3`，右侧应显示：
  - **模型** `glm-4-flash`
  - **token** 例如 `242 prompt → 70 completion (∑ 312)`
  - **延迟** 例如 `4.62s`
  - **成本** `$0.00`（GLM 免费额度，这是对的）
  - **Input** = 发给模型的完整提示词
  - **Output** = 模型改完的文本
  - **metadata.label** = 这块内容的 chunk id

**记录**：LLM 调用次数、token 合计、最慢那次多少秒、单次成本。

**这里要体会的**：Langfuse 的价值是"账本" —— token、钱、延迟，跨时间聚合。它看不到 chunk 原文，Streamlit 看得到。

---

## Step 5 · 跑一次查询

```bash
.venv/bin/python scripts/query.py \
  --query "chunking 策略有哪些参数" \
  --collection demo-run \
  --no-rerank
```

**这一步在做什么**：分词抽关键词 → 语义检索 + 关键词检索（两路并行）→ RRF 融合 → 返回 top-k。

**期望**：打印 `RESULTS (top_k=10, returned=N)`，每条带分数、来源、chunk 文本。

**记录**：返回条数、总耗时。

---

## Step 6 · 验证「冷启动陷阱」（重要的一步）

同一个进程里连跑 3 次同样的查询：

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.bootstrap import build_query_components

settings = load_settings()
hs, _ = build_query_components(settings, "demo-run")
for i in range(3):
    tr = TraceContext()
    hs.search("chunking 策略有哪些参数", top_k=10, trace=tr, return_details=True)
    t = {s["stage"]: s.get("elapsed_ms") for s in tr.stages}
    print(f"第 {i+1} 次: query_processing={t.get('query_processing')}ms  "
          f"dense={t.get('dense_retrieval')}ms  sparse={t.get('sparse_retrieval')}ms")
PY
```

**期望**：第 1 次的 `query_processing` 是几百毫秒，第 2、3 次掉到 1 毫秒以下。

**为什么**：那是 jieba 分词词典的首次加载，不是真实查询开销。dense 第一次也偏慢（要建 HTTP 连接）。

**记录**：第 1 次 vs 第 2 次的数字。

**这里要体会的**：`query.py` 每次都是新进程，所以 CLI 单次查询**永远**付这笔钱；Streamlit / MCP 是长驻进程，只付一次。
**别拿单次 CLI 的冷启动数据当稳态性能** —— 这是最容易下错结论的地方。

---

## Step 7 · 查询侧也已经在云上了

同样**不用敲命令**，查询跑完 trace 就自动上报了。回 Langfuse → Traces → 点最新那条（名字 `query`）：

**期望看到**：`query_processing` / `dense_retrieval` / `sparse_retrieval` / `fusion` 四个阶段的时间线，
`dense_retrieval` 和 `sparse_retrieval` 的类型标成 **retriever**，展开能看到 `result_count`、`top_k` 等元数据。

**记录**：最慢的阶段是哪个、占多少。

---

## Step 8 · 看降级率报表

```bash
.venv/bin/python scripts/degradation_report.py
```

**期望**：`检索降级 0 次 0.0%`，最后一行写"窗口内没有发生降级（不是没统计，是真的没挂）"。

**记录**：降级次数 / 降级率。

---

## Step 9 · 亲手制造一次降级

把嵌入服务的地址指到一个没人监听的端口（模拟 Ollama 挂掉）：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:9 .venv/bin/python scripts/query.py \
  --query "chunking 策略有哪些参数" \
  --collection demo-run \
  --no-rerank
```

**期望**：日志里出现 `Dense retrieval error: ... 502` 和 `Dense retrieval failed, using sparse only`，
但**命令仍然返回了结果**（因为关键词那路顶上了）。

然后：

```bash
.venv/bin/python scripts/degradation_report.py
.venv/bin/python scripts/export_langfuse.py --last 1 --type query
```

**期望**：报表变成 `检索降级 1 次 …%`，并写明 `dense 路径挂掉`。
到 Langfuse 点最新那条 trace，能看到一个 **`retrieval_fallback`** 节点。

**记录**：降级率、哪一路挂了、系统还能用吗。

**这里要体会的**：**降级不报错**。用户感受到的只是"结果变少了/变差了"。
没有这一步的度量，线上悄悄变差你永远不知道。

---

## Step 10 · 换一个可插拔组件，看追踪是否照旧

编辑 `config/settings.yaml`，把重排从"不重排"换成"用 LLM 重排"：

```yaml
rerank:
  enabled: true          # 原来是 false
  provider: "llm"        # 原来是 "none"
```

然后跑同一句查询（这次**不加** `--no-rerank`）：

```bash
.venv/bin/python scripts/query.py \
  --query "chunking 策略有哪些参数" \
  --collection demo-run
```

**期望**：命令会慢很多（约 30–60 秒，因为对每个候选都问了一次 GLM），最后仍然返回结果。

再发上云看：

```bash
.venv/bin/python scripts/export_langfuse.py --last 1 --type query
```

**期望在 Langfuse 看到**：trace 里**多出一个 `rerank` 阶段**（`provider: llm`），
它下面挂着一个 **`rerank#1`** 的 generation，带模型名和 token。

**记录**：trace 里多了什么阶段？有没有 token？

**这里要体会的（这就是你问的宪法问题）**：
你**只改了 config 两行，没动任何代码**，追踪就自动跟上了 ——
因为打点写在编排层（`pipeline.py` / `HybridSearch` / `CoreReranker`），
LLM 计费写在 `BaseLLM.chat()` 一个入口。换成别人写的插件，同一套追踪照旧生效。

**看完记得改回去**（`enabled: false`、`provider: "none"`），否则之后每次查询都慢一分钟。

---

## Step 11 · 收尾

`config/settings.yaml` 改回原样。测试用的集合想删就在 Streamlit → **Ingestion Manager** → Collections 里删掉 `demo-run`。

---

## 跑完之后，把这四句话填完整（面试时就是这么讲的）

1. **数据在哪个环节花掉**：摄取里 ____% 的时间在 ____ 环节；查询里最慢的是 ____。
2. **成本**：一次摄取 = ____ 次 LLM 调用 / ____ token / $____。
3. **可靠性**：我制造了 ____ 的故障，系统 ____（崩了 / 降级继续服务），我能在 ____ 看到它。
4. **可插拔**：我改了 ____ 行配置，从 ____ 换成 ____，追踪____（需要 / 不需要）改代码。

把这张表和上面「记录表」发我，我帮你校一遍数字，然后一起写进简历。
