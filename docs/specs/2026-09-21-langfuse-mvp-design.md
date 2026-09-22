# Langfuse 可观测 MVP — 设计文档

日期：2026-09-21 · 分支：dev · 状态：执行中

## 1. 背景与目标

给本 RAG 项目接入 **Langfuse**（LLM 应用可观测平台），实现查询链路的 trace 上报与
可视化：每次 `query_knowledge_hub` 调用在 Langfuse UI 中可看到——检索各阶段
（bm25/dense/RRF/重排）耗时、各重排器的分数与概率、最终返回的 chunks 与引用。

MVP 边界：**能看**。 Tonight 的目标是"跑一个查询 → 打开 Langfuse → 看到完整链路"，
不做看板定制与评估上报。

## 2. 非目标（本期不做）

- 评估结果（Ragas/基准分数）上报为 Langfuse Scores
- 成本看板定制、用户反馈（thumbs up/down）采集
- 既有 `traces.jsonl` / Streamlit Dashboard 的替换（两者并存）
- Prompt 管理与 A/B 实验

## 3. 方案

- **自托管** Langfuse v3（Docker Compose，本机 3000 端口）——数据不出本机，
  明天可复跑；云版作为备选（用户注册 cloud.langfuse.dev 取 key）
- **复用既有观测层**：`TraceContext`/`TraceCollector` 已记录全部阶段，不重写埋点
- **导出器**（新建 `scripts/export_langfuse.py`）：读 `logs/traces.jsonl`，
  把每条 trace 映射为 Langfuse trace + 每阶段一个 span（输入/输出/耗时/元数据），
  经 `langfuse` Python SDK 上报
- **实时挂钩**（最小侵入）：`scripts/query.py` 加 `--langfuse` 开关，查询后同步导出
  该次 trace（MCP server 主链路零改动）

## 4. 配置与凭证

| 项 | 位置 | 说明 |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `.env` | 注册后在 Langfuse UI 创建 |
| `LANGFUSE_HOST` | `.env` | `http://localhost:3000` |
| `langfuse` SDK | `pyproject [project.optional-dependencies].observability` | 本期不进主依赖 |

## 5. 测试策略

- 导出器单测：给定一行合成 trace JSON → 断言生成的 Langfuse span 结构（mock SDK）
- 端到端：`scripts/query.py --query "..." --langfuse` → Langfuse UI/`/api/public/traces` 可查

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| v3 栈资源占用高（clickhouse/minio） | 用官方 compose；Mac 16GB 可承 |
| SDK 与自托管版本不匹配 | compose 固定 langfuse/langfuse:3 镜像版本，SDK 装稳定版 |

## 📊 进度跟踪表 (Progress Tracking)

> **状态说明**：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成
>
> **更新时间**：每完成一个子任务后更新对应状态

### 阶段 K：Langfuse 可观测 MVP

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| K1 | Docker Compose 起 Langfuse 本机栈 | [x] | 2026-09-22 | v4 六服务 healthy，localhost:3000 就绪（云模式已接管） |
| K2 | SDK 接入 + .env 凭证位 | [x] | 2026-09-22 | langfuse 4.15.4 + observability extra |
| K3 | traces.jsonl → Langfuse 导出器 | [x] | 2026-09-22 | OTLP 传输层 + 映射单测；实时 --langfuse/Playground 开关已接 |
| K4 | 复跑 runbook | [x] | 2026-09-22 | 下方「云模式 Runbook」为可跑通版本（已含 OTLP 排障） |

## 9. 验收标准

- [x] 一条真实查询后 Langfuse UI 出现含 4 阶段 span 的 trace（OTLP 实测 50 条 observations 落库）
- [x] 单测无新增失败；runbook 让"明天的你"10 分钟内复现
- [x] （自托管可选）`docker compose ps` 全服务 healthy

---

## 📖 Runbook（从零到看到 trace）

### 方案 A：Langfuse Cloud（推荐，当前使用）

```bash
cd /Users/hong/projects/active/MODULAR-RAG-PLAYGROUND

# 1. 一次性配置：把三个变量写进 .env（已被 gitignore）
#    LANGFUSE_PUBLIC_KEY=pk-lf-...
#    LANGFUSE_SECRET_KEY=sk-lf-...
#    LANGFUSE_HOST=https://jp.cloud.langfuse.com     # 日本区；欧/美区改域名

# 2. 跑一条真实查询，并实时上报（无需事后导出）
.venv/bin/python scripts/query.py --query "远程接入指南里怎么上传文档？" --langfuse

# 3. 或者事后批量导出最近 5 条
.venv/bin/python scripts/export_langfuse.py --last 5

# 4. 浏览器打开 LANGFUSE_HOST → Traces → 点开任意一条看阶段瀑布
#    Sessions 页按集合名分组（trace metadata 里的 collection）
```

**Playground 等效操作**：Streamlit 的 🧪 Query Playground 打开「实时上报 Langfuse」开关，查询后数秒内云端可查。

#### 排障：新组织的 410 错误

新建的 Langfuse 组织（2026-09-16 之后创建）调用 legacy API 会返回：

```
410 LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION
GET /api/public/traces is a legacy API ...
replacement: GET /api/public/v2/observations
```

因此本项目导出器**必须走 OTLP**（`POST /api/public/otel/v1/traces`，basic auth），
不能用 `POST /api/public/ingestion`（返回 200 但数据对新组织不可读）。
读取侧同理：用 `GET /api/public/v2/observations?fromStartTime=...&toStartTime=...`。

### 方案 B：自托管（可选）

```bash
docker compose -f infra/langfuse/docker-compose.yml up -d
sleep 45 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000   # 期望 200
# 注册 → 建 API keys → .env 里 LANGFUSE_HOST=http://localhost:3000
# 之后步骤与方案 A 的第 2-4 步完全一致
```

若 `docker compose` 报未知命令：`brew install docker-compose && mkdir -p ~/.docker/cli-plugins && ln -sf $(brew --prefix)/opt/docker-compose/bin/docker-compose ~/.docker/cli-plugins/docker-compose`

> 首次生成 `infra/langfuse/.env` 密钥：`cd infra/langfuse && for v in SALT CLICKHOUSE_PASSWORD MINIO_ROOT_PASSWORD; do echo "$v=$(openssl rand -hex 16)"; done; echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"; echo "NEXTAUTH_SECRET=$(openssl rand -hex 32)"`
> 注意：postgres 首次初始化后密码即固定为 `.env` 中 `POSTGRES_PASSWORD`（未设置则 `postgres`），`DATABASE_URL` 必须与其一致。
