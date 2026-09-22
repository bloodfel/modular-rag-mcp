---
name: setup
description: "Interactive project setup wizard. From a clean codebase, guides user through provider selection (OpenAI/Azure/DeepSeek/Ollama/Qwen/Gemini/etc.), API key configuration, dependency installation, config generation, and launches the dashboard. If user selects an unimplemented provider, auto-scaffolds the provider code following the plugin architecture. Auto-diagnoses and fixes startup failures with up to 3 retry rounds. Use when user says 'setup', 'set up', 'configure', 'init project', '初始化', '环境配置', '项目配置', 'first run', 'get started', 'quick start', or wants to configure and launch the project from scratch."
---

# Setup

Interactive wizard: configure providers → install deps → generate config → launch dashboard → auto-fix issues.

---

## Pipeline

```
Preflight → Ask User → Generate Config → Install Deps → Validate → Launch → Usage Guide
```

> Auto-fix loop: if any step fails, diagnose → fix → retry (≤3 rounds).

---

## Step 1: Preflight Checks

Verify prerequisites before asking the user anything:

### 1.1 Check Python Version

```powershell
python --version          # Require >=3.10
```

If Python < 3.10, stop and inform user to install a supported version.

### 1.2 Check & Create Virtual Environment

Check if `.venv` directory already exists. If it does, skip creation and just activate it. If it does NOT exist, create it and activate it.

**Important:** Use `--without-pip` to avoid the slow `ensurepip` step that can hang on Windows, then bootstrap pip manually with `ensurepip` after activation.

```powershell
# Step 1: Check if .venv exists
Test-Path ".venv"

# Step 2: If .venv does NOT exist, create it (fast, no pip bundling)
python -m venv .venv --without-pip

# Step 3: Activate the virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# Step 4: Bootstrap pip inside the venv (only needed after --without-pip)
python -m ensurepip --upgrade

# Step 5: Verify
pip --version             # Should show pip path inside .venv
```

If `.venv` already exists, only run Step 3 (activate) and Step 5 (verify).

---

## Step 2: Ask User for Configuration

Use the `ask_questions` tool to gather provider choices. Ask in batches (max 4 questions per call).

### Batch 1: Core Providers

Ask these questions together:

1. **LLM Provider** — Which LLM provider?
   - Options: `OpenAI`, `Azure OpenAI`, `DeepSeek`, `GLM (Zhipu AI)`, `Ollama (local)`, `Qwen (Alibaba Cloud)`, `Gemini (Google)`
   - Recommended: `GLM (Zhipu AI)`（glm-4-flash 免费档，国内访问快）
   - Built-in: OpenAI, Azure, DeepSeek, GLM, Ollama. Others require auto-scaffolding (see Step 2.5).

2. **Embedding Provider** — Which embedding provider?
   - Options: `OpenAI`, `Azure OpenAI`, `Ollama (local)`, `Qwen (Alibaba Cloud)`, `Gemini (Google)`
   - Recommended: `Ollama (local)`（免费、离线；GLM 的 embedding-3 为付费接口）
   - Built-in: OpenAI, Azure, Ollama. Others require auto-scaffolding (see Step 2.5).
   - ⚠️ 换 embedding 模型后需重新灌库（向量空间不兼容）。

3. **Vision** — Enable vision/image captioning?
   - Options: `Yes`, `No`
   - Recommended: `Yes`（GLM 用户可用免费的 glm-4v-flash）

4. **Rerank** — Enable reranking?
   - Options: `No (fastest)`, `Cross-Encoder (local, free)`, `LLM-based (chat model, slow)`, `Jev (TypeSafe API, best quality/speed)`
   - Recommended: `Jev`（BEIR 实测 6 格 5 个第一，~$0.17/千次，无 JSON 解析失败路径；见 reports/BENCHMARK_REPORT.md）
   - Cross-Encoder 在 Apple Silicon CPU 上必须串行执行（torch MPS 并发段错误），代码已处理。

### Batch 1.5: Jev credentials (if Jev rerank selected)

- Ask: TypeSafe API Key（typesafe.ai 申请）
- Jev 不需要 embedding/LLM 配套，独立使用 `TYPESAFE_API_KEY`

### Batch 2: Credentials (based on Batch 1 answers)

Ask for credentials based on selected providers. Refer to [references/provider_profiles.md](references/provider_profiles.md) for required fields per provider.

**If OpenAI selected:**
- Ask: OpenAI API Key
- Ask: LLM model (default: `gpt-4o`)
- Ask: Embedding model (default: `text-embedding-ada-002`)

**If Azure OpenAI selected:**
- Ask: Azure API Key
- Ask: Azure Endpoint URL
- Ask: LLM deployment name (default: `gpt-4o`)
- Ask: Embedding deployment name (default: `text-embedding-ada-002`)

**If DeepSeek selected:**
- Ask: DeepSeek API Key
- Ask: Embedding provider separately (DeepSeek has no embeddings — must use OpenAI/Ollama)

**If GLM selected:**
- Ask: GLM API Key（open.bigmodel.cn 获取）
- Ask: LLM model (default: `glm-4-flash`，免费档)
- Embedding: GLM 的 embedding-3 为**付费**接口（无余额报 429/1113）——推荐 embedding 仍选 Ollama
- 凭证写入 `.env` 的 `GLM_API_KEY`，settings.yaml 用 `${GLM_API_KEY}` 占位

**If Jev rerank selected (Batch 1.5):**
- Ask: TypeSafe API Key
- 凭证写入 `.env` 的 `TYPESAFE_API_KEY`，settings.yaml 的 rerank 块用 `${TYPESAFE_API_KEY}` 占位

**If Ollama selected:**
- Ask: Ollama base URL (default: `http://localhost:11434`)
- Ask: LLM model name (default: `llama3`)
- Ask: Embedding model name (default: `nomic-embed-text`)
- Verify Ollama is running: `curl http://localhost:11434/api/tags` or equivalent

**If Qwen selected:**
- Ask: Qwen API Key (DashScope)
- Ask: LLM model (default: `qwen-turbo`)
- Ask: Embedding model (default: `text-embedding-v3`) — if Qwen also chosen for embedding
- Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1` (OpenAI-compatible)

**If Gemini selected:**
- Ask: Gemini API Key (Google AI Studio)
- Ask: LLM model (default: `gemini-2.0-flash`)
- Ask: Embedding model (default: `text-embedding-004`) — if Gemini also chosen for embedding
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/` (OpenAI-compatible)

### Batch 3: Vision Credentials (if vision enabled)

Vision LLM has its **own independent config section** (`vision_llm`) with separate `provider`, `api_key`, `azure_endpoint`, etc. Do NOT assume it shares credentials with the main LLM.

Ask up to 2 questions:

1. **Vision Provider** — Which provider for vision/image captioning?
   - Options: same as LLM provider list, but default to user's LLM choice
   - Built-in vision: OpenAI, Azure, GLM, Ollama. Others require auto-scaffolding.

2. **Vision credentials** — based on provider:
   - If vision provider == LLM provider: ask "Reuse the same API key/endpoint for vision?" (default: Yes)
     - If Yes: copy LLM credentials to vision config
     - If No: ask for separate vision API key / endpoint
   - If vision provider != LLM provider: ask for vision-specific API key / endpoint / model

Vision models per provider (recommended model listed first):

| Provider | Recommended | Other Options | Notes |
|----------|------------|---------------|-------|
| GLM | `glm-4v-flash` | `glm-4v-plus` | glm-4v-flash 免费档，中文图表友好 |
| OpenAI | `gpt-4o` | `gpt-4o-mini`, `gpt-4-turbo` | gpt-4o-mini 更便宜，适合简单图片描述 |
| Azure | `gpt-4o` | `gpt-4o-mini`, `gpt-4-turbo` | 需要 deployment_name + azure_endpoint |
| Ollama | `llava` | `llava:13b`, `llava:34b`, `llava-llama3`, `bakllava`, `moondream` | moondream 最轻量，llava:34b 质量最高 |
| Qwen | `qwen-vl-max` | `qwen-vl-plus`, `qwen2.5-vl-72b-instruct`, `qwen2.5-vl-7b-instruct` | qwen-vl-plus 性价比高 |
| Gemini | `gemini-2.0-flash` | `gemini-1.5-pro`, `gemini-2.0-flash-lite`, `gemini-1.5-flash` | gemini-1.5-pro 质量最高但较慢 |
| DeepSeek | ❌ 无 Vision 模型 | — | 需选择其他 provider 作为 Vision LLM |

---

## Step 2.5: Scaffold Unimplemented Providers (if needed)

If the user selected a provider not yet built-in (e.g., Qwen, Gemini, or any custom provider), auto-scaffold the implementation before proceeding to config generation.

Refer to [references/new_provider_guide.md](references/new_provider_guide.md) for the complete scaffolding procedure.

Summary:
1. Create LLM class in `src/libs/llm/{name}_llm.py` (extend `BaseLLM`)
2. Create Embedding class in `src/libs/embedding/{name}_embedding.py` (extend `BaseEmbedding`) — if needed
3. Create Vision LLM class in `src/libs/llm/{name}_vision_llm.py` (extend `BaseVisionLLM`) — if needed
4. Register in `src/libs/llm/__init__.py` and `src/libs/embedding/__init__.py`
5. Add `base_url` field to `LLMSettings` / `EmbeddingSettings` if the provider uses a custom endpoint
6. Install provider SDK: `pip install <sdk>` if needed
7. Add provider profile to `references/provider_profiles.md`

Many providers (Qwen, Gemini, Groq, Mistral, etc.) are OpenAI-compatible — simply subclass `OpenAILLM` / `OpenAIEmbedding` and override `DEFAULT_BASE_URL` + auth logic.

---

## Step 3: Generate Config

Read the template from [references/settings_template.yaml](references/settings_template.yaml) and fill in values based on user answers.

Key rules:
- **凭证一律走 `.env`**：settings.yaml 里写 `${ENV_VAR}` 占位符（如 `${GLM_API_KEY}`），真实 key 写入仓库根 `.env`（已被 gitignore）。settings 启动时自动加载并展开，缺失会 fail-fast
- Look up `dimensions` from the model→dimensions table in [references/provider_profiles.md](references/provider_profiles.md)
- For Ollama: set `base_url`, leave `api_key`/`azure_endpoint`/`deployment_name` empty
- For OpenAI: leave `azure_endpoint`/`deployment_name`/`api_version` empty
- If vision disabled: set `vision_llm.enabled: false`
- For rerank: set `enabled`, `provider`, and `model` accordingly (`none | cross_encoder | llm | jev`；jev 建议配 `min_score: 1.0`)

Write the generated config to `config/settings.yaml`.

Also ensure required directories exist:

```powershell
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/db/chroma', 'data/images/default', 'logs', 'config/prompts']]"
```

---

## Step 4: Install Dependencies

```powershell
pip install -e ".[dev]"
```

If specific providers need extra packages:
- **Cross-Encoder rerank**: `pip install sentence-transformers`
- **Streamlit dashboard**: `pip install streamlit`
- **OpenAI**: `pip install openai`

Verify critical imports:

```powershell
python -c "import chromadb; import mcp; import yaml; print('Core deps OK')"
python -c "import streamlit; print('Streamlit OK')"
python -c "import openai; print('OpenAI SDK OK')"
```

---

## Step 5: Validate Configuration

Test that the config loads correctly:

```powershell
python -c "from src.core.settings import load_settings; s = load_settings(); print(f'Config OK: LLM={s.llm.provider}/{s.llm.model}, Embed={s.embedding.provider}/{s.embedding.model}')"
```

If this fails, enter **auto-fix loop**:

### Auto-Fix Loop (≤3 rounds)

```
Round 0..2:
  Read error message
  Diagnose root cause (missing field, wrong type, bad provider name, etc.)
  Fix config/settings.yaml or install missing dependency
  Re-validate
  If pass → continue to Step 6
  If fail → next round
```

Common fixes:
- `SettingsError: Missing required field` → add the field to settings.yaml
- `ModuleNotFoundError` → `pip install <package>`
- `Connection refused` (Ollama) → inform user to start Ollama service
- Wrong `dimensions` value → look up correct value from provider_profiles.md

If 3 rounds fail, report the issue to the user with diagnosis and ask for help.

---

## Step 6: Launch Dashboard

```powershell
python scripts/start_dashboard.py --port 8501
```

Run this as a **background process**. Wait a few seconds, then verify it's accessible:

```powershell
python -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:8501/_stcore/health')
    print('Dashboard is running!' if r.status == 200 else f'Status: {r.status}')
except Exception as e:
    print(f'Dashboard not yet ready: {e}')
"
```

If the dashboard fails to start, enter auto-fix loop:
- Read the error output from the background terminal
- Common issues: missing `streamlit`, port already in use, import errors
- Fix and retry

---

## Step 7: Usage Guide

After successful launch, present this to the user:

```
🎉 Setup Complete!

Dashboard: http://localhost:8501

Quick Start:
  1. Ingest documents:  python scripts/ingest.py <path-to-pdf-or-folder>
  2. Query:             python scripts/query.py --query "your question here"
  3. Dashboard:         python scripts/start_dashboard.py
  4. MCP Server (stdio): python -m src.mcp_server.server
  5. MCP Server (HTTP):  python -m src.mcp_server.server --http --port 8000  # 远程/多客户端

Configuration: config/settings.yaml
Logs:          logs/traces.jsonl

Provider: {provider} / Model: {model}
```

Adapt the message based on the user's chosen providers and language (Chinese if user communicates in Chinese).
