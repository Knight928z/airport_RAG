# 机场知识库 RAG 助手

一个面向机场业务文档的中文 RAG（检索增强生成）助手：

- 支持文档入库：PDF / Markdown / TXT / 无后缀纯文本文档（如 `出发指南-国内出发`）
- 支持图片文档 OCR 入库：PNG / JPG / JPEG / WEBP / TIFF / BMP（管理端批量上传后自动生成 OCR 文本侧车文件）
- 基于 Chroma 的向量检索
- 支持可配置 reranker（cross-encoder / heuristic fallback）优化检索重排
- 支持 LoRA 微调任务管理（后台提交 + 状态追踪）
- 使用 LangChain 增强语义切分与问答生成链路
- 回答自带可追溯引用
- 输出风格符合机场业务专业规范（结论、依据、执行建议、风险提示）
- 提供 FastAPI 接口与 CLI

## 目录结构

- `src/airport_rag/`：核心实现
- `tests/`：基础测试
- `data/`：运行时数据目录（文档、补丁、反馈、向量库）

### 文档目录约定（机场 / 航司）

系统会基于 `data/documents` 的一级子目录自动识别规则归属：

- `data/documents/airport/`：广州白云机场统一规定
- `data/documents/CZ/`：南航（China Southern）航司规定
- `data/documents/<其他航司代码>/`：其他航司同级目录（例如 `MU`、`CA`），无需改代码即可扩展

检索策略会自动应用以下优先级：

- 问题明确包含航司（如“南航”“CZ”）时：优先并限制匹配对应航司目录
- 问题为机场运行规范（海关、边检、航站楼等）时：优先机场目录
- 问题为泛航司政策（如“航空公司行李规定”）时：优先航司目录

### 补丁与反馈目录约定

- `data/patches/`：用户纠错补丁（按航司/主题/置信分层归档）
  - 结构：`data/patches/{airport|航司代码}/{主题}/{high|medium|low}/{YYYY-MM}-用户纠错补丁.md`
- `data/feedback/`：反馈与治理日志
  - `answer_feedback.jsonl`：用户点赞/点踩/纠错原始日志
  - `patch_registry.jsonl`：补丁去重指纹注册表
  - `patch_audit.jsonl`：补丁治理审计日志（applied/deduplicated/merged/review-merged）
  - `uncovered_questions.jsonl`：未覆盖问题记录

### 本地私有明文 API（仅本机，不入 Git）

- 项目默认会额外读取一个本地私有配置文件：`data/private/local.secrets.env`
- 该目录在 `.gitignore` 下，不会被提交；适合存放明文 API Key（如 MCP、第三方航班服务）
- 可通过 `RAG_PRIVATE_SECRETS_FILE` 指定自定义私有文件路径

示例（`data/private/local.secrets.env`）：

```bash
VARIFLIGHT_MCP_URL=https://ai.variflight.com/servers/aviation/mcp/
VARIFLIGHT_MCP_API_KEY=your-local-key
VARIFLIGHT_MCP_TIMEOUT=10
```

## 快速开始

1. 安装依赖
2. 复制环境变量模板：`.env.example -> .env`
3. 入库机场文档
4. 启动 API 或使用 CLI 提问

> 说明：`data/documents` 目录下无文件后缀名的中文规章文档也会自动识别并入库。

### 性能相关环境变量（建议）

- `RAG_ALLOW_MODEL_DOWNLOAD`（默认 `false`）：
  - `false`：仅加载本地已缓存 embedding 模型，未命中即快速回退到 hashing，避免网络超时拖慢首问。
  - `true`：允许在线拉取模型（首次可能较慢，受外网影响）。

- `RAG_RERANKER_BACKEND`（默认 `cross_encoder`）：
  - `cross_encoder`：优先使用 cross-encoder 模型重排
  - `heuristic`：使用启发式重排（无需模型）

- `RAG_RERANKER_MODEL`（默认 `BAAI/bge-reranker-v2-m3`）：
  - 当 `RAG_RERANKER_BACKEND=cross_encoder` 时生效。

## API 概览

- `GET /health`：健康检查
- `GET /app`：普通人员可用的问答前端页面
- `GET /admin`：管理人员文档后台页面（可视化文档管理）
- `GET /admin/ai-lab`：AI 调优实验窗（LoRA + reranker 可视化）
- `GET /admin/patches`：补丁治理面板（统计与审核）
- `GET /admin/ocr-review`：OCR 人工校对面板（侧车文本复核与入库同步）
- `POST /flight/realtime`：实时航班查询（MCP，返回标准化航班卡片字段）
- `POST /ingest`：文档入库
- `POST /ingest/default`：一键同步 `data/documents` 到知识库
- `POST /ask`：RAG 问答
- `GET /self-test`：100题混合集回归（应答 + 应拒答），包含按主题分项分数（海关/边防/出发/行李/航司）

### 管理后台（文档 CRUD + 自动分类）

管理端 API：

- `GET /admin/docs`：列出当前知识库文档清单
- `GET /admin/docs/content?path=...`：读取文档内容
- `POST /admin/docs/classify`：预览新增文档自动分类（`airport` 或对应航司代码目录）
- `POST /admin/docs`：新增文档（支持自动分类与自动同步入库）
- `PUT /admin/docs/content?path=...`：更新文档内容
- `DELETE /admin/docs?path=...`：删除文档
- `GET /admin/tree`：目录树视图数据（用于树形浏览）
- `GET /admin/search?q=...`：关键词搜索（路径 + 内容片段）
- `POST /admin/docs/bulk`：批量上传（支持拖拽上传，多文件自动分类）
  - 当上传图片文件时，会自动执行 OCR，并生成同目录侧车文本：`<原文件名>.ocr.md`，该文本会参与后续检索与问答。

实时航班（MCP）说明：

- 当问题包含航班号（如 `MU2456`）或实时航班关键词（如“延误/起飞/到达/动态”）时，`POST /ask` 会优先调用 VariFlight MCP。
- `/ask` 的响应会增加 `realtime_flight` 字段（可能为 `null`），用于前端实时航班卡片展示。
- 标准化字段：`flight_no`、`planned_departure`、`actual_departure`、`planned_arrival`、`actual_arrival`、`delay_minutes`、`terminal`、`gate`、`status`。
- 命中实时航班问答后，会自动新建/追加记录到 `data/documents/airport/实时航班/`，并立即触发入库，供后续检索追溯。

建议使用环境变量配置 MCP（不要在代码中硬编码 API Key）：

- `VARIFLIGHT_MCP_URL`（默认 `https://ai.variflight.com/servers/aviation/mcp/`）
- `VARIFLIGHT_MCP_API_KEY`（必填，推荐）
- `VARIFLIGHT_MCP_TIMEOUT`（默认 `10` 秒）

OCR 人工校对 API：

- `GET /admin/ocr-review/items`：列出 OCR 侧车文本、关联原文件、是否过期（原文件更新后未复核）
- `PUT /admin/ocr-review/content?path=...`：保存 OCR 校对文本（可选自动同步入库）

补丁治理 API：

- `GET /admin/patches/stats`：查看主题补丁数量、去重率、合并次数
- `POST /admin/patches/review-merge?cleanup=true`：一键审核后回写主文档并清理补丁（可关闭 cleanup 仅回写）

AI 调优实验窗 API：

- Reranker：
  - `GET /admin/ai-lab/options`：获取推荐 backend / 模型可选项（前端下拉使用）
  - `GET /admin/reranker/config`：读取当前重排配置
  - `PUT /admin/reranker/config`：更新重排后端与模型
  - `POST /admin/reranker/preview`：输入问题与候选文本，返回打分与排序预览

- LoRA：
  - `POST /admin/lora/train`：提交 LoRA 微调任务
  - `GET /admin/lora/jobs`：查看任务列表
  - `GET /admin/lora/jobs/{job_id}`：查看单任务状态

默认推荐模型可选项：

- Reranker backend：`cross_encoder`、`heuristic`
- Reranker model：
  - `BAAI/bge-reranker-v2-m3`（默认）
  - `BAAI/bge-reranker-base`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `jinaai/jina-reranker-v2-base-multilingual`
- LoRA base model：
  - `Qwen/Qwen2.5-0.5B-Instruct`（默认）
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

> 调优页面依然支持手工输入自定义模型名；上面列表是开箱即用的推荐组合。

## AI 调优使用方法（Reranker + LoRA）

推荐按“先重排、后微调、再回归”的顺序执行：

1. 进入 `GET /admin/ai-lab` 打开调优实验窗。

1. 在 **Reranker 调试** 中，先点击页面内置示例（如“机场 vs 航司”“海关申报”），执行 `重排预览` 后重点看 Top1/Top3 与 `top_score_gap`，满意后再点“保存配置”。

1. 在 **LoRA 微调任务** 中，优先使用小样本热身（10~50 条先验证流程）；参数建议先保持默认：`epochs=0.5~1`、`batch_size=1~2`、`learning_rate=2e-4`；提交后观察任务状态 `queued -> running -> succeeded/failed`。

1. 任务成功后做小回归，重点抽查高频问题（行李、海关、边检、实时航班边界问题），并保留至少 1 组固定测试题与历史结果做横向对比。

### LoRA 训练文件路径说明（Docker 部署必看）

- 若服务跑在 Docker 容器内，训练文件路径应使用容器路径：`/app/data/...`
- 常见映射：本机 `/Users/<you>/RAG/data/lora/train_xxx.jsonl` 对应容器 `/app/data/lora/train_xxx.jsonl`
- 调优页面已内置路径映射提示与预览，推荐直接使用 `/app/...` 路径提交

### 推荐的首轮微调主题

- 先选单一高频主题（如“海关申报”或“行李托运”）做一轮小样本验证；
- 单主题验证通过后，再扩展到多主题联合训练，避免一次性范围过大导致定位困难。

调优参数建议（起步值）：

- LoRA：`epochs=1~2`、`batch_size=2~8`、`learning_rate=1e-4~3e-4`
- Reranker：优先 `BAAI/bge-reranker-v2-m3`，资源受限时可切 `heuristic`

建议的验收标准（可按团队指标调整）：

- 关键业务题 Top1 命中率提升或持平；
- `low-confidence` 比例不上升；
- 无新增误答（特别是“非实时问题误引用实时航班存档”这类历史问题）。

新增文档的自动分类优先级：

1. 识别 `scope_hint/carrier_hint`
2. 基于 `data/documents/airport/航司代码` 自动匹配航司名称与 IATA 代码
3. 未命中航司时归档到 `airport/`

## Docker 打包与运行

已提供：

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`

启动（容器内即提供普通前端 `/app` 与管理前端 `/admin`）：

```bash
docker compose up -d --build
```

访问：

- 普通前端：`http://127.0.0.1:8000/app`
- 管理前端：`http://127.0.0.1:8000/admin`

> OCR 说明：Docker 运行已内置 `tesseract-ocr` 与 `chi_sim` 中文语言包；本地裸机运行请确保系统可用 Tesseract（并建议安装中文语言包）。

停止：

```bash
docker compose down
```

## 面向普通人员的前端使用

启动 API 后，在浏览器访问 `http://127.0.0.1:8000/app`：

- 点击“同步最新文档”即可重建知识库索引
- 或勾选“提问前自动同步文档”，每次提问前自动读取最新文档
- 输入问题后点击“开始问答”
- 页面会显示规范化回答和引用依据

前端还提供了：

- 常见问题快捷按钮（出发、海关、充电宝）
- 引用条数可选（Top-K）
- 置信状态显示（`retrieval-extractive` / `low-confidence`）

## 专业规范输出约定

所有回答按以下结构组织：

1. 结论
2. 依据（含引用）
3. 执行建议
4. 风险提示

如果检索证据不足，会明确提示“需人工复核”，避免越权推断。
