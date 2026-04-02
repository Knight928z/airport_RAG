# 机场知识库 RAG 助手

一个面向机场业务文档的中文 RAG（检索增强生成）助手：

- 支持文档入库：PDF / Markdown / TXT / 无后缀纯文本文档（如 `出发指南-国内出发`）
- 基于 Chroma 的向量检索
- 使用 LangChain 增强语义切分与问答生成链路
- 回答自带可追溯引用
- 输出风格符合机场业务专业规范（结论、依据、执行建议、风险提示）
- 提供 FastAPI 接口与 CLI

## 目录结构

- `src/airport_rag/`：核心实现
- `tests/`：基础测试
- `data/`：默认向量库目录（运行后自动生成）

### 文档目录约定（机场 / 航司）

系统会基于 `data/documents` 的一级子目录自动识别规则归属：

- `data/documents/airport/`：广州白云机场统一规定
- `data/documents/CZ/`：南航（China Southern）航司规定
- `data/documents/<其他航司代码>/`：其他航司同级目录（例如 `MU`、`CA`），无需改代码即可扩展

检索策略会自动应用以下优先级：

- 问题明确包含航司（如“南航”“CZ”）时：优先并限制匹配对应航司目录
- 问题为机场运行规范（海关、边检、航站楼等）时：优先机场目录
- 问题为泛航司政策（如“航空公司行李规定”）时：优先航司目录

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

## API 概览

- `GET /health`：健康检查
- `GET /app`：普通人员可用的问答前端页面
- `GET /admin`：管理人员文档后台页面（可视化文档管理）
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

## 开发提交流程（自动写日志 + 规范提交 + 推送）

仓库内提供一键提交脚本：`scripts/commit_and_push.sh`。

- 功能：
  - 自动 `git add -A`
  - 自动向 `CHANGELOG_LOG.md` 追加时间戳与本次变更文件
  - 校验 commit message 是否符合 Conventional Commits
  - 自动 `git commit` 并 `git push origin main`

- 用法：

```bash
bash scripts/commit_and_push.sh "feat(api): add feedback loop"
```

可选第二个参数用于写入日志备注：

```bash
bash scripts/commit_and_push.sh "fix(rules): improve battery evidence" "避免引用酒精条款"
```
