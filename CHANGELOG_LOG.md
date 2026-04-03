# Change Log

> 记录代码改动内容与时间戳（按时间倒序追加）。

## 2026-04-03 08:19:49 +0800

- 提交：`fix(rules): support airport hotline contact retrieval`
- 变更文件：
  - `src/airport_rag/rules.py`
  - `tests/test_service_rerank.py`
- 备注：新增机场客服热线问题规则命中与回归测试

## 2026-04-03 08:10:57 +0800

- 提交：`fix(rag): improve airport routing and vector query fallback stability`
- 变更文件：
  - `src/airport_rag/rules.py`
  - `src/airport_rag/service.py`
  - `src/airport_rag/vector_store.py`
  - `tests/test_service_rerank.py`
- 备注：修复国内出发与150Wh问答准确性并提升检索容错

## 2026-04-03 07:53:39 +0800

- 提交：`feat(feedback): add rating/correction loop and uncovered-question logging`
- 变更文件：
  - `.githooks/commit-msg`
  - `.github/copilot-instructions.md`
  - `CHANGELOG_LOG.md`
  - `README.md`
  - `scripts/commit_and_push.sh`
  - `src/airport_rag/api.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_feedback_logging.py`
- 备注：新增自动提交脚本与commit message规范校验

## 2026-04-03 07:50:37 +0800

- 新增：用户反馈闭环能力。
  - `POST /feedback` 支持点赞/点踩与纠错答案提交。
  - `AskResponse` 新增 `answer_id` 字段，用于每条回答反馈关联。
- 新增：未覆盖问题自动记录。
  - 对 `low-confidence` / `index-empty` 的问答自动写入 `data/feedback/uncovered_questions.jsonl`。
  - 对用户点踩或纠错提交的问答，同步写入未覆盖问题队列用于知识库扩充优先级。
- 新增：前端问答页反馈控件。
  - 在 `static/index.html` 增加点赞、点踩、纠错输入框与提交按钮。
- 新增：回归测试 `tests/test_feedback_logging.py`，覆盖低置信自动记录与反馈日志写入。

## 2026-04-03 07:36:02 +0800

- 新增：`CHANGELOG_LOG.md`，用于持续记录代码改动与时间戳。
- 新增：`.gitignore`，忽略本地环境与数据产物（如 `.env`、`data/`、`.venv/`、临时脚本等）。
- 调整：RAG 回答准确性策略优化（规则抽取与证据引用修正），并补充回归测试。
