# Change Log

> 记录代码改动内容与时间戳（按时间倒序追加）。

## 2026-04-09 12:26:39 +0800

- 提交：`feat(cli): enrich command interface and built-in self-test modes`
- 变更文件：
  - `src/airport_rag/cli.py`
- 备注：优化 CLI 接口内容：补充更清晰的 help/示例说明；`ingest` 支持默认路径与文本/JSON 双输出；`ask` 增加 answer-only、citations、meta 与 JSON 输出模式；新增 `self-test` 子命令并支持 default/batch1/batch2/all 题集批次、limit 与失败样例输出，直接复用集中题库定义进行本地回归。

## 2026-04-09 09:49:55 +0800

- 提交：`fix(retrieval): avoid realtime-archive citations for generic arrival intent`
- 变更文件：
  - `src/airport_rag/service.py`
  - `tests/test_service_rerank.py`
- 备注：修复“国际到达是什么”误引用 `data/documents/airport/实时航班/*.md` 存档问题：仅当问题具备实时航班意图（明确航班号或“航班+实时状态”语义）时才允许实时航班存档参与检索；补充正反两类回归测试（通用到达问题排除实时存档、实时航班问题保留实时存档）。

## 2026-04-08 01:29:46 +0800

- 提交：`refactor(eval): centralize tested question banks and harden customs fallback`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/eval_cases.py`
  - `src/airport_rag/rules.py`
  - `tests/test_service_rerank.py`
- 备注：将已测试题目统一集中到 `eval_cases.py`（含 `ALL_TESTED_CASES` 与两组不重复 `200` 题分片）；`/self-test` 改为直接复用集中题库；修复海关热线与海关排队时长误答为 low-confidence 防护；新增回归测试并完成“第二组不一样的200题”回归（200/200）。

## 2026-04-08 01:18:30 +0800

- 提交：`fix(rules): tighten battery parsing and add regression guards`
- 变更文件：
  - `.gitignore`
  - `src/airport_rag/rules.py`
  - `tests/test_service_rerank.py`
- 备注：修复 `120wh` 这类无“充电宝/锂电池”关键词问法仍应命中电池规则；增强仅给 `mAh` 场景按 3.7V 估算 Wh 并在结论中显式提示估算口径；新增两条回归测试覆盖上述场景。

## 2026-04-08 01:05:40 +0800

- 提交：`fix(rules): stabilize battery answers and expand eval regression`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/eval_cases.py`
  - `src/airport_rag/rules.py`
  - `src/airport_rag/static/index.html`
- 备注：前端反馈交互改为答后显示+点踩后单次纠错提交，实时字段映射改为前端传统内置表；修复锂电池/充电宝问答在无证据召回时的规则兜底（含mAh+电压换算），新增多条电池专项eval样例并完成回归验证。

## 2026-04-05 15:58:18 +0800

- 提交：`fix(realtime-ui): enforce confirm flow and server label mapping`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：修复实时航班详情仍显示英文键名与二次确认未出现问题：ask新增enable_realtime开关默认关闭自动实时，前端统一走二次确认；/flight/realtime返回realtime_flight_labels并优先用于详情渲染；补充并通过回归测试，刷新egg-info元数据。

## 2026-04-05 15:54:32 +0800

- 提交：`fix(mapping): harden realtime field label resolution`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：修复实时航班详细字段映射稳定性：前端增加规范化/无缓存加载/键形态兼容匹配，后端解析字段表时清理引号与反引号；新增回归测试覆盖FlightNo/FlightState/AssistFlightState映射场景。

## 2026-04-05 15:48:25 +0800

- 提交：`fix(ui): ensure realtime detail labels use mapping table`
- 变更文件：
  - `src/airport_rag/static/index.html`
  - `src/airport_rag_assistant.egg-info/PKG-INFO`
  - `src/airport_rag_assistant.egg-info/SOURCES.txt`
- 备注：前端实时详情字段名映射增加规范化匹配与加载兜底，确保FlightNo/FlightState/AssistFlightState显示字段表中文名；同步刷新egg-info元数据(PKG-INFO/SOURCES.txt)。

## 2026-04-04 11:30:39 +0800

- 提交：`fix(realtime): require confirm for non-flight-no queries`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：ask接口仅对含明确航班号的问题自动调用实时航班；前端新增‘系统暂无此航班信息，是否查询？’二次确认按钮后再触发/flight/realtime；补充防误触回归测试。

## 2026-04-04 11:22:45 +0800

- 提交：`refactor(realtime): load field labels from documents mapping`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：移除前端硬编码字段映射，新增/flight/field-mappings从data/documents/airport/实时航班/航班字段读取映射并前端动态加载；补充接口测试。

## 2026-04-04 11:12:42 +0800

- 提交：`feat(realtime): map detailed flight fields and add detail toggle UI`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/realtime_flight.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：实时航班详情支持字段中文映射与优先排序；默认简洁卡片展示，按钮切换详细字段；后端补充realtime_flight_details并兼容旧返回格式。

## 2026-04-04 11:01:58 +0800

- 提交：`feat(eval): add 100 more self-test cases and optimize rules`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/eval_cases.py`
  - `src/airport_rag/rules.py`
- 备注：新增25个seed题（展开后+100题，总200题）；修复边检窗口/排队与登机口开放时间低置信口径、补充100Wh充电宝随身规则、self-test预热入库；回归200/200通过

## 2026-04-04 10:51:27 +0800

- 提交：`refactor(eval): centralize qa test cases into one module`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/eval_cases.py`
- 备注：将问答评测用例集中到 src/airport_rag/eval_cases.py，api自测与评测流程统一引用该模块

## 2026-04-04 10:47:41 +0800

- 提交：`fix(eval): improve random-50 accuracy on edge intents`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/rules.py`
- 备注：随机50题评测从45/50提升到50/50；优化吸烟区低置信、春秋客服电话谨慎答复、打火机携带规则兜底；self-test将index-empty并入低置信口径

## 2026-04-04 10:38:45 +0800

- 提交：`fix(realtime): parse chinese flight-no and avoid unknown records`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/realtime_flight.py`
  - `tests/test_realtime_flight_api.py`
- 备注：修复CZ325航班状态识别失败：支持中文黏连航班号提取；实时工具报错时不写入错误卡片；UNKNOWN落盘回退提取问题中的航班号；补充回归测试

## 2026-04-04 10:33:02 +0800

- 提交：`fix(realtime): support MCP redirect and accept headers`
- 变更文件：
  - `src/airport_rag/realtime_flight.py`
- 备注：使用本地私有API Key联调，修复307重定向与406 Accept头问题；航班号查询优先searchFlightsByNumber并自动补当天日期；增强文本内嵌结构化结果解析

## 2026-04-04 10:24:35 +0800

- 提交：`feat(realtime): persist flight status answers into kb`
- 变更文件：
  - `README.md`
  - `src/airport_rag/api.py`
  - `tests/test_realtime_flight_api.py`
- 备注：航班状态问答命中实时模块后自动新建文档到 data/documents/airport/实时航班 并立即入库；补充对应测试与文档

## 2026-04-04 10:17:33 +0800

- 提交：`feat(realtime): integrate MCP flight card and local private secrets`
- 变更文件：
  - `.env.example`
  - `README.md`
  - `src/airport_rag/api.py`
  - `src/airport_rag/config.py`
  - `src/airport_rag/realtime_flight.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_realtime_flight_api.py`
- 备注：接入 VariFlight MCP 实时航班标准化卡片；新增本地私有明文密钥文件加载(data/private)并保持不入Git

## 2026-04-04 08:09:41 +0800

- 提交：`feat(admin): add ocr manual review panel`
- 变更文件：
  - `README.md`
  - `src/airport_rag/api.py`
  - `src/airport_rag/static/admin.html`
  - `src/airport_rag/static/ocr_review.html`
  - `tests/test_admin_ocr_review.py`
- 备注：新增 OCR 人工校对面板、API、测试与文档

## 2026-04-04 07:57:27 +0800

- 提交：`feat(admin): add OCR support for image document ingestion`
- 变更文件：
  - `Dockerfile`
  - `README.md`
  - `requirements.txt`
  - `src/airport_rag/api.py`
  - `src/airport_rag/ingest.py`
  - `src/airport_rag/static/admin.html`
  - `tests/test_admin_ocr_upload.py`
  - `tests/test_ingest_documents.py`
- 备注：admin批量上传支持图片OCR，自动生成.ocr.md侧车文本并参与检索；ingest新增图片解析链路；Docker安装tesseract中文包并补测试与README说明

## 2026-04-03 09:58:42 +0800

- 提交：`refactor(patches): move patch storage to data/patches root`
- 变更文件：
  - `README.md`
  - `src/airport_rag/api.py`
  - `tests/test_admin_patches.py`
  - `tests/test_feedback_logging.py`
- 备注：补丁目录从data/documents/*/patches迁移至data/patches，新增兼容迁移逻辑并更新治理测试与README说明

## 2026-04-03 09:52:43 +0800

- 提交：`feat(admin): add patches governance dashboard and review merge action`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/prompts.py`
  - `src/airport_rag/static/admin.html`
  - `src/airport_rag/static/patches.html`
  - `tests/test_admin_patches.py`
- 备注：新增/admin/patches管理页与统计接口，支持主题补丁数量/去重率/合并次数展示及一键审核回写主文档并清理补丁；同步优化prompts专业表达

## 2026-04-03 09:47:19 +0800

- 提交：`feat(feedback): add patch governance with dedup tiering and merge`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_feedback_logging.py`
- 备注：新增补丁治理：同问题去重、按主题与置信分层归档、阈值触发自动合并到知识补丁合并稿并归档轮转，前端展示patch_status

## 2026-04-03 09:36:21 +0800

- 提交：`feat(feedback): add auto knowledge patch closed-loop`
- 变更文件：
  - `src/airport_rag/api.py`
  - `src/airport_rag/schemas.py`
  - `src/airport_rag/static/index.html`
  - `tests/test_feedback_logging.py`
- 备注：点踩或纠错自动写入用户纠错补丁文档并触发自动入库，反馈接口返回patch状态与迭代答案，前端同步展示闭环结果

## 2026-04-03 09:27:58 +0800

- 提交：`fix(api): keep admin pdf preview inline instead of auto-download`
- 变更文件：
  - `src/airport_rag/api.py`
  - `tests/test_admin_binary_upload.py`
- 备注：admin /admin/docs/raw 改为 inline Content-Disposition，修复前端 PDF 预览自动跳下载；补充回归测试校验响应头

## 2026-04-03 09:17:48 +0800

- 提交：`fix(rules): improve accuracy for 8 key passenger policy questions`
- 变更文件：
  - `src/airport_rag/rules.py`
  - `src/airport_rag/service.py`
  - `tests/test_service_rerank.py`
- 备注：新增保险退保专项规则、文档直读兜底、意图检索回退与多场景规则，修复军残票/公务舱行李/9C餐食与保险/旅游团入境/国内液体/锂电池托运/机场热线问答准确性

## 2026-04-03 08:36:34 +0800

- 提交：`refactor(rules): centralize maps keywords and hints from service`
- 变更文件：
  - `src/airport_rag/rules.py`
  - `src/airport_rag/service.py`
- 备注：将zhtoen/entozh、intent rules、topic map、source-policy keywords、carrier alias hint集中到rules.py，service仅保留编排与调用

## 2026-04-03 08:30:33 +0800

- 提交：`fix(rules): add cabin baggage allowance rule for class queries`
- 变更文件：
  - `src/airport_rag/rules.py`
  - `tests/test_service_rerank.py`
- 备注：修复公务舱行李额问答误命中补偿金额的问题，新增回归测试锁定30公斤答案

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
