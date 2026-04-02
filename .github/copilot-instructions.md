# 机场知识库 RAG 助手 - 执行清单

- [x] Verify that the copilot-instructions.md file in the .github directory is created.  
  已创建并开始维护本清单。

- [x] Clarify Project Requirements  
  需求已明确：机场业务文档 RAG、准确回答、带引用、符合机场专业规范。

- [x] Scaffold the Project  
  已在当前目录创建 Python 项目结构与核心代码。

- [x] Customize the Project  
  已实现：文档入库（PDF/MD/TXT）、Chroma 检索、引用溯源、规范化回答、FastAPI 接口、CLI。

- [x] Install Required Extensions  
  已检查：未提供必须安装的扩展，按规则跳过。

- [x] Compile the Project  
  已完成依赖安装与测试验证（`pytest` 通过）。

- [x] Create and Run Task  
  已创建并运行 VS Code 任务：`Run Airport RAG API`。

- [x] Launch the Project  
  已完成本地启动与 Docker 启动验证（`/health`、`/app`、`/admin` 均可访问）。

- [x] Ensure Documentation is Complete  
  已补齐并更新 `README.md`、`.env.example`、`requirements.txt`、`pyproject.toml` 与本清单。

---

执行原则：
- 使用当前目录作为项目根。
- 优先最小改动与可运行性。
- 完成每个步骤后更新清单。