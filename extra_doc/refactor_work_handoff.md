# 重构协作约定 & 工作交接日志（Codex）

本文件用于记录我（Codex）在本仓库按 `extra_doc/refactor_plan_to_new_arch.md` 推进重构时的**协作约定**与**每次提交的工作日志**，便于你随时接手/回溯。

---

## 1) 变更与提交（Commit）约定

1. **每次修改只聚焦一个部分（一个 Milestone/一个子模块/一个问题）**  
   - 避免“顺手改一下”导致 PR/commit 不可审计、难回滚。
2. **该部分修改完成后必须提交（commit）**，并在 commit message 中写清楚：  
   - 改动范围（哪个模块/哪个 milestone）  
   - 解决了什么问题 / 为什么要改  
   - 是否包含测试与测试范围

> 约定：如果某次改动需要拆成多个 commit（例如“重构 + 迁移测试 + 修复 lint”），也要保证每个 commit 都是可运行/可回滚的最小单元。

---

## 2) 每次更新后的日志要求

每次我提交（commit）后，都要在本文件的 **「4) 工作日志」** 追加一条记录，至少包含：

- 做了什么（What）
- 为什么（Why，可选但推荐）
- 影响范围（Impact）
- 运行了哪些测试（Mock-only）
- 下一步计划（Next）

---

## 3) 测试约定（非常重要）

### 3.1 我实际运行的测试：必须全量 Mock（不真实调用 LLM）

- **我在本地/CI 中实际执行的所有测试，都必须通过 mock 来模拟 LLM 返回**，不得真实请求模型服务。
- 我需要为关键路径构造足够覆盖的用例与 mock（含异常/边界），确保：
  - 测试稳定、可复现、离线可跑
  - 同时保证“真实 LLM 返回”在协议/结构上也能跑通（即 mock 必须贴近真实返回形态）

### 3.2 需要同时提供两类测试：Mock 测试 + Real LLM 测试（实现但不运行）

- **Mock 测试（必须运行）**：默认测试集，保证功能正确性与回归保护。
- **Real LLM 测试（只实现，不需要运行）**：用于验证端到端连通性。约定：
  - 用 pytest marker（例如 `@pytest.mark.real_llm`）隔离
  - 默认不执行（例如需要显式 `-m real_llm` 才会跑）
  - 测试中要清晰声明依赖（环境变量/密钥/网络），并在缺失时自动 skip

### 3.3 测试目录约定

- **所有新增/重构后的测试统一放在 `tests/example/`** 下（不再新增到其他目录）。
- 如果仓库里历史测试目录不一致（例如当前存在 `test/`），会在后续某个独立 commit 中完成迁移/兼容，保证约定落地。

---

## 4) 工作日志（逐次追加）

### 2026-02-12

- What: 初始化本文件（协作约定 & 工作交接日志）。
- Tests: N/A（仅文档变更）。
- Next: 按 `extra_doc/refactor_plan_to_new_arch.md` 从 Milestone A（query-only）开始逐步评估与改造代码，并建立 `tests/example/` 的 mock/real_llm 测试骨架。

- What: Milestone A（query-only）落地：dataset_synthesis 强制 query-only；修复 generator 对 non-query/mixed 的隐式依赖；提升 LLM 输出 JSON 解析鲁棒性；新增 mock 测试与 real-LLM（不默认跑）测试骨架；注册 `real_llm` marker。
- Commit: e8b8f6a
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 继续 Milestone B（固定单 Expert 入口）：在评估/示例执行提交任务时补齐 `assigned_expert_name`，并实现“仅一个 expert 时自动绑定入口 expert”的 SDK 能力；同时把历史 `test/` 下的 real-LLM 测试逐步迁移/标记为 `real_llm`，避免默认跑到网络。

- What: Milestone B（固定单 Expert 入口）第一步：实现 single-expert 模式下的“入口 Expert 自动绑定”，并让 MCTS `LLMEvaluator` 提交任务时强制走 entry expert，避免 Leader 分解触发额外 LLM 调用；新增对应 mock 测试与 real-LLM（不默认跑）测试。
- Commit: f74c208
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: Milestone B 继续：把 workflow_generator 的示例/runner 统一补齐 `assigned_expert_name`（若仍有直接 submit 的入口），并考虑将 `pyproject.toml` 的 pytest `testpaths` 逐步迁移到 `tests/`（需要一个独立 commit 处理历史 `test/` 目录）。

- What: 修复 MCTS workflow_generator 的 init template：提供可执行的单 Expert（Main Expert）入口，并移除默认 MCP 工具定义（BrowserUsing/FileTool），避免环境不可用时初始化失败；补充模板结构断言测试。
- Commit: a0ce714
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: Milestone C：约束 Expander（只允许 1 个 expert 且 name 固定为 Main Expert），并在候选 YAML 校验中检查 expert 数量/name/单尾约束，减少无效 round。

- What: Milestone C（部分）：约束 expander 的专家生成逻辑为单 Expert 模式（必须且只能输出一个 `Main Expert`），并在 prompt 与解析 filter 中加硬约束，避免生成多 expert 导致 Leader 分解语义回流。
- Commit: 97ab25e
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: Milestone C 继续：增加候选 YAML 校验（expert 数量/name + workflow 单尾约束 + operator 引用合法性），并在生成器侧落地 main_expert_name 配置贯穿 expander/evaluator。

- What: Milestone C（继续）：新增候选 workflow.yml 校验器（single expert + DAG + 单 tail + operator 引用合法），并在 MCTSGenerator 保存候选后、评估前强制校验，减少无效 round 进入执行/评估。
- Commit: a0bda9f
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 将 `main_expert_name` 从硬编码提升为贯穿 MCTS 组件的配置（expander/evaluator/validator/init template），并补充更严格的引用校验（比如 operators 去重/必备字段）。

- What: 将 `main_expert_name` 贯穿 MCTS 组件：Expander/Prompt/Evaluator/Generator 都以配置驱动并做一致性校验（避免未来改名导致隐式分解或校验失效）。
- Commit: f398f90
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 把 init template 的 expert 名称也参数化（或在生成器里自动从模板读取并设置 `main_expert_name`），再继续做单尾约束/引用校验的增强（例如 operator 字段完整性、workflow 引用唯一性）。

- What: 生成器支持从 init template 自动推断单 Expert 入口名称（single-expert 时无需再手动传 `main_expert_name`），并补充推断函数与单测。
- Commit: 38eb3a1
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 继续增强候选 YAML 校验（operator 字段完整性/唯一性、workflow 引用一致性），并把示例 `test/example/workflow_generator/workflow_generator_example.py` 迁移到 `tests/example/`（按测试约定）。

- What: 增强 workflow 校验器：operator 的 actions 引用校验（基于 action.name），并改进 operator canonicalization（从 action dict 中提取 name），减少误合并/误判。
- Commit: 4526bb4
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 迁移 `test/` 下的示例与测试到 `tests/example/`（尤其是 `test/example/test_minimal_sdk_yaml.py`），并将 real-LLM/外部依赖测试全部标记为 opt-in（默认不跑）。

- What: 将历史 `test/example/test_minimal_sdk_yaml.py` 的 live-LLM 用例改为 opt-in：增加 `real_llm` marker，并要求 `CHAT2GRAPH_RUN_REAL_LLM_TESTS=1` 才会执行，避免默认测试触网调用模型。
- Commit: baeb776
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 把 `test/example/test_minimal_sdk_yaml.py` 迁移到 `tests/example/`（并移除旧路径），再评估是否需要把 `pyproject.toml` 的 pytest `testpaths` 调整为 `tests`（独立 commit，避免影响现有 CI）。

- What: 迁移 minimal SDK YAML 测试到 `tests/example/`，并保持 real-LLM 用例为 opt-in（`real_llm` + `CHAT2GRAPH_RUN_REAL_LLM_TESTS=1`）。
- Commit: 743edb2
- Tests (mock-only): `.venv/bin/pytest tests/example -m "not real_llm"`
- Next: 评估是否将 pytest `testpaths` 从 `test` 迁移到 `tests`（独立 commit），并逐步把 `test/unit` 下的用例也迁移到 `tests/example`（或建立 `tests/unit` 子目录并更新约定）。

- What: 将 pytest 默认发现路径切换到 `tests/`（不再默认收集 `test/`），避免 legacy 测试因历史依赖缺失导致 collection 失败；后续按约定逐步迁移/重写旧用例。
- Commit: 02ad207
- Tests (mock-only): `.venv/bin/pytest -m "not real_llm"`
- Next: 将 `test/unit` 下仍有价值的用例按模块迁移到 `tests/`（建议 `tests/unit/`），并在迁移过程中把所有外部依赖（LLM/MCP/DB）测试都改为 opt-in。
