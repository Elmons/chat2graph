# Task1 任务复盘与项目改造方案（v2）

## 0. 文档目的

这份文档只回答 5 个问题：
1. 我观察到了什么（事实）
2. 我得到了什么结论（判断）
3. 我收到了什么启发（方法来源）
4. 我想怎么解决问题（策略）
5. 我们的项目要怎么改（模块级改造）

范围：
- `test/example/workflow_generator/workflow_space/case1/`
- `test/example/workflow_generator/workflow_space/case2/`

---

## 1. 我观察到了什么

## 1.1 搜索过程中的关键事实

| 事实编号 | 观察 | 证据 | 影响 |
| --- | --- | --- | --- |
| O1 | `case1` 的最佳分数出现在 `round8` 和 `round16`，均为 `2.1111` | `test/example/workflow_generator/workflow_space/case1/log/log.json` | 当前能力上限已出现，但不稳定 |
| O2 | `case1 round9` 加入 `pre_execution_validation` 后，输出变成验证文案而非任务答案，分数跌到 `-0.0667` | `test/example/workflow_generator/workflow_space/case1/round9/workflow.yml`, `test/example/workflow_generator/workflow_space/case1/round9/results.json` | 出现“输出契约破坏”，并非纯语义能力退化 |
| O3 | `case1 round16` 线性链 `ambiguity -> query -> graph` 达到最佳点，`round17/18` 再次回落到 `1.4889` | `test/example/workflow_generator/workflow_space/case1/round16/results.json`, `test/example/workflow_generator/workflow_space/case1/round17/results.json`, `test/example/workflow_generator/workflow_space/case1/round18/results.json` | 当前优化属于“脆弱改进”，不是稳态改进 |
| O4 | `case2 round4` 为最佳 `1.6667`；`round5` 直接 `-2.2` 且 `error_rate=1.0` | `test/example/workflow_generator/workflow_space/case2/round4/results.json`, `test/example/workflow_generator/workflow_space/case2/round5/results.json` | 存在系统层故障污染搜索信号 |
| O5 | `case2 round5` 主要错误是 `sqlite database is locked` | `test/example/workflow_generator/workflow_space/case2/round5/results.json` | 该轮不可用于评估算法优劣 |
| O6 | `case2 round9` 插入 `result_validation_operator` 后回归明显（score 降到 `0.2222`，回归率上升） | `test/example/workflow_generator/workflow_space/case2/round9/workflow.yml`, `test/example/workflow_generator/workflow_space/case2/round9/results.json` | “后置验证节点”在当前实现下会伤害最终答案产出 |

## 1.2 持续难题不是随机波动

| 难题 | 统计观察 | 判断 |
| --- | --- | --- |
| `shortest path` | `case1` 中 `14/17` 轮非正分；`case2` 中 `8/8` 轮非正分 | 路径策略/约束表达长期不稳定 |
| `largest single transfer amount` | `case1/case2` 基本长期非正分 | 聚合+排序+方向约束模板缺失 |

## 1.3 错误类型主次

当前更值得优先解决的是：
- `E3` 聚合/排序查询构造问题
- `E4` 路径策略与约束错配
- `E5` 输出契约破坏（验证文本替代最终答案）
- `E6` 系统异常（锁冲突、长度上限）

---

## 2. 我得到了什么结论

1. 当前瓶颈不是“模型不会推理”，而是“workflow 契约不稳定 + hardest intent 缺少确定性模板 + 运行时噪声未隔离”。
2. 把验证算子放在尾部但不强制答案契约，会直接把“验证结果”误当“最终答案”。
3. `shortest path` 和 `largest transfer` 不能继续依赖自由生成，必须改成半确定性模板策略。
4. 运行时错误（如 DB lock）如果不从搜索反馈中剥离，会导致 MCTS 学到错误偏好。
5. 目前最接近可复用的拓扑是线性链（`ambiguity -> query -> graph`），但缺乏护栏时仍会退化。

---

## 3. 我收到了什么启发

## 3.1 方法层启发

- ReAct / ToT：适合“分步思考”，但不自动保证“输出契约正确”。
- Reflexion / Self-Refine：反思必须结构化为可执行 action，而不是长文本建议。
- Execution-Guided / PICARD：生成约束要和执行反馈联动，才能稳定查询质量。
- Structured Outputs：能约束格式，但不能替代语义验证和执行验证。

## 3.2 对本项目的直接翻译

1. “会反思”不等于“会改对”，必须把反思写入固定字段。
2. “加验证节点”不等于“答案更可靠”，必须先定义 tail answer contract。
3. hardest intents 应采用“模板优先、自由生成兜底”而非反过来。

---

## 4. 我想怎么解决问题

## P0（先做，决定成败）

1. 建立 Tail Answer Contract
- 规则：终端节点必须输出任务答案结构，验证文本不能作为 final output。
- 目标：消除 `E5`。

2. hardest intents 模板化
- 对 `shortest path`、`largest single transfer amount` 提供确定性查询模板与参数槽位。
- 目标：降低 `E3/E4`。

3. 运行时噪声隔离
- 将 `bootstrap/runtime` 错误与语义错误分开打分与统计。
- 目标：避免 `E6` 污染搜索策略。

## P1（稳定化）

1. ID-first 实体消歧与类型约束。
2. 关系方向/角色 pre-check（生成前校验）。
3. 输出规范化（list/scalar/字段名统一）。

## P2（增益项）

1. 拓扑先验评分：惩罚绕过答案节点的拓扑。
2. Reflection 结构化：反思输出固定为 action slots。

---

## 5. 怎么改造我们的项目（模块级）

下面是“文档级改造清单”，对应到仓库模块，不涉及本轮代码改动。

| 改造项 | 目标 | 建议落点 | 具体改造 |
| --- | --- | --- | --- |
| Tail Answer Contract Checker | 防止验证文本替代答案 | `app/core/workflow/workflow_generator/mcts_workflow_generator/validator.py`, `app/core/workflow/workflow_generator/mcts_workflow_generator/evaluator.py` | 在候选 workflow 验证阶段新增“末节点输出类型检查”；在评估阶段对契约违例直接降权或拒收 |
| Hard-intent Template Engine | 稳定 hardest intents | `app/core/prompt/workflow_generator.py`, `app/core/prompt/reasoner.py` | 为 path/max-intent 增加模板分支；自由生成仅作 fallback |
| Direction/Role Pre-check | 生成前拦截语义错配 | `app/core/workflow/workflow_generator/mcts_workflow_generator/constraints.py` | 增加关系方向、实体类型、聚合合法性约束 |
| Runtime Error Isolation | 把系统错与语义错分离 | `app/core/workflow/workflow_generator/mcts_workflow_generator/runner.py`, `app/core/workflow/workflow_generator/mcts_workflow_generator/evaluator.py` | error 标签拆分为 `runtime_error`/`semantic_error`，分别计分 |
| Structured Reflection Action | 把反思变更可执行化 | `app/core/workflow/workflow_generator/mcts_workflow_generator/expander.py` | 反思结果输出固定字段（operator_change/query_template_change/runtime_change） |

---

## 6. 实施顺序与验收

## 6.1 实施顺序

1. 第 1 周：Tail Answer Contract + Runtime Error Isolation
2. 第 2 周：Hard-intent Template Engine + Direction/Role Pre-check
3. 第 3 周：Structured Reflection Action + 拓扑先验评分

## 6.2 验收指标

- `invalid_output_rate` 显著下降（重点看“验证文案冒充答案”是否归零）
- `shortest path` 或 `largest transfer` 至少一项出现可重复提升
- `regression_rate` 在新增验证节点场景不再异常飙升
- runtime 错误与语义错误可分离统计

---

## 7. 当前结论的置信度与风险

- 高置信：O1/O2/O4/O5（有明确 round 级证据）
- 中置信：hard-intent 模板可显著提升（需要消融验证）
- 主要风险：
  - 如果只改 prompt 不改契约，回归仍会反复出现。
  - 如果不隔离 runtime error，MCTS 仍会学偏。

---

## 8. 立即可执行的文档化行动

1. 把后续 Task1 讨论统一追加到本文件，不再拆分子文档。
2. 每次新增实验记录必须按“观察->结论->改造项->指标变化”四段式追加。
3. 任何拟实施改造必须能回链到本文件的 O1~O6 证据之一。
