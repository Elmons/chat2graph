# 数据合成质量问题清单与修复方案（目标错误率 < 5%）

## 1. 当前问题清单（基于 30 条样本）
数据源：`test/example/workflow_generator/generated_datasets/20260215_190900_query_real_30_stage30_merged_v10_v11/question_answerability_review.json`

- `query_defect`（9/30）
  - 索引：`[0,1,2,10,11,15,16,20,26]`
  - 典型问题：ID 被当作 name 匹配、out-degree 被写成无向 degree、verifier 退化为 `MATCH (n) RETURN n LIMIT 10`、布尔问题返回多行。

- `question_defect`（8/30）
  - 索引：`[3,5,6,9,12,14,21,24]`
  - 典型问题：单数问法但多解、`top/limit` 无排序导致非确定性、开放枚举问题不可稳定作答。

- `label_defect`（5/30）
  - 索引：`[7,8,17,18,23]`
  - 典型问题：`expected_global` 与实时执行结果漂移、数值类型不一致（float vs string）。

- `ok`（8/30）
  - 索引：`[4,13,19,22,25,27,28,29]`

---

## 2. 修复总策略（已落地到管道）

### A. 先修“问题本身”可答性
1. 对“有上界但无排序”的问题加硬门控：拒绝 `LIMIT` 但无 `ORDER BY` 的 bounded list。
2. 对“单数问法但多结果”加运行时语义门控：执行后若多唯一结果，触发改写为复数/有界版本。
3. 对开放枚举优先改写（不是直接拒绝）：改写成 bounded + deterministic 版本。

### B. 修 query 语义漂移
1. 禁止 generic fallback verifier 通过（`MATCH (n) RETURN n [LIMIT ...]` 直接拦截）。
2. 引入 literal 语义提示（`id/name/nickname/companyName/...` + label hint）来编译 verifier：
   - 解决 shortest-path 用 ID 却按 `name` 匹配的问题。
3. topology/degree 支持方向语义：`out-degree/in-degree` 分别编译为有向模式。
4. 布尔/标量问题执行后做 shape 检查：多行布尔/标量结果触发重编译或改写。

### C. 修 expected 漂移
1. 对 global-scope 行，执行 verifier 后 **总是** 回填 `expected_global = executed`（不再仅 missing 时填充）。
2. 去掉“执行失败但静态放行”的宽松兜底，避免脏样本被接收。

---

## 3. 为每类问题提供的处理机制

### 3.1 `question_defect`
- 机制：`runtime semantic gate + rewrite`
- 处理：
  - `singular_question_multi_rows` -> 复数化问法 + 加 order/limit
  - `missing_order_for_bounded_list` -> 自动补 deterministic `ORDER BY + LIMIT`
- 结果：保留枚举/排名任务，不用“一刀切”拒绝。

### 3.2 `query_defect`
- 机制：`strict compile + strict gate`
- 处理：
  - `generic_global_verifier` -> 重编译；不可编译则拒绝
  - path 端点字段推断（ID/name）+ label 提示
  - topology degree 方向修正
  - 布尔/标量 shape 异常触发重编译

### 3.3 `label_defect`
- 机制：`execution truth overwrite`
- 处理：每次执行成功后覆盖 `expected_global`，保证与 verifier 对齐。

---

## 4. 多样性保障（避免“修复=任务类型塌缩”）

1. 不采用“全拒绝枚举/聚合/路径”的硬规则。
2. 对 soft reject 先改写再验证，尽量保留样本类型。
3. 只拒绝：
   - 语义无法对齐（问题/查询脱钩）
   - 运行时无法稳定作答（多行布尔、单数多解且不可改写）
   - generic fallback 或执行失败
4. ranking/aggregation/path 仍可保留，前提是可答且可复验。

---

## 5. 质量目标与验收标准（错误率 < 5%）

设 `error = query_defect + question_defect + label_defect`。
目标：`error / total <= 0.05`。

执行验收：
1. `mock/unit` 先通过（已通过）。
2. 真实图先跑 15 条，逐条审查 `progress_raw_candidates.jsonl / progress_decisions.jsonl / progress_rejections.jsonl / progress_dataset.jsonl`。
3. 定位残余规则盲点后微调，再跑 30 条。
4. 用一致性脚本复核每条 `task/query/expected/actual`，统计错误占比。

