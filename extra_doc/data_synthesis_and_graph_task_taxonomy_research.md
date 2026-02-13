# 数据合成方案调研 + 图数据库任务难度/类型分级（落地版）

> 目标：补全 [extra_doc/refactor_plan_to_new_arch.md](refactor_plan_to_new_arch.md) 第 8 章提出的“数据合成改进方向（8.1/8.2）”与“任务分类 taxonomy 调研（8.3）”。
>
> 本文面向工程落地：给出可执行的分类口径、可对齐的外部基准参考、以及对本仓库当前实现（`app/core/workflow/dataset_synthesis`）的具体改造建议。

---

## 1. 现状回顾（仓库当前实现）

当前数据合成链路（详见 [extra_doc/workflow_auto_generation_system.md](workflow_auto_generation_system.md)）：

- 子图采样：`RandomWalkSampler.get_random_subgraph()` 从全局 GraphDb 抽取局部子图（JSON）。
- 生成：`SamplingDatasetGenerator.generate_pairs()` 把 `task_desc + 子图 + 任务分级/统计` 注入 prompt，LLM 输出若干条 Row。
- 过滤：`SamplingDatasetGenerator.filter()` 再用 LLM 过滤不可用样本。
- Row schema（见 `app/core/workflow/dataset_synthesis/model.py`）：
  - `level`: `L1..L4`
  - `task_subtype`: str
  - `task`: 题目（自然语言）
  - `verifier`: **当前是“标准答案文本/或查询草稿文本”**（非强约束、不可执行）

目前全仓库已经统一 **query-only**（任务类型固定为 `query`），避免 non-query/mixed 带来的接口与实现不一致。

### 当前痛点（与 8.1/8.3 强相关）

1) **局部子图 ≠ 全局图**
- 生成题目时给的是“局部子图”，但 agent 执行阶段面对的是“全局图”。
- 如果题目语义是全局范围（隐含 `all/list/top-k/ranking/distribution`），就会天然不可验证或答案不一致。

2) **`verifier` 不可执行，质量门槛不可量化**
- LLM 既出题又出答案，再由 LLM 过滤，系统缺少 deterministic ground-truth。
- 一旦要做“可复现实验/论文对比/模型迭代”，会受到 LLM 偏差与漂移影响。

3) **taxonomy 目前是“概念型分层”，缺 capability 绑定**
- `task_subtypes.py` 定义了 L1..L4 与子类型，但 `REGISTER_LIST` 只启用到 L3；L4 存在但默认不参与 prompt/统计。
- 子类型的定义没有显式标注“需要哪些工具/动作/验证器类型”，会导致：生成了但跑不了（工具缺失/DB 不支持）。

---

## 2. 外部基准调研：常见任务分类与代表 benchmark

本节不是“把 benchmark 全搬进来”，而是提炼可对齐的分类维度，用来校准本仓库的 `L1..L4` 与 `task_subtype`。

### 2.1 OLTP/邻域查询类（读写/事务场景）

- LDBC SNB（Social Network Benchmark）
  - Interactive workload：围绕“给定起点节点”的邻域检索、路径/关系类复杂读查询，并含持续更新。
  - BI workload：聚合/Join-heavy 复杂分析查询 + micro-batch 写入。
  - 参考：https://ldbcouncil.org/benchmarks/snb/

- LDBC FinBench（Financial Benchmark）
  - 交易型 workload：金融/反欺诈/风控场景，复杂读查询 + 持续写入/删除。
  - 参考：https://ldbcouncil.org/benchmarks/finbench/

> 对齐到本文 taxonomy：这类 benchmark 的核心是 **查询与模式匹配**（L1/L2 为主），以及部分“路径/可达性/最短路”（可落到 L2/L3 的边界）。

### 2.2 图算法/图分析类（离线或批处理平台）

- LDBC Graphalytics
  - 工业级图分析 benchmark：包含若干核心算法、标准数据集与 reference outputs。
  - 参考：https://ldbcouncil.org/benchmarks/graphalytics/

- MIT GraphChallenge
  - 面向图分析/稀疏数据的社区挑战赛，覆盖 triangle counting、subgraph matching 等典型图计算任务。
  - 参考：https://graphchallenge.mit.edu/

> 对齐：这类 benchmark 对应本文的 **L3/L4（算法能力）**。若你的 toolset 没有算法动作（例如 Neo4j GDS、TuGraph algo），就不应启用这些子类型。

### 2.3 RDF/SPARQL 查询多样性类（结构多样、选择率多样）

- WatDiv（Waterloo SPARQL Diversity Test Suite）
  - 目标是覆盖不同结构特征与选择率类别的查询模板（linear/star/snowflake/complex）。
  - 参考：https://dsg.uwaterloo.ca/watdiv/

- LUBM（Lehigh University Benchmark）
  - 单一领域本体 + 合成数据生成器 + 一组固定查询，面向语义仓库性能评测。
  - 参考：http://swat.cse.lehigh.edu/projects/lubm/

- LDBC SPB（Semantic Publishing Benchmark）
  - RDF 引擎 benchmark（出版/媒体领域），包含复杂查询、推理与持续更新。
  - 参考：https://ldbcouncil.org/benchmarks/spb/

> 对齐：RDF 的 query-shape 分类（star/snowflake/linear）对“难度结构化指标”很有启发，可借用到 property graph/Cypher 的 pattern 结构分级。

### 2.4 NL2GraphQuery / KGQA（自然语言到查询）

- LC-QuAD（Complex Question Answering over Knowledge Graph）
  - 自然语言问题 + SPARQL 查询，覆盖复杂组合与多跳。
  - 参考：https://github.com/AskNowQA/LC-QuAD

> 对齐：如果我们未来把 `task` 侧重点从“给出 Cypher”扩展到“自然语言 → 可执行查询”，这类数据集提供了可参考的数据形态（question + query）。

---

## 3. 本仓库推荐 taxonomy：两层拆分（Task vs Capability）

这是 [extra_doc/refactor_plan_to_new_arch.md](refactor_plan_to_new_arch.md) 8.3 提到的拆分思想的“落地版”。

### 3.1 Task Taxonomy（任务类型：问的是什么）

建议把图数据库任务拆成可组合的原子能力（便于合成、计数、难度控制）：

- **Entity Lookup**：实体属性/label 查询
- **1-hop Neighborhood**：直接邻居/边类型/关系存在性
- **Predicate Filtering**：属性过滤（单条件/多条件）
- **Pattern Matching**：小模式匹配（star/chain/cycle 等）
- **Path Existence / Reachability**：可达性、是否存在路径
- **Path Retrieval**：返回路径/路径长度（注意：返回全路径往往不可控）
- **Aggregation**：count/sum/avg/group-by（全局 vs 范围必须显式）
- **Ranking / Top-K**：排序与 Top-K（同上，范围必须显式）
- **Algorithmic Analysis**（可选）：shortest path、centrality、community、triangle counting…
- **Anomaly / Risk Pattern**（可选）：异常模式、风控团伙结构

### 3.2 Capability Taxonomy（能力上界：怎么做得到）

把“能否做”绑定到 toolset 与图平台：

- **Cypher-only (Read)**：只需要 Cypher 读查询
- **Cypher+Procedures**：需要 APOC/自定义 procedure
- **GDS / Algo**：需要 Neo4j GDS 或 TuGraph/其他平台的图算法能力
- **Write**（当前禁用）：需要写入/更新

对每个 subtype 显式标注：

- `required_actions`: 例如 `graph_db.run_cypher`、`graph_db.schema`、`graph_db.shortest_path`（取决于工具封装）
- `verifier_type`: `cypher` / `procedure` / `python` / `llm_judge`（尽量不要以 LLM 作为主 verifier）

---

## 4. 难度分级：从“文案级”升级为“结构级指标”

保持现有 `L1..L4`，但让它变成可计算的结构约束（便于合成/评估/统计）。

### 4.1 结构指标（建议落盘到 metadata）

对每条样本计算并记录：

- `hop_max`: 最高跳数（pattern 中最长链）
- `predicate_count`: 过滤条件数量（属性谓词 + label/rel type 约束）
- `pattern_shape`: `linear` / `star` / `snowflake` / `cycle` / `multi-path`
- `uses_aggregation`: 是否包含聚合（count/sum/avg/group-by）
- `uses_ordering`: 是否包含排序/Top-K
- `uses_algorithm`: 是否调用图算法（GDS/最短路/社区/中心性…）
- `scope`: `local_subgraph` / `global_graph`（建议默认 global；local 仅做对照实验）

### 4.2 L1..L4 的建议判定规则（query-only 视角）

- **L1（简单查询）**
  - `hop_max <= 1`
  - `predicate_count <= 1`
  - `uses_aggregation == false`
  - 典型：实体属性、1-hop 关系存在性、单条件过滤

- **L2（复杂查询，无算法）**
  - `hop_max >= 2` 或 `predicate_count >= 2` 或 `pattern_shape in {star,snowflake,cycle}`
  - `uses_algorithm == false`
  - `uses_aggregation` 可选（但必须是“可验证、范围明确”的聚合）

- **L3（单算法应用）**
  - `uses_algorithm == true` 且只需要 1 个算法结果（例如 shortest path / degree / pagerank@local）

- **L4（复杂算法/多阶段）**
  - `uses_algorithm == true` 且多算法流水线/多阶段推理
  - 或需要复杂模式（subgraph isomorphism）/异常检测/预测类

> 重要：如果当前 toolset 不支持算法动作，应当在 dataset 合成时 **裁剪掉 L3/L4**，或者把 L3 仅限于“Cypher 可表达的最短路/可达性”这类近似能力。

---

## 5. 数据合成方案：从“LLM 出题/出答案”升级为“可执行 verifier + ground-truth”

这部分对应 [extra_doc/refactor_plan_to_new_arch.md](refactor_plan_to_new_arch.md) 8.1 的路线 B，给出更细的可落地实现。

### 5.1 Row 的推荐升级（RowV2，不破坏旧接口的最小增量）

现有 Row：`task` + `verifier(str)`。

建议升级为（示意）：

- `task`: 自然语言题目
- `verifier_type`: `cypher`（默认）
- `verifier`: Cypher 查询（或结构化 tool plan）
- `expected`: JSON（执行 verifier 得到的标准结果，作为 ground-truth）
- `expected_schema`: 结果字段/类型约束（可选）
- `metrics`: 结构指标（见 4.1，可选）

> 如果暂时不想改 Pydantic model，可以先把 `expected` JSON 序列化塞进 `verifier`（不推荐长期这样做），或新增并行字段并保持向后兼容。

### 5.2 合成链路三种实现路线（从易到难）

**路线 1：LLM-first + 执行校验（最贴近当前实现）**

1. 采样子图（仅用于锚点/出题，不是答案来源）
2. LLM 生成：`task` + `cypher`（并标注 `level/subtype`）
3. 系统执行 `cypher` 于全局 GraphDb，拿到 `expected`
4. 校验/过滤：
   - cypher 能执行且结果不为空/符合格式
   - 不允许隐含全局枚举（除非你明确把“全局聚合/Top-K”作为允许的 subtype 并可验证）
   - 去重与覆盖率控制

优点：改动小；缺点：仍依赖 LLM 的查询正确性，reject rate 可能高。

**路线 2：Template-first（确定性生成 verifier） + LLM 负责语言化**

1. 为每个 subtype 维护 3-10 个 Cypher 模板（参数化）
2. 从采样子图中抽取候选实体/属性值作为填充（entity grounding）
3. 生成确定性 cypher 并执行获得 `expected`
4. 让 LLM 把“结构化意图 + 参数”改写成自然语言 `task`

优点：高可控、低幻觉、覆盖可控；缺点：需要你维护模板库与 grounding 规则。

**路线 3：Hybrid（结构可控 + 语言多样性）**

- 模板产生意图骨架 + cypher
- LLM 做 paraphrase/上下文包装
- 同时保留“hard constraints”：scope、禁止词、禁止 all/list 等

> 工程建议：先落地路线 1（最少改动），同时为关键 subtype 逐步引入路线 2 的模板，最终演进到 Hybrid。

### 5.3 全局/局部 scope 的处理建议

- 训练/评估目标如果是“线上可执行图任务”，建议全部样本 **以 global graph 为准**，子图只做锚点。
- 若要做对照实验（路线 A），必须在 `task` 文本显式写清楚范围：“在给定子图中…”，并在评估时将执行环境也限制在子图（否则和线上不一致）。

---

## 6. 任务子类型（task_subtype）建议表：对齐 L1/L2/L3/L4 + 能力裁剪

下面给出一份“面向 query-only + Cypher 可验证”的建议子类型 ID（**建议使用稳定的 snake_case id**，避免与展示文本耦合）。

### L1（hop<=1 / 单条件 / 无聚合）

- `entity_attribute_lookup`
- `entity_label_lookup`
- `direct_relationship_exists`
- `one_hop_neighbors`
- `simple_attribute_filter`

### L2（多跳 / 多条件 / 模式匹配 / 可验证聚合）

- `multi_hop_reachability`
- `bounded_path_length`（例如 *..2、*..3 这种有上界）
- `pattern_star_join`
- `pattern_chain_join`
- `pattern_cycle_exists`
- `combined_attribute_filters`
- `scoped_aggregation_count`（必须明确范围，例如“在某一类关系邻域内 count”）

### L3（需要算法/或平台内置最短路）

> 若没有 GDS/算法工具，可只保留：

- `shortest_path_length`（若你的 Cypher/平台支持 `shortestPath` 且可控）

### L4（高级算法/预测/异常）

> 默认建议关闭，除非 toolset 明确支持并有可执行 verifier。

- `community_detection_explain`
- `centrality_topk_explain`
- `anomaly_pattern_detection`

---

## 7. 对仓库代码的最小改造建议（不改变整体架构）

### 7.1 `task_subtypes.py` 的结构建议

- 为 `SubTaskType` 增加稳定字段 `id`（snake_case），把当前 `name` 作为展示名。
- 为每个 subtype 增加：
  - `required_actions`
  - `verifier_type`
  - `default_scope`
- `GraphTaskTypesInfo` 的计数与 prompt 注入基于 `id`，避免 LLM 输出的 `task_subtype` 与展示名不一致导致全进 `unknown`。

### 7.2 Dataset synthesis 的 validator（路线 B 的关键）

新增 `validator.py`（或放在 evaluation pipeline 中），实现：

- `compile_verifier(row) -> cypher/tool_plan`
- `execute_verifier(graph_db, verifier) -> expected`
- `validate_row(row, expected, policy) -> ok/reason`

policy 包含：

- 禁止模式：`all/list` 之类隐含全局枚举
- 结果格式约束：字段、类型、最大返回行数
- scope 与 subtype 是否一致

---

## 8. 建议的落地顺序（1-2 周可交付的版本）

1) **RowV2（可选字段）+ verifier 执行校验**
- 让 `verifier` 统一成为 Cypher（或工具调用计划），并在合成阶段执行拿到 `expected`。

2) **taxonomy 裁剪与计数稳定化**
- 引入 subtype `id`；按 toolset 裁剪 L3/L4。

3) **模板库 + grounding**
- 对 L1/L2 选 5-10 个高频 subtype 先做 template-first，显著降低 reject。

4) **结构指标落盘**
- 给 dataset + eval 输出增加 difficulty metrics（利于论文与迭代）。

---

## 9. 附：与本仓库现有 `L1..L4` 的映射建议

当前 `QueryTaskSubtypes` 的描述可以保留为“展示文本”，但建议改造为：

- L1 ↔ entity/1-hop/filter（对应 SNB Interactive 的低复杂度查询）
- L2 ↔ multi-hop/pattern/combined filters（对应 SNB Interactive / FinBench 中更复杂的邻域与模式查询）
- L3 ↔ shortest path / basic centrality（对应 Graphalytics 的一部分）
- L4 ↔ anomaly/prediction/multi-algo（对应 Graphalytics + GraphChallenge + 风控真实场景，但需要 toolset 支持）

---

## 10. 你可以直接拿来用的 checklist

- [ ] 每条样本的 `verifier` 能在 GraphDb 上执行并拿到 `expected`
- [ ] `task` 文本中明确 scope（默认 global，且避免隐式 all/list）
- [ ] `task_subtype` 使用稳定 id，且能映射到 `required_actions`
- [ ] dataset 输出包含 level/subtype 分布 + 结构指标分布
- [ ] L3/L4 默认按 toolset 裁剪（没有算法动作就不要生成）
