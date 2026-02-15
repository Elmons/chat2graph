# Task2 总体改造方案（v5，精简主线版）

## 0. 文档目的

这份文档只保留 Task2 的主线内容，回答 4 个问题：
1. 我们到底要解决什么问题。
2. 为什么要这样改造。
3. 具体怎么改（端到端流程 + 模块落点）。
4. 依据是什么（论文/工业/开源/标准）。

> 说明：历史版本已归档到 `extra_doc/task2_integrated_research_report_archive_2026-02-15.md`。

---

## 1. 目标与硬约束（先对齐）

### 1.1 目标

用“围绕真实用户问题”的合成数据集，驱动智能体能力提升，让智能体在全局图数据库上更稳定地回答用户问题。

### 1.1.1 当前执行范围（冻结说明）

- 当前只推进 `pure QA` 主线（查询问答类任务）。
- 图表/风险分析/报告生成类复合任务暂缓，不进入当前迭代实施范围。

### 1.2 硬约束

- 问题合成：基于局部子图采样（成本原因）。
- 问题回答：必须在全局图执行并回答。

### 1.3 关键原则

- `local_subgraph` 只负责发现候选问题。
- `global_graph` 才是标准答案来源。

---

## 2. 现状问题与诊断结论

### 2.1 现状问题

1. 合成样本里存在不可执行 verifier / 叙事型 verifier。
2. scope 语义不清（局部生成语义与全局回答语义混淆）。
3. path/ranking/aggregation 类题目最易产生不可回答样本。
4. 样本质量门槛和分布控制不稳定，容易“量变但无效”。

### 2.2 诊断结论

1. 主要矛盾不是“题目不够多”，而是“真值不可靠”。
2. 只做 prompt 改写无法根治，必须改数据协议与执行流程。
3. verifier 必须是离线真值工具，不是线上回答器。
4. 意图识别必须是多标签映射，否则无法覆盖真实复合问题。

---

## 3. 总体改造思路（核心）

### 3.1 改造总览

围绕“用户问题驱动闭环”改造：
1. 用户问题 -> 意图识别（多标签）
2. 意图扩展 -> 同类问题合成（覆盖各子类/难度）
3. 每条样本生成可执行 `global_verifier`
4. 全局执行得到 `expected_global`
5. 通过 QA Gate 后入库
6. 用数据集驱动智能体优化
7. 回答原用户问题并观测增益

### 3.2 为什么是这条路线

1. 与业务目标一致：最终提升真实用户问题成功率。
2. 可闭环量化：每一步都有可测指标。
3. 可迭代落地：先模板化高风险意图，再扩展长尾。

---

## 4. 关键改造点（做什么 + 为什么 + 依据）

| 改造点 | 具体做法 | 为什么要做 | 依据 |
| --- | --- | --- | --- |
| A. 意图识别多标签化 | 从单标签改为 `L0/L1/L2` 层级标签（语义任务/执行能力/约束标签） | 用户问题经常是复合意图（如 path+ranking），单标签会丢信息 | NAT-NL2GQL, RAT-SQL |
| B. 任务分类三轴化 | `Task Axis + Capability Axis + Scope Axis` | 防止“语义分类正确但引擎不可执行” | ISO GQL/openCypher + Neptune 兼容性文档 |
| C. verifier 生成编译化 | Template-first -> IR-first -> LLM-repair fallback | 直接 LLM 自由生成 verifier 不稳定，需可控生成路径 | PICARD, Execution-Guided |
| D. 全局真值回填 | `global_verifier` 在全局执行产出 `expected_global` | 解决局部生成与全局回答不一致 | 你的业务约束 + 工程可验真需求 |
| E. QA Gate 硬门槛 | 执行性、范围一致性、语义一致性、方言兼容、一票否决 | 没有硬门槛会把噪声样本喂给训练，污染智能体 | Mind the Query, Test Suite Accuracy |
| F. 分布与覆盖控制 | 按 `subtype x difficulty x operation` 设目标配额 | 防止数据偏科，保证能力覆盖而非堆量 | CypherBench, Neo4j Text2Cypher |

---

## 5. 意图识别与多分类映射方案（可执行版）

### 5.1 标签体系（最小可用）

- L0 语义族：`lookup/filter/pattern/path/aggregation/ranking/algorithm`
- L1 子类：如 `path.shortest`, `ranking.topk`, `aggregation.group_count`
- L2 约束：`requires_hop_bound`, `requires_tie_break`, `global_answer_required`, `needs_procedure`

### 5.2 识别流程

1. Intent Frame 抽取：从用户问题抽出实体、关系、条件、操作、范围需求。
2. Schema Linking：将抽取槽位对齐到真实 schema。
3. 多标签判定：输出 `intent_set`，允许复合标签。
4. 能力裁剪：结合引擎能力矩阵去除不可执行标签。
5. 映射到合成计划：生成同类扩展样本计划。

### 5.3 映射输出格式（建议）

`intent_plan` 至少包含：
- `core_intents`
- `neighbor_intents`
- `difficulty_plan`
- `capability_constraints`
- `template_routes`

---

## 6. verifier 生成方案（你关心的重点）

### 6.1 设计原则

- verifier 是离线真值工具，不是线上回答模型。
- verifier 目标是“可执行、可复现、可验真”，不是“语言优雅”。

### 6.2 生成路径（推荐顺序）

1. Template-first（首选）
- 按意图选择参数化 Cypher 模板，填充实体/条件/排序规则。

2. IR-first（中期）
- 先产出结构化 IR，再编译成 Cypher，便于静态检查。

3. LLM-repair（兜底）
- 当模板/IR 不覆盖时，LLM 生成草稿并走执行修复循环；失败则拒收。

### 6.3 verifier 通过标准

1. 全局图可执行。
2. 返回字段与任务预期一致。
3. path 任务包含 hop/方向规则。
4. ranking 任务包含 tie-break。
5. 通过方言兼容检查。

---

## 7. 数据过滤与QA Gate（发布前）

任一命中即拒收：
1. `global_verifier` 不可执行。
2. `expected_global` 缺失或非全局执行所得。
3. `task` 与 `global_verifier` scope 不一致。
4. path/ranking 缺必要边界规则。
5. 标签与 verifier 操作不一致（标为 topk 但查询无排序）。
6. 存在自相矛盾答案或验证叙述。
7. 目标图引擎不支持对应语法/函数。

---

## 8. 子图采样改造思路（用于候选问题发现）

### 8.1 原则

子图采样只服务“问题发现覆盖”，不参与真值定义。

### 8.2 建议策略（SHS）

`Stratified Hybrid Sampling`：
1. 分层 seed：实体类型、度分位、社区簇、时间片。
2. 混合采样：
- MHRW/RWRW（降偏）
- cluster-based 子图（保结构）
- motif/path 定向采样（补 hardest intents）
3. 覆盖约束：每批保证 subtype/hop/operation 最小覆盖。

### 8.3 如何“保证采样子图可提出目标类别问题”

结论：做不到一次采样的绝对保证，但可以通过“可提问性约束 + 接受拒绝机制”实现稳定可控的高命中率。

做法分 4 步：
1. 为每个目标类别定义 `可提问性前置条件`（feasibility predicate）。
- 例：`ranking.topk` 需要候选集合规模 >= k 且有可比较指标。
- 例：`path.shortest` 需要至少一对候选点存在路径且可定义 hop/方向约束。
- 例：`aggregation.group_count` 需要至少 2 个 group 且 group 分布非退化。
2. 子图采样后先跑“轻量可提问性探针”（cheap probes），而不是直接交给 LLM 出题。
3. 不满足前置条件的子图直接丢弃或补采样（accept-reject sampling）。
4. 对低命中类别做定向采样（motif/path/关系类型定向），直到达到每类最低样本配额。

核心思想：
- 先证明“这张子图可以出某类题”，再让模型生成该类题。
- 这样可以把“碰运气采样”改成“受约束采样”。

### 8.4 建议增加的采样指标（用于保障）

每个类别 `c` 持续跟踪：
1. `feasible_hit_rate(c)`：采样子图中满足类别前置条件的比例。
2. `accepted_rate(c)`：满足前置条件且通过 QA Gate 的比例。
3. `coverage_gap(c)`：目标配额与当前已接收样本数差值。
4. `attempt_budget(c)`：达到配额前允许的最大补采样次数。

调度策略：
1. 若 `coverage_gap(c)` 大且 `feasible_hit_rate(c)` 低，提升该类定向采样权重。
2. 若 `accepted_rate(c)` 低，优先修模板/约束，不盲目加采样量。
3. 仅当 `coverage_gap(c)=0` 才降低该类采样预算。

---

## 9. 项目改造落点（仅文档方案，不改代码）

| 模块路径 | 改造内容 |
| --- | --- |
| `app/core/workflow/dataset_synthesis/model.py` | 增加 RowV2.1 字段：`generation_scope`, `answer_scope`, `intent_set`, `global_verifier`, `expected_global` |
| `app/core/workflow/dataset_synthesis/sampler.py` | 实施 SHS 采样策略与覆盖统计 |
| `app/core/workflow/dataset_synthesis/generator.py` | 流程改为“意图计划 -> verifier 生成 -> 全局执行 -> task 渲染” |
| `app/core/prompt/data_synthesis.py` | 收紧输出合约，强制结构化字段 |
| `app/core/workflow/evaluation/eval_yaml_pipeline.py` | 增加 QA Gate 硬门槛与拒收原因统计 |
| `app/core/workflow/dataset_synthesis/utils.py` | 增加 engine capability matrix 检查 |

---

## 10. 指标体系（如何判断改造有效）

### 10.1 数据质量指标

1. `global_executable_rate`
2. `scope_alignment_rate`
3. `intent_verifier_alignment_rate`
4. `local_global_consistency_rate`
5. `data_retention_rate`

### 10.2 业务效果指标

1. `user_intent_coverage_rate`
2. `user_question_success_rate`
3. `post_synthesis_gain`

### 10.3 建议门槛（首版）

- `global_executable_rate >= 95%`
- `scope_alignment_rate >= 95%`
- `intent_verifier_alignment_rate >= 95%`
- `unanswerable_rate_on_global <= baseline * 50%`

---

## 11. 分阶段落地计划

### Phase-S（1周）

1. 定义意图标签字典（L0/L1/L2）
2. 定义 verifier 模板最小集（先覆盖 path/filter/aggregation/ranking）
3. 上线 QA Gate v1（可执行+scope+方言）

### Phase-M（1~2周）

1. 引入 IR 生成与静态检查
2. 上线 SHS 采样与覆盖控制
3. 上线“用户问题驱动扩展合成”流程

### Phase-L（持续）

1. 维护拒收案例库与规则迭代
2. 监控意图覆盖漂移
3. 以真实用户问题成功率驱动数据策略调整

---

## 12. 依据（精选）

### 标准与规范

1. ISO GQL: https://www.iso.org/standard/76120.html
2. openCypher: https://github.com/opencypher/openCypher
3. Neo4j GQL conformance: https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/

### 任务分类与基准

1. LDBC SNB: https://ldbcouncil.org/benchmarks/snb/
2. LDBC Graphalytics: https://ldbcouncil.org/benchmarks/graphalytics/algorithms/
3. CypherBench: https://arxiv.org/abs/2412.18702
4. Mind the Query (IBM): https://research.ibm.com/publications/mind-the-query-a-benchmark-dataset-towards-text2cypher-task

### 意图识别/可执行生成/过滤

1. NAT-NL2GQL: https://arxiv.org/abs/2412.10434
2. RAT-SQL: https://arxiv.org/abs/1911.04942
3. PICARD: https://arxiv.org/abs/2109.05093
4. Execution-Guided Decoding: https://arxiv.org/abs/1807.03100
5. Test Suite Accuracy: https://arxiv.org/abs/2010.02840
6. OpenAI Structured Outputs: https://openai.com/index/introducing-structured-outputs-in-the-api/

### 采样与引擎差异

1. A Walk in Facebook: https://arxiv.org/abs/0906.0060
2. GraphSAINT: https://arxiv.org/abs/1907.04931
3. Cluster-GCN: https://arxiv.org/abs/1905.07953
4. node2vec: https://arxiv.org/abs/1607.00653
5. AWS Neptune openCypher compliance: https://docs.aws.amazon.com/neptune/latest/userguide/feature-opencypher-compliance.html

---

## 13. （暂缓）从 QA 扩展到“图表+分析+报告”任务（归档保留）

> 状态：Paused。该节内容仅保留为后续迭代参考，不纳入当前 pure QA 计划与验收。

### 13.1 问题本质

你给的例子不是单轮 QA，而是复合任务：
1. 数据查询（按时间窗/对象）
2. 风险检测（非法交易/洗钱等）
3. 可视化生成（图表）
4. 结论分析（文本解释）
5. 文档输出（报告）

所以不能再用“单答案样本（question-answer）”作为唯一数据形态，必须升级为“任务包（task bundle）”。

### 13.2 分类体系升级（新增 Deliverable 轴）

在现有 `Task/Capability/Scope` 三轴外，新增第 4 轴：
1. `Deliverable Axis`
- `qa_answer`
- `table_output`
- `chart_output`
- `risk_assessment`
- `report_document`

意义：
- 同一个用户问题可以映射成多个产出物；智能体要学的是“多产出协同”而非单点回答。

### 13.3 数据协议升级（RowV3 / Bundle）

建议从 RowV2.1 升级到 `TaskBundleV1`，核心字段：
1. `user_task`：用户原始任务
2. `sub_tasks`：拆解后的步骤（query/detect/chart/report）
3. `global_verifiers`：每个子任务的可执行 verifier 列表
4. `expected_artifacts`：
- `expected_table`（结构化表）
- `expected_chart_spec`（如 Vega-Lite/Plotly schema）
- `expected_findings`（风险点与证据）
- `expected_report_outline`（报告结构）
5. `evidence_links`：每条分析结论对应的数据证据位置

关键点：
- 真值不再是单值答案，而是一组“可验证产物”。

### 13.4 生成与执行流程（复合任务版）

1. 用户问题意图识别：识别为 `query + risk_detection + chart + report` 复合标签。
2. 按标签生成子任务计划（DAG）：
- `query_subtask -> risk_subtask -> chart_subtask -> report_subtask`
3. 每个子任务都生成并执行 `global_verifier`，得到中间真值产物。
4. 只有全部关键子任务通过 QA gate，整条任务包才入库。
5. 用任务包训练/评测智能体，最终回到真实用户问题。

### 13.5 “非法交易/洗钱检测”怎么做（落地口径）

检测子任务建议拆为两层：
1. 规则层（高精度）
- 大额分拆、短期循环转账、异常多跳资金回流、黑名单邻接、结构化拆分转账。
2. 图模式层（高召回）
- 环路资金流、hub-spoke 异常聚集、短时团簇共现。

输出时必须区分：
- `rule_hit`（命中规则）
- `risk_score`（模型评分）
- `evidence_path`（对应交易链路证据）

这样报告里“怀疑洗钱”的每一句都能追溯到图证据，而不是模型主观判断。

### 13.6 verifier 在复合任务里的角色（避免矛盾）

在这种任务下 verifier 不是“替代智能体”，而是：
1. 对查询子任务给可执行真值。
2. 对图表子任务验证“图表是否基于正确数据”。
3. 对分析子任务验证“结论是否被数据支持”。
4. 对报告子任务验证“结构完整性与证据引用完整性”。

也就是：verifier 负责离线验收，智能体负责线上泛化执行。

### 13.7 复合任务 QA Gate（新增）

整包拒收条件（任一命中）：
1. 任一关键子任务 verifier 不可执行。
2. 图表 spec 与查询结果字段不一致。
3. 风险结论无证据链（claim 无 evidence link）。
4. 报告结论与前序表格/图表冲突。
5. 仅有文本结论，无结构化中间产物。

### 13.8 评估指标（复合任务版）

新增指标：
1. `artifact_completeness_rate`：表/图/结论/报告是否齐全。
2. `chart_data_consistency_rate`：图表数据与查询结果一致率。
3. `claim_evidence_grounding_rate`：结论有证据支撑比例。
4. `risk_detection_precision_recall`：风险检测精确率/召回率。
5. `end_to_end_task_success_rate`：整任务闭环成功率。

### 13.9 实施建议（不改代码版）

1. 先挑 2 类复合任务模板试点：
- “时间窗交易汇总 + 可视化 + 分析”
- “反洗钱可疑链路检测 + 可视化 + 分析报告”
2. 先做模板化 verifier 与报告骨架，不追求语言多样性。
3. 跑通 end-to-end 后再放开语言改写和长尾任务。
