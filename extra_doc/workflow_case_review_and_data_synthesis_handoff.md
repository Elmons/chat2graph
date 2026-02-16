# Workflow Case 复盘 + 数据合成问题治理（交接文档）

## 文档目的
用于跟踪以下两个任务在不同 session 间的交接状态，确保后续工作可无缝续做。

- 任务1：复盘 `case1/case2` 的 workflow 搜索日志与结果，提炼可学习 case，输出算法与 prompt 优化建议。
- 任务2：治理数据合成中“不可回答问题”（局部子图答案与全局图执行不一致）的问题，明确可落地方案（过滤/改写/结构升级）。

## 任务定义

### 任务1：`case1/case2` 复盘与优化建议
目标：
- 从 `log.json`、`edges.json`、`results.json`、`workflow.yml` 识别高价值 case。
- 输出可执行的优化建议，重点放在：
- 算法策略（路径、聚合、实体消歧、执行稳定性）
- prompt 约束（operator 指令、output schema、workflow 拓扑）

输入范围：
- `test/example/workflow_generator/workflow_space/case1/`
- `test/example/workflow_generator/workflow_space/case2/`

预期产出：
- 一份按优先级排序的优化建议（P0/P1/P2）
- 每条建议附证据路径（具体 round + 现象）
- 建议至少覆盖：算法逻辑、prompt 结构、执行稳定性

### 任务2：数据合成不可回答问题治理
目标：
- 解决由“局部子图生成题目 + 全局图执行评测”导致的答案不一致/不可回答问题。
- 给出可落地方案：过滤规则、问法改写规则、schema 升级方案。

输入范围：
- `test/example/workflow_generator/data_example.json`
- `test/example/workflow_generator/generated_datasets/20260213_144328_query_small_100/dataset.json`
- `extra_doc/data_synthesis_and_graph_task_taxonomy_research.md`

预期产出：
- 问题类型划分（至少：范围不一致、隐式全量枚举、不可执行 verifier）
- 解决方案决策（过滤 vs 改写 vs Row schema 升级）
- 最小落地路线（先做什么、如何验收）


## 任务1独立研究计划：Case学习 + 算法/Prompt优化

### 1. 研究目标与边界

目标：
1. 系统复盘 `case1/case2` 搜索过程，识别“有效改进路径”和“退化触发模式”。
2. 基于证据提出算法与 prompt 的可执行优化方案，并给出优先级。
3. 形成可复现实验结论，而不是仅凭单轮观察。

边界：
1. 仅研究 query-only 场景。
2. 仅使用当前可用工具链（`graph_only` toolset）进行方法验证。
3. 不在本阶段改业务目标，只做方法优化与验证。

### 2. 核心研究问题（Task1-RQ）

1. `Task1-RQ1`：哪些 workflow 拓扑变化会稳定提升分数，哪些会诱发退化？
2. `Task1-RQ2`：prompt 中哪些约束（关系方向、实体消歧、聚合模板）对正确率提升最关键？
3. `Task1-RQ3`：错误反馈（reflection）如何改写为“下一轮可执行动作”才能真正带来提升？
4. `Task1-RQ4`：运行时异常（超时、超长输入、数据库锁）如何影响搜索信号，如何消偏？

### 3. 研究步骤（详细执行版）

#### Step 1：Case证据整理（Case Forensics）
1. 建立 `round-level` 证据表，字段至少包含：
- `round_number`, `parent_round`, `score`, `regression_rate`, `error_rate`
- `workflow_topology_signature`
- `prompt_signature`（指令关键约束）
- `error_type`（语义错/执行错/系统错）
2. 构建“关键路径案例集”：
- 高分路径：如 `case1 round8/round16`, `case2 round4`
- 退化路径：如 `case1 round9`, `case2 round9`
- 异常路径：如 `case2 round5`（bootstrap_error）
3. 输出初版文档：
- `extra_doc/task1_case_evidence_registry.md`

#### Step 2：相关领域文献与工程方案调研（Task1专用）
1. 研究方向A：Agent 搜索与反思
- ReAct, Reflexion, Self-Refine, Tree of Thoughts
2. 研究方向B：受约束生成与执行反馈
- PICARD, Execution-Guided Decoding, Structured Outputs
3. 研究方向C：Text2Cypher/Graph Query 评测与错误分析
- Neo4j Text2Cypher, IBM Mind the Query, CypherBench
4. 每篇/每个项目输出统一卡片：
- `方法摘要` / `可迁移点` / `不适配点` / `落地成本`
5. 输出文档：
- `extra_doc/task1_literature_review.md`

#### Step 3：可验证假设设计（Hypothesis Design）
1. 针对 `Task1-RQ1~RQ4` 定义假设与判定标准。
2. 至少设计三组消融：
- 拓扑消融：线性链 vs 增加验证尾节点
- 指令消融：去/留实体消歧、方向约束、聚合模板约束
- 反馈消融：free-text reflection vs structured action list
3. 定义指标：
- `answer_score`, `execution_success_rate`, `invalid_output_rate`
- `time_to_best_round`, `best_score@k`, `error_breakdown`

#### Step 4：实验执行与误差归因
1. 固定实验配置（seed、top_k、max_rounds、样本集）。
2. 逐组运行并记录结果。
3. 对退化轮次做反事实分析：
- 如果移除某算子/某约束，分数是否恢复？
- 退化是否来自语义错、输出格式错、或系统异常？
4. 产出：
- `extra_doc/task1_ablation_results.md`
- `extra_doc/task1_error_taxonomy.md`

#### Step 5：优化方案收敛与工程落地建议
1. 形成 P0/P1/P2 方案清单（每条必须带证据回链）。
2. 对每条方案给出：
- `修改点`
- `预期收益`
- `风险`
- `验证方法`
3. 产出最终报告：
- `extra_doc/task1_case_study_and_optimization_report.md`

### 4. Task1里程碑与时间预算

1. M1（Day 1-2）：完成证据表与关键 case 集合。
2. M2（Day 3-4）：完成文献调研卡片与可迁移映射。
3. M3（Day 5-7）：完成消融实验与误差归因。
4. M4（Day 8）：提交最终优化报告与实施优先级。

### 5. Task1最终输出文档清单

1. `extra_doc/task1_integrated_research_report.md`
2. 注：原 Task1 子文档内容已并入该文档的合并附录，分散文档已清理。

### 6. Task1验收标准

1. 至少 3 条 P0 优化建议有明确证据与实验支撑。
2. 能解释至少 80% 的退化轮次原因（可归类到既定 taxonomy）。
3. 给出可执行的 prompt/operator 变更清单，不是泛化描述。

## 任务2独立研究计划：数据合成不可回答问题治理

### 1. 研究目标与边界

目标：
1. 系统识别“不可回答样本”成因，尤其是局部子图与全局图答案不一致问题。
2. 建立数据合成质量控制策略：过滤、改写、schema 升级三路径对比。
3. 形成可复现实验，明确最小可落地方案与中长期路线。

边界：
1. 先聚焦 query-only dataset。
2. 优先解决可执行性与一致性，不先追求语言多样性。
3. 以“执行可验证”作为第一质量标准。

### 2. 核心研究问题（Task2-RQ）

1. `Task2-RQ1`：哪些问法模式最容易导致不可回答（如 all/list/every/top-k 无边界）？
2. `Task2-RQ2`：过滤与改写相比，哪种方式在质量/成本上更优？
3. `Task2-RQ3`：引入 RowV2（`scope/verifier/expected`）后，能否显著提升样本可执行率？
4. `Task2-RQ4`：如何设计一套稳定的 dataset QA pipeline，避免后置 LLM judge 的不稳定性？

### 3. 研究步骤（详细执行版）

#### Step 1：样本审计与风险分层
1. 扫描 `data_example.json` 与 `generated_datasets/.../dataset.json`。
2. 标注风险类型：
- `RISK-A` 隐式全量枚举（all/list/every）
- `RISK-B` scope 缺失（未声明 global/local）
- `RISK-C` verifier 不可执行（纯文本答案）
- `RISK-D` 类型/方向语义易错（路径、极值、聚合）
3. 输出审计文档：
- `extra_doc/task2_dataset_risk_audit.md`

#### Step 2：相关领域文献与开源方案调研（Task2专用）
1. Text2Cypher 数据构建与评测：
- Neo4j Text2Cypher, IBM Mind the Query, CypherBench
2. 执行可验证数据生成：
- execution-guided / constrained generation 相关方法
3. 工程框架实践：
- GraphRAG、Neo4j GraphRAG、LangChain GraphCypherQA
4. 输出文档：
- `extra_doc/task2_literature_review.md`

#### Step 3：三方案设计（过滤/改写/结构升级）
1. 方案A：过滤优先
- 规则：屏蔽高风险问法与 scope 缺失样本
- 优点：上线快
- 风险：覆盖率下降
2. 方案B：改写优先
- 将无边界问题改写为有边界问题（限定节点集合、路径上界、时间范围）
- 优点：保留样本量
- 风险：改写质量依赖模板
3. 方案C：RowV2 升级
- 新字段：`scope`, `verifier_type`, `verifier`, `expected`
- 生成后强制执行 verifier 获取 `expected`
- 优点：根治可验证性问题
- 风险：工程改造成本较高

#### Step 4：实验对比与决策
1. 对 A/B/C 在同一基线上做对比实验。
2. 指标体系：
- `unanswerable_rate`
- `verifier_executable_rate`
- `scope_consistency_rate`
- `downstream_eval_score`
- `data_retention_rate`
3. 输出：
- `extra_doc/task2_solution_comparison.md`
- `extra_doc/task2_decision_record.md`

#### Step 5：落地方案与迁移计划
1. 形成“短期+中期+长期”路线图：
- 短期：A（过滤）+ 最小 scope 约束
- 中期：B（改写模板）
- 长期：C（RowV2 + 执行校验）
2. 输出最终报告：
- `extra_doc/task2_data_synthesis_governance_report.md`

### 4. Task2里程碑与时间预算

1. M1（Day 1-2）：完成样本审计与风险分层。
2. M2（Day 3-4）：完成文献调研与方案设计。
3. M3（Day 5-7）：完成 A/B/C 对比实验。
4. M4（Day 8）：提交决策记录与治理报告。

### 5. Task2最终输出文档清单

1. `extra_doc/task2_integrated_research_report.md`
2. 注：原 Task2 子文档内容已并入该文档的合并附录，分散文档已清理。

### 6. Task2验收标准

1. 不可回答样本率显著下降（需给出量化对比）。
2. 新样本 verifier 可执行率达到目标阈值（阈值在实验前固定）。
3. 至少一套方案具备可直接工程化实施的改造清单。

## 计划更新机制（两个任务通用）

### 1. 允许动态更新，但必须满足

1. 新证据出现（新 case、异常模式、文献结论冲突）时可以改计划。
2. 每次改计划必须记录：
- 改动原因
- 被替换的原计划项
- 新计划项
- 影响范围

### 2. 更新记录格式

在本文件追加：
1. `Update Date`
2. `Task`
3. `Change Summary`
4. `Reason`
5. `Impact on Timeline`
6. `New Deliverables (if any)`

## 任务专用参考资料池（初版）

### Task1 重点参考
1. ReAct: https://arxiv.org/abs/2210.03629
2. Reflexion: https://arxiv.org/abs/2303.11366
3. Self-Refine: https://arxiv.org/abs/2303.17651
4. Tree of Thoughts: https://arxiv.org/abs/2305.10601
5. PICARD: https://aclanthology.org/2021.emnlp-main.779/
6. Execution-Guided Decoding: https://arxiv.org/abs/1807.03100
7. Structured Outputs: https://openai.com/index/introducing-structured-outputs-in-the-api/

### Task2 重点参考
1. Neo4j Text2Cypher dataset: https://neo4j.com/blog/developer/introducing-neo4j-text2cypher-dataset/
2. Neo4j benchmark: https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/
3. IBM Mind the Query: https://research.ibm.com/publications/mind-the-query-a-benchmark-dataset-towards-text2cypher-task
4. CypherBench: https://arxiv.org/abs/2412.18702
5. GraphRAG paper: https://arxiv.org/abs/2404.16130
6. GraphRAG repo: https://github.com/microsoft/graphrag
7. Neo4j GraphRAG docs: https://neo4j.com/docs/neo4j-graphrag-python/current/index.html

## 约定
后续你完成了什么需要写到这个文档，便于交接
- 从 2026-02-14 开始，执行模式切换为“文档/计划/分析模式”：仅允许产出文档、计划、idea、分析，不再进行任何代码实现或代码修改。
- 若后续出现实现类需求，先在本文件记录并等待明确解除该限制后再执行。
- 2026-02-15：用户已明确下达实现与测试指令（含真实 LLM 15/30 条验证），上述限制对 Task2 实施阶段已临时解除，并已记录在 Update Record。

## 当前状态

- 已完成：
  - Task1 Step1 初版（case 证据归集 + 错误 taxonomy）
  - Task2 Step1 与 Step4 初版（风险审计 + A/B/C 对比 + 决策记录）
  - Task1 Step2 初版（文献调研 v0）
  - Task2 Step2 初版（文献调研 v0）
  - Task2 Step5 初版（治理报告 v0）
  - Task1 Step5 初版（最终优化报告 v0）
  - Task2 改写模板规范与验收清单 v0
  - 已按要求清理周报/运营模板类扩展文档，仅保留研究计划主线文档
  - Task1/Task2 各自集成为单文档主报告（便于集中阅读）
  - Task1/Task2 分散子文档已全文并入 integrated 文档并删除
- 进行中：
  - Task1 Step3/Step4（假设与消融设计细化、待实验）
  - Task2 模板质量评估口径细化（发布阈值与抽检策略）

## Update Record

### Update Date
2026-02-14

### Task
Task1（case 复盘与优化）

### Change Summary
- 新增 `extra_doc/task1_case_evidence_registry.md`
- 新增 `extra_doc/task1_error_taxonomy.md`
- 新增 `extra_doc/task1_literature_review.md`
- 新增 `extra_doc/task1_ablation_results.md`
- 新增 `extra_doc/task1_case_study_and_optimization_report.md`

### Reason
- 完成 Task1 从证据归集到优化建议收敛的核心研究文档链路。

### Impact on Timeline
- Task1 M1/M2/Step5 初版文档齐备，后续聚焦实验设计细化与验证。

### New Deliverables (if any)
- `extra_doc/task1_case_evidence_registry.md`
- `extra_doc/task1_error_taxonomy.md`
- `extra_doc/task1_literature_review.md`
- `extra_doc/task1_ablation_results.md`
- `extra_doc/task1_case_study_and_optimization_report.md`

---

### Update Date
2026-02-14

### Task
Task2（数据治理）

### Change Summary
- 新增 `extra_doc/task2_dataset_risk_audit.md`
- 新增 `extra_doc/task2_literature_review.md`
- 新增 `extra_doc/task2_solution_comparison.md`
- 新增 `extra_doc/task2_decision_record.md`
- 新增 `extra_doc/task2_rewrite_template_spec.md`
- 新增 `extra_doc/task2_dataset_acceptance_checklist.md`
- 新增 `extra_doc/task2_data_synthesis_governance_report.md`

### Reason
- 完成 Task2 从风险识别到治理决策、模板规范、验收标准的主线文档闭环。

### Impact on Timeline
- Task2 M1/M2/M4 初版文档齐备，后续聚焦口径细化与实验验证设计。

### New Deliverables (if any)
- `extra_doc/task2_dataset_risk_audit.md`
- `extra_doc/task2_literature_review.md`
- `extra_doc/task2_solution_comparison.md`
- `extra_doc/task2_decision_record.md`
- `extra_doc/task2_rewrite_template_spec.md`
- `extra_doc/task2_dataset_acceptance_checklist.md`
- `extra_doc/task2_data_synthesis_governance_report.md`

---

### Update Date
2026-02-14

### Task
文档集清理

### Change Summary
- 删除周报/模板/运营类扩展文档，仅保留研究计划主线产物。

### Reason
- 用户要求清理与研究计划无关内容，减少交接噪音。

### Impact on Timeline
- 文档集合更聚焦，便于后续按研究计划推进。

### New Deliverables (if any)
- 无（清理动作）

---

### Update Date
2026-02-14

### Task
Task1/Task2 集成单文档

### Change Summary
- 新增 `extra_doc/task1_integrated_research_report.md`
- 新增 `extra_doc/task2_integrated_research_report.md`

### Reason
- 用户要求将 Task1 与 Task2 内容分别集中到单一文档，减少分散阅读成本。

### Impact on Timeline
- 不改变研究节奏；仅优化文档组织方式与可读性。

### New Deliverables (if any)
- `extra_doc/task1_integrated_research_report.md`
- `extra_doc/task2_integrated_research_report.md`

---

### Update Date
2026-02-15

### Task
Task1/Task2 文档收敛与清理

### Change Summary
- 将 Task1 原子文档全文并入 `extra_doc/task1_integrated_research_report.md`（新增 Full Source Merge 附录）。
- 将 Task2 原子文档全文并入 `extra_doc/task2_integrated_research_report.md`（新增 Full Source Merge 附录）。
- 删除 Task1/Task2 分散子文档，仅保留两个 integrated 主文档。

### Reason
- 用户要求“Task1、Task2 各自只保留一个集中文档，不要分散”。

### Impact on Timeline
- 不改变研究节奏；交接与查阅成本进一步降低，后续维护入口固定为 2 个主文档。

### New Deliverables (if any)
- `extra_doc/task1_integrated_research_report.md`（已扩展为主报告 + 全量合并附录）
- `extra_doc/task2_integrated_research_report.md`（已扩展为主报告 + 全量合并附录）

---

### Update Date
2026-02-15

### Task
Task1/Task2 集成文档重构（清晰化）

### Change Summary
- 重写 `extra_doc/task1_integrated_research_report.md` 为 v2 结构：`观察 -> 结论 -> 启发 -> 解决方案 -> 项目改造`。
- 重写 `extra_doc/task2_integrated_research_report.md` 为 v2 结构：`观察 -> 结论 -> 启发 -> 解决方案 -> 项目改造`。
- 两份文档均新增“模块级改造清单 + 实施顺序 + 验收标准/发布标准”。

### Reason
- 用户反馈旧版逻辑不清晰，无法直接读出观察、结论、启发与项目改造方向。

### Impact on Timeline
- 不改变任务目标，显著提升后续评审与执行对齐效率。

### New Deliverables (if any)
- `extra_doc/task1_integrated_research_report.md`（v2）
- `extra_doc/task2_integrated_research_report.md`（v2）

---

### Update Date
2026-02-15

### Task
Task2 外部调研深化与方案重构

### Change Summary
- 将 `extra_doc/task2_integrated_research_report.md` 升级为 v4。
- 按用户提出的 5 个问题逐条重构：任务分类标准、意图识别、多标签映射、子图采样改进、质量分布控制、问题样本过滤。
- 明确“局部采样生成 + 全局范围回答”的协议为核心前提，并给出 RowV2.1 与 QA Gate 方案。
- 新增论文/官方标准/大厂文档/开源实现的依据链接清单。

### Reason
- 用户要求继续深入调研（论文、工业界、大厂、开源）并给出带依据的解决方案。

### Impact on Timeline
- 不改动代码实现；提升 Task2 决策依据完整性，可直接进入评审与实施排期。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（v4 深度调研版）

---

### Update Date
2026-02-15

### Task
Task2 目标闭环对齐（用户问题驱动）

### Change Summary
- 在 `extra_doc/task2_integrated_research_report.md` 增补“用户问题驱动闭环”定义。
- 明确主流程：`用户问题 -> 意图识别 -> 同类扩展合成 -> 智能体优化 -> 全局回答用户问题`。
- 新增闭环指标：`user_intent_coverage_rate`, `user_question_success_rate`, `post_synthesis_gain`。

### Reason
- 用户明确 Task2 的最终目标是用合成数据集驱动智能体提升真实用户问题完成率，而不是仅做数据治理。

### Impact on Timeline
- 不新增代码工作；提升方案与业务目标一致性，后续评审重点更清晰。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（补充用户问题驱动闭环）

---

### Update Date
2026-02-15

### Task
Task2 意图识别与多任务分类映射重调研

### Change Summary
- 在 `extra_doc/task2_integrated_research_report.md` 新增 `Appendix B`。
- 专项整理“如何做意图识别并映射到多个图任务分类”，包含：
  - 层级多标签 taxonomy（L0/L1/L2）
  - 5-stage 识别与映射流程（Intent Frame -> Schema Linking -> Multi-label -> Capability 裁剪 -> 回写闭环）
  - 用户问题驱动的同类扩展合成策略
  - 意图层指标与拒收规则
  - 重调研依据链接清单（论文/标准/工业/开源）

### Reason
- 用户要求“重新调研、整理”该主题，并需要可落地的映射方案与依据。

### Impact on Timeline
- 无代码改动；直接提升 Task2 方案可执行性与评审清晰度。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（新增 Appendix B）

---

### Update Date
2026-02-15

### Task
Task2 主文档重构整理（保留历史归档）

### Change Summary
- 将 `extra_doc/task2_integrated_research_report.md` 重构为 v5 精简主线版。
- 主文档仅保留：改造思路、改造理由、依据、落地流程、指标与实施计划。
- 将此前所有历史扩展内容完整归档到：
  - `extra_doc/task2_integrated_research_report_archive_2026-02-15.md`

### Reason
- 用户要求“把整体改造思路、依据、为什么改造写清楚，并去掉无关内容”，同时避免历史信息丢失。

### Impact on Timeline
- 不涉及代码；文档可读性显著提升，且保留历史可追溯性。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（v5 精简主线）
- `extra_doc/task2_integrated_research_report_archive_2026-02-15.md`（历史归档）

---

### Update Date
2026-02-15

### Task
Task2 子图采样“类别可提问性保障”补充

### Change Summary
- 在 `extra_doc/task2_integrated_research_report.md` 第 8 节新增：
  - `8.3 如何保证采样子图可提出目标类别问题`
  - `8.4 采样保障指标与调度策略`
- 引入“前置条件探针 + 接受拒绝采样 + 定向补采样 + 类别配额”机制。

### Reason
- 用户关注“如何保证采样子图支持期望类别问题”，需要从概念升级到可执行机制。

### Impact on Timeline
- 不涉及代码；补齐采样阶段的可控性设计，便于后续实施。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（新增 8.3/8.4）

---

### Update Date
2026-02-15

### Task
Task2 从 QA 扩展到复合分析任务（图表+风险检测+报告）

### Change Summary
- 在 `extra_doc/task2_integrated_research_report.md` 新增第 13 节：
  - 复合任务定义与分类升级（新增 Deliverable 轴）
  - RowV3/TaskBundle 数据协议建议
  - 查询/检测/图表/报告的端到端流程
  - “非法交易/洗钱检测”任务拆解与证据化输出
  - 复合任务 QA Gate 与评估指标

### Reason
- 用户提出目标任务不止 QA，还包括图表制作、分析与文档生成，需要扩展治理和合成方案。

### Impact on Timeline
- 不涉及代码；明确了从 QA 到复合分析任务的升级路线，便于后续实施排期。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md`（新增第 13 节）

---

### Update Date
2026-02-15

### Task
Task2 范围收敛（回到 pure QA 主线）

### Change Summary
- 按用户指示，当前迭代只保留 pure QA 任务主线。
- 在 `extra_doc/task2_integrated_research_report.md` 明确新增“当前执行范围（冻结说明）”。
- 将第 13 节（图表/风险分析/报告复合任务）标记为 `Paused`，仅归档保留。

### Reason
- 用户要求“先把这个部分暂缓，我们先做之前的纯qa”。

### Impact on Timeline
- 聚焦当前主线，减少范围扩散；复合任务后续再单独排期。

### New Deliverables (if any)
- 无（范围与状态更新）

---

### Update Date
2026-02-15

### Task
Task2 工程实现落地 + 真实 LLM 验证 + 后续优化建议

### Change Summary
- 完成 Task2 主链路代码落地（query-only）：
  - `app/core/workflow/dataset_synthesis/model.py`
  - `app/core/workflow/dataset_synthesis/generator.py`
  - `app/core/workflow/dataset_synthesis/utils.py`
  - `app/core/workflow/dataset_synthesis/sampler.py`
  - `app/core/prompt/data_synthesis.py`
  - `app/core/workflow/evaluation/eval_yaml_pipeline.py`
- 完成关键质量门槛修复：
  - intent alias 归一 + 强约束意图集合
  - path 必须有 hop 上界
  - 过滤 Neo4j 不兼容旧语法（`size((...))`）
  - shortestPath 变量遮蔽检测
  - 生成期全局 verifier 执行校验
- 完成 mock 回归验证：
  - `test/example/workflow_generator/mock` 全量通过（30 passed）
  - `test/unit/workflow_generator/test_dataset_generator.py` 扩展并通过
- 完成真实 LLM 测试：
  - 15 条：`test/example/workflow_generator/generated_datasets/20260215_162618_query_real_15/quality_audit.json`
  - 30 条：`test/example/workflow_generator/generated_datasets/20260215_163339_query_real_30/quality_audit.json`

### Reason
- 用户要求先跑低成本真实验证（15 条），确认稳定后再跑 30 条，并要求写入可交接文档。

### Impact on Timeline
- Task2 已从“文档方案”推进到“可运行实现 + 实测闭环”。
- 后续可直接进入“分布均衡与留存率提升”迭代，而非基础稳定性修复。

### New Deliverables (if any)
- 新增真实测试产物目录：
  - `test/example/workflow_generator/generated_datasets/20260215_162618_query_real_15/`
  - `test/example/workflow_generator/generated_datasets/20260215_163339_query_real_30/`
- 更新主报告：
  - `extra_doc/task2_integrated_research_report.md`（增加实现状态、实测结果、问题与优化、taxonomy 评估）

### Known Issues / Next Actions
1. 留存率有提升空间：当前约 `68%~79%`，拒收主要集中在 `implicit_full_enumeration`、`path_missing_hop_bound`、`intent_verifier_mismatch`。
2. 分布控制仍为软约束：30 条样本 `L1=16, L2=10, L3=4`，存在 L1 偏高。
3. 同质化仍存在：无完全重复 task，但存在模板化高相似问法。
4. `task_subtype` 体系结论：当前“可用但不完整”。
5. 下一步优先级：
   - P0：上线 level/subtype 硬配额
   - P0：统一 subtype 与 intent 的规范映射（QueryTaxonomy v2）
   - P1：加多样性惩罚与实体复用上限
   - P1：增强 path/ranking/aggregation 定向采样

---

### Update Date
2026-02-15

### Task
Task2 完整化需求确认（采样意图 + 分类体系 + 全链路闭环）

### Change Summary
- 根据用户最新要求，新增“完整化改造范围”并写入主报告：
  - 完整采样意图字典（SamplingIntentV2）
  - 完整分类体系（QueryTaxonomy v2）
  - 完整任务清单（P0/P1/P2）与验收标准
- 目标从“可运行”提升为“可解释 + 可控 + 可考核”。

### Reason
- 用户明确要求“把采样意图补完整、分类体系补完整，把数据合成整体补完整”，并要求先形成文档化问题与待办。

### Impact on Timeline
- 下一阶段工作中心已明确：优先做 taxonomy 与统计口径统一，再做硬配额与多样性治理。
- 可直接按文档 P0 列表推进实现，不需要再次收敛需求边界。

### New Deliverables (if any)
- `extra_doc/task2_integrated_research_report.md` 新增：
  - `## 14. 下一阶段：采样意图与分类体系“完整化”改造清单（Query-only）`

### Scope Freeze for Next Coding Session
1. 先做 P0，不跨到复合任务（图表/报告）范围。
2. 所有新增字段与指标必须能进入 `quality_audit` 与 `meta`。
3. 先保证 size=15 低成本验证通过，再放大到 size=30。
