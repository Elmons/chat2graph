# Task2 深入调研与方案（v4）

## 0. 先对齐真实业务约束（你强调的关键点）

我们的真实链路是：
- 生成阶段：只能在局部子图上采样并构造候选题目。
- 回答/评测阶段：必须在全局图上回答。

因此 Task2 的目标不是“局部可答”，而是：
- `local_subgraph` 只负责发现候选问题。
- `global_graph` 负责生成标准答案（ground truth）。

如果题目是局部生成但答案不是全局执行得到，样本就不应进入发布集。

## 0.1 用户问题驱动闭环（本项目主目标）

你给出的目标闭环应作为 Task2 的主流程：
1. 输入一个真实用户问题。
2. 对该问题做意图识别，提取任务模式（可多标签）。
3. 基于该模式做“同类扩展合成”，覆盖各个相关类别与难度层。
4. 用扩展后的合成数据集驱动智能体生成/优化。
5. 用优化后的智能体回到真实用户问题做全局回答。

这意味着数据合成不是独立任务，而是“面向用户问题能力提升”的训练数据引擎。

## 0.2 本文后续方案如何服务该闭环

- 第 2/3 节：解决“如何识别模式并映射分类”。
- 第 4/5 节：解决“如何高覆盖合成同类问题且保证质量分布”。
- 第 6 节：解决“如何过滤坏样本，避免训练污染”。
- 第 7 节：把上面三件事整合成可执行流程，最终服务“全局回答用户问题”。

---

## 1. 调研说明（回答你第1个问题）

## 1.1 我这轮具体调研了什么

本轮补充调研覆盖了 4 类证据：
1. 标准与规范：ISO GQL、SQL/PGQ、openCypher。
2. 学术论文：Text2Cypher/NL2GQL、执行约束、评测方法、图采样。
3. 工业与大厂文档：Microsoft GraphRAG、AWS Neptune、OpenAI Structured Outputs、IBM Research、Neo4j。
4. 开源实现：LlamaIndex、LangChain、Neo4j text2cypher 仓库。

## 1.2 关于“是否已做大量全面调研”的诚实结论

- 已达到：可用于制定工程方案的“系统化代表性调研”。
- 尚未达到：覆盖所有论文与所有公司内部实践的“穷尽式调研”。

这个结论是基于公开可得资料做出的工程判断。

---

## 2. 问题1：图数据库任务怎么分类？有规范吗？有官方标准吗？

## 2.1 调研结论

1. 有官方标准的是“查询语言标准”，不是“任务分类标准”。
- ISO/IEC 39075:2024（GQL）已发布。
- SQL:2023 Part 16（SQL/PGQ）定义了在 SQL 中查询属性图的机制。
- openCypher 项目明确以“向 GQL 演进”为目标，并提供 grammar/TCK。

2. 业界有“benchmark workload 分类”，但不是统一 taxonomy 标准。
- LDBC SNB 把 workload 分为 BI 与 Interactive。
- LDBC Graphalytics按算法集合评测（BFS/PR/WCC/CDLP/LCC/SSSP）。

3. Text2Cypher 社区有“复杂度分层/任务分层”实践，但目前多为各数据集自定义。
- Mind the Query 强调 schema/runtime/value checks 与复杂度分层。
- CypherBench 提供大规模多领域 benchmark 与系统化生成管线。

## 2.2 对我们当前分类的优化建议

结论（推断）：我们当前分类可以从“单轴任务类型”升级为“三轴分类”，更适配你的场景。

建议三轴：
1. `Task Axis`（语义任务轴）：lookup/filter/pattern/path/aggregation/ranking/algo。
2. `Capability Axis`（可执行能力轴）：cypher-only / procedure / algo。
3. `Scope Axis`（范围轴，必须显式）：
- `generation_scope=local_subgraph`
- `answer_scope=global_graph`

这三轴组合，才能防止“子图生成语义”混入“全局答案语义”。

---

## 3. 问题2：如何做意图识别并映射到多个图任务分类？

## 3.1 调研启发

1. NL2GQL 最近工作已采用多阶段/多代理结构。
- NAT-NL2GQL：Preprocessor(实体识别/重写/路径链接/schema抽取) -> Generator -> Refiner(执行反馈修正)。

2. Text2SQL 经典结论可迁移：schema linking 是泛化核心。
- RAT-SQL 把 schema encoding + schema linking 作为关键瓶颈。

3. 开源实现已显示“自由生成 + 受约束模板”可并存。
- LlamaIndex 同时提供 `TextToCypherRetriever` 与更受约束的 `CypherTemplateRetriever`。

## 3.2 我们建议的意图识别架构（面向多标签）

1. 第一步：多标签意图识别（非单标签）。
- 一条问题可同时命中多个标签，例如：
- `path + aggregation`
- `filter + ranking`

2. 第二步：schema linking。
- 从问题抽取实体、关系、属性，再与全局 schema 对齐。

3. 第三步：计划路由。
- 高风险意图（path/ranking/aggregation）走模板优先。
- 低风险意图走自由生成+校验。

4. 第四步：执行反馈回写。
- 若全局执行失败或不一致，回写到 Refiner 阶段做重写/降级。

5. 第五步：模式扩展合成（新增，服务你的闭环）。
- 针对当前用户问题识别出的主标签，做“邻近标签扩展”与“难度扩展”：
- 例如用户问题命中 `path + ranking`，则至少扩展到：
- `path_exists`, `shortest_path`, `k_hop_path`, `path_with_filter`, `ranking_on_path_result`
- 并按 `easy/medium/hard` 形成训练批次。
- 扩展后样本仍必须走 `global_verifier -> expected_global`。

---

## 4. 问题3：子图采样算法可以如何改进？

## 4.1 调研要点

1. 随机游走/BFS 的偏差问题是已知问题。
- A Walk in Facebook 指出：BFS/朴素RW有明显偏差；MHRW/RWRW更接近均匀采样。

2. 子图采样可以做“偏差校正 + 方差控制”。
- GraphSAINT：按子图采样并做 bias normalization / variance reduction。
- Cluster-GCN：基于图聚类采样稠密子图块。
- node2vec：通过偏置随机游走在局部/全局邻域探索间调节。

## 4.2 我们场景下的采样升级方案（local proposal 用）

目标不是让子图“替代全局”，而是让子图“更好发现候选问题”。

建议算法：`Stratified Hybrid Sampling (SHS)`
1. 分层种子：按实体类型、度分位、社群簇、时间片分层抽 seed。
2. 混合采样：
- 40% MHRW/RWRW（降 degree bias）
- 40% cluster/社区子图采样（保结构）
- 20% motif/path 定向采样（补 hardest intents）
3. 覆盖约束：控制每批次在 subtype/hop/aggregation/path 上的最小覆盖。
4. 全局回填：任何样本都必须走全局 verifier 得 `expected_global`。

---

## 5. 问题4：大模型合成问题时如何控质量和分布？

## 5.1 调研要点

1. 产业侧数据构建已强调“自动校验 + 人工复核 + 复杂度分层”。
- Mind the Query 明确采用 schema/runtime/value checks + manual review。
- Neo4j Text2Cypher (2024) 数据构建有清洗、去重、语法校验流程。

2. 生成约束与执行约束都重要。
- PICARD：解码期拒绝不合法 token。
- Execution-Guided：执行期剔除错误程序。
- Structured Outputs：可把输出固定到 schema 结构，但不替代语义正确性验证。

## 5.2 我们的质量与分布控制方案

1. 先定义目标分布（配额化）。
- 按 `task_subtype x difficulty x hop x operation` 设目标比例。
- 每批次必须与目标分布的偏差在阈值内。

2. 生成时强制结构输出。
- 输出必须带：
- `generation_scope`, `answer_scope`, `task_subtype`, `global_verifier`, `expected_global`。

3. 执行前后双门控。
- 前门：语法、schema 链接、危险词、范围一致性。
- 后门：全局执行一致性、结果类型一致性、重复样本检测。

4. 模板优先策略。
- 对 path/ranking/aggregation 先模板化，再做语言多样化改写。

---

## 6. 问题5：如何找出有问题的任务并过滤？

## 6.1 过滤规则（发布前硬门槛）

任一命中即拒收：
1. `global_verifier` 不可执行。
2. `expected_global` 缺失或不是全局执行结果。
3. `task` 与 `global_verifier` scope 不一致。
4. 枚举/排序任务缺 cohort 或 tie-break 规则。
5. path 任务缺 hop 上界或方向约束。
6. 出现自相矛盾叙述（答案/验证冲突）。
7. 数据库方言不兼容（例如目标引擎不支持对应语法）。

## 6.2 为什么要加“方言兼容过滤”

- AWS Neptune 文档明确列出 openCypher 支持差异（如 `shortestPath()`、`allShortestPaths()`不支持）。
- 所以同一个任务在不同图引擎上的“可执行性”会不同，必须引入 engine capability matrix。

## 6.3 语义正确性过滤（从评测研究迁移）

- 参考 Test Suite Accuracy 思路：
- 对同一语义构造多个等价查询/等价数据库状态做一致性验证。
- 避免“字符串看起来不一样但语义正确”或“字符串看起来相似但语义错误”。

---

## 7. 最终方案（针对你的场景）

## 7.1 协议级改造（核心）

每条样本至少包含：
- `generation_scope=local_subgraph`
- `answer_scope=global_graph`
- `seed_entities`
- `global_verifier`
- `expected_global`
- `local_evidence`（可选）

解释：local 只负责“出题线索”，global 才是“答案真值来源”。

## 7.2 流程级改造（核心）

统一流程：
1. local sampling 生成候选任务。
2. 编译/生成 `global_verifier`。
3. 在全局图执行，得到 `expected_global`。
4. 再渲染自然语言 task。
5. 通过 QA Gate 才能入库。

## 7.2.1 用户问题驱动版本（你要求的流程）

1. 输入真实用户问题 `Q_user`。
2. 对 `Q_user` 做多标签意图识别，得到 `intent_set`。
3. 按 `intent_set` 触发合成器生成“同类覆盖数据集” `D_intent`。
4. `D_intent` 经全局执行校验后，用于驱动/优化智能体。
5. 用优化后的智能体重新回答 `Q_user`（全局图执行）。

关键判定：
- 如果 `Q_user` 回答失败，要回看 `intent_set` 覆盖是否不足，而不是只调 prompt。

## 7.3 治理级改造（核心）

上线指标：
- `global_executable_rate`
- `scope_alignment_rate`
- `local_global_consistency_rate`
- `unanswerable_rate_on_global`
- `data_retention_rate`

建议首版门槛：
- `global_executable_rate >= 95%`
- `scope_alignment_rate >= 95%`
- `local_global_consistency_rate >= 95%`
- `unanswerable_rate_on_global <= baseline * 50%`

## 7.4 新增闭环指标（面向用户问题成功率）

除数据质量指标外，新增任务效果指标：
1. `user_intent_coverage_rate`：给定用户问题，其相关意图是否被合成数据集充分覆盖。
2. `user_question_success_rate`：智能体在真实用户问题上的全局回答成功率。
3. `post_synthesis_gain`：引入 `D_intent` 前后，用户问题成功率提升幅度。

---

## 8. 项目改造建议（仅文档方案，不改代码）

| 改造项 | 目标 | 建议落点 |
| --- | --- | --- |
| RowV2.1 数据契约 | 固化 local-generate/global-answer | `app/core/workflow/dataset_synthesis/model.py` |
| SHS 采样策略 | 降偏+提覆盖 | `app/core/workflow/dataset_synthesis/sampler.py` |
| 生成流程重排 | expected 必须全局执行得到 | `app/core/workflow/dataset_synthesis/generator.py` |
| Prompt 合约收紧 | 结构化输出 + 范围显式 | `app/core/prompt/data_synthesis.py` |
| QA Gate 升级 | 可执行/一致性一票否决 | `app/core/workflow/evaluation/eval_yaml_pipeline.py` |
| 方言能力矩阵 | 引擎差异过滤 | `app/core/workflow/dataset_synthesis/utils.py` |

---

## 9. 证据与依据（按来源类型）

## 9.1 官方标准与规范

1. ISO GQL 标准页面：
- https://www.iso.org/standard/76120.html
2. openCypher 规范仓库（含 grammar/TCK，目标对齐 GQL）：
- https://github.com/opencypher/openCypher
3. SQL/PGQ（SQL:2023 Part 16）与 PGQL 对齐说明：
- https://pgql-lang.org/
4. Neo4j 对 GQL 一致性说明：
- https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/

## 9.2 基准与任务分类参考

1. LDBC SNB（BI + Interactive workloads）：
- https://ldbcouncil.org/benchmarks/snb/
2. LDBC Graphalytics（BFS/PR/WCC/CDLP/LCC/SSSP）：
- https://ldbcouncil.org/benchmarks/graphalytics/algorithms/
3. CypherBench（11个大规模多领域图 + 1万+问题）：
- https://arxiv.org/abs/2412.18702
4. Mind the Query（schema/runtime/value checks + complexity）：
- https://research.ibm.com/publications/mind-the-query-a-benchmark-dataset-towards-text2cypher-task

## 9.3 意图识别与生成流程

1. NAT-NL2GQL（Preprocessor/Generator/Refiner）：
- https://arxiv.org/abs/2412.10434
2. RAT-SQL（schema linking 关键性）：
- https://arxiv.org/abs/1911.04942
3. LlamaIndex TextToCypherRetriever / CypherTemplateRetriever：
- https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/
4. LangChain GraphCypherQAChain（权限与安全提示）：
- https://api.python.langchain.com/en/latest/community/chains/langchain_community.chains.graph_qa.cypher.GraphCypherQAChain.html

## 9.4 采样与偏差控制

1. A Walk in Facebook（BFS/RW 偏差，MHRW/RWRW）：
- https://arxiv.org/abs/0906.0060
2. GraphSAINT（subgraph sampling + bias normalization）：
- https://arxiv.org/abs/1907.04931
3. Cluster-GCN（cluster-based subgraph block sampling）：
- https://arxiv.org/abs/1905.07953
4. node2vec（biased random walk 探索策略）：
- https://arxiv.org/abs/1607.00653

## 9.5 质量控制与过滤

1. Execution-Guided Decoding：
- https://arxiv.org/abs/1807.03100
2. PICARD（约束解码拒绝非法 token）：
- https://arxiv.org/abs/2109.05093
3. Test Suite Accuracy（语义级评测思路）：
- https://arxiv.org/abs/2010.02840
4. OpenAI Structured Outputs（schema 约束输出）：
- https://openai.com/index/introducing-structured-outputs-in-the-api/
5. AWS Neptune openCypher 兼容性差异（方言过滤依据）：
- https://docs.aws.amazon.com/neptune/latest/userguide/feature-opencypher-compliance.html

## 9.6 大厂/工业实践补充

1. Microsoft GraphRAG（Local/Global query 分工）：
- https://microsoft.github.io/graphrag/
- https://microsoft.github.io/graphrag/query/global_search/
2. Neo4j Text2Cypher 2024 数据集：
- https://neo4j.com/blog/developer/introducing-neo4j-text2cypher-dataset/
- https://huggingface.co/datasets/neo4j/text2cypher-2024v1
3. Neo4j text2cypher 开源仓库：
- https://github.com/neo4j-labs/text2cypher

---

## 10. 下一步文档任务（不写代码）

1. 输出《Task2 RowV2.1 字段字典与样例规范（20条）》。
2. 输出《Task2 QA Gate 判定手册（含拒收案例库）》。
3. 输出《Task2 采样分布看板定义（指标+阈值+周报模板）》。

---

## Appendix A: 历史版本保留（v3 原文，未删减）

# Task2 数据合成治理与项目改造方案（v3）

## 0. 先校正一个前提（关键）

你给的业务约束是：
- 数据生成阶段：受限于图规模，只能基于局部采样子图构造候选问题。
- 评测/回答阶段：必须在全局图范围回答。

这意味着 Task2 的核心不是“局部自洽”，而是“局部生成 -> 全局可答”的一致性。

## 0.1 调研覆盖度说明（对齐你的问题1）

当前调研结论：
1. 已做的是“代表性调研”，不是“大规模穷尽调研”。
2. 已覆盖类型：Text2Cypher 数据集实践、execution-guided/constrained generation 思路、GraphCypherQA/GraphRAG 工程经验。
3. 未完成部分：
- 系统化论文矩阵（按任务类型/评测协议/可执行验证机制分层）
- 大厂内部未开源产出的完整对照（公开信息之外无法完全覆盖）

因此，当前方案是“可落地工程方案”，不是“文献穷尽后的最优理论解”。

---

## 1. 我观察到了什么（基于当前样本）

| 事实编号 | 观察 | 影响 |
| --- | --- | --- |
| O1 | `RISK-C (verifier 不可执行)` 近乎全量高风险 | 无法稳定验证“全局答案”是否正确 |
| O2 | 大量样本未声明 scope 或 scope 语义混乱 | 容易把“局部发现”误当“全局真值” |
| O3 | path/rank/transfer 等方向敏感任务风险高 | 局部图与全局图的边/路径差异会被放大 |
| O4 | 存在叙事型 verifier 和自相矛盾描述 | 无法进入自动化全局验真流水线 |

关键新理解：
- 不是局部子图错，而是“把局部生成结果直接当全局答案依据”的协议错。

---

## 2. 我得到了什么结论

1. 题目可以来自局部子图，但 `expected answer` 必须来自全局执行结果。
2. 局部子图在协议中应降级为“候选问题发现器（proposal generator）”，不是“真值来源”。
3. 约束重点应从“限制问法文本”升级为“限制数据契约 + 执行路径”。
4. 只做过滤/改写不足以解决局部-全局错位，必须引入全局执行回填。

---

## 3. 约束到底怎么加（你关心的重点）

## 3.1 先加“范围契约约束”

每条样本新增/固定以下字段语义：
- `generation_scope`: 固定为 `local_subgraph`
- `answer_scope`: 固定为 `global_graph`
- `seed_entities`: 局部采样时的锚点实体 ID 集
- `global_verifier`: 必须可执行（Cypher/Procedure）
- `expected_global`: `global_verifier` 在全局图执行后的结果
- `local_evidence`（可选）：局部图里触发该问题的证据片段

硬规则：
- 没有 `global_verifier` 或执行失败 -> 拒收
- `expected_global` 不是全局执行结果 -> 拒收

## 3.2 再加“生成流程约束”

新流程必须是：
1. 在局部子图中挖掘候选任务（只负责“发现”）。
2. 将候选任务编译成全局可执行查询（`global_verifier`）。
3. 在全局图执行，得到 `expected_global`。
4. 再生成自然语言 `task`（与全局执行语义一致）。
5. 通过 QA gate 后入库。

不是：`task -> 文本 verifier -> 人工猜 expected`。

## 3.3 最后加“语义一致性约束”

1. `task` 不得隐含“只在局部图成立”的限定词。
2. 若是全局枚举/排序题：
- 必须显式 cohort（人群定义）和排序规则（tie-break）。
3. 若是 path 题：
- 必须显式 hop 上限、方向约束和并列处理规则。
4. `local_evidence` 若存在，只能作为“证据样例”，不能覆盖 `expected_global`。

---

## 4. 具体拒收规则（发布前）

任一命中即拒收：
1. `global_verifier` 不可执行。
2. `expected_global` 缺失或与任务预期语义不一致。
3. `task` 与 `global_verifier` scope 不一致。
4. 枚举/排序任务缺 cohort 或 tie-break。
5. path 任务缺 hop/方向边界。
6. verifier 或答案出现自相矛盾。

---

## 5. 怎么改造我们的项目（模块级，不改代码只给方案）

| 改造项 | 目标 | 建议落点 | 方案摘要 |
| --- | --- | --- | --- |
| RowV2.1 数据契约 | 固化 local-generate/global-answer 协议 | `app/core/workflow/dataset_synthesis/model.py` | 增加 `generation_scope/answer_scope/seed_entities/global_verifier/expected_global/local_evidence` |
| 生成管线重排 | 确保 expected 来源于全局执行 | `app/core/workflow/dataset_synthesis/generator.py`, `app/core/workflow/dataset_synthesis/utils.py`, `app/core/workflow/dataset_synthesis/sampler.py` | 改为 `local proposal -> global verifier -> global execute -> render task` |
| Prompt 合约收紧 | 减少叙事型 verifier 与 scope 漂移 | `app/core/prompt/data_synthesis.py` | 明确禁止 narrative verifier，强制输出结构化字段 |
| QA Gate 升级 | 发布前强制一致性审核 | `app/core/workflow/evaluation/eval_yaml_pipeline.py` | 新增 `global_executable_rate`, `scope_alignment_rate`, `local_global_consistency_rate` |
| 风险策略配置化 | 分阶段治理可切换 | `app/core/workflow/operator_config.py`, `app/core/workflow/dataset_synthesis/task_subtypes.py` | 将拒收阈值和模板策略参数化 |

---

## 6. 指标与门槛（围绕“全局回答”）

必须长期跟踪：
1. `global_executable_rate`（全局可执行率）
2. `scope_alignment_rate`（task 与 verifier 范围一致率）
3. `local_global_consistency_rate`（局部证据与全局结果不冲突率）
4. `unanswerable_rate_on_global`（全局不可回答率）
5. `data_retention_rate`（保留率）

建议门槛（首版）：
- `global_executable_rate >= 95%`
- `scope_alignment_rate >= 95%`
- `local_global_consistency_rate >= 95%`
- `unanswerable_rate_on_global <= baseline * 50%`

---

## 7. 分阶段实施（只写计划）

1. Phase-S（1周）
- 定义 RowV2.1 字段契约
- 上线拒收规则草案
- 跑第一版风险与保留率报告

2. Phase-M（1~2周）
- 全量切换到 `global_verifier -> expected_global`
- 重点覆盖 path/rank/transfer 三类高风险模板

3. Phase-L（持续）
- 建立漂移监控和分层评测集
- 每周复盘“拒收原因 TopN”和“全局不可回答 TopN”

---

## 8. 下一步（你确认后我继续）

1. 我补一份“调研矩阵文档”（论文/工业/开源，按问题-方法-可迁移性分栏），把“代表性调研”升级为“系统调研”。
2. 我补一份 Task2 的“RowV2.1 字段字典 + 20 条样例规范”（全是文档，不写代码）。
3. 我补一份“局部生成-全局回答”QA 审核清单（供你评审）。

---

## Appendix B: 意图识别并映射到多个图任务分类（重调研整理版，2026-02-15）

本附录在不删除原内容的前提下，单独回答：
- 如何做意图识别？
- 如何映射到多个图任务分类？
- 如何用于你定义的“用户问题驱动闭环”？

## B1. 先说结论

1. 不应做单标签分类，必须做“层级多标签”意图识别。
- 一个用户问题常同时包含多个子意图：例如 `path + filter + ranking`。

2. 映射应分 3 轴而不是 1 轴：
- `语义任务轴`：lookup/filter/pattern/path/aggregation/ranking/algo
- `能力轴`：cypher-only / procedure / algo
- `范围轴`：`generation_scope=local_subgraph`, `answer_scope=global_graph`

3. 映射输出不是“类别名称”，而是“可执行计划槽位（intent frame）”。
- 这样才能直接驱动数据合成与智能体优化。

## B2. 为什么要层级多标签（调研依据）

1. 图查询与 SQL/NL2GQL 一样，复杂问题本质上是组合任务，不是单一任务。
- NAT-NL2GQL 采用 Preprocessor/Generator/Refiner 多阶段，Preprocessor 包含实体识别、重写、path linking、schema 提取，说明单步单标签不足。

2. schema linking 是跨库泛化关键点。
- RAT-SQL 的核心贡献就是 relation-aware schema encoding + linking，验证了“先对齐 schema 再生成查询”的必要性。

3. 工业开源也在走“自由生成 + 受约束模板”并行路线。
- LlamaIndex 同时提供 `TextToCypherRetriever`（自由生成）和 `CypherTemplateRetriever`（模板受约束）。

## B3. 多图任务分类（建议的层级体系）

## B3.1 L0（顶层族）

- `lookup`：实体/属性查找
- `filter`：条件过滤
- `pattern`：模式匹配（star/chain/cycle）
- `path`：可达性/最短路径/路径约束
- `aggregation`：count/sum/avg/group
- `ranking`：order by/top-k/tie-break
- `algorithm`：中心性/社区/图算法（仅当能力支持）

## B3.2 L1（可执行子类）

示例：
- `path.exists`, `path.shortest`, `path.k_hop`
- `aggregation.count`, `aggregation.group_count`, `aggregation.sum`
- `ranking.topk`, `ranking.argmax`
- `pattern.chain`, `pattern.star`, `pattern.cycle`

## B3.3 L2（执行约束标签）

- 方向约束：`directed` / `undirected`
- 范围约束：`global_answer_required`
- 稳定性约束：`requires_tie_break`, `requires_hop_bound`
- 引擎约束：`needs_shortest_path_builtin`, `needs_procedure`

解释：
- L0/L1 解决“这是什么任务”。
- L2 解决“怎么安全、可执行地完成任务”。

## B4. 意图识别与映射流程（推荐落地）

## Stage 1: 结构化解析（Intent Frame Extraction）

输入：用户原问题 `Q`
输出：
- `entities`
- `relations`
- `constraints`（时间、金额、路径、排序）
- `operations`（count/top-k/shortest/...）
- `scope_requirement`（默认全局回答）

要求：
- 用结构化输出（JSON Schema）约束，避免自由文本漂移。

## Stage 2: Schema Linking

- 将 `entities/relations/properties` 映射到真实图 schema。
- 输出候选映射及置信度。
- 低置信项进入澄清或保守模板路径。

## Stage 3: 层级多标签分类

- 基于 Stage1+2 结果预测 `L0/L1/L2` 多标签。
- 不是互斥分类，而是集合预测：`intent_set = {path.shortest, filter, ranking.topk, requires_hop_bound}`。

## Stage 4: 能力与方言裁剪

- 用 engine capability matrix 过滤不可执行标签。
- 例如 Neptune 不支持 `shortestPath()` / `allShortestPaths()`，则相关计划必须降级。

## Stage 5: 计划生成与闭环回写

- 将 `intent_set` 映射到合成模板与执行计划。
- 回答失败时，先检查“意图覆盖不足”而不是盲目调 prompt。

## B5. 从“用户问题”到“多类合成”的映射策略

给定用户问题 `Q_user`：
1. 识别主意图 `core_intents`。
2. 在意图图中扩展邻近子类 `neighbor_intents`（同语义族、同操作族、同难度层）。
3. 生成合成训练集 `D_intent = core + neighbor`。
4. 每条样本必须满足：
- `generation_scope=local_subgraph`
- `answer_scope=global_graph`
- `global_verifier` 可执行
- `expected_global` 来自全局执行

这样能保证：
- 训练数据是“围绕真实用户问题”的，而不是无关随机题。

## B6. 质量控制与过滤（针对意图映射）

在现有 QA gate 之外，新增 4 个意图层指标：
1. `intent_precision`：预测标签中真正相关的比例。
2. `intent_recall`：真实需要标签被覆盖比例。
3. `intent_coverage_rate`：目标意图图中的覆盖率。
4. `cross_intent_confusion`：易混标签对（如 path.exists vs path.shortest）的混淆率。

拒收新增规则：
- 若样本标签与 `global_verifier` 的操作不一致（例如标为 ranking 但查询无排序），拒收。
- 若标为 path.shortest 但无 hop/方向/并列规则，拒收。

## B7. 你这个场景下的最小可行实现（文档版）

1. 建立 `intent taxonomy v1`（L0/L1/L2）与标签字典。
2. 定义 `intent frame schema`（结构化输出规范）。
3. 建立 `engine capability matrix`（至少覆盖 Neo4j / Neptune）。
4. 建立 `intent->template` 映射表（先覆盖 path/filter/aggregation/ranking）。
5. 建立 `intent QA` 指标报表（每周更新）。

## B8. 常见误区（这次重点规避）

1. 误区：把意图识别当作单标签分类。
- 修正：采用层级多标签 + 约束标签。

2. 误区：只识别语义，不做能力裁剪。
- 修正：增加 capability/方言矩阵。

3. 误区：只看语法正确，不看语义可执行。
- 修正：必须对齐 `intent_set` 与 `global_verifier` 执行行为。

4. 误区：合成数据和真实用户问题脱节。
- 修正：以 `Q_user` 为锚点，做意图邻域扩展合成。

## B9. 依据（重调研来源）

1. NAT-NL2GQL（多阶段 NL2GQL：Preprocessor/Generator/Refiner）
- https://arxiv.org/abs/2412.10434

2. RAT-SQL（schema linking 对泛化的重要性）
- https://arxiv.org/abs/1911.04942

3. PICARD（约束解码）
- https://arxiv.org/abs/2109.05093

4. Execution-Guided Decoding（执行反馈过滤）
- https://arxiv.org/abs/1807.03100

5. OpenAI Structured Outputs（结构化输出约束）
- https://openai.com/index/introducing-structured-outputs-in-the-api/

6. LlamaIndex TextToCypherRetriever / CypherTemplateRetriever（自由+模板并行）
- https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/

7. LDBC SNB Interactive（图查询 workload 参考）
- https://ldbcouncil.org/benchmarks/snb/interactive/

8. LDBC Graphalytics Algorithms（图算法 workload 参考）
- https://ldbcouncil.org/benchmarks/graphalytics/algorithms/

9. ISO/IEC 39075:2024 GQL（语言标准）
- https://www.iso.org/standard/76120.html

10. openCypher（向 GQL 演进）
- https://opencypher.org/

11. Neo4j GQL conformance（实现层合规）
- https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/

12. AWS Neptune openCypher compliance（方言差异约束）
- https://docs.aws.amazon.com/neptune/latest/userguide/feature-opencypher-compliance.html

13. Mind the Query（Text2Cypher 数据构建与复杂度分层）
- https://research.ibm.com/publications/mind-the-query-a-benchmark-dataset-towards-text2cypher-task

14. Neo4j Text2Cypher 2024（数据清洗与语法校验流程）
- https://neo4j.com/blog/developer/introducing-neo4j-text2cypher-dataset/

15. CypherBench（大规模 Text2Cypher benchmark 与生成管线）
- https://arxiv.org/abs/2412.18702
