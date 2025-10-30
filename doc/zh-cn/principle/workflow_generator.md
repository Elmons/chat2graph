---
title: 工作流自动生成
---

## 1. 介绍

工作流自动生成模块能够根据具体的”任务描述“，端到端完成工作流的自动生成与优化。具体来说，该模块可以把“任务描述”转化为可执行的agentic sdk配置，配置描述了智能体的定义及工作流，配置的具体方式参考[sdk](./sdk.md)。工作流自动生成模块由两个部分组成：
- **数据合成器**（`app/core/workflow/dataset_synthesis`）：数据合成器的作用是为工作流的自动生成提供数据基础。其核心思想是通过子图采样，基于大模型进行数据合成，并对合成的数据进行过滤，形成数据集`WorkflowTrainDataset` 供后续评估。如果你已经有了自己的数据集，可以使用自己的数据集完成工作流的自动生成。
- **工作流生成器**（`app/core/workflow/workflow_generator/mcts_workflow_generator`）：工作流生成器的作用是根据数据集和任务描述，从一个基本的yaml配置模板开始，通过 MCTS进行迭代搜索，每一轮迭代通过大模型优化循优化operators / experts，最终给出搜索过程中最优轮次的工作流（以yaml配置文件形式给出）。
  
## 2. 设计
### 2.1 总体流程
整体的框架与流程如下：
![](../../asset/image/workflow_generation_arch.png)

具体步骤：
1. **提出需求/任务描述**：用户提出一个任务描述/任务需求，例如图中的“你的任务是完成图数据库查询相关的任务”。
2. **数据合成（可选）**：根据用户的需求以及**准备好的图数据库**，基于LLM合成数据集，用于后续的训练。**如果已有数据，可以省略该步骤**。
3. **指定初始工作流模板**：指定一个yaml配置文件模板作为初始工作流。
4. **工作流自动生成与优化**：根据数据集和初始工作流，通过蒙特卡洛树搜索的方法，自动生成与优化工作流。
5. **输出工作流**：最终优化得到的工作流，通过sdk配置文件的形式保存。


### 2.2 数据合成器
数据合成器主要负责从任务描述合成数据集，为工作流的自动生成提供数据基础，弥补训练数据的空白。
其接口格式如下：
```Python
class DatasetGenerator(ABC):
    @abstractmethod
    async def generate(
        self, task_desc: str, dataset_name: str, size: int
    ) -> WorkflowTrainDataset: ...
```

#### 2.2.1 数据集格式
如下是一个具体的数据集的示例：
```JSON
[
  {
    "level": "L1",
    "task_type": "query",
    "task_subtype": "Entity Attribute and Label Query",
    "task": "What is the city of residence for the person named Dietsce?",
    "verifier": "Chengde"
  },
  {
    "level": "L2",
    "task_type": "query",
    "task_subtype": "Multi-Step Chain Reasoning Query",
    "task": "Who is the owner of the account that received a deposit from the loan with ID '4866420872350534055'?",
    "verifier": "The loan with ID '4866420872350534055' was deposited into the account owned by the person named Yudhoyono."
  },
  ...
]
```
其中每个字段含义如下：
- `level`：任务的难度，目前定义有L1/L2/L3/L4难度，具体可以参考`app/core/workflow/dataset_synthesis/task_subtypes.py`文件。
- `task_type`：任务的类型，可以是查询类（query）和非查询类（non-query），**目前non-query暂未实现**。
- `task_subtype`：更加细粒度的任务类型
- `task`：任务的文本描述。
	- query类任务：实际上等同于问题（question）
- `verifier`：验证器，用于后续评估时验证任务是否执行正确。
	- query类任务：verifier等同于问题的答案（anwser）。
#### 2.2.2 基于大模型和子图采样的数据合成器
我们具体实现了一种基于大模型和子图采样的数据合成方法，**核心思路**是通过采样子图获得一个局部子图，让大模型根据局部子图合成数据得到初步的数据集，再对初步的数据进行过滤得到最终的数据集。

基于大模型和子图采样的数据合成器主要由三部分组成：`sampler`，`generator`，`filter`，分别负责子图采样、数据合成、数据过滤。
![](../../asset/image/data_synthesis.png)

关键步骤：
1. **子图采样**：默认使用 `RandomWalkSampler`，通过随机游走的方式在图数据库中采样子图，可通过替换为自定义实现。
    - 选起始点：随机挑一个未被采样过的节点作为起点。
	 - 多轮游走（最多 `max_depth` 轮）：
		- 从当前“前沿节点”出发，用 Cypher 查询邻居；
		- 根据随机生成的 `dfs_bias` 权衡“深入探索”（DFS）和“广度扩展”（BFS）
		- 按权重随机排序，选新节点和边，不超过剩余的节点数和边数
	- 将采样到的子图格式化为字符
2. **数据生成**：将采样到的子图嵌入到Prompt中，调用大模型进行数据合成。
3. **过滤**：通过大模型过滤出现幻觉、不符合条件的数据。


### 2.3 工作流生成器
工作流生成器负责生成并优化工作流，最终输出一份优化好的sdk声明式配置文件`workflow.yml`，该配置文件描述了系统中的`expert`及其对应的`workflow`。

我们具体实现参考了Aflow论文，**核心思想**是基于蒙特卡洛搜索树的搜索框架进行workflow的生成与优化。
- 在原基础上适配到chat2graph的sdk声明式配置文件，实现从“生成与优化代码”到”生成与优化配置文件“的转变。
- 在对工作流进行拓展/优化阶段，采用了分层优化的策略，通过”上下文工程“控制每一层优化的上下文，缓解幻觉问题。

#### 2.3.1 建模
整体建模如下所示：
![](../../asset/image/mcts_search.png)
具体来说：
- **工作流/智能体系统的表示**：通过chat2graph的声明式配置文件，描述了整个智能体系统/工作流，**在本小节中工作流实际等同于配置文件，统一用工作流来描述**。配置文件的约定请参考[sdk](./sdk.md)
- **搜索树中的节点**：搜索树中每一个节点都代表了**一份完整的配置文件**，即一个完整的工作流/智能体系统。
    - 除此之外，每个节点还存储了评估的分数、反馈信息等其他上下文，用于指导后续的优化
- **搜索框架**：基于蒙特卡洛树搜索，通过选择-拓展-评估-反馈的循环，逐步优化初始的配置文件。

#### 2.3.2 关键步骤/迭代过程
-  **初始化**
-  **迭代优化**
	1. **选择工作流**：`selector`负责从当前的搜索树中选择一个节点。
	2. **优化工作流**：基于LLM，对选择的节点进行优化。具体实现我们采取了分层优化的策略，先生成优化建议，再分别单独优化工作流的`operator`和`expert`部分。
	3. **执行工作流**：在数据集上执行新生成的工作流，记录执行结果。
	4. **评估工作流**：基于大模型，根据执行结果进行评估，并进行反思，生成迭代建议
	5. **反向传播**：保存得分、反思、修改、反馈、动作建议等到日志中，并将一些反馈信息反向传播到父节点。
-  **输出最优的工作流**

## 3. API与使用示例
### 3.1数据合成器
数据合成器提供的接口如下：
```Python
class DatasetGenerator(ABC):
    @abstractmethod
    async def generate(
        self, task_desc: str, dataset_name: str, size: int
    ) -> WorkflowTrainDataset: ...
```
参数说明：
- `task_desc`：任务描述
- `dataset_name`：合成的数据集名称
- `size`：合成的数据集规模


我们具体实现的数据合成器的构造函数如下：
```Python
class SamplingDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        graph_db: GraphDb,
        sampler: SubGraphSampler,
        strategy: GENERATOR_STRATEGY = None,
        max_depth: int = 2,
        max_noeds: int = 10,
        max_edges: int = 20,
        nums_per_subgraph: int = 10,
```
**参数说明**：
- `graph_db`：图数据库连接
- `sampler`：具体子图采样器，可以自定义为自己的子图采样算法，是一种策略设计模式。
- `strategy`：这里是生成数据集的策略，有三种“query", "non-query", "mixed"，分别对应只生成查询类数据，只生成非查询类数据，混合数据。
- `max_depth`：采样的子图的最大深度
- `max_noeds`：采样的子图的最大节点数
- `max_edges`：采样的子图的最大边数
- `nums_per_subgraph`：每个采样的子图进行数据合成时生成的数据量大小。

**使用示例：**
```Python
db = register_and_get_graph_db()
dataset_generator: DatasetGenerator = SamplingDatasetGenerator(
    graph_db=db,
    sampler=RandomWalkSampler(),
    strategy="query",
    max_depth=5,
    max_noeds=15,
    max_edges=30,
    nums_per_subgraph=10,
)

train_set = await dataset_generator.generate(
    "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等",
    dataset_name="test",
    size=10,s
)
print(f"train_set={train_set}")
```


### 3.2 工作流生成器
工作流生成器的抽象接口如下：
```Python
class WorkflowGenerator(ABC):
    """Interface that all workflow generators must implement."""

    @abstractmethod
    async def generate(self) -> WorkflowGenerationResult:
        """Execute the generation procedure and return the best found workflow."""
```
主要考量到后续可能有不同的实现方式，因此具体的参数传递会通过初始化具体实现的时候传递。


我们具体实现的MCTSWorkflowGenerator的构造函数如下：
```Python
class MCTSWorkflowGenerator:
    def __init__(
        self,
        db: GraphDb, 
        dataset: WorkflowTrainDataset, 
        selector: Selector, 
        expander: Expander, 
        evaluator: Evaluator, 
        optimize_grain: List[AgenticConfigSection], 
        init_template_path: str = "./init_template/basic_template.yml", 
        max_rounds: int = 30, 
        optimized_path: str = "workflow_space", #
        top_k: int = 5, 
        max_retries: int = 5, 
    ):
```
**参数说明：**
- `db`：图数据库
- `dataset`：数据集的内容，包括`name`、`task_desc`、`data`三个字段，分别对应了数据集的名称、任务描述、具体的数据。
	- **注意**：这里的`name`会和`optimized_path`一起组合，共同构成一个路径`${optimized_path}_{name}`作为工作目录，记录整个执行过程中输出的所有内容
- `selector`：负责选择一个节点。
- `expander`：负责从选择的节点进行拓展
- `evaluator`：负责对拓展后的工作流进行评估
- `optimize_grain`：优化粒度主要指优化op、expert、actions、tools中的哪几个部分。**暂未用到**，目前只支持优化op和expert
- `init_template_path`：初始模板的路径
- `max_rounds`：最大迭代次数
- `optimized_path`：与`dataset`的`name`组合构成工作目录，存储了执行过程中的产出内容，包括：每一轮迭代输出的`workflow.yaml`及对应的评估结果`result.json` 
	- **例如**：假设执行脚本的工作目录为`/home/example.py`，`dataset.name = test`，`optimized_path`= `"workflow_space"`，那么执行过程中第4轮，会生成`/home/example/workflow_space/test/round4`文件夹，并存储了该轮生成的`workflow.yaml`和评估结果`result.json`
- `top_k`：从树中采样的节点的规模
- `max_retries`：执行过程如果遇到错误最大重试次数

**使用示例：**
```Python
selector = MixedProbabilitySelector()
expander = LLMExpander()
evaluator = LLMEvaluator()

# 定义mcts搜索框架
workflow_generator = MCTSWorkflowGenerator(
    db=db, 
    dataset=dataset, 
    selector= selector, 
    expander=expander, 
    evaluator=evaluator,
    max_rounds=3,
    optimized_path=Path(__file__).resolve().parent / "workflow_space",
    top_k=5,
    max_retries=5,
    optimize_grain=None,
    init_template_path="app/core/workflow/workflow_generator/mcts_workflow_generator/init_template/basic_template.yml"
    )

# mcts主流程入口，开始进行工作流生成与优化
await workflow_generator.generate()   
```
## 4. 更具体的使用示例与测试
**完整使用示例**
>**示例运行前提**
>1. 配置好图数据库
>2. 图数据库中存在一定的数据，取决于使用时子图采样的规模
>   
>   可以使用下面的neo4j docker镜像来获取我们实验中使用的图数据库，该数据库导入了Finbench sf1数据：
>   [todo]()
>

- `test/example/workflow_generator/dataset_generator_example.py`：给出了完整的使用数据合成器的示例。
- `test/example/workflow_generator/workflow_generator_example.py`：给出了完整的端到端的使用工作流生成器的示例。


**单元测试**：使用了Mock的形式测试，不需要连接数据库
- `test/unit/workflow_generator/test_dataset_generator.py`：数据合成的单元测试
- `test/unit/workflow_generator/test_workflow_generator.py`：工作流自动生成的单元测试

 
