import asyncio
import json
import logging
from pathlib import Path
import sys

from app.core.sdk.init_server import init_server
from app.core.workflow.dataset_synthesis.generator import DatasetGenerator, SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.sampler import RandomWalkSampler
from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset
from app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator import LLMEvaluator
from app.core.workflow.workflow_generator.mcts_workflow_generator.expander import LLMExpander
from app.core.workflow.workflow_generator.mcts_workflow_generator.generator import (
    MCTSWorkflowGenerator,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.selector import (
    MixedProbabilitySelector,
)
from test.example.workflow_generator.utils import register_and_get_graph_db

logger = logging.getLogger(__name__)

"""Workflow Generator Example (MCTS)

本示例用于演示：在一个已注册的图数据库上，通过 MCTS（Monte Carlo Tree Search）框架生成/优化工作流。

重要说明（与“代理/网络/环境变量”相关）：
- 本文件不再读取任何环境变量（例如 `os.getenv`），所有配置都在代码中“写死”。
- 如果你的 LLM/Embedding/GraphDB 连接依赖环境变量，请去对应的组件配置处修改（例如模型服务配置、插件配置等），
 但本示例脚本本身不会再读取它们。

配置项说明（逐项）：

1) 任务与数据集
- `TASK_DESC`: 训练/生成数据时使用的任务描述（自然语言）。它会影响数据合成的分布，以及评估器对“好工作流”的偏好。
- `DATASET_PATH`: 训练数据集 JSON 文件路径。
    - 如果文件存在：直接加载并用于工作流优化。
    - 如果文件不存在：会调用 `generate_dataset()` 合成数据并写入该路径。
- `DATASET_LIMIT`: 加载/生成后截断数据条数（仅用于快速调试）。
    - `<= 0` 表示不截断。

2) MCTS 搜索/优化相关
- `MAX_ROUNDS`: MCTS 优化迭代轮数上限。轮数越大，搜索越充分但耗时越长。
- `TOP_K`: 每轮保留/导出前 K 个候选结果（防止设置为 0，代码里会 `max(1, TOP_K)`）。
- `NO_IMPROVEMENT_PATIENCE`: 连续多少轮“没有提升”就触发提前停止（早停）。
- `MAX_RETRIES`: 某些 LLM 调用或节点扩展失败时的重试次数。

3) 断点续跑（resume）
- `RESUME`: 是否开启断点续跑。
- `RESUME_RUN_PATH`: 断点目录（例如某次 run 的输出目录）。
    - `RESUME=True` 且该值为空时，具体行为取决于实现（可能从默认目录找最近一次）。

4) 训练/测试划分（用于评估/统计）
- `TRAIN_TEST_SPLIT_RATIO`: 将数据集划分为测试集的比例（0.0 ~ 1.0）。
    - `0.0`：不划分，全部用于训练/优化。
    - `0.2`：20% 作为测试集，其余训练。

5) 文件/目录相关
- `OPTIMIZED_PATH`: MCTS 搜索产生的中间结果、工作流空间（workflow_space）等输出目录。
- `INIT_TEMPLATE_PATH`: 初始化模板路径（YAML），用于提供“初始工作流结构/约束”。

6) 数据合成参数（见 `generate_dataset()`）
- `SYNTHETIC_DATASET_NAME`: 生成数据集的名字标签。
- `SYNTHETIC_DATASET_SIZE`: 生成数据条数。

7) 日志（本文件新增）
- `LOG_IN_RUN_DIR`: 是否把日志写入“本次 run 的输出目录”下。
    - 本项目的 `MCTSWorkflowGenerator` 每次运行会生成一个独立输出目录（`optimized_path`），下面会有 `round*`、`log/` 等子目录。
    - 设为 `True` 时，本脚本会把日志写到：`<optimized_path>/log/<LOG_FILE_NAME>`。
    - 设为 `False` 时，日志写到 `LOG_FILE_PATH` 指定的位置（固定路径，方便长期追踪）。
- `LOG_FILE_NAME`: 日志文件名（仅在 `LOG_IN_RUN_DIR=True` 时使用）。
- `LOG_FILE_PATH`: 固定日志文件路径（仅在 `LOG_IN_RUN_DIR=False` 时使用）。
    - 目录不存在会自动创建。
    - 默认会覆盖写入（`LOG_FILE_MODE="w"`）。
- `LOG_LEVEL`: 日志级别，例如：`"DEBUG" | "INFO" | "WARNING" | "ERROR"`。
    - 作用：控制“哪些日志会被输出”。级别越低（DEBUG）越啰嗦；级别越高（WARNING/ERROR）越安静。
    - 例如：`LOG_LEVEL="INFO"` 时，`logger.debug(...)` 会被过滤掉。
- `LOG_TO_CONSOLE`: 是否同时输出到控制台。
- `LOG_FILE_MODE`: `"w"` 覆盖写入；`"a"` 追加写入。

常见问题：
- `ping` 不通并不代表代理不可用：`ping` 走 ICMP，不走 HTTP 代理。本示例主要依赖 HTTP(S) 的模型调用。
"""

CONFIG = {
    # 任务与数据集
    "TASK_DESC": "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等",
    "DATASET_PATH": (
        "test/example/workflow_generator/data_example.json"
    ),
    "DATASET_LIMIT": 10,

    # MCTS 搜索/优化
    "MAX_ROUNDS": 20,
    "TOP_K": 3,
    "NO_IMPROVEMENT_PATIENCE": 10,
    "MAX_RETRIES": 5,

    # 断点续跑
    "RESUME": False,
    "RESUME_RUN_PATH": None,  # e.g. "test/example/workflow_generator/workflow_space/runs/20260214_101010"

    # 数据集切分
    "TRAIN_TEST_SPLIT_RATIO": 0.0,

    # 输出目录与模板
    "OPTIMIZED_PATH": str(Path(__file__).resolve().parent / "workflow_space"),
    "INIT_TEMPLATE_PATH": (
        "app/core/workflow/workflow_generator/mcts_workflow_generator/"
        "init_template/basic_template.yml"
    ),

    # 数据合成参数
    "SYNTHETIC_DATASET_NAME": "test",
    "SYNTHETIC_DATASET_SIZE": 10,

    # 日志
    "LOG_IN_RUN_DIR": True,
    "LOG_FILE_NAME": "python.log",
    "LOG_FILE_PATH": str(
        Path(__file__).resolve().parent
        / "workflow_space"
        / "workflow_generator_example.log"
    ),
    "LOG_LEVEL": "INFO",
    "LOG_TO_CONSOLE": True,
    "LOG_FILE_MODE": "w",
}

TASK_DESC: str = CONFIG["TASK_DESC"]
DATASET_PATH: str = CONFIG["DATASET_PATH"]
DATASET_LIMIT: int = CONFIG["DATASET_LIMIT"]
MAX_ROUNDS: int = CONFIG["MAX_ROUNDS"]
TOP_K: int = CONFIG["TOP_K"]
NO_IMPROVEMENT_PATIENCE: int = CONFIG["NO_IMPROVEMENT_PATIENCE"]
MAX_RETRIES: int = CONFIG["MAX_RETRIES"]
RESUME: bool = CONFIG["RESUME"]
RESUME_RUN_PATH: str | None = CONFIG["RESUME_RUN_PATH"]
TRAIN_TEST_SPLIT_RATIO: float = CONFIG["TRAIN_TEST_SPLIT_RATIO"]
OPTIMIZED_PATH = Path(CONFIG["OPTIMIZED_PATH"])
INIT_TEMPLATE_PATH: str = CONFIG["INIT_TEMPLATE_PATH"]
SYNTHETIC_DATASET_NAME: str = CONFIG["SYNTHETIC_DATASET_NAME"]
SYNTHETIC_DATASET_SIZE: int = CONFIG["SYNTHETIC_DATASET_SIZE"]
LOG_IN_RUN_DIR: bool = CONFIG["LOG_IN_RUN_DIR"]
LOG_FILE_NAME: str = CONFIG["LOG_FILE_NAME"]
LOG_FILE_PATH = Path(CONFIG["LOG_FILE_PATH"])
LOG_LEVEL: str = CONFIG["LOG_LEVEL"]
LOG_TO_CONSOLE: bool = CONFIG["LOG_TO_CONSOLE"]
LOG_FILE_MODE: str = CONFIG["LOG_FILE_MODE"]


def setup_logging(
    *,
    log_file_path: Path | None,
    level: str = "INFO",
    to_console: bool = True,
    file_mode: str = "w",
) -> None:
    handlers: list[logging.Handler] = []
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, mode=file_mode, encoding="utf-8"))
    if to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers if handlers else None,
        force=True,
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logger.info(
        "Logging initialized: file=%s level=%s console=%s mode=%s",
        str(log_file_path) if log_file_path is not None else None,
        level,
        to_console,
        file_mode,
    )


async def test():
    # 注意：init_server() 里可能会初始化模型/插件/配置，建议在 logging 初始化后调用
    init_server()

    # 1. 注册图数据库相关的信息，请查看test/example/workflow_generator/utils.py
    db = register_and_get_graph_db()
    dataset_generator: DatasetGenerator = SamplingDatasetGenerator(
        sampler=RandomWalkSampler(),
        graph_db=db,
    )

    # 2. 获取数据集：如果指定的数据集已经存在，则直接加载；否则，通过数据合成的方式合成数据集
    data_file_path = Path(DATASET_PATH)
    logger.info("dataset path: %s", data_file_path)
    if Path.exists(data_file_path):
        logger.info("loading data...")
        dataset = load_workflow_train_dataset(
            task_desc=TASK_DESC,
            path=data_file_path,
        )
        if DATASET_LIMIT > 0:
            dataset.data = dataset.data[:DATASET_LIMIT]
        logger.info("dataset loaded: total=%s (limit=%s)", len(dataset.data), DATASET_LIMIT)
    else:
        dataset = await generate_dataset(generator=dataset_generator, file_path=data_file_path)
        if DATASET_LIMIT > 0:
            dataset.data = dataset.data[:DATASET_LIMIT]
        logger.info("dataset generated: total=%s (limit=%s)", len(dataset.data), DATASET_LIMIT)

    # 3. 定义mcts搜索所需的组件，包括：selector、expander、evaluator
    selector = MixedProbabilitySelector()
    expander = LLMExpander()
    evaluator = LLMEvaluator()
    
    # 4. 定义mcts搜索框架
    workflow_generator = MCTSWorkflowGenerator(
        db=db,
        dataset=dataset,
        selector=selector,
        expander=expander, 
        evaluator=evaluator,
        max_rounds=MAX_ROUNDS,
        optimized_path=OPTIMIZED_PATH,
        top_k=max(1, TOP_K),
        max_retries=MAX_RETRIES,
        no_improvement_patience=NO_IMPROVEMENT_PATIENCE,
        resume=RESUME,
        resume_run_path=RESUME_RUN_PATH,
        train_test_split_ratio=TRAIN_TEST_SPLIT_RATIO,
        optimize_grain=None,
        init_template_path=INIT_TEMPLATE_PATH,
    )

    # 将日志落到本次 run 的输出目录下（每次运行一个新文件夹，方便回溯）
    if LOG_IN_RUN_DIR:
        run_log_file = Path(workflow_generator.optimized_path) / "log" / LOG_FILE_NAME
        setup_logging(
            log_file_path=run_log_file,
            level=LOG_LEVEL,
            to_console=LOG_TO_CONSOLE,
            file_mode=LOG_FILE_MODE,
        )
        logger.info("Run log file: %s", run_log_file)

    # 5. mcts主流程入口，开始进行工作流生成与优化
    await workflow_generator.generate()   

async def generate_dataset(generator: DatasetGenerator, file_path: Path | str):
    # 没有数据的时候，通过数据合成的方式来合成数据集
    train_set = await generator.generate(
        task_desc=TASK_DESC,
        dataset_name=SYNTHETIC_DATASET_NAME,
        size=SYNTHETIC_DATASET_SIZE,
    )
    logger.info("synthetic dataset generated: %s", train_set)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([row.model_dump() for row in train_set.data], f, indent=2, ensure_ascii=False)
    return train_set
    
if __name__ == "__main__":
    # 先用控制台日志启动；run 目录在 MCTSWorkflowGenerator 初始化后才能确定
    setup_logging(log_file_path=None, level=LOG_LEVEL, to_console=True, file_mode=LOG_FILE_MODE)
    asyncio.run(test())
