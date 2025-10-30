import asyncio

from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset
from app.core.workflow.workflow_generator.mcts_workflow_generator.evaluator import LLMEvaluator


async def main():
    dataset_paths = [
        "test/example/workflow_generator/data_example.json"
    ]
    optimized_path = "test/example/workflow_generator/workflow_space/test_just"

    for dataset in dataset_paths:
        await eval(dataset, optimized_path, round_name=0)


async def eval(dataset_path: str, optimized_path: str, round_name: int):
    evaluator = LLMEvaluator(need_reflect=False)
    dataset = load_workflow_train_dataset(
        task_desc="你的主要任务是完成图数据库的查询任务", path=dataset_path
    )
    await evaluator.evaluate_workflow(
        optimized_path=optimized_path,
        round_num=round_name,
        parent_round=-1,
        dataset=dataset.data,
        modifications=[],
    )


if __name__ == "__main__":
    asyncio.run(main())
