import asyncio
import json
import time

from app.core.sdk.init_server import init_server
from app.core.workflow.dataset_synthesis.generator import DatasetGenerator, SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.sampler import (
    RandomWalkSampler,
)
from test.example.workflow_generator.utils import register_and_get_graph_db

init_server()



async def test_generate_dataset(generator: DatasetGenerator):
    train_set = await generator.generate(
        "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等",
        dataset_name="test",
        size=10,
    )
    print(f"end, train_set={train_set}")

    dataset_name = "data" + str(int(time.time())) + ".json"
    with open(dataset_name, "w", encoding="utf-8") as f:
        json.dump([row.model_dump() for row in train_set.data], f, indent=2, ensure_ascii=False)


async def test():
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
    tests = [
        (test_generate_dataset, [DatasetGenerator]),
    ]

    for test_func, allow_types in tests:
        for t in allow_types:
            if isinstance(dataset_generator, t):
                await test_func(dataset_generator)


if __name__ == "__main__":
    asyncio.run(test())
