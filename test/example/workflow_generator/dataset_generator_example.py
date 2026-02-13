import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.sdk.init_server import init_server
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.sampler import RandomWalkSampler
from test.example.workflow_generator.utils import register_and_get_graph_db

init_server()

TASK_DESC = "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等"
DATASET_NAME = "query_small"
DATASET_SIZE = 100
OVERALL_TIMEOUT_SECONDS = 300


def _prepare_output_paths(dataset_name: str, size: int) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent / "generated_datasets"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{run_id}_{dataset_name}_{size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "dataset.json", run_dir / "meta.json"


async def example() -> None:
    db = register_and_get_graph_db()
    generator = SamplingDatasetGenerator(
        graph_db=db,
        sampler=RandomWalkSampler(),
        strategy="query",
        max_depth=5,
        max_noeds=5,
        max_edges=30,
        nums_per_subgraph=10,
    )

    try:
        dataset = await asyncio.wait_for(
            generator.generate(
                task_desc=TASK_DESC,
                dataset_name=DATASET_NAME,
                size=DATASET_SIZE,
            ),
            timeout=OVERALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(f"dataset generation timeout: {OVERALL_TIMEOUT_SECONDS}s")
        return

    dataset_path, meta_path = _prepare_output_paths(dataset_name=DATASET_NAME, size=DATASET_SIZE)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump([row.model_dump() for row in dataset.data], f, indent=2, ensure_ascii=False)

    meta = {
        "dataset_name": dataset.name,
        "task_desc": dataset.task_desc,
        "target_size": DATASET_SIZE,
        "actual_size": len(dataset.data),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"dataset saved: {dataset_path}")
    print(f"meta saved: {meta_path}")
    print(f"target_size={DATASET_SIZE}, actual_size={len(dataset.data)}")


if __name__ == "__main__":
    asyncio.run(example())
