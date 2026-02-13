from __future__ import annotations

import asyncio
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset
from app.core.workflow.evaluation.eval_yaml_pipeline import evaluate_yaml

UTILS_PATH = Path(__file__).resolve().parents[1] / "utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("workflow_generator_utils", UTILS_PATH)
assert UTILS_SPEC and UTILS_SPEC.loader
_utils_module = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(_utils_module)
register_and_get_graph_db = _utils_module.register_and_get_graph_db
DB_CONFIG = _utils_module.DB_CONFIG

TASK_DESC = "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等"
DATASET_PATH = "test/example/workflow_generator/data_example.json"
YAML_PATH = "test/example/workflow_generator/eval_suite/yamls/[codex]_graph_query_single_expert_builtin.yml"
MAIN_EXPERT_NAME = "Main Expert"
SCORE_MODE = "llm"
OUT_DIR = (
    Path(__file__).resolve().parent
    / "eval_runs"
    / datetime.now().strftime("%Y%m%d_%H%M%S_single_yaml_eval")
)


async def main() -> None:
    register_and_get_graph_db()
    dataset = load_workflow_train_dataset(task_desc=TASK_DESC, path=DATASET_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"yaml: {YAML_PATH}")
    print(f"dataset: {DATASET_PATH}")
    print(f"task_desc: {TASK_DESC}")
    print(f"score_mode: {SCORE_MODE}")
    print(f"output_dir: {OUT_DIR}")
    print("=" * 80)

    summary = await evaluate_yaml(
        dataset=dataset,
        yaml_path=YAML_PATH,
        out_dir=OUT_DIR,
        graph_db_config=DB_CONFIG,
        main_expert_name=MAIN_EXPERT_NAME,
        score_mode=SCORE_MODE,
    )

    overview = {
        "yaml_path": YAML_PATH,
        "dataset_path": DATASET_PATH,
        "task_desc": TASK_DESC,
        "score_mode": SCORE_MODE,
        "dataset_rows": len(dataset.data),
        "avg_score": summary.avg_score,
        "success_rate": summary.success_rate,
        "avg_latency_ms": summary.avg_latency_ms,
        "total_tokens": summary.total_tokens,
        "error": summary.error,
        "results_path": summary.results_path,
        "summary_path": summary.summary_path,
        "run_meta_path": summary.run_meta_path,
    }
    overview_path = OUT_DIR / "overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(f"dataset_rows={overview['dataset_rows']}")
    print(f"avg_score={overview['avg_score']}")
    print(f"success_rate={overview['success_rate']}")
    print(f"overview={overview_path}")
    print(f"summary={summary.summary_path}")
    print(f"results={summary.results_path}")


if __name__ == "__main__":
    asyncio.run(main())
