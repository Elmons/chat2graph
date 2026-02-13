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

from app.core.workflow.dataset_synthesis.utils import (  # noqa: E402
    load_workflow_train_dataset,
)
import app.core.workflow.evaluation.eval_yaml_pipeline as eval_yaml_pipeline_module  # noqa: E402
from app.core.workflow.evaluation.eval_yaml_pipeline import evaluate_yaml  # noqa: E402
from app.core.workflow.workflow_generator.mcts_workflow_generator.runner import (  # noqa: E402
    WorkflowRunner,
)

UTILS_PATH = Path(__file__).resolve().parents[1] / "utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("workflow_generator_utils", UTILS_PATH)
assert UTILS_SPEC and UTILS_SPEC.loader
_utils_module = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(_utils_module)
register_and_get_graph_db = _utils_module.register_and_get_graph_db
DB_CONFIG = _utils_module.DB_CONFIG

# ===== Editable run config =====
# Change only these values when evaluating another YAML/dataset/task.
TASK_DESC = "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等"
DATASET_PATH = "test/example/workflow_generator/data_example.json"
YAML_PATH = (
    "test/example/workflow_generator/eval_suite/yamls/"
    "[codex]_from_scratch_single_expert_graph_query.yml"
)
# ===============================

# None means auto-infer from YAML at runtime.
# If YAML has exactly one expert, entry expert is inferred automatically.
# If YAML has multiple experts, set this explicitly.
MAIN_EXPERT_NAME: str | None = None
SCORE_MODE = "llm"
OUT_DIR = (
    Path(__file__).resolve().parent
    / "eval_runs"
    / datetime.now().strftime("%Y%m%d_%H%M%S_single_yaml_eval")
)


class _SerialWorkflowRunner(WorkflowRunner):
    """Force serial execution to reduce upstream LLM rate-limit failures."""

    def __init__(self, *args, **kwargs):
        kwargs["batch_size"] = 1
        super().__init__(*args, **kwargs)


async def main() -> None:
    yaml_file = Path(YAML_PATH)
    dataset_file = Path(DATASET_PATH)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_file}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    register_and_get_graph_db()
    dataset = load_workflow_train_dataset(task_desc=TASK_DESC, path=DATASET_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"yaml: {YAML_PATH}")
    print(f"dataset: {DATASET_PATH}")
    print(f"task_desc: {TASK_DESC}")
    print(f"main_expert_name: {MAIN_EXPERT_NAME if MAIN_EXPERT_NAME else 'AUTO'}")
    print(f"score_mode: {SCORE_MODE}")
    print(f"output_dir: {OUT_DIR}")
    print("=" * 80)

    # Override default runner batch size (5) for this benchmark script only.
    eval_yaml_pipeline_module.WorkflowRunner = _SerialWorkflowRunner

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
