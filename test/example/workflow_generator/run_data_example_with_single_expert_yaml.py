from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.model.message import HybridMessage, TextMessage
from app.core.sdk.agentic_service import AgenticService
from app.core.sdk.init_server import init_server
from app.core.workflow.dataset_synthesis.utils import load_workflow_train_dataset

UTILS_PATH = Path(__file__).resolve().with_name("utils.py")
UTILS_SPEC = importlib.util.spec_from_file_location("workflow_generator_utils", UTILS_PATH)
assert UTILS_SPEC and UTILS_SPEC.loader
_utils_module = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(_utils_module)
register_and_get_graph_db = _utils_module.register_and_get_graph_db

init_server()

YAML_PATH = "test/example/workflow_generator/eval_suite/yamls/[codex]_graph_query_single_expert_builtin.yml"
DATASET_PATH = "test/example/workflow_generator/data_example.json"
TASK_DESC = "你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等"


def _unwrap_payload(message: object) -> str:
    if isinstance(message, HybridMessage):
        return str(message.get_instruction_message().get_payload() or "")
    if isinstance(message, TextMessage):
        return str(message.get_payload() or "")
    payload_func = getattr(message, "get_payload", None)
    if callable(payload_func):
        return str(payload_func() or "")
    return str(message)


def main() -> None:
    register_and_get_graph_db()

    service = AgenticService.load(YAML_PATH)
    entry_expert = service.entry_expert_name() or "Main Expert"
    dataset = load_workflow_train_dataset(task_desc=TASK_DESC, path=DATASET_PATH)

    for i, row in enumerate(dataset.data, start=1):
        reply = service.session().submit(
            TextMessage(payload=row.task, assigned_expert_name=entry_expert)
        ).wait()
        output = _unwrap_payload(reply).strip()
        print(f"[Task {i}] {row.task_subtype}")
        print(f"Q: {row.task}")
        print(f"Expected: {row.verifier}")
        print(f"Output: {output}")
        print("-" * 80)


if __name__ == "__main__":
    main()
