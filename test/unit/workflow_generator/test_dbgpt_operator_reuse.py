from pathlib import Path

from app.core.common.type import WorkflowPlatformType
from app.core.model.agentic_config import AgenticConfig
from app.core.sdk.agentic_service import AgenticService
from app.core.sdk.wrapper.workflow_wrapper import WorkflowWrapper


def test_dbgpt_workflow_reuses_operator_instances_for_fanin(tmp_path: Path) -> None:
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text(
        """
app: {name: "x"}
plugin: {workflow_platform: "DBGPT"}
reasoner: {type: "DUAL"}
tools: []
actions: []
toolkit: []
operators:
  - &query_understanding {instruction: "u", output_schema: "u_out", actions: []}
  - &schema_retrieval {instruction: "s", output_schema: "s_out", actions: []}
  - &cypher_refinement {instruction: "r", output_schema: "r_out", actions: []}
  - &qa {instruction: "q", output_schema: "q_out", actions: []}
experts:
  - profile: {name: "Main Expert", desc: "d"}
    workflow:
      - [*query_understanding, *cypher_refinement]
      - [*schema_retrieval, *cypher_refinement]
      - [*cypher_refinement, *qa]
""".lstrip(),
        encoding="utf-8",
    )

    config = AgenticConfig.from_yaml(yaml_path)
    expert = config.experts[0]

    # The same operator referenced in multiple chains should share one logical id.
    assert expert.workflow[0][1].id == expert.workflow[1][1].id
    assert expert.workflow[0][1].id == expert.workflow[2][0].id

    workflow_items = AgenticService._build_expert_workflow(expert, config)
    workflow = WorkflowWrapper(platform=WorkflowPlatformType.DBGPT).chain(*workflow_items).workflow
    graph = workflow._operator_graph  # pylint: disable=protected-access

    tails = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    assert len(tails) == 1
    assert graph.number_of_nodes() == 4
