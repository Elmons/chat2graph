import asyncio

from app.core.workflow.workflow_generator.mcts_workflow_generator.expander import LLMExpander
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import WorkflowLogFormat
from app.core.workflow.workflow_generator.mcts_workflow_generator.utils import (
    load_config_dict,
)


async def main():
    config_dict = load_config_dict(
        path="app/core/workflow/workflow_generator/mcts_workflow_generator/init_template/basic_template.yml", 
        skip_section=[]
    )
    expander = LLMExpander()
    suggestion, resp = await expander.expand(
        task_tesc="主要任务是完成一切关于图数据库的查询任务",
        current_config=config_dict,
        round_context=WorkflowLogFormat(
            round_number=1, score=1, modifications=[], reflection="", feedbacks=[]
        ),
    )
    print(resp.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
