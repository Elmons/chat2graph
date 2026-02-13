import json
from pathlib import Path
import re
from typing import Callable, Dict, List, Union

import yaml

from app.core.common.logger import Chat2GraphLogger
from app.core.common.type import MessageSourceType
from app.core.common.util import parse_jsons
from app.core.model.message import ModelMessage
from app.core.reasoner.model_service_factory import ModelService
from app.core.sdk.agentic_service import AgenticService
from app.core.workflow.workflow_generator.mcts_workflow_generator.config_assembler import (
    assemble_workflow_file_from_candidate_yaml,
)
from app.core.workflow.workflow_generator.mcts_workflow_generator.model import (
    AgenticConfigSection,
    ExecuteResult,
)

logger = Chat2GraphLogger.get_logger(__name__)


def load_agentic_service(
    optimized_path: str,
    round_num: int,
    *,
    base_template_path: str | None = None,
    toolset_path: str | None = None,
) -> AgenticService:
    """Load the agentic service definition for a given optimisation round."""
    round_dir = Path(optimized_path) / f"round{round_num}"
    workflow_path = round_dir / "workflow.yml"
    if base_template_path and toolset_path:
        assembled_path = round_dir / "runtime_workflow.yml"
        assemble_workflow_file_from_candidate_yaml(
            base_template_path=base_template_path,
            toolset_path=toolset_path,
            candidate_yaml_path=workflow_path,
            output_path=assembled_path,
        )
        workflow_path = assembled_path
    mas = AgenticService.load(workflow_path)
    return mas


def load_config_dict(path: str, skip_section: List[AgenticConfigSection]) -> Dict[str, str]:
    """Parse workflow YAML content into a dictionary keyed by section name.

    Args:
        path (str): Path to the workflow YAML file.
        skip_section (List[AgenticConfigSection]): List of sections to skip during parsing.
        
    Returns:
        Dict[str, str]: A dictionary mapping section names to their YAML content.
    """
    try:
        with open(path, encoding="utf-8") as file:
            content = file.read()
            results = {}

        for section in AgenticConfigSection:
            if section in skip_section:
                continue
            section_name = str(section.value)
            # Match a specific key to the next top-level key or the end of the file
            pattern = re.compile(rf"(^|\n){section_name}:(.*?)(?=\n\w+:|\Z)", re.DOTALL)
            match = pattern.search(content)
            if match:
                results[section_name] = match.group(0).strip()

        return results
    except FileNotFoundError:
        logger.warning("Not found file: %s", path)
        return {}
    except Exception:
        logger.exception("Error while reading file: %s", path)
        return {}


def format_yaml_with_anchor(
    text: str, key: str, fields: List[str], need_anchor_name: bool = True
) -> str:
    """Convert YAML containing anchors into a JSON string with required fields.
    
    Args:
        text (str): The YAML content as a string.
        key (str): The top-level key to extract the list from. such as "actions".
        fields (List[str]): Additional fields to extract from each item. such as ["desc", "name"].
        need_anchor_name (bool): Whether to include anchor names in the output.
        
    Returns:
        str: A JSON-formatted string representing the extracted data.
    """
    if fields is None:
        fields = []
    anchor_pattern = re.compile(r"-\s*&(\w+)\s*\n")
    anchors: List[str] = []

    def capture_anchor(match: re.Match):
        anchor_name = match.group(1)
        anchors.append(anchor_name)
        return "- \n"

    text_without_anchor_def = anchor_pattern.sub(capture_anchor, text)

    text_cleaned = re.sub(r"\*(\w+)", r"\1", text_without_anchor_def)

    try:
        parsed_data = yaml.safe_load(text_cleaned)
        if not isinstance(parsed_data, Dict) or key not in parsed_data:
            raise ValueError(f"Cann't find {key} field.")
        yaml_list = parsed_data[key]
        if not isinstance(yaml_list, List):
            raise ValueError("'actions' is not a valid list")
    except yaml.YAMLError as e:
        raise ValueError(f"parse failed：{str(e)}") from e

    if need_anchor_name and len(anchors) != len(yaml_list):
        raise ValueError(
            f"length of anchors（{len(anchors)} unmatch length of actions list {len(yaml_list)}"
        )

    new_text: List[Dict] = []
    for idx, item in enumerate(yaml_list):
        info = {}
        if need_anchor_name:
            info["name"] = anchors[idx]
        extra_info: Dict = {field: item.get(field, "") for field in fields}
        info.update(extra_info)
        new_text.append(info)
    return json.dumps(new_text, indent=4, ensure_ascii=False)


JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], List["JsonValue"]]


async def generate_json(
    model: ModelService,
    sys_prompt: str,
    messages: List[ModelMessage],
    max_retry: int = 3,
    filter: Callable[[List[JsonValue]], JsonValue] = lambda data: True,
    need_parse: bool = True,
) -> JsonValue:
    """Call the model, parse JSON-like responses, and apply a validation filter."""
    times = 0
    while times < max_retry:
        times += 1
        resp_str = ""
        try:
            response = await model.generate(sys_prompt=sys_prompt, messages=messages)
            resp_str = response.get_payload()
            if need_parse:
                parsed_strs = parse_jsons(resp_str)
                for strs in parsed_strs:
                    if isinstance(strs, json.JSONDecodeError):
                        raise Exception(f"{strs}")
                valid_strs: List[JsonValue] = [
                    strs for strs in parsed_strs if not isinstance(strs, json.JSONDecodeError)
                ]
            else:
                valid_strs = [resp_str]
            return filter(valid_strs)
        except Exception as e:
            logger.warning("[generate_json] failed (times=%s): %s", times, e)
            logger.debug("[generate_json] raw response: %s", resp_str)
            messages.append(
                ModelMessage(
                    payload=resp_str,
                    source_type=MessageSourceType.MODEL,
                    job_id=messages[-1].get_job_id(),
                    step=messages[-1].get_step(),
                )
            )
            messages.append(
                ModelMessage(
                    payload=f"When parse and filter json encounter exception={e}, \
                        please output the right json format.",
                    source_type=MessageSourceType.MODEL,
                    job_id=messages[-1].get_job_id(), 
                    step=messages[-1].get_step() + 1,
                )
            )
    return None


def load_execute_result(path: Path) -> List[ExecuteResult]:
    """Load execution results from disk and convert them into model instances."""
    with open(path, encoding="utf-8") as f:
        reuslts = json.load(f)

    execute_results: List[ExecuteResult] = []
    for result in reuslts:
        execute_results.append(ExecuteResult.model_validate(result))

    return execute_results
