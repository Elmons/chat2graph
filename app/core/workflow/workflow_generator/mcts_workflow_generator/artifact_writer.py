from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from app.core.workflow.workflow_generator.mcts_workflow_generator.config_assembler import (
    assemble_workflow_file,
)


class MCTSArtifactWriter:
    """Persist workflow rounds and MCTS logs under a single artifact root."""

    def __init__(self, root: str | Path):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def round_dir(self, round_num: int) -> Path:
        path = self._root / f"round{round_num}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_round_workflow(
        self,
        *,
        round_num: int,
        base_template_path: str | Path,
        toolset_path: str | Path,
        candidate_sections: Mapping[str, str],
    ) -> Path:
        output = self.round_dir(round_num) / "workflow.yml"
        return assemble_workflow_file(
            base_template_path=base_template_path,
            toolset_path=toolset_path,
            candidate_sections=candidate_sections,
            output_path=output,
        )

    def write_round_json(self, *, round_num: int, filename: str, payload: Any) -> Path:
        output = self.round_dir(round_num) / filename
        with output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output

    def write_logs(
        self,
        *,
        logs: Mapping[int, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Path]:
        log_dir = self._root / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log.json"
        edges_path = log_dir / "edges.json"
        config_path = log_dir / "config.json"

        log_rows = [v.model_dump(mode="json") for _, v in sorted(logs.items(), key=lambda kv: kv[0])]
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log_rows, f, ensure_ascii=False, indent=2)

        edges = []
        for _, log in sorted(logs.items(), key=lambda kv: kv[0]):
            parent_round = getattr(log, "parent_round", None)
            round_number = getattr(log, "round_number", None)
            if parent_round is None or round_number is None:
                continue
            edges.append({"parent_round": parent_round, "child_round": round_number})
        with edges_path.open("w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)

        with config_path.open("w", encoding="utf-8") as f:
            json.dump([config], f, ensure_ascii=False, indent=2)

        return {
            "log_path": log_path,
            "edges_path": edges_path,
            "config_path": config_path,
        }
