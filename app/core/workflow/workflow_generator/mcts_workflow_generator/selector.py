from abc import abstractmethod
from math import log, sqrt
from typing import Dict, List

from app.core.workflow.workflow_generator.mcts_workflow_generator.model import WorkflowLogFormat


class Selector:
    """Choose the next workflow candidate to expand."""

    @abstractmethod
    def select(self, top_k: int, logs: Dict[int, WorkflowLogFormat]) -> WorkflowLogFormat:
        """Select a workflow candidate according to the strategy implementation."""


class MixedProbabilitySelector(Selector):
    """Select candidates with a UCB/PUCT-style score for stabler improvement."""

    def __init__(
        self,
        *,
        c_ucb: float = 0.35,
        c_puct: float = 0.45,
        failure_penalty: float = 0.35,
    ):
        self.c_ucb = c_ucb
        self.c_puct = c_puct
        self.failure_penalty = failure_penalty

    def select(self, top_k: int, logs: Dict[int, WorkflowLogFormat]) -> WorkflowLogFormat:
        """Select by maximizing puct-like value among top-k candidates."""
        if not logs:
            raise ValueError("logs is empty")

        list_items = sorted(logs.values(), key=lambda x: x.score, reverse=True)
        candidates: List[WorkflowLogFormat] = list_items[: max(1, top_k)]

        # keep baseline node searchable to avoid local lock-in
        if 1 in logs and all(item.round_number != 1 for item in candidates):
            candidates.append(logs[1])

        total_visits = sum(self._node_visits(item.round_number, logs) for item in candidates) + 1
        priors = self._score_priors(candidates)

        best = None
        best_value = float("-inf")
        for item in candidates:
            visits = self._node_visits(item.round_number, logs)
            exploit = self._node_value(item.round_number, logs)
            ucb_bonus = self.c_ucb * sqrt(log(total_visits + 1) / (visits + 1))
            puct_bonus = self.c_puct * priors[item.round_number] * sqrt(total_visits) / (visits + 1)
            fail_penalty = self.failure_penalty * self._failure_ratio(item)
            value = exploit + ucb_bonus + puct_bonus - fail_penalty

            if best is None or value > best_value or (
                value == best_value and item.score > (best.score if best else float("-inf"))
            ):
                best = item
                best_value = value

        return best if best is not None else candidates[0]

    @staticmethod
    def _node_visits(round_number: int, logs: Dict[int, WorkflowLogFormat]) -> int:
        return 1 + sum(1 for _, node in logs.items() if node.parent_round == round_number)

    @staticmethod
    def _node_value(round_number: int, logs: Dict[int, WorkflowLogFormat]) -> float:
        node = logs[round_number]
        child_scores = [n.score for _, n in logs.items() if n.parent_round == round_number]
        if not child_scores:
            return node.score
        best_child = max(child_scores)
        mean_child = sum(child_scores) / len(child_scores)
        # prefer nodes that both perform well and spawn good descendants
        return 0.6 * node.score + 0.25 * best_child + 0.15 * mean_child

    @staticmethod
    def _score_priors(candidates: List[WorkflowLogFormat]) -> Dict[int, float]:
        if not candidates:
            return {}
        min_score = min(item.score for item in candidates)
        shifted = [item.score - min_score + 1e-6 for item in candidates]
        total = sum(shifted)
        if total <= 0:
            uniform = 1.0 / len(candidates)
            return {item.round_number: uniform for item in candidates}
        return {
            item.round_number: shifted[idx] / total
            for idx, item in enumerate(candidates)
        }

    @staticmethod
    def _failure_ratio(node: WorkflowLogFormat) -> float:
        if not node.feedbacks:
            return 0.0
        failed = 0
        for fb in node.feedbacks:
            succeed = str(fb.get("succeed", "")).strip().lower()
            if succeed not in {"true", "yes"}:
                failed += 1
        return failed / max(len(node.feedbacks), 1)
