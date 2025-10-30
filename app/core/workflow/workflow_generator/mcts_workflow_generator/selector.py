from abc import abstractmethod
from typing import Dict, List

import numpy as np

from app.core.workflow.workflow_generator.mcts_workflow_generator.model import WorkflowLogFormat


class Selector:
    """Choose the next workflow candidate to expand."""

    @abstractmethod
    def select(self, top_k: int, logs: Dict[int, WorkflowLogFormat]) -> WorkflowLogFormat:
        """Select a workflow candidate according to the strategy implementation."""


class MixedProbabilitySelector(Selector):
    """Sample candidates using a mixed probability distribution.

    For details, combines uniform distribution and softmax distribution based on scores
    """

    def select(self, top_k: int, logs: Dict[int, WorkflowLogFormat]) -> WorkflowLogFormat:
        """Select a workflow entry based combines uniform and score-weighted probabilities."""

        # get sample top scored workflows, including the initial workflow
        list_items = [log_format for _, log_format in logs.items()]
        top_items: List[WorkflowLogFormat] = []
        list_items.sort(key=lambda x: x.score, reverse=True)
        top_items.extend(list_items[: top_k - 1])
        has_round1 = False
        for item in top_items:
            if item.round_number == 1:
                has_round1 = True
                break

        if not has_round1:
            top_items.append(logs[1])

        elif top_k <= len(list_items):
            top_items.append(list_items[top_k - 1])

        # calculate probability distribution
        scores = [item.score * 20 for item in top_items]
        probabilities = self._compute_probabilities(scores)

        # randomly select based on the computed probabilities
        index = np.random.choice(len(top_items), p=probabilities)
        return top_items[index]

    def _compute_probabilities(
        self, scores: List[float], alpha: float = 0.2, lambda_: float = 0.3
    ) -> np.ndarray:
        """Blend uniform and softmax distributions for candidate sampling."""
        scores_np = np.array(scores, dtype=np.float64)
        n = len(scores_np)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores_np)
        shifted_scores = scores_np - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob
