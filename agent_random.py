# agent_random.py
from __future__ import annotations

from typing import List, Optional
import numpy as np


class RandomAgent:
    """
    Placeholder agent: picks a random legal action.
    Expected interface for student agents:
      - act(board: np.ndarray, legal_actions: List[str]) -> str
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:
        if not legal_actions:
            return "up"
        return legal_actions[int(self.rng.integers(0, len(legal_actions)))]
