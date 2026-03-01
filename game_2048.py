from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


Action = Union[int, str]  # 0/1/2/3 or "up"/"down"/"left"/"right"


@dataclass
class StepResult:
    obs: np.ndarray
    reward: int
    done: bool
    info: Dict


class Game2048:
    """
    Core 2048 game logic (no UI, no agents).

    - Board is a size x size numpy int array.
    - Empty cells are 0.
    - Tiles are powers of two (2,4,8,...).
    - On each valid move (board changes), a new tile appears:
        2 with prob p_two (default 0.9), else 4.
    - Reward is the sum of merged tile values created in that move
      (standard 2048 scoring).
    """

    ACTIONS = ("up", "down", "left", "right")

    def __init__(
        self,
        size: int = 4,
        seed: Optional[int] = None,
        p_two: float = 0.9,
        spawn_initial: int = 2,
    ) -> None:
        if size < 2:
            raise ValueError("size must be >= 2")
        if not (0.0 < p_two < 1.0):
            raise ValueError("p_two must be in (0,1)")
        if spawn_initial < 0:
            raise ValueError("spawn_initial must be >= 0")

        self.size = int(size)
        self.p_two = float(p_two)
        self.spawn_initial = int(spawn_initial)

        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int64)

        self.reset()

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        for _ in range(self.spawn_initial):
            self._spawn_tile()
        return self.board.copy()

    def step(self, action: Action) -> StepResult:
        action_str = self._normalize_action(action)

        before = self.board.copy()
        moved, reward = self._apply_move(action_str)

        if not moved:
            done = self.is_done()
            return StepResult(
                obs=self.board.copy(),
                reward=0,
                done=done,
                info={"moved": False, "legal_actions": self.legal_actions()},
            )

        self._spawn_tile()
        done = self.is_done()
        return StepResult(
            obs=self.board.copy(),
            reward=reward,
            done=done,
            info={"moved": True, "legal_actions": self.legal_actions()},
        )

    def legal_actions(self) -> List[str]:
        legals: List[str] = []
        for a in self.ACTIONS:
            if self._would_change(a):
                legals.append(a)
        return legals

    def is_done(self) -> bool:
        return len(self.legal_actions()) == 0

    # ----------------------- Internals -----------------------

    def _normalize_action(self, action: Action) -> str:
        if isinstance(action, int):
            if action < 0 or action >= 4:
                raise ValueError("int action must be in {0,1,2,3}")
            return self.ACTIONS[action]
        if isinstance(action, str):
            a = action.strip().lower()
            alias = {"u": "up", "d": "down", "l": "left", "r": "right"}
            a = alias.get(a, a)
            if a not in self.ACTIONS:
                raise ValueError(f"unknown action: {action!r}")
            return a
        raise TypeError("action must be int or str")

    def _spawn_tile(self) -> bool:
        empties = np.argwhere(self.board == 0)
        if empties.size == 0:
            return False
        idx = self._rng.integers(0, len(empties))
        r, c = empties[idx]
        self.board[r, c] = 2 if self._rng.random() < self.p_two else 4
        return True

    def _would_change(self, action: str) -> bool:
        tmp = self.board.copy()
        moved, _ = self._apply_move(action, board=tmp)
        return moved

    def _apply_move(self, action: str, board: Optional[np.ndarray] = None) -> Tuple[bool, int]:
        """
        Apply move to `board` (defaults to self.board).
        Returns (moved, reward).
        """
        b = self.board if board is None else board

        reward_total = 0
        moved_any = False

        if action in ("left", "right"):
            for i in range(self.size):
                row = b[i, :]
                if action == "right":
                    row = row[::-1]
                new_row, moved, reward = self._merge_line(row)
                if action == "right":
                    new_row = new_row[::-1]
                if moved:
                    moved_any = True
                reward_total += reward
                b[i, :] = new_row
        else:  # up/down -> operate on columns
            for j in range(self.size):
                col = b[:, j]
                if action == "down":
                    col = col[::-1]
                new_col, moved, reward = self._merge_line(col)
                if action == "down":
                    new_col = new_col[::-1]
                if moved:
                    moved_any = True
                reward_total += reward
                b[:, j] = new_col

        return moved_any, reward_total

    def _merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Merge a 1D line to the left (standard 2048 rule):
        - compress non-zeros
        - merge equal adjacent once
        - compress again
        Returns (new_line, moved, reward)
        """
        original = line.copy()

        nonzero = original[original != 0].tolist()
        merged: List[int] = []
        reward = 0

        i = 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                v = nonzero[i] * 2
                merged.append(v)
                reward += v
                i += 2
            else:
                merged.append(nonzero[i])
                i += 1

        # pad with zeros
        new_line = np.zeros_like(original)
        new_line[: len(merged)] = merged

        moved = not np.array_equal(new_line, original)
        return new_line, moved, reward
