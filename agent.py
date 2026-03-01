# agent.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class Agent:
    def __init__(self, seed: int | None = None) -> None:
        # Matriz de pesos (score logaritmico)
        self.weights = np.array([
            [100, 80, 70, 60],
            [10,  20, 30, 40],
            [9,   8,  7,  6],
            [2,   3,  4,  5]
        ])

    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:
        if not legal_actions:
            return "up"

        best_score = -float('inf')
        best_action = legal_actions[0]
        
        # Poda por prob.: if rama < 0.001 de prob., bye bye.
        depth = 4 

        for action in legal_actions:
            sim_board, moved, _ = self._simulate_move(board, action)
            if moved:
                # probabilidad inicial 1.0
                score = self._expectimax(sim_board, depth - 1, False, 1.0) #<-Poda
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def _expectimax(self, board: np.ndarray, depth: int, is_agent_turn: bool, prob: float) -> float:
        # Poda por probabilidad
        if depth == 0 or prob < 0.001:
            return self._evaluate(board)

        if is_agent_turn:
            max_val = -float('inf')
            moved_any = False
            for action in ["up", "down", "left", "right"]:
                sim_board, moved, _ = self._simulate_move(board, action)
                if moved:
                    moved_any = True
                    val = self._expectimax(sim_board, depth - 1, False, prob)
                    if val > max_val:
                        max_val = val
            return max_val if moved_any else self._evaluate(board)
        else:
            empty_cells = np.argwhere(board == 0)
            if empty_cells.size == 0:
                return self._evaluate(board)

            expected_val = 0
            # me "expando" por las celdas cercanas
            num_cells = len(empty_cells)
            
            for r, c in empty_cells:
                # Prob. de que aparezca un 2 es 0.9
                p2 = prob * (0.9 / num_cells)
                board[r, c] = 2
                expected_val += self._expectimax(board, depth - 1, True, p2) * (0.9 / num_cells)
                
                # Probabilidad de que aparezca un 4 es 0.1 (con fe)
                p4 = prob * (0.1 / num_cells)
                board[r, c] = 4
                expected_val += self._expectimax(board, depth - 1, True, p4) * (0.1 / num_cells)
                
                board[r, c] = 0 # Backtrack
            return expected_val

    def _evaluate(self, board: np.ndarray) -> float:
        """Heurística optimizada para la métrica final_score."""
        if np.max(board) == 0: return 0
        
        # 1. Alineación con pesos (Maximiza T_mean y L_mean)
        score = np.sum(board * self.weights)
        
        # 2. Bonus por celdas vacías (Crucial para sobrevivir y reducir K_mean)
        empty_count = np.count_nonzero(board == 0)
        score += (empty_count ** 2) * 10 
        
        # 3. Penalización drástica si la ficha máxima no está en la esquina superior izquierda
        if board[0, 0] != np.max(board):
            score -= 500

        return float(score)

    def _simulate_move(self, board: np.ndarray, action: str) -> Tuple[np.ndarray, bool, int]:
        """Yakub es mi guía"""
        b = board.copy()
        reward_total = 0
        moved_any = False
        size = b.shape[0]

        if action in ("left", "right"):
            for i in range(size):
                row = b[i, :]
                if action == "right": row = row[::-1]
                new_row, moved, reward = self._merge_line(row)
                if action == "right": new_row = new_row[::-1]
                if moved: moved_any = True
                reward_total += reward
                b[i, :] = new_row
        else:
            for j in range(size):
                col = b[:, j]
                if action == "down": col = col[::-1]
                new_col, moved, reward = self._merge_line(col)
                if action == "down": new_col = new_col[::-1]
                if moved: moved_any = True
                reward_total += reward
                b[:, j] = new_col
        return b, moved_any, reward_total

    def _merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """Lógica de mezcla (2 + 2 = pez)"""
        nonzero = line[line != 0].tolist()
        merged = []
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
        new_line = np.zeros_like(line)
        new_line[:len(merged)] = merged
        return new_line, not np.array_equal(new_line, line), reward