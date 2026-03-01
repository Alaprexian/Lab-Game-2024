# agent_ppo.py
"""
PPO Agent para 2048.

Interfaz requerida:
    agent = PPOAgent(model_path="ppo_2048.pt")
    action = agent.act(board, legal_actions)   # -> str

Encoding del tablero:
    Cada celda se codifica como one-hot sobre su nivel log2
    (0 → índice 0, 2 → 1, 4 → 2, ..., 32768 → 15).
    Shape resultante: (16, 4, 4).

Red neuronal:
    CNN  →  Flatten  →  FC  →  Cabezas de política (4) y valor (1).
    ~300 KB de parámetros; < 5 MB de VRAM en inferencia.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constantes ────────────────────────────────────────────────────────────────
N_LEVELS = 16          # niveles log2: 2^1 .. 2^15 = 32768, índice 0 = vacío
ACTIONS   = ("up", "down", "left", "right")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ── Encoding ──────────────────────────────────────────────────────────────────
def encode_board(board: np.ndarray) -> torch.Tensor:
    """
    Convierte un tablero (H, W) int en un tensor one-hot (N_LEVELS, H, W).

    Args:
        board: np.ndarray de shape (size, size) con valores 0, 2, 4, 8, …

    Returns:
        FloatTensor de shape (N_LEVELS, size, size).
    """
    size = board.shape[0]
    out  = np.zeros((N_LEVELS, size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            v = int(board[r, c])
            if v > 0:
                level = min(int(np.log2(v)), N_LEVELS - 1)
            else:
                level = 0
            out[level, r, c] = 1.0
    return torch.from_numpy(out)


# ── Red neuronal ──────────────────────────────────────────────────────────────
class PPONet(nn.Module):
    """
    Red actor-crítico para 2048.

    Input:  (batch, N_LEVELS, size, size)
    Output: logits (batch, 4)  +  value (batch, 1)
    """

    def __init__(self, board_size: int = 4, n_levels: int = N_LEVELS):
        super().__init__()
        self.board_size = board_size

        # --- Troncal convolucional -------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(n_levels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Después del conv: (batch, 128, size, size)
        flat_dim = 128 * board_size * board_size   # = 2048 para size=4

        # --- Capas FC compartidas --------------------------------------------
        self.fc_shared = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # --- Cabezas ----------------------------------------------------------
        self.policy_head = nn.Linear(256, 4)
        self.value_head  = nn.Linear(256, 1)

        # Inicialización ortogonal (recomendada para PPO)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # La cabeza de política con ganancia más pequeña
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, N_LEVELS, size, size)
        Returns:
            logits: (batch, 4)
            value:  (batch, 1)
        """
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        h = self.fc_shared(h)
        return self.policy_head(h), self.value_head(h)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Muestrea (o evalúa) una acción con su log-prob, entropía y valor.

        Args:
            x:           (batch, N_LEVELS, H, W)
            action:      (batch,) int64  — si None, se muestrea
            legal_mask:  (batch, 4) bool  — True en acciones legales

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(x)

        # Enmascarar acciones ilegales
        if legal_mask is not None:
            logits = logits + (~legal_mask).float() * (-1e9)

        dist      = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        log_prob  = dist.log_prob(action)
        entropy   = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# ── Agente de inferencia ──────────────────────────────────────────────────────
class PPOAgent:
    """
    Agente listo para usar en run_2048.py.

    Uso:
        agent = PPOAgent(model_path="ppo_2048.pt")
        action = agent.act(board, legal_actions)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        board_size: int = 4,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net = PPONet(board_size=board_size).to(self.device)

        if model_path and os.path.isfile(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.net.load_state_dict(state)
            print(f"[PPOAgent] Modelo cargado desde '{model_path}'")
        else:
            if model_path:
                print(f"[PPOAgent] AVISO: '{model_path}' no encontrado. Usando pesos aleatorios.")
            else:
                print("[PPOAgent] Sin modelo cargado. Usa train_ppo.py para entrenar.")

        self.net.eval()

    # ── Método principal ──────────────────────────────────────────────────────
    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:
        """
        Args:
            board:         np.ndarray (size, size) con el estado actual.
            legal_actions: Lista de acciones legales, ej. ["up", "left"].

        Returns:
            Una acción en {"up", "down", "left", "right"}.
        """
        if not legal_actions:
            return "up"

        with torch.no_grad():
            x = encode_board(board).unsqueeze(0).to(self.device)   # (1, 16, 4, 4)

            # Máscara de legalidad
            mask = torch.zeros(1, 4, dtype=torch.bool, device=self.device)
            for a in legal_actions:
                mask[0, ACTION_TO_IDX[a]] = True

            action_t, _, _, _ = self.net.get_action_and_value(x, legal_mask=mask)
            return ACTIONS[action_t.item()]

    def act_greedy(self, board: np.ndarray, legal_actions: List[str]) -> str:
        """Versión determinista (argmax sin samplear). Útil para evaluación."""
        if not legal_actions:
            return "up"

        with torch.no_grad():
            x = encode_board(board).unsqueeze(0).to(self.device)
            logits, _ = self.net(x)
            logits = logits[0]

            # Enmascarar ilegales
            mask = torch.full((4,), float("-inf"), device=self.device)
            for a in legal_actions:
                mask[ACTION_TO_IDX[a]] = 0.0

            logits = logits + mask
            return ACTIONS[logits.argmax().item()]