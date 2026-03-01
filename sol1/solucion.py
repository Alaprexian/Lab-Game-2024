import numpy as np
import torch
import torch.nn as nn
import os

# Arquitectura de red
class DQN_2048(nn.Module):
    def __init__(self):
        super(DQN_2048, self).__init__()
        # Entrada: 1 canal (tablero 4x4 logarítmico)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # 4 acciones posibles (Arriba, Abajo, Izquierda, Derecha)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Agent para evaluación
class Agent:
    def __init__(self, model_path="best_agent_2048_12k.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN_2048().to(self.device)
        
        # Mapeo de acciones de salida de red (0,1,2,3)
        self.action_mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Advertencia: No se encontró el modelo {model_path}. Se usará red aleatoria.")
            
        self.model.eval()

    def preprocess(self, board): #Convierte el tablero a log2 y lo normaliza para la red neuronal
        board_log = np.log2(np.maximum(board, 1.0)) # evita log(0)

        board_norm = board_log / 16.0 
        # (Batch, Channels, Height, Width) -> (1, 1, 4, 4)
        state_tensor = torch.FloatTensor(board_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return state_tensor

    def act(self, board, legal_actions) -> str:
        with torch.no_grad():
            state = self.preprocess(board)
            q_values = self.model(state)[0].cpu().numpy() 
            
        # Si la red predice una acción ilegal, la penalizamos fuertemente para que elija la mejor entre las legales.
        # Lista de legal_actions -> índices numéricos
        legal_indices = [self.reverse_mapping[a] for a in legal_actions]
        
        best_action_idx = None
        best_q = -float('inf')
        
        # Encontrar la acción (legal) con mayor Q-Value
        for idx in legal_indices:
            if q_values[idx] > best_q:
                best_q = q_values[idx]
                best_action_idx = idx
                
        # Fall-back ante fallo
        if best_action_idx is None:
            return np.random.choice(legal_actions)

        return self.action_mapping[best_action_idx]