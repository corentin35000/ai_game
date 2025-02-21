import torch
import torch.nn as nn

class MapGenerator(nn.Module):
    def __init__(self):
        super(MapGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 32),  # 10 entrées aléatoires
            nn.ReLU(),
            nn.Linear(32, 256), # 16x16 pixels (256 valeurs)
            nn.Sigmoid()        # Valeurs entre 0 et 1
        )

    def forward(self, x):
        return self.model(x)
