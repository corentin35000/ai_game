import torch
import torch.nn as nn

class MapGenerator(nn.Module):
    def __init__(self, input_size=64, output_channels=1):
        super(MapGenerator, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_size * input_size * output_channels),
            nn.Sigmoid()  # Sortie entre 0 et 1 pour représenter la carte
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Remettre en forme pour obtenir une carte 2D (ex: 64x64)
        x = x.view(-1, 1, self.input_size, self.input_size)
        return x

# Test rapide
if __name__ == "__main__":
    model = MapGenerator()
    noise = torch.randn(1, 64)  # Entrée : bruit aléatoire
    output = model(noise)
    print(f"Shape de la carte générée : {output.shape}")  # Devrait être [1, 1, 64, 64]