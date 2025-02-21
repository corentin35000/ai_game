import torch
import numpy as np
import onnx
from model import MapGenerator

# Création du modèle
model = MapGenerator()

# Simuler un entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for _ in range(1000):  # Faux entraînement pour l'exemple
    input_data = torch.rand(1, 10)
    target_output = torch.rand(1, 256)
    optimizer.zero_grad()
    loss = criterion(model(input_data), target_output)
    loss.backward()
    optimizer.step()

# Exporter en ONNX
dummy_input = torch.rand(1, 10)
torch.onnx.export(model, dummy_input, "../../models/map_generator.onnx",
                  input_names=['input'], output_names=['output'])

print("Modèle exporté en ONNX : ../../models/map_generator.onnx")
