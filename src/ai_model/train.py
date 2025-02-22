import torch
import torch.nn as nn
import torch.optim as optim
from model import MapGenerator

def train_model():
    # Hyperparamètres
    input_size = 64
    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Initialisation
    model = MapGenerator(input_size=input_size)
    criterion = nn.MSELoss()  # Exemple de perte (à adapter selon ton objectif)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Simulation d'entraînement (remplace avec tes vraies données)
    for epoch in range(epochs):
        noise = torch.randn(batch_size, input_size)
        target = torch.randn(batch_size, 1, input_size, input_size)  # Données fictives
        optimizer.zero_grad()
        output = model(noise)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Exportation en ONNX
    model.eval()
    dummy_input = torch.randn(1, input_size)  # Entrée fictive pour l'export
    torch.onnx.export(
        model,
        dummy_input,
        "../../models/map_generator.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("Modèle exporté vers ../../models/map_generator.onnx")

if __name__ == "__main__":
    train_model() # Lancer l'entraînement