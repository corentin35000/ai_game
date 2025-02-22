import torch

def normalize_map(tensor):
    """Normalise une carte entre 0 et 1."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def tensor_to_map(tensor):
    """Convertit un tenseur en carte utilisable (par exemple, arrondi à 0 ou 1)."""
    return (tensor > 0.5).float()  # Exemple simple : seuil à 0.5

if __name__ == "__main__":
    # Test rapide
    fake_map = torch.randn(1, 64, 64)
    normalized = normalize_map(fake_map)
    binary_map = tensor_to_map(normalized)
    print(f"Shape de la carte binaire : {binary_map.shape}")