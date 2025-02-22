import pygame
import torch
from src.ai_model.model import MapGenerator
from src.ai_model.utils import tensor_to_map

# Initialisation globale de Pygame
pygame.init()

# Constantes
WINDOW_HEIGHT_SIZE = 800  # Taille de la fenêtre en pixels
WINDOW_WIDTH_SIZE = 800   # Taille de la fenêtre en pixels
TILE_HEIGHT_SIZE = 8      # Taille d'une tuile en pixels
TILE_WIDTH_SIZE = 8       # Taille d'une tuile en pixels
MAP_HEIGHT_SIZE = 64      # Taille de la carte en tuiles
MAP_WIDTH_SIZE = 64       # Taille de la carte en tuiles

# Variables globales
screen = pygame.display.set_mode((WINDOW_WIDTH_SIZE, WINDOW_HEIGHT_SIZE))
pygame.display.set_caption("Procedural Map Generator")
clock = pygame.time.Clock()
model = None
current_map = None

def load():
    """Charge les ressources initiales (modèle, etc.)."""
    global model
    model = MapGenerator(input_size=64)
    model.eval()
    print("Modèle chargé avec succès.")

def handle_events():
    """Gère les événements utilisateur."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            update()  # Générer une nouvelle carte quand on appuie sur ESPACE
    return True

def update():
    """Met à jour l'état du jeu (génère une nouvelle carte)."""
    global current_map
    noise = torch.randn(1, 64)
    with torch.no_grad():
        map_tensor = model(noise).squeeze(0).squeeze(0)  # [64, 64]
        current_map = tensor_to_map(map_tensor)

def draw():
    """Dessine la carte à l'écran."""
    if current_map is None:
        screen.fill((0, 0, 0))  # Fond noir si pas de carte
    else:
        screen.fill((0, 0, 0))  # Fond noir
        for y in range(MAP_HEIGHT_SIZE):
            for x in range(MAP_WIDTH_SIZE):
                color = (255, 255, 255) if current_map[y, x] == 1 else (0, 0, 0)
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(x * TILE_WIDTH_SIZE, y * TILE_HEIGHT_SIZE, TILE_WIDTH_SIZE, TILE_HEIGHT_SIZE)
                )

    pygame.display.flip()

def main():
    """Boucle principale du jeu."""
    # Récupérer le taux de rafraîchissement du moniteur
    display_info = pygame.display.Info()
    monitor_refresh_rate = display_info.current_hz if display_info.current_hz > 0 else 60 
    print(f"Taux de rafraîchissement du moniteur : {monitor_refresh_rate} Hz")

    # Charger les ressources
    load()

    # Boucle de jeu
    running = True
    while running:
        running = handle_events()
        draw()
        clock.tick(monitor_refresh_rate) # Limiter au taux de rafraîchissement du moniteur

    # Quitter le jeu
    pygame.quit()

if __name__ == "__main__":
    main()