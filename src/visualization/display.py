import pygame
import numpy as np
import onnxruntime as ort

# Initialisation
def init():
    """Initialise Pygame, charge ONNX Runtime et configure l'affichage."""
    pygame.init()
    global screen, clock, session, monitor_refresh_rate
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    # Récupérer la fréquence de rafraîchissement du moniteur
    display_info = pygame.display.Info()
    monitor_refresh_rate = display_info.current_w

    # Charger le modèle ONNX
    session = ort.InferenceSession("../../models/map_generator.onnx")

# Génération de la carte
def generate_map():
    """Génère une nouvelle map à partir du modèle ONNX."""
    input_data = np.random.rand(1, 10).astype(np.float32)
    output = session.run(None, {"input": input_data})[0]
    return output.reshape(16, 16)  # Reshape en 16x16

# Gestion des événements
def handle_events():
    """Gère les entrées utilisateur (fermeture, touche ESPACE)."""
    global running, output
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            output = generate_map()  # Générer une nouvelle map en appuyant sur ESPACE

# Mise à jour
def update():
    """Met à jour la logique du jeu (actuellement rien à mettre à jour)."""
    pass

# Affichage de la carte
def draw():
    """Affiche la carte générée dans Pygame."""
    screen.fill((0, 0, 0))
    tile_size = 20
    for y in range(16):
        for x in range(16):
            color = (0, int(255 * output[y, x]), 0)
            pygame.draw.rect(screen, color, (x * tile_size, y * tile_size, tile_size, tile_size))
    
    pygame.display.flip()

# Boucle principale du jeu
def main():
    """Boucle principale du jeu : gestion des événements, mise à jour et affichage."""
    global running, output
    running = True
    output = generate_map()  # Première génération de map

    while running:
        handle_events()  # Gérer les entrées utilisateur
        update()         # Mettre à jour les éléments (si nécessaire)
        draw()           # Afficher la map

        # Synchronisation avec la fréquence du moniteur
        clock.tick_busy_loop(monitor_refresh_rate)

    pygame.quit()

# Exécution du jeu
if __name__ == "__main__":
    init()
    main()
