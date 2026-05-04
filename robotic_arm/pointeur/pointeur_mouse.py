import pygame
import numpy as np
import argparse
from stable_baselines3 import SAC, PPO
from pointeur_env import RoboticArmPointeurEnv

parser = argparse.ArgumentParser(description="Mode interactif pour bras robotisé.")
parser.add_argument("model_path", type=str, help="Chemin vers le fichier .zip du modèle")
parser.add_argument(
    "--algo", 
    type=str, 
    choices=["sac", "ppo"], 
    default="ppo", 
    help="L'algorithme utilisé pour l'entraînement (SAC ou PPO). Par défaut: SAC"
)
args = parser.parse_args()

env = RoboticArmPointeurEnv(segment_lengths=[1.0, 1.0, 1.0])

algo_class = SAC if args.algo.upper() == "SAC" else PPO

print(f"Chargement du cerveau {args.algo.upper()} depuis '{args.model_path}'...")
try:
    model = algo_class.load(args.model_path)
except Exception as e:
    print(f"Erreur : Impossible de charger le modèle. Vérifie le chemin et l'algo.")
    print(f"Détails : {e}")
    exit()

obs, info = env.reset()
env.render()

window_size = env.unwrapped.window_size
scale = env.unwrapped.scale
max_reach = sum(env.unwrapped.segment_lengths)

print("\n--- Mode Interactif Activé ! ---")
print(f"Algorithme : {args.algo.upper()}")
print("Bouge ta souris dans la fenêtre pour guider le bras.")
print("Ferme la fenêtre pour quitter.")

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mouse_x_px, mouse_y_px = pygame.mouse.get_pos()
    target_x = (mouse_x_px - window_size / 2) / scale
    target_y = (window_size / 2 - mouse_y_px) / scale

    distance_souris = np.sqrt(target_x**2 + target_y**2)
    if distance_souris > max_reach:
        target_x = (target_x / distance_souris) * max_reach
        target_y = (target_y / distance_souris) * max_reach

    env.unwrapped.target_pos = np.array([target_x, target_y])
    env.unwrapped.current_step = 0 

    angles = env.unwrapped.angles
    main_pos = env.unwrapped.get_end_arm_pos()
    obs = np.concatenate([angles, env.unwrapped.target_pos, main_pos]).astype(np.float32)

    action, _ = model.predict(obs, deterministic=True)
    env.step(action)
    
    env.render()

print("--- Fin du programme ---")
env.close()