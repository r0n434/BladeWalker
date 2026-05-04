import pygame
import numpy as np
from pointeur_env import RoboticArmPointeurEnv
from stable_baselines3 import SAC
import argparse

parser = argparse.ArgumentParser(description="Tester un modèle SAC sur l'environnement RoboticArm.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Le chemin vers le fichier .zip du modèle (ex: models/sac_robot_arm)"
)
args = parser.parse_args()

print(f"Chargement du modèle depuis : '{args.model_path}'...")
try:
    model = SAC.load(args.model_path)
except Exception as e:
    print(f"Erreur : Impossible de charger le modèle à l'emplacement '{args.model_path}'.")
    print(f"Détails : {e}")
    exit()

env = RoboticArmPointeurEnv(segment_lengths=[1.0, 1.0])

obs, info = env.reset()
env.difficulty = 1.0
env.render()

success_count = 0
num_episodes = 50

print(f"\n--- Début du test visuel sur {num_episodes} épisodes ---")

for episode in range(num_episodes):
    terminated = False
    truncated = False
    touched_target = False
    
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(f"Épisode {episode + 1} interrompu par l'utilisateur.")
                    truncated = True
        if truncated:
            break
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        end_pos = env.get_end_arm_pos()
        if np.linalg.norm(end_pos - env.target_pos) < 0.1:
            touched_target = True
            
        env.render()
        
    if touched_target:
        success_count += 1
        print(f"Épisode {episode + 1} : SUCCÈS ! 🟢 (La cible a été atteinte et tenue)")
    else:
        print(f"Épisode {episode + 1} : ÉCHEC 🔴")
        
    obs, info = env.reset()

print(f"\n--- Bilan : {success_count} réussites sur {num_episodes} ---")
env.close()