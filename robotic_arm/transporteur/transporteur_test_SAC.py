import pygame
import numpy as np
from transporteur_env import RoboticArmTransporteurEnv
from stable_baselines3 import SAC
import argparse
import time

parser = argparse.ArgumentParser(description="Tester un modèle SAC sur l'environnement Pick & Place.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Le chemin vers le fichier .zip du modèle (ex: models/sac_pick_place)"
)
args = parser.parse_args()

print(f"Chargement du modèle depuis : '{args.model_path}'...")
try:
    model = SAC.load(args.model_path)
except Exception as e:
    print(f"Erreur : Impossible de charger le modèle à l'emplacement '{args.model_path}'.")
    print(f"Détails : {e}")
    exit()

env = RoboticArmTransporteurEnv(segment_lengths=[1.0, 1.0])

obs, info = env.reset()
env.difficulty = 1.0
env.render()

success_count = 0
num_episodes = 50

print(f"\n--- Début du test visuel sur {num_episodes} épisodes ---")

for episode in range(num_episodes):
    terminated = False
    truncated = False
    success = False
    
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
        
        # La condition de victoire est portée par terminated dans notre nouvel environnement
        if terminated:
            success = True
            
        env.render()
        
    if success:
        success_count += 1
        print(f"Épisode {episode + 1} : SUCCÈS ! 🟢 Objet livré.")
    else:
        print(f"Épisode {episode + 1} : ÉCHEC 🔴")
        
    obs, info = env.reset()
    time.sleep(0.5)

print(f"\n--- Bilan : {success_count} réussites sur {num_episodes} ---")
env.close()