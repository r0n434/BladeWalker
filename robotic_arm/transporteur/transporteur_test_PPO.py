import argparse
import pygame
import numpy as np
from transporteur_env import RoboticArmTransporteurEnv
from stable_baselines3 import PPO
import time

parser = argparse.ArgumentParser(description="Tester un modèle PPO sur l'environnement Pick & Place.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Le chemin vers le fichier .zip du modèle (ex: models/ppo_pick_place)"
)
args = parser.parse_args()

print(f"Chargement du modèle depuis : '{args.model_path}'...")
try:
    model = PPO.load(args.model_path)
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
    success = False # Nouvelle variable pour suivre le succès
    
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
        
        # Le succès est maintenant défini par le fait que l'épisode se termine (terminated=True)
        # ce qui arrive uniquement quand l'objet est déposé dans la zone.
        if terminated:
            success = True
            
        env.render()
        
    if success:
        success_count += 1
        print(f"Épisode {episode + 1} : SUCCÈS ! 🟢 Objet livré.")
    else:
        print(f"Épisode {episode + 1} : ÉCHEC (Temps écoulé) 🔴")
        
    obs, info = env.reset()
    time.sleep(0.5) # Petite pause pour observer le succès avant de recommencer

print(f"\n--- Bilan : {success_count} réussites sur {num_episodes} ---")
env.close()