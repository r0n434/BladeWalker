import argparse
import pygame
import numpy as np
from robot_arm_env import RoboticArmEnv
from stable_baselines3 import PPO
import time

parser = argparse.ArgumentParser(description="Tester un modèle PPO sur l'environnement RoboticArm.")
parser.add_argument(
    "model_path", 
    type=str, 
    help="Le chemin vers le fichier .zip du modèle (ex: models/ppo_robot_arm)"
)
args = parser.parse_args()
    

print(f"Chargement du modèle depuis : '{args.model_path}'...")
try:
    model = PPO.load(args.model_path)
except Exception as e:
    print(f"Erreur : Impossible de charger le modèle à l'emplacement '{args.model_path}'.")
    print(f"Détails : {e}")
    exit()

env = RoboticArmEnv(segment_lengths=[1.0, 1.0])

obs, info = env.reset()
env.render()

success_count = 0
num_episodes = 50

print(f"\n--- Début du test visuel sur {num_episodes} épisodes ---")

for episode in range(num_episodes):
    terminated = False
    truncated = False
    
    while not (terminated or truncated):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
                
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        end_pos = env.get_end_arm_pos()
        if np.linalg.norm(end_pos - env.target_pos) < 0.1:
            touched_target = True
            
        env.render()
        
    if touched_target:
        success_count += 1
        print(f"Épisode {episode + 1} : SUCCÈS ! 🟢")
    else:
        print(f"Épisode {episode + 1} : ÉCHEC (Temps écoulé) 🔴")
        
    obs, info = env.reset()
    time.sleep(0.5)

print(f"\n--- Bilan : {success_count} réussites sur {num_episodes} ---")
env.close()