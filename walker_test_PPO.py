import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf

from envs.walker_env import WalkerEnv 

from models.policy_network import ActorCritic


def evaluate_ppo(weights_path="checkpoints/iter_0950.weights.h5", episodes=5):
    print("🤖 Initialisation de l'environnement...")
    env = WalkerEnv(render_mode="human")
    
    state_dim = env.observation_space.shape[0]  # 14
    action_dim = env.action_space.shape[0]      # 4
    
    print("🧠 Création du modèle PPO (ActorCritic)...")

    model = ActorCritic(obs_dim=state_dim, act_dim=action_dim)
    
    dummy_obs = tf.zeros((1, state_dim), dtype=tf.float32)
    model(dummy_obs)
    
    try:
        model.load_weights(weights_path)
        print(f"✅ Poids chargés avec succès depuis {weights_path} !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des poids : {e}")
        env.close()
        return

    print("\n🚀 Lancement de la simulation !")
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Préparation de l'observation pour TensorFlow (ajout de la dimension Batch)
            obs_tensor = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), axis=0)
            
            # On demande au modèle de ton collègue de choisir l'action
            action_tf, log_prob, value = model.get_action(obs_tensor)
            
            # On convertit le tenseur d'action en tableau NumPy
            action_np = action_tf.numpy()[0]
            
            # Le robot effectue l'action
            obs, reward, terminated, truncated, info = env.step(action_np)
            env.render()
            
            total_reward += reward
            step_count += 1
            
            # Sécurité si on ferme la fenêtre Pygame
            if env.screen is None:
                print("Fenêtre fermée par l'utilisateur.")
                return

        print(f"🏁 Épisode {ep + 1}/{episodes} terminé en {step_count} pas | Récompense totale : {total_reward:.2f}")

    env.close()
    print("👋 Fin du test.")

if __name__ == "__main__":
    evaluate_ppo("checkpoints/iter_0950.weights.h5")