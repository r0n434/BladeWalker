import numpy as np
import tensorflow as tf
from envs.walker_env import WalkerEnv
from models.policy_network import ActorCritic
from training.rollout_buffer import RolloutBuffer
# from training.ppo import PPO  <-- Le futur fichier de ton collègue

# 1. Initialisation des composants
env = WalkerEnv(render_mode=None) # Mode headless pour la vitesse
obs_dim = env.observation_space.shape[0] # 14
act_dim = env.action_space.shape[0] # 4

model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
# Ton collègue fournira sûrement une classe PPO prenant le modèle et des hyperparamètres (lr=3e-4, etc.)
# ppo_agent = PPO(model=model, lr=3e-4, clip_eps=0.2) 

# Hyperparamètres recommandés
n_steps = 2048 # Steps par rollout
batch_size = 64 # Taille des mini-batches
n_epochs = 10 # Passes sur le buffer par update

buffer = RolloutBuffer(n_steps=n_steps, obs_dim=obs_dim, act_dim=act_dim)

MAX_ITERATIONS = 500 # Environ 1 million de steps total

for iteration in range(MAX_ITERATIONS):
    obs, _ = env.reset()
    buffer.reset() # On vide le buffer à chaque itération
    
    # ========================================================
    # PHASE 1 : COLLECTE DU ROLLOUT
    # ========================================================
    for step in range(n_steps):
        # /!\ Appliquer le normaliseur sur 'obs' ici
        obs_tensor = tf.expand_dims(obs, axis=0)
        
        # Le modèle sort l'action, sa probabilité et l'estimation de la valeur
        action, log_prob, value = model.get_action(obs_tensor)
        
        # On passe du tenseur au numpy pour Box2D
        action_np = action.numpy()[0]
        log_prob_np = float(log_prob.numpy()[0])
        value_np = float(value.numpy()[0])
        
        # On interagit avec l'environnement
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        # On stocke la transition dans le buffer
        buffer.add(obs, action_np, reward, done, log_prob_np, value_np)
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()

    # ========================================================
    # PHASE 2 : CALCUL DES AVANTAGES (GAE)
    # ========================================================
    # On a besoin de la valeur du tout dernier état pour la récursion GAE
    last_obs_tensor = tf.expand_dims(obs, axis=0)
    _, _, last_value = model.get_action(last_obs_tensor)
    last_value_np = float(last_value.numpy()[0])
    
    # Calcul des returns et avantages
    buffer.compute_returns_and_advantages(last_value=last_value_np)

    # ========================================================
    # PHASE 3 : UPDATE PPO (La partie de ton collègue)
    # ========================================================
    # La méthode de ton collègue devrait boucler sur les epochs et les mini-batchs
    """
    for epoch in range(n_epochs):
        for batch in buffer.get_batches(batch_size=batch_size):
            # batch contient : obs, actions, log_probs, returns, advantages
            metrics = ppo_agent.update(batch)
    """
    
    # ========================================================
    # PHASE 4 : LOGGING ET SAUVEGARDE
    # ========================================================
    # Enregistrer la mean_reward, policy_loss, etc. via TensorBoard
    
    if iteration % 50 == 0:
        # Sauvegarder les poids du modèle
        model.save_weights(f"checkpoints/iter_{iteration:04d}.weights.h5")
        print(f"Iteration {iteration} sauvegardée.")