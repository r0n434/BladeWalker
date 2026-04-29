import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from datetime import datetime

from envs.walker_env import WalkerEnv
from models.policy_network import ActorCritic
from training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """Entraîneur PPO pour le robot Walker"""
    
    def __init__(self, env, model, buffer, 
                 n_steps=2048, n_epochs=10, batch_size=64,
                 learning_rate=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5):
        """
        Args:
            env: Environnement Gymnasium
            model: Réseau ActorCritic
            buffer: RolloutBuffer
            n_steps: Nombre d'étapes par rollout
            n_epochs: Nombre d'époques d'optimisation par update
            batch_size: Taille des mini-batches
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            lam: Coefficient GAE
            clip_ratio: Ratio de clipping PPO
            ent_coef: Coefficient d'entropie
            vf_coef: Coefficient de la valeur
        """
        self.env = env
        self.model = model
        self.buffer = buffer
        
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def compute_ppo_loss(self, batch):
        """Calcule la perte PPO pour un mini-batch
        
        Args:
            batch: (obs, actions, old_log_probs, returns, advantages)
        
        Returns:
            loss: Perte totale (actor + critic - entropie)
        """
        obs, actions, old_log_probs, returns, advantages = batch
        
        with tf.GradientTape() as tape:
            # Forward pass
            new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions)
            
            # PPO Clipped Loss (Surrogate Objective)
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Value (Critic) Loss
            value_loss = tf.reduce_mean(tf.square(values - returns))
            
            # Entropy Bonus
            entropy_loss = -tf.reduce_mean(entropy)
            
            # Loss totale
            total_loss = actor_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # Backprop
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, actor_loss, value_loss, entropy_loss
    
    def update(self):
        """Effectue une étape d'optimisation PPO sur tous les mini-batches"""
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                total_loss, actor_loss, value_loss, entropy_loss = self.compute_ppo_loss(batch)
                
                losses.append(float(total_loss))
                actor_losses.append(float(actor_loss))
                value_losses.append(float(value_loss))
                entropy_losses.append(float(entropy_loss))
        
        return {
            'loss': np.mean(losses),
            'actor_loss': np.mean(actor_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def collect_rollout(self):
        """Collecte n_steps transitions en interagissant avec l'environnement"""
        obs, _ = self.env.reset()
        episode_return = 0.0
        returns = []
        
        for step in range(self.n_steps):
            # Récupérer action et value du modèle
            obs_tensor = tf.expand_dims(obs, axis=0)
            action, log_prob, value = self.model.get_action(obs_tensor)
            
            # Convertir en numpy
            action_np = action.numpy()[0]
            log_prob_np = float(log_prob.numpy()[0])
            value_np = float(value.numpy()[0])
            
            # Exécuter l'action
            obs_next, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated
            
            episode_return += reward
            
            # Stocker dans le buffer
            self.buffer.add(obs, action_np, reward, done, log_prob_np, value_np)
            
            obs = obs_next
            
            if done:
                obs, _ = self.env.reset()
                returns.append(episode_return)
                episode_return = 0.0
        
        # Calculer la valeur du dernier état
        obs_tensor = tf.expand_dims(obs, axis=0)
        _, _, last_value = self.model.get_action(obs_tensor)
        last_value_np = float(last_value.numpy()[0])
        
        # Calculer les returns et advantages
        self.buffer.compute_returns_and_advantages(last_value_np)
        
        return {
            'mean_return': np.mean(returns) if returns else 0.0,
            'n_episodes': len(returns)
        }
    
    def train(self, n_updates=100):
        """Boucle principale d'entraînement
        
        Args:
            n_updates: Nombre d'itérations de mise à jour
        """
        print("=" * 70)
        print(" DÉMARRAGE DE L'ENTRAÎNEMENT PPO - BLADEWALKER")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  • n_steps: {self.n_steps}")
        print(f"  • n_epochs: {self.n_epochs}")
        print(f"  • batch_size: {self.batch_size}")
        print(f"  • clip_ratio: {self.clip_ratio}")
        print(f"=" * 70 + "\n")
        
        start_time = datetime.now()
        
        for update in range(1, n_updates + 1):
            # Collecte des trajectoires
            rollout_info = self.collect_rollout()
            
            # Mise à jour du modèle
            loss_info = self.update()
            
            # Reset du buffer pour la prochaine itération
            self.buffer.reset()
            
            # Affichage des statistiques
            if update % 10 == 0 or update == 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"[Update {update:3d}/{n_updates}] | "
                      f"Return: {rollout_info['mean_return']:7.2f} | "
                      f"Episodes: {rollout_info['n_episodes']:2d} | "
                      f"Loss: {loss_info['loss']:.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        print("\n" + "=" * 70)
        print(" ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)


def main():
    """Fonction principale : orchestre tout"""
    
    # 1️ INSTANCIATION DE L'ENVIRONNEMENT
    print("1️  Instanciation de l'environnement...")
    env = WalkerEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"    Environnement créé")
    print(f"     - Observation space: {obs_dim}")
    print(f"     - Action space: {act_dim}\n")
    
    #  INSTANCIATION DU MODÈLE
    print(" Instanciation du modèle ActorCritic...")
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    # Effectuer un forward pass pour initialiser les poids
    dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
    model(dummy_obs)
    print(f"    Modèle créé et initialisé\n")
    
    # INSTANCIATION DU BUFFER
    print("  Instanciation du RolloutBuffer...")
    n_steps = 2048
    buffer = RolloutBuffer(n_steps=n_steps, obs_dim=obs_dim, act_dim=act_dim)
    print(f"    Buffer créé (n_steps={n_steps})\n")
    
    #  INSTANCIATION DE L'ENTRAÎNEUR PPO
    print(" Instanciation de l'entraîneur PPO...")
    trainer = PPOTrainer(
        env=env,
        model=model,
        buffer=buffer,
        n_steps=n_steps,
        n_epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        clip_ratio=0.2,
        ent_coef=0.01,
        vf_coef=0.5
    )
    print(f"    Entraîneur PPO créé\n")
    
    #  LANCEMENT DE LA BOUCLE D'ENTRAÎNEMENT
    print("  Lancement de la boucle d'entraînement...\n")
    trainer.train(n_updates=100)


if __name__ == "__main__":
    main()
