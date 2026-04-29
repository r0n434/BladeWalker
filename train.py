import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.policy_network import ActorCritic
from training.rollout_buffer import RolloutBuffer
from training.ppo import PPO
from envs.walker_env import WalkerEnv

CONFIG = {
    # Env
    "obs_dim"        : 14,
    "act_dim"        : 4,

    # Rollout
    "n_steps"        : 2048,

    # PPO
    "lr"             : 3e-4,
    "clip_eps"       : 0.2,
    "c1"             : 0.5,
    "c2"             : 0.01,
    "n_epochs"       : 10,
    "batch_size"     : 64,
    "gamma"          : 0.99,
    "lam"            : 0.95,

    # Entraînement
    "max_iterations" : 1000,
    "save_every"     : 50,
    "checkpoint_dir" : "checkpoints",
}


# ============================================================
# INSTANCIATION
# ============================================================

env = WalkerEnv()
model = ActorCritic(obs_dim=CONFIG["obs_dim"], act_dim=CONFIG["act_dim"])

dummy = tf.zeros((1, CONFIG["obs_dim"]))
model(dummy)

buffer = RolloutBuffer(
    n_steps=CONFIG["n_steps"],
    obs_dim=CONFIG["obs_dim"],
    act_dim=CONFIG["act_dim"],
    gamma=CONFIG["gamma"],
    lam=CONFIG["lam"],
)
ppo = PPO(
    model = model,
    lr=CONFIG["lr"],
    clip_eps=CONFIG["clip_eps"],
    c1=CONFIG["c1"],
    c2=CONFIG["c2"],
    n_epochs=CONFIG["n_epochs"],
    batch_size=CONFIG["batch_size"],
)
# Construire le modèle en appelant une fois avec un dummy input
dummy_obs = tf.zeros((1, CONFIG["obs_dim"]), dtype=tf.float32)
model.get_action(dummy_obs)

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)


# ============================================================
# ROLLOUT
# ============================================================

def collect_rollout(env, model, buffer):
    buffer.reset()
    obs, _ = env.reset()

    for step in range(CONFIG["n_steps"]):
        obs_tensor = tf.convert_to_tensor(obs[np.newaxis], dtype=tf.float32)
        action, log_prob, value = model.get_action(obs_tensor)

        action_np = action.numpy()[0]
        log_prob_np = float(log_prob.numpy()[0])
        value_np = float(value.numpy()[0])

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        buffer.add(obs, action_np, reward, done, log_prob_np, value_np)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    last_obs_tensor = tf.convert_to_tensor(obs[np.newaxis], dtype=tf.float32)
    _, _, last_value = model.get_action(last_obs_tensor)
    last_value_np = float(last_value.numpy()[0])

    buffer.compute_returns_and_advantages(last_value=last_value_np)

    return buffer

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
print("Début de l'entraînement...")

for iteration in range(CONFIG["max_iterations"]):
    collect_rollout(env, model, buffer)

    metrics = ppo.update(buffer)

    mean_reward = float(np.mean(buffer.rewards))
    print(
        f"iter {iteration:04d} | "
        f"reward: {mean_reward:+.3f} | "
        f"loss_total {metrics['loss_total']:+.4f} | "
        f"loss_actor {metrics['loss_actor']:+.4f} | "
        f"loss_critic {metrics['loss_critic']:+.4f} | "
        f"loss_entropy {metrics['loss_entropy']:+.4f} |"
    )

    if iteration % CONFIG["save_every"] == 0:
        path = os.path.join(CONFIG["checkpoint_dir"], f"iter_{iteration:04d}.weights.h5")
        model.save_weights(path)
        print(f"  → checkpoint sauvegardé : {path}")