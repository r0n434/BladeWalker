import numpy as np
import os
# reduce TensorFlow verbose logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.walker_env import WalkerEnv
from models.sac_networks import GaussianPolicy, QNetwork
from training.replay_buffer import ReplayBuffer
from training.sac_tf import SACAgentTF

CONFIG = {
    # Env
    "obs_dim"        : 14,
    "act_dim"        : 4,

    # SAC
    "gamma"          : 0.99,
    "tau"            : 0.005,
    "alpha"          : 0.2,
    "auto_alpha"     : True,
    "actor_lr"       : 3e-4,
    "critic_lr"      : 3e-4,
    "alpha_lr"       : 3e-4,
    "target_entropy" : None,
    "hidden"         : (64, 64),

    # Entraînement
    "total_steps"    : 200000,
    "start_steps"    : 1000,
    "update_after"   : 1000,
    "update_every"   : 50,
    "batch_size"     : 256,
    "save_every"     : 50000,
    "save_dir"       : "models",
}


# ============================================================
# INSTANCIATION
# ============================================================

env = WalkerEnv()

state_dim = CONFIG["obs_dim"]
action_dim = CONFIG["act_dim"]

actor_net = GaussianPolicy(state_dim, action_dim, hidden=CONFIG["hidden"])
critic1_net = QNetwork(state_dim, action_dim, hidden=CONFIG["hidden"])
critic2_net = QNetwork(state_dim, action_dim, hidden=CONFIG["hidden"])
critic1_target = QNetwork(state_dim, action_dim, hidden=CONFIG["hidden"])
critic2_target = QNetwork(state_dim, action_dim, hidden=CONFIG["hidden"])

agent = SACAgentTF(
    state_dim,
    action_dim,
    gamma=CONFIG["gamma"],
    tau=CONFIG["tau"],
    alpha=CONFIG["alpha"],
    auto_alpha=CONFIG["auto_alpha"],
    actor_lr=CONFIG["actor_lr"],
    critic_lr=CONFIG["critic_lr"],
    alpha_lr=CONFIG["alpha_lr"],
    target_entropy=CONFIG["target_entropy"],
    hidden=CONFIG["hidden"],
    actor=actor_net,
    critic_1=critic1_net,
    critic_2=critic2_net,
    critic_1_target=critic1_target,
    critic_2_target=critic2_target,
)

buffer = ReplayBuffer()

# rolling window of episode returns for monitoring (comparable to PPO mean_reward)
ep_returns = deque(maxlen=100)
# keep recent per-step rewards so we can show a step-average reward
step_rewards = deque(maxlen=CONFIG["update_every"])

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

print("Début de l'entraînement...")

obs, _ = env.reset()
state = obs
episode_reward = 0.0
episode_len = 0

try:
    for step in range(1, CONFIG["total_steps"] + 1):
        if step < CONFIG["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.push(state, action, reward, next_state, float(done))

        # track per-step reward for reporting even before full episodes complete
        step_rewards.append(reward)

        state = next_state
        episode_reward += reward
        episode_len += 1

        if done:
            # print immediately when episode ends (PPO-like behavior)
            state, _ = env.reset()
            print(f"Episode done | step={step} | ep_len={episode_len} | ep_rew={episode_reward:.2f}")
            # store episode return for rolling mean monitoring
            ep_returns.append(episode_reward)
            episode_reward = 0.0
            episode_len = 0

        if step >= CONFIG["update_after"] and step % CONFIG["update_every"] == 0:
            metrics_list = []
            for _ in range(CONFIG["update_every"]):
                m = agent.update(buffer, batch_size=CONFIG["batch_size"])
                if m is not None:
                    metrics_list.append(m)

            if metrics_list:
                # moyenne des métriques sur le bloc d'updates
                mean_metrics = {}
                keys = list(metrics_list[0].keys())
                for k in keys:
                    mean_metrics[k] = float(np.mean([m[k] for m in metrics_list]))

                # rolling mean of episode returns is kept but not shown here
                mean_step = float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0
                update_idx = step // CONFIG["update_every"]

                print(
                    f"iter {update_idx:05d} | "
                    f"buf:{len(buffer):06d} | "
                    f"mean_step_rew {mean_step:+.3f} | "
                    f"mean_loss_total {mean_metrics['loss_total']:+.4f} | "
                    f"mean_loss_actor {mean_metrics['loss_actor']:+.4f} | "
                    f"mean_loss_critic {mean_metrics['loss_critic']:+.4f} | "
                    f"mean_loss_entropy {mean_metrics['loss_entropy']:+.4f} |"
                )
            else:
                # Pas de métriques retournées (cas improbable), afficher au moins la taille du buffer
                print(f"iter {step // CONFIG['update_every']:05d} | buf:{len(buffer):06d}")

            # episode prints are immediate; nothing else to do here

        if CONFIG["save_every"] and step % CONFIG["save_every"] == 0:
            path = os.path.join(CONFIG["save_dir"], f"sac_{step}")
            agent.save(path)
            print(f"  → checkpoint sauvegardé : {path}")

finally:
    env.close()