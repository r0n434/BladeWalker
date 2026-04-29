import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les messages INFO/WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Supprime les messages oneDNN

import numpy as np

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam

        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        #calculés après le rollout
        self.returns = []
        self.advantages = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float):
        advantages = []
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            reward = self.rewards[t]
            done = self.dones[t]
            value = self.values[t]

            if t == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = reward + self.gamma * next_value * (1 - done) - value

            gae = delta + self.gamma * self.lam * (1 - done) * gae

            advantages.insert(0, gae)

        self.advantages = np.array(advantages, dtype=np.float32)
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        n = len(self.obs)
        indices = np.random.permutation(n)

        obs = np.array(self.obs, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        returns = np.array(self.returns, dtype=np.float32)
        advantages = np.array(self.advantages, dtype=np.float32)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield (
                obs[batch_idx],
                actions[batch_idx],
                log_probs[batch_idx],
                returns[batch_idx],
                advantages[batch_idx],
            )


if __name__ == "__main__":
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from models.policy_network import ActorCritic
    import tensorflow as tf

    # instanciation
    buf = RolloutBuffer(n_steps=20, obs_dim=14, act_dim=4)
    model = ActorCritic(obs_dim=14, act_dim=4)

    #simulation rollout
    obs = np.random.randn(14).astype(np.float32)

    for step in range(20):
        obs_tensor = tf.expand_dims(obs, axis=0)
        action, log_prob, value = model.get_action(obs_tensor)

        #scalaire --> array
        action_np = action.numpy()[0]
        log_prob_np = float(log_prob.numpy()[0])
        value_np = float(value.numpy()[0])

        #reward alléatoire
        reward = float(np.random.randn())
        done = False

        buf.add(obs, action_np, reward, done, log_prob_np, value_np)

        #prochain obs fictif
        obs = np.random.randn(14).astype(np.float32)

    last_obs = tf.expand_dims(obs, axis=0)
    _, _, last_value = model.get_action(last_obs)
    buf.compute_returns_and_advantages(last_value=float(last_value.numpy()[0]))

    print("advantages: ", buf.advantages.shape)
    print("returns: ", buf.returns.shape)
    print("mean adv: ", buf.advantages.mean())
    print("std adv: ", buf.advantages.std())

    # --- batches ---
    print("\n--- batches ---")
    for i, batch in enumerate(buf.get_batches(batch_size=4)):
        obs_b, act_b, lp_b, ret_b, adv_b = batch
        print(f"batch {i} — obs: {obs_b.shape}, actions: {act_b.shape}, advantages: {adv_b.shape}")

    # --- reset ---
    buf.reset()
    print("\naprès reset — steps stockés :", len(buf.obs))  # 0