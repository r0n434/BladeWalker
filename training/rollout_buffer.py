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


if __name__ == "__main__":
    import numpy as np

    buf = RolloutBuffer(n_steps=2048, obs_dim=14, act_dim=4)

    for _ in range(5):
        obs = np.random.randn(14).astype(np.float32)
        action = np.random.uniform(-1, 1, (4,)).astype(np.float32)
        reward = float(np.random.rand())
        done = False
        log_prob = float(np.random.rand())
        value = np.random.rand()

        buf.add(obs, action, reward, done, log_prob, value)

    buf.compute_returns_and_advantages(last_value=0.0)

    print("advantages :", buf.advantages.shape)
    print("returns    :", buf.returns.shape)
    print("mean adv   :", buf.advantages.mean())
    print("std adv    :", buf.advantages.std())