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

    print("steps stockés :", len(buf.obs))
    print("shape obs[0]  :", buf.obs[0].shape)
    print("shape act[0]  :", buf.actions[0].shape)