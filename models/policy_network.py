import tensorflow as tf

class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
        ])

        self.actor_mean = tf.keras.layers.Dense(act_dim)
        self.critic = tf.keras.layers.Dense(1)

    def call(self, obs):
        x = self.backbone(obs)
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, value

if __name__ == "__main__":
    import numpy as np

    model = ActorCritic(obs_dim=14, act_dim=4)

    batch = np.random.randn(4, 14).astype(np.float32)
    mean, value = model(batch)

    print("mean: ", mean.shape)
    print("value: ", value.shape)