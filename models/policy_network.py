import Tensorflow as tf

class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
        ])

    def call(self, obs):
        x = self.backbone(obs)
        return x

if __name__ == "__main__":
    import numpy as np

    model = ActorCritic(obs_dim=14, act_dim=2)

    batch = np.randn(4, 14).astype(np.float32)
    x = model(batch)

    print(x.shape)