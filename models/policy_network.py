import tensorflow as tf
import tensorflow_probability as tfp

class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
        ])

        self.actor_mean = tf.keras.layers.Dense(act_dim)
        self.critic = tf.keras.layers.Dense(1)

        self.log_std = tf.Variable(tf.zeros(act_dim), trainable=True)

    def call(self, obs):
        x = self.backbone(obs)
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, value

    def get_action(self, obs):
        mean, value = self.call(obs)

        std = tf.exp(self.log_std)
        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample()
        action = tf.clip_by_value(action, -1.0, 1.0)
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        value = tf.squeeze(value, axis=-1)

        return action, log_prob, value

if __name__ == "__main__":
    import numpy as np

    model = ActorCritic(obs_dim=14, act_dim=4)
    batch = np.random.randn(4, 14).astype(np.float32) #valeur random à remplacer par les vrais observations du walker

    action, log_prob, value = model.get_action(batch)

    print("action: ", action.shape)
    print("log_prob: ", log_prob.shape)
    print("value: ", value.shape)