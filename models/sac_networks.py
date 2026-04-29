import tensorflow as tf
import tensorflow_probability as tfp

from .common_backbone import MLP

tfd = tfp.distributions


class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden=(64, 64)):
        super().__init__()
        self.net = MLP(1, hidden=hidden)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        q = self.net(x)
        return tf.squeeze(q, axis=-1)


class GaussianPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden=(64, 64), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = MLP(action_dim * 2, hidden=hidden)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def call(self, state):
        out = self.net(state)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=-1)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state, eps=1e-6):
        mu, log_std = self.call(state)
        std = tf.exp(log_std)
        normal = tfd.Normal(loc=mu, scale=std)
        z = normal.sample()
        action = tf.tanh(z)
        log_prob = normal.log_prob(z) - tf.math.log(1 - tf.square(action) + eps)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        return action, log_prob, tf.tanh(mu)
