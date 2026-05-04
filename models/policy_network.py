import tensorflow as tf
import tensorflow_probability as tfp
from .common_backbone import MLP


class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Use the shared MLP backbone as a feature extractor. We set
        # output_dim=None so the MLP returns the last hidden activations
        # (two Dense(64, tanh) layers) preserving original behavior.
        self.backbone = MLP(output_dim=None, hidden=(64, 64), activation=tf.nn.tanh)

        self.actor_mean = tf.keras.layers.Dense(act_dim)
        self.critic = tf.keras.layers.Dense(1)

        self.log_std = tf.Variable(tf.zeros(act_dim), trainable=True)

    def call(self, obs):
        x = self.backbone(obs)
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, value

    @tf.function
    def get_action(self, obs):
        mean, value = self.call(obs)

        std = tf.exp(self.log_std)
        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample()
        action = tf.clip_by_value(action, -1.0, 1.0)
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        value = tf.squeeze(value, axis=-1)

        return action, log_prob, value

    @tf.function
    def evaluate_actions(self, obs, actions):
        mean, value = self.call(obs)

        std = tf.exp(self.log_std)
        dist = tfp.distributions.Normal(mean, std)

        log_prob = tf.reduce_sum(dist.log_prob(actions), axis=-1)
        entropy = tf.reduce_sum(dist.entropy(), axis=-1)
        value = tf.squeeze(value, axis=-1)

        return log_prob, value, entropy

if __name__ == "__main__":
    import numpy as np

    model = ActorCritic(obs_dim=14, act_dim=4)
    batch = np.random.randn(4, 14).astype(np.float32) # valeur random à remplacer par les vrais observations du walker
    actions = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)

    action, log_prob, value = model.get_action(batch)
    print("get_action")
    print("action: ", action.shape)
    print("log_prob: ", log_prob.shape)
    print("value: ", value.shape)

    log_prob, value, entropy = model.evaluate_actions(batch, actions)
    print("evaluate_actions")
    print("log_prob: ", log_prob.shape)
    print("value: ", value.shape)
    print("entropy: ", entropy.shape)