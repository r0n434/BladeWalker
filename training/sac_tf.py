import math
import random
import os
# reduce TensorFlow verbose logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import numpy as np
import tensorflow as tf

from envs.walker_env import WalkerEnv
from models.sac_networks import GaussianPolicy, QNetwork
from .replay_buffer import ReplayBuffer

def soft_update(target_weights, source_weights, tau):
    for (t, s) in zip(target_weights, source_weights):
        t.assign(t * (1.0 - tau) + s * tau)

class SACAgentTF:
    def __init__(self, state_dim, action_dim, action_range=1.0,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 target_entropy=None, hidden=(64,64),
                 actor=None, critic_1=None, critic_2=None,
                 critic_1_target=None, critic_2_target=None):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.action_range = action_range

        # allow injecting custom networks (keeps backward compatibility)
        if actor is None:
            self.actor = GaussianPolicy(state_dim, action_dim, hidden)
        else:
            self.actor = actor

        if critic_1 is None:
            self.critic_1 = QNetwork(state_dim, action_dim, hidden)
        else:
            self.critic_1 = critic_1

        if critic_2 is None:
            self.critic_2 = QNetwork(state_dim, action_dim, hidden)
        else:
            self.critic_2 = critic_2

        # target critics: if provided use them, otherwise create new instances
        if critic_1_target is None:
            self.critic_1_target = QNetwork(state_dim, action_dim, hidden)
        else:
            self.critic_1_target = critic_1_target

        if critic_2_target is None:
            self.critic_2_target = QNetwork(state_dim, action_dim, hidden)
        else:
            self.critic_2_target = critic_2_target

        # build networks (call once to create weights)
        dummy_s = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_a = tf.zeros((1, action_dim), dtype=tf.float32)
        # ensure variables are created for all networks
        self.actor(dummy_s)
        self.critic_1(dummy_s, dummy_a)
        self.critic_2(dummy_s, dummy_a)
        self.critic_1_target(dummy_s, dummy_a)
        self.critic_2_target(dummy_s, dummy_a)

        # copy weights from critics to their targets to initialize
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

        # optimizers
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_1_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # alpha (entropy)
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        if auto_alpha:
            self.log_alpha = tf.Variable(math.log(alpha), dtype=tf.float32)
            self.alpha_opt = tf.keras.optimizers.Adam(learning_rate=alpha_lr)
        else:
            self._alpha = alpha

    @property
    def alpha(self):
        if self.auto_alpha:
            return tf.exp(self.log_alpha)
        return self._alpha

    def select_action(self, state, deterministic=False):
        s = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        if deterministic:
            _, _, mu = self.actor.sample(s)
            return (mu[0].numpy() * self.action_range)
        else:
            a, _, _ = self.actor.sample(s)
            return (a[0].numpy() * self.action_range)

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done = replay_buffer.sample_arrays(batch_size)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward[:, None], dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        done = tf.convert_to_tensor(done[:, None], dtype=tf.float32)

        # critic update
        with tf.GradientTape(persistent=True) as tape:
            next_action, next_logp, _ = self.actor.sample(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)[:, None]
            target_q2 = self.critic_2_target(next_state, next_action)[:, None]
            target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_logp
            q_target = reward + (1.0 - done) * self.gamma * target_q

            q1 = self.critic_1(state, action)[:, None]
            q2 = self.critic_2(state, action)[:, None]
            critic1_loss = tf.reduce_mean(tf.square(q1 - q_target))
            critic2_loss = tf.reduce_mean(tf.square(q2 - q_target))

        grads1 = tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        grads2 = tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(grads1, self.critic_1.trainable_variables))
        self.critic_2_opt.apply_gradients(zip(grads2, self.critic_2.trainable_variables))
        del tape

        # actor update
        with tf.GradientTape() as tape2:
            action_pi, log_pi, _ = self.actor.sample(state)
            q1_pi = self.critic_1(state, action_pi)[:, None]
            q2_pi = self.critic_2(state, action_pi)[:, None]
            q_pi = tf.minimum(q1_pi, q2_pi)
            actor_loss = tf.reduce_mean(self.alpha * log_pi - q_pi)

        actor_grads = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # alpha update
        if self.auto_alpha:
            with tf.GradientTape() as tape3:
                _, log_pi, _ = self.actor.sample(state)
                alpha_loss = -tf.reduce_mean(self.log_alpha * (log_pi + self.target_entropy))
            alpha_grads = tape3.gradient(alpha_loss, [self.log_alpha])
            self.alpha_opt.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        else:
            alpha_loss = tf.constant(0.0, dtype=tf.float32)

        # soft update targets
        soft_update(self.critic_1_target.variables, self.critic_1.variables, self.tau)
        soft_update(self.critic_2_target.variables, self.critic_2.variables, self.tau)

        return {
            "loss_total": float((critic1_loss + critic2_loss + actor_loss).numpy()),
            "loss_actor": float(actor_loss.numpy()),
            "loss_critic": float(((critic1_loss + critic2_loss) * 0.5).numpy()),
            "loss_entropy": float(alpha_loss.numpy()),
        }

    def save(self, path_prefix):
        # save weights
        self.actor.save_weights(path_prefix + '_actor.weights.h5')
        self.critic_1.save_weights(path_prefix + '_critic1.weights.h5')
        self.critic_2.save_weights(path_prefix + '_critic2.weights.h5')
        if self.auto_alpha:
            np.save(path_prefix + '_log_alpha.npy', self.log_alpha.numpy())

    def load(self, path_prefix):
        self.actor.load_weights(path_prefix + '_actor.weights.h5')
        self.critic_1.load_weights(path_prefix + '_critic1.weights.h5')
        self.critic_2.load_weights(path_prefix + '_critic2.weights.h5')
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        if self.auto_alpha:
            alpha_path = path_prefix + '_log_alpha.npy'
            if os.path.exists(alpha_path):
                self.log_alpha.assign(np.load(alpha_path).item())


def train_sac(
    total_steps=200000,
    start_steps=1000,
    update_after=1000,
    update_every=50,
    batch_size=256,
    save_every=50000,
    save_dir="models",
    render_mode=None,
    seed=None,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    print("Début de l'entraînement...")

    env = WalkerEnv(render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # construct networks explicitly and inject into the agent (keeps parity with PPO style)
    actor_net = GaussianPolicy(state_dim, action_dim, hidden=(64,64))
    critic1_net = QNetwork(state_dim, action_dim, hidden=(64,64))
    critic2_net = QNetwork(state_dim, action_dim, hidden=(64,64))
    critic1_target = QNetwork(state_dim, action_dim, hidden=(64,64))
    critic2_target = QNetwork(state_dim, action_dim, hidden=(64,64))

    agent = SACAgentTF(
        state_dim,
        action_dim,
        actor=actor_net,
        critic_1=critic1_net,
        critic_2=critic2_net,
        critic_1_target=critic1_target,
        critic_2_target=critic2_target,
    )
    buffer = ReplayBuffer()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    state = obs
    episode_reward = 0.0
    episode_len = 0

    try:
        for step in range(1, total_steps + 1):
            if step < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward
            episode_len += 1

            if done:
                state, _ = env.reset()
                print(f"Episode done | step={step} | ep_len={episode_len} | ep_rew={episode_reward:.2f}")
                episode_reward = 0.0
                episode_len = 0

            if step >= update_after and step % update_every == 0:
                for _ in range(update_every):
                    agent.update(buffer, batch_size=batch_size)

            if save_every and step % save_every == 0:
                agent.save(str(save_path / f"sac_{step}"))

    finally:
        env.close()

    return agent
