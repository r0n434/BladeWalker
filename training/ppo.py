import tensorflow as tf

class PPO:
    def __init__(
            self,
            model,
            lr: float = 3e-4,
            clip_eps: float = 0.2,
            c1: float = 0.5,
            c2: float = 0.01,
            n_epochs: int = 10,
            batch_size: int = 64,
    ):
        self.model = model
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _compute_losses(self, obs, actions, old_log_probs, returns, advantages):
        log_probs, values, entropy = self.model.evaluate_actions(obs, actions)

        ratio = tf.exp(log_probs - old_log_probs)

        unclipped = ratio * advantages
        clipped = tf.clip_by_value(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss_actor = -tf.reduce_mean(tf.minimum(unclipped, clipped))

        loss_critic = tf.reduce_mean(tf.square(values - returns))

        loss_entropy = -tf.reduce_mean(entropy)

        loss_total = loss_actor + self.c1 * loss_critic + self.c2 * loss_entropy

        return loss_total, loss_actor, loss_critic, loss_entropy

    def update(self, buffer):
        metrics = {
            "loss_total" : [],
            "loss_actor" : [],
            "loss_critic" : [],
            "loss_entropy" : [],
        }

        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs, actions, old_log_probs, returns, advantages = batch

                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
                returns = tf.convert_to_tensor(returns, dtype=tf.float32)
                advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    loss_total, loss_actor, loss_critic, loss_entropy = \
                        self._compute_losses(obs, actions, old_log_probs, returns, advantages)

                grads = tape.gradient(loss_total, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                metrics["loss_total"].append(loss_total.numpy())
                metrics["loss_actor"].append(loss_actor.numpy())
                metrics["loss_critic"].append(loss_critic.numpy())
                metrics["loss_entropy"].append(loss_entropy.numpy())

        return {k: sum(v) / len(v) for k, v in metrics.items()}


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from models.policy_network import ActorCritic
    from training.rollout_buffer import RolloutBuffer
    import tensorflow as tf

    model  = ActorCritic(obs_dim=14, act_dim=4)
    ppo    = PPO(model)
    buf    = RolloutBuffer(n_steps=128, obs_dim=14, act_dim=4)

    # rollout fictif
    obs = np.random.randn(14).astype(np.float32)
    for _ in range(128):
        obs_t                    = tf.expand_dims(obs, axis=0)
        action, log_prob, value  = model.get_action(obs_t)
        buf.add(
            obs      = obs,
            action   = action.numpy()[0],
            reward   = float(np.random.randn()),
            done     = False,
            log_prob = float(log_prob.numpy()[0]),
            value    = float(value.numpy()[0]),
        )
        obs = np.random.randn(14).astype(np.float32)

    last_obs = tf.expand_dims(obs, axis=0)
    _, _, last_value = model.get_action(last_obs)
    buf.compute_returns_and_advantages(last_value=float(last_value.numpy()[0]))

    # update PPO
    weights_before = model.trainable_variables[0].numpy().copy()
    metrics        = ppo.update(buf)
    weights_after  = model.trainable_variables[0].numpy().copy()

    print("loss_total  :", metrics["loss_total"])
    print("loss_actor  :", metrics["loss_actor"])
    print("loss_critic :", metrics["loss_critic"])
    print("loss_entropy:", metrics["loss_entropy"])
    print("poids changés:", not np.allclose(weights_before, weights_after))