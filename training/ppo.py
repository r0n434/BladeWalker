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


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.policy_network import ActorCritic

    model = ActorCritic(obs_dim=14, act_dim=4)
    ppo = PPO(model)

    print("clip_eps: ", ppo.clip_eps)
    print("c1: ", ppo.c1)
    print("c2: ", ppo.c2)
    print("n_epochs: ", ppo.n_epochs)
    print("optimizer: ", ppo.optimizer)