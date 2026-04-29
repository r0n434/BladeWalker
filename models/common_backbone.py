import tensorflow as tf


class MLP(tf.keras.Model):
    """Small configurable MLP backbone used by model heads.

    Behavior:
    - If `output_dim` is None, the MLP returns the last hidden layer's
      activations (feature extractor). This preserves the original PPO
      `ActorCritic` backbone semantics (two Dense(64, tanh) layers -> features).
    - If `output_dim` is provided, a final linear layer is appended and its
      activation can be controlled via `final_activation` (used by SAC heads).

    Defaults: `hidden=(64, 64)`, `activation=tf.nn.tanh` to match PPO.
    """

    def __init__(self, output_dim=None, hidden=(64, 64), activation=tf.nn.tanh, final_activation=None):
        super().__init__()
        self._layers = []
        for units in hidden:
            self._layers.append(tf.keras.layers.Dense(units, activation=activation))
        if output_dim is not None:
            self._layers.append(tf.keras.layers.Dense(output_dim, activation=final_activation))

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
