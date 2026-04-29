import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Simple deque-based replay buffer.

    This is the original, easy-to-read implementation used for prototyping.
    Kept as a separate module so it can be swapped with a numpy ring buffer later.
    """

    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def sample_arrays(self, batch_size):
        batch = self.sample(batch_size)
        return (
            np.asarray(batch.state, dtype=np.float32),
            np.asarray(batch.action, dtype=np.float32),
            np.asarray(batch.reward, dtype=np.float32),
            np.asarray(batch.next_state, dtype=np.float32),
            np.asarray(batch.done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
