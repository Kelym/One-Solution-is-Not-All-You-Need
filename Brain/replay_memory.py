import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        random.seed(self.seed)

    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, fn_path):
        torch.save(self.buffer, fn_path)

    def load_buffer(self, fn_path):
        buffer = torch.load(fn_path)
        self.buffer = buffer

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
