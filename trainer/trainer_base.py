import torch


class BaseTrainer(object):
    def __init__(self):
        super().__init__()

    def train(self, *inputs):
        raise NotImplementedError

    def evaluate(self, *inputs):
        raise NotImplementedError
