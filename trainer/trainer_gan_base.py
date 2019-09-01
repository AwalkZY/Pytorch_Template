import torch


class BaseTrainer(object):
    def __init__(self):
        super().__init__()

    def train(self, *inputs):
        raise NotImplementedError

    def generator_train(self, *inputs):
        raise NotImplementedError

    def discriminator_train(self, *inputs):
        raise NotImplementedError

    def generator_evaluate(self, *inputs):
        raise NotImplementedError

    def discriminator_evaluate(self, *inputs):
        raise NotImplementedError
