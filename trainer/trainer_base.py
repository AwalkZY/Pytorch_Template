class BaseTrainer(object):
    def __init__(self):
        super().__init__()

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
