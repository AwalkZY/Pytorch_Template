class BaseExtractor(object):
    def __init__(self):
        super().__init__()

    def extract_all(self):
        raise NotImplementedError

    def extract_item(self, *inputs):
        raise NotImplementedError
