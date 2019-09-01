from utils.accessor import load_json


class ConfigOrganizer(object):
    def __init__(self):
        super(ConfigOrganizer, self).__init__()
        self.config = {}

    def load_config(self, path, name):
        config = load_json(path)
        self.config.update({name: config})

    def fetch_config(self, name):
        assert name in self.config, "Invalid config name!"
        return self.config[name]


configOrganizer = ConfigOrganizer()
