from utils.accessor import load_json, load_yaml


class ConfigContainer(object):
    def __init__(self):
        super(ConfigContainer, self).__init__()
        self.config = {}

    def load_config(self, path, name):
        config = load_yaml("./config/" + path + ".yaml")
        self.config.update({name: config})
        return self.config[name]

    def flatten_config(self, path, prefix, common_name):
        config = load_yaml("./config/" + path + ".yaml")
        assert common_name in config, "Invalid common config name!"
        common_config = config[common_name]
        for configName in config:
            if configName != common_name:
                config[configName].update(common_config)
                self.config.update({prefix + "_" + configName: config[configName]})

    def fetch_config(self, name):
        assert name in self.config, "Invalid config name!"
        return self.config[name]

    def concat_config(self, target_name, source_name):
        assert target_name in self.config, "Invalid target config name!"
        assert source_name in self.config, "Invalid source config name!"
        self.config[target_name].update(self.config[source_name])


class ModelContainer(object):
    def __init__(self):
        super(ModelContainer, self).__init__()
        self.model = {}
        self.data = {}

    def save_model(self, model, name, data, criterion, greater_better=True):
        assert criterion in data, "Incompatible criterion name!"
        if (name not in self.model) or ((self.data[name][criterion] <= data[criterion]) == greater_better):
            self.model.update({name: model})
            self.data.update({name: data})
        return self.data[name]

    def fetch_model(self, name):
        assert name in self.model, "Invalid model name!"
        return self.model[name], self.data[name]


def merge_dicts(a, b):
    for key, value in b.items():
        a[key] = value + a[key] if key in a else value
    return a


class MetricsContainer(object):
    def __init__(self):
        super().__init__()
        self.data = {}

    def reset(self, model_name):
        self.data[model_name] = {"count": 0}

    def update(self, model_name, metrics):
        if model_name not in self.data:
            self.reset(model_name)
        self.data[model_name] = merge_dicts(self.data[model_name], metrics)
        self.data[model_name]["count"] += 1

    def calculate_average(self, model_name):
        result = {}
        cum_result = self.data[model_name]
        for key in cum_result:
            result[key] = cum_result[key] / cum_result["count"]
        return result


configContainer = ConfigContainer()
modelContainer = ModelContainer()
metricsContainer = MetricsContainer()
