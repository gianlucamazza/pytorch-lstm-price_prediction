import json


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def save(self, path):
        with open(path, 'w') as config_file:
            json.dump(self.__dict__, config_file, indent=4)


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)
    return Config(config_dict)


def update_config(config, key, value):
    keys = key.split('.')
    d = config.__dict__
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return config
