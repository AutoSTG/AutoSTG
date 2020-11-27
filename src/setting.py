from collections import namedtuple

import ruamel.yaml as yaml

data_path = '../dataset'
param_path = '../param'


def dict_to_namedtuple(dic: dict):
    return namedtuple('tuple', dic.keys())(**dic)


class Config:
    def __init__(self):
        pass

    def load_config(self, config):
        with open(config, 'r') as f:
            setting = yaml.load(f, Loader=yaml.RoundTripLoader)

        self.sys = dict_to_namedtuple(setting['sys'])
        self.data = dict_to_namedtuple(setting['data'])
        self.model = dict_to_namedtuple(setting['model'])
        self.trainer = dict_to_namedtuple(setting['trainer'])


config = Config()
