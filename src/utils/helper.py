import numpy as np


class Scaler:
    def __init__(self, data, missing_value=np.inf):
        values = data[data != missing_value]
        self.mean = values.mean()
        self.std = values.std()

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * self.std + self.mean


def add_indent(str_, num_spaces):
    s = str_.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    return '\n'.join(s)


def num_parameters(layer):
    def prod(arr):
        cnt = 1
        for i in arr:
            cnt = cnt * i
        return cnt

    cnt = 0
    for p in layer.parameters():
        cnt += prod(p.size())
    return cnt
