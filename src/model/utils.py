import torch
import torch.nn as nn


def normalize_adj_mats(adj_mats):
    mask = (adj_mats > 1e-3).float()
    adj_mats = torch.softmax(adj_mats, dim=1) * mask
    adj_mats = (1.0 / (adj_mats.sum(dim=1, keepdim=True) + 1e-8)) * adj_mats
    return adj_mats


def create_activation(activation):
    if activation == 'Sigmoid': return nn.Sigmoid()
    if activation == 'ReLU': return nn.ReLU()
    if activation == 'Tanh': return nn.Tanh()
    raise Exception('unknown activation!')


class MLP(nn.Module):
    def __init__(self, in_hidden, hiddens, activation='ReLU'):
        super(MLP, self).__init__()
        self._in_hidden = in_hidden
        self._hiddens = hiddens
        self._activation = activation
        self._layers = nn.ModuleList()
        for i, h in enumerate(hiddens):
            self._layers += [nn.Linear(in_hidden if i == 0 else hiddens[i - 1], h)]
            if i != len(hiddens) - 1:
                self._layers += [create_activation(activation)]

    def forward(self, x):
        for i, l in enumerate(self._layers):
            x = l(x)
        return x
