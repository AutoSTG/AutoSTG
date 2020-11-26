import torch
import torch.nn as nn

from model.utils import normalize_adj_mats


class GMKLearner(nn.Module):
    def __init__(self, node_hiddens, edge_hiddens):
        super(GMKLearner, self).__init__()

        self._num_iters = len(node_hiddens) - 1
        self._node_hiddens = node_hiddens
        self._edge_hiddens = edge_hiddens

        self._node_learners = nn.ModuleList()
        self._edge_learners = nn.ModuleList()
        for i in range(1, self._num_iters + 1):
            self._node_learners += [GConv(self._node_hiddens[i - 1], self._node_hiddens[i], self._edge_hiddens[i - 1])]
            self._edge_learners += [
                nn.Linear(self._edge_hiddens[i - 1] + self._node_hiddens[i] * 2, self._edge_hiddens[i])]

    def forward(self, node_fts, adj_mats):
        n = adj_mats.size(0)
        mask = (adj_mats.abs().sum(dim=2, keepdim=True) > 1e-5).float()
        for i in range(self._num_iters):
            node_fts = torch.relu(self._node_learners[i](node_fts, adj_mats))
            _mat1 = torch.repeat_interleave(node_fts.view(n, 1, -1), n, dim=1)
            _mat2 = torch.repeat_interleave(node_fts.view(1, n, -1), n, dim=0)
            adj_mats = torch.cat([adj_mats, _mat1, _mat2], dim=2)
            adj_mats = torch.relu(self._edge_learners[i](adj_mats)) * mask
        return node_fts, adj_mats


class GConv(nn.Module):
    def __init__(self, in_hidden, out_hidden, num_adj_mats, order=2):
        super(GConv, self).__init__()
        self._in_hidden = in_hidden
        self._out_hidden = out_hidden
        self._num_adj_mats = num_adj_mats
        self._order = order
        self._linear = nn.Linear(self._in_hidden * (self._num_adj_mats * self._order + 1), self._out_hidden)

    def forward(self, x, adj_mats):
        adj_mats = normalize_adj_mats(adj_mats)
        out = [x]
        for i in range(self._num_adj_mats):
            _x = x
            for k in range(self._order):
                _x = torch.mm(adj_mats[:, :, i], _x)
                out += [_x]
        h = torch.cat(out, dim=-1)
        h = self._linear(h)
        return h
