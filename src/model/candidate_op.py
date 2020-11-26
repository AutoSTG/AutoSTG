import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import MLP, normalize_adj_mats


def create_op(op_name, in_channels, out_channels, setting):
    name2op = {
        'Zero': lambda: Zero(),
        'Identity': lambda: Identity(),
        'Conv': lambda: Conv(in_channels, out_channels, **setting),
        'GraphConv': lambda: GraphConv(in_channels, out_channels, **setting),
        'MetaConv': lambda: MetaConv(in_channels, out_channels, **setting),
        'MetaGraphConv': lambda: MetaGraphConv(in_channels, out_channels, **setting),
    }
    op = name2op[op_name]()
    return op


class BasicOp(nn.Module):
    def __init__(self, **kwargs):
        super(BasicOp, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        cfg = []
        for (key, value) in self.setting:
            cfg += [str(key) + ': ' + str(value)]
        return str(self.type) + '(' + ', '.join(cfg) + ')'

    @property
    def type(self):
        raise NotImplementedError

    @property
    def setting(self):
        raise NotImplementedError


class Identity(BasicOp):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, **kwargs):
        x = 0
        for i in inputs: x += i
        return x

    @property
    def type(self):
        return 'identity'

    @property
    def setting(self):
        return []


class Zero(BasicOp):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, inputs, **kwargs):
        return torch.zeros_like(inputs[0])

    @property
    def type(self):
        return 'zero'

    @property
    def setting(self):
        return []


class Conv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1, use_bn=True, dropout=0.3,
                 type_name='conv3'):
        super(Conv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._type_name = type_name
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._use_bn = use_bn
        self._dropout = dropout
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, **kwargs):
        x = 0
        for i in inputs: x += i
        x = torch.relu(x)
        x = self._conv(x)
        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('kernel_size', self._kernel_size),
            ('stride', self._stride),
            ('padding', self._padding),
            ('dilation', self._dilation),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


class MetaConv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 node_in_hidden, meta_hiddens,
                 kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1, use_bn=True, dropout=0.3,
                 type_name='tc'):
        super(MetaConv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._node_in_hidden = node_in_hidden
        self._meta_hiddens = meta_hiddens

        self._type_name = type_name
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._use_bn = use_bn
        self._dropout = dropout

        self._conv = nn.Conv2d(in_channels, out_channels * meta_hiddens[-1], kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self._node_mlp = MLP(node_in_hidden, meta_hiddens)
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, node_fts, **kwargs):
        x = 0
        for i in inputs: x += i
        x = torch.relu(x)
        x = self._conv(x)  # [batch_size, out_channels * meta_dims[-1], n, T]
        B, _, N, T = x.size()

        x = x.view(B, self._out_channels, self._meta_hiddens[-1], N, T)
        y = self._node_mlp(node_fts).view(-1, 1, self._meta_hiddens[-1], 1, 1).transpose(0, 3).contiguous()
        x = torch.sum(x * y, dim=2)

        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('node_in_hidden', self._node_in_hidden),
            ('meta_hiddens', self._meta_hiddens),
            ('kernel_size', self._kernel_size),
            ('stride', self._stride),
            ('padding', self._padding),
            ('dilation', self._dilation),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


def gconv(x, A):
    x = torch.einsum('ncvl,vw->ncwl', (x, A))
    return x.contiguous()


class GraphConv(BasicOp):
    def __init__(self, in_channels, out_channels, num_graphs, order, use_bn=True, dropout=0.3):
        super(GraphConv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_graphs = num_graphs
        self._order = order
        self._use_bn = use_bn
        self._dropout = dropout

        self._linear = nn.Conv2d(in_channels * (num_graphs * order + 1), out_channels, kernel_size=(1, 1),
                                 stride=(1, 1))
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, adj_mats, **kwargs):
        adj_mats = normalize_adj_mats(adj_mats)

        x = 0
        for i in inputs: x += i
        x = torch.relu(x)

        outputs = [x]
        for i in range(self._num_graphs):
            y = x
            for j in range(self._order):
                y = gconv(y, adj_mats[:, :, i].squeeze())
                outputs += [y]

        x = torch.cat(outputs, dim=1)
        x = self._linear(x)
        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return 'gconv'

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('num_graphs', self._num_graphs),
            ('order', self._order),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


class MetaGraphConv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 edge_in_hidden, meta_hiddens, num_graphs, order, use_bn=True, dropout=0.3):
        super(MetaGraphConv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._edge_in_hidden = edge_in_hidden
        self._meta_hiddens = meta_hiddens

        self._num_graphs = num_graphs
        self._order = order
        self._use_bn = use_bn
        self._dropout = dropout

        self._edge_mlp = MLP(edge_in_hidden, meta_hiddens + [num_graphs])
        self._linear = nn.Conv2d(in_channels * (num_graphs * order + 1), out_channels, kernel_size=(1, 1),
                                 stride=(1, 1))
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, adj_mats, **kwargs):

        mask = (adj_mats.abs().sum(dim=2, keepdim=True) > 1e-5).float()
        adj_mats = self._edge_mlp(adj_mats) * mask
        adj_mats = normalize_adj_mats(adj_mats)

        x = 0
        for i in inputs: x += i
        x = torch.relu(x)

        outputs = [x]
        for i in range(self._num_graphs):
            y = x
            for j in range(self._order):
                y = gconv(y, adj_mats[:, :, i].squeeze())
                outputs += [y]

        x = torch.cat(outputs, dim=1)
        x = self._linear(x)

        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return 'sc'

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('meta_hiddens', self._meta_hiddens),
            ('num_graphs', self._num_graphs),
            ('order', self._order),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]
