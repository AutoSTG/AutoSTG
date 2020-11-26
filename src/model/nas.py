import torch
import torch.nn as nn

from model.cell import STCell
from model.gmk_learner import GMKLearner
from model.mode import Mode

def create_layer(name, hidden_channels, num_mixed_ops, candidate_op_profiles):
    if name == 'STCell':
        return STCell(hidden_channels, num_mixed_ops, candidate_op_profiles)
    if name == 'Pooling':
        return nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
    raise Exception('unknown layer name!')


class AutoSTG(nn.Module):

    def __init__(self,
                 in_length, out_length,
                 node_hiddens, edge_hiddens,
                 in_channels, out_channels, hidden_channels, skip_channels, end_channels,
                 layer_names,
                 num_mixed_ops, candidate_op_profiles):
        super(AutoSTG, self).__init__()

        self._in_length = in_length
        self._out_length = out_length

        self._node_hiddens = node_hiddens
        self._edge_hiddens = edge_hiddens

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._hidden_channels = hidden_channels
        self._skip_channels = skip_channels
        self._end_channels = end_channels

        self._num_mixed_ops = num_mixed_ops
        self._candidate_op_profiles = candidate_op_profiles

        self._gmk_learner = GMKLearner(self._node_hiddens, self._edge_hiddens)
        self._layers = nn.ModuleList()
        self._skip_convs = nn.ModuleList()
        for name in layer_names:
            self._layers += [
                create_layer(name, self._hidden_channels, self._num_mixed_ops, self._candidate_op_profiles)]
            self._skip_convs += [nn.Conv2d(self._hidden_channels, self._skip_channels, kernel_size=(1, 1))]

        self._start_conv = nn.Conv2d(self._in_channels, self._hidden_channels, kernel_size=(1, 1))
        self._end_conv1 = nn.Conv2d(self._skip_channels, self._end_channels, kernel_size=(1, 1))
        self._end_conv2 = nn.Conv2d(self._end_channels, self._out_channels * self._out_length, kernel_size=(1, 1))

        self.set_mode(Mode.NONE)

    def forward(self, x, node_fts, adj_mats, mode):
        adj_mats = torch.from_numpy(adj_mats).to(device=x.device, dtype=torch.float)
        node_fts = torch.from_numpy(node_fts).to(device=x.device, dtype=torch.float)
        self.set_mode(mode)

        # forward process
        node_fts, adj_mats = self._gmk_learner(node_fts, adj_mats)

        skip = None
        x = self._start_conv(x)
        for i, l in enumerate(self._layers):
            x = l(x, node_fts, adj_mats) if isinstance(l, STCell) else l(x)
            try:
                skip = skip[:, :, :, -x.size(3):]
            except:
                skip = 0
            skip = skip + self._skip_convs[i](x)

        x = skip.mean(dim=3, keepdim=True)
        x = torch.relu(x)
        x = self._end_conv1(x)
        x = torch.relu(x)
        x = self._end_conv2(x)
        x = x.view(x.size(0), self._out_channels, self._out_length, x.size(2))
        x = x.transpose(2, 3).contiguous()  # [b, c, n, t]

        self.set_mode(Mode.NONE)
        return x

    def set_mode(self, mode):
        self._mode = mode
        for l in self._layers:
            if isinstance(l, STCell):
                l.set_mode(mode)

    def weight_parameters(self):
        for m in [self._gmk_learner, self._start_conv, self._end_conv1, self._end_conv2]:
            for p in m.parameters():
                yield p
        for m in self._skip_convs:
            for p in m.parameters():
                yield p
        for m in self._layers:
            if isinstance(m, STCell):
                for p in m.weight_parameters():
                    yield p
            else:
                for p in m.parameters():
                    yield p

    def arch_parameters(self):
        for m in self._layers:
            if isinstance(m, STCell):
                for p in m.arch_parameters():
                    yield p

    def num_weight_parameters(self):
        from utils.helper import num_parameters
        current_mode = self._mode
        self.set_mode(Mode.ONE_PATH_FIXED)
        count = 0
        for m in [self._gmk_learner, self._start_conv, self._end_conv1, self._end_conv2]:
            count += num_parameters(m)
        for m in self._skip_convs:
            count += num_parameters(m)
        for m in self._layers:
            if isinstance(m, STCell):
                count += m.num_weight_parameters()
            else:
                count += num_parameters(m)

        self.set_mode(current_mode)
        return count

    def __repr__(self):
        out_str = []
        for l in self._layers:
            out_str += [str(l)]

        from utils.helper import add_indent
        out_str = 'NAS {\n%s\n}\n' % add_indent('\n'.join(out_str), 4)
        return out_str
