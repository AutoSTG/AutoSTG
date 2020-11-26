import torch.nn as nn

from model.mixed_op import MixedOp
from model.mode import Mode


# from graphviz import Digraph


class Cell(nn.Module):
    def __init__(self, channels, num_mixed_ops, candidate_op_profiles):
        super(Cell, self).__init__()
        # create mixed operations
        self._channels = channels
        self._num_mixed_ops = num_mixed_ops
        self._mixed_ops = nn.ModuleList()
        for i in range(self._num_mixed_ops):
            self._mixed_ops += [MixedOp(self._channels, self._channels, candidate_op_profiles)]

        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        for op in self._mixed_ops:
            op.set_mode(mode)

    def arch_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].arch_parameters():
                yield p

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p

    def num_weight_parameters(self):
        count = 0
        for i in range(self._num_mixed_ops):
            count += self._mixed_ops[i].num_weight_parameters()
        return count

    def forward(self, x, node_fts, adj_mats):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

# def render_arch(self):
# 	raise NotImplementedError


class STCell(Cell):
    def __init__(self, channels, num_mixed_ops, candidate_op_profiles):
        super(STCell, self).__init__(channels, num_mixed_ops, candidate_op_profiles)

    def forward(self, x, node_fts, adj_mats):
        # calculate outputs
        node_idx = 0
        current_output = 0

        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            current_output += self._mixed_ops[i]([node_outputs[node_idx]], node_fts, adj_mats)
            if node_idx + 1 >= len(node_outputs):
                node_outputs += [current_output]
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1

        if node_idx != 0:
            node_outputs += [current_output]

        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def __repr__(self):
        edge_cnt = 0
        out_str = []
        for i in range(self._num_mixed_ops):
            out_str += ['mixed_op: %d\n%s' % (i, self._mixed_ops[i])]

        from utils.helper import add_indent
        out_str = 'STCell {\n%s\n}' % add_indent('\n'.join(out_str), 4)
        return out_str

# def render_arch(self, save_dir, save_name):
# 	g = Digraph(comment='architecture of %s' % save_name, format='png')
# 	g.node_attr['style'] = 'filled'

# 	node_idx = 0
# 	node_names = ['input']
# 	for i in range(self._num_mixed_ops):
# 		g.edge(node_names[node_idx], str(len(node_names)), label=self._mixed_ops[i].render_name())
# 		if node_idx + 1 >= len(node_names):
# 			node_names += [str(len(node_names))]
# 			node_idx = 0
# 		else:
# 			node_idx += 1
# 	if node_idx != 0:
# 		node_names += [str(len(node_names))]

# 	for i in node_names[:]:
# 		g.edge(i, 'output')
# 	g.render(os.path.join(save_dir, '%s' % save_name))
