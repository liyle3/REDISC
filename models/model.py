from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import (
  to_dense_adj,
  add_self_loops,
  is_torch_sparse_tensor,
  remove_self_loops,
  softmax,
  spmm
)
from torch_geometric.typing import (
  Adj,
  OptPairTensor,
  OptTensor,
  SparseTensor,
  Size,
  torch_sparse,
)
from torch_geometric.utils.sparse import set_sparse_value
from models.layers import DenseGCNConv, MLP
import math

def SinusoidalPosEmb(x, num_steps, dim, rescale=4):
    x = x / num_steps * num_steps * rescale
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


def weight_init_xavier_uniform(submodule):
    torch.nn.init.xavier_normal_(submodule.weight)


def _one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)


class GATsepConv(MessagePassing):
    """
    Based on https://github.com/yandex-research/heterophilous-graphs/blob/a431395582e929d88271309716bea4fe24ce6318/modules.py#L120
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        #self.lin_phi = Linear(2 * heads * out_channels, heads * out_channels, False,
        #                      weight_initializer='glorot')
        self.lin_phi = MLP(num_layers=2, input_dim=2 * heads * out_channels, hidden_dim=heads * out_channels, output_dim=heads * out_channels,
                           use_bn=False, activate_func=F.gelu, apply_dr = True)

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        #self.lin_phi.reset_parameters()
        for module in self.lin_phi.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        # x_origin = x
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def update(self, inputs, x):
        inputs = inputs.view(-1, self.heads * self.out_channels)
        central_node_emb = x[0].view(-1, self.heads * self.out_channels)
        emb = torch.cat([inputs, central_node_emb], dim=-1)
        return self.lin_phi(emb).view(-1, self.heads, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Simple_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid, nhead=8, alpha = 0):
        super(Simple_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(nn.Linear(self.nfeat, self.nhid))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat, self.nhid, nhead, concat=True))
                elif model == 'GATsepConv':
                    self.layers.append(GATsepConv(self.nfeat, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(GCN2Conv(channels = self.nhid, alpha = 0.1, layer = i, theta = 0.5))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid, out_channels =self.nhid, normalize = True))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid, self.nhid, nhead, concat=True))
                elif self.model == 'GATsepConv':
                    self.layers.append(GATsepConv(self.nhead*self.nhid, self.nhid, nhead, concat=True))

        self.activation = torch.nn.ReLU()

        if self.model == 'GATConv' or self.model == 'GATsepConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()


        self.fdim = self.nhid
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel,
                         use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)

        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=0.5)
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, adj):

        save_inital = None
        for i in range(self.depth):
            if self.model == 'GCN2Conv':
                if i == 0:
                    x_before_act = self.layers[i](x)
                else:
                    x_before_act = self.layers[i](x, save_inital, edge_index = adj)
            else:
                x_before_act = self.layers[i](x, edge_index = adj)

            x = self.activation(x_before_act)
            x = self.dr(x)

            if i == 0:
                save_inital = x

        pred_y = self.final(x)
        return F.log_softmax(pred_y, dim=1)


class Denoising_Model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, nhid, nhead=8):
        super(Denoising_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
                elif model == 'GATsepConv':
                    self.layers.append(GATsepConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(GCN2Conv(channels = self.nhid, alpha = 0.1, layer = i, theta = 0.5))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))
                elif self.model == 'GATsepConv':
                    self.layers.append(GATsepConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))

        self.activation = torch.nn.ReLU()
        if self.model == 'GATConv' or self.model == 'GATsepConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()

        self.fdim = self.nhid + self.nlabel
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.nhid)
        )

        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel,
                        use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)

        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=0.5)  # 0.5 for continuous version
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, q_Y_sample, adj, t, num_steps, train=False):
        t = SinusoidalPosEmb(t, num_steps, 128)
        t = self.time_mlp(t)
        x = torch.cat([x, q_Y_sample], dim = -1)

        for i in range(self.depth):

            if self.model == 'GCN2Conv':
                if i == 0:
                    x_before_act = self.layers[i](x)
                else:
                    x_before_act = self.layers[i](x, save_inital, edge_index = adj)
            else:
                x_before_act = self.layers[i](x, adj) +  t

            x = self.activation(x_before_act)
            if train:
                x = self.dr(x)

            if self.model != 'GCN2Conv':
                x = torch.cat([x, q_Y_sample], dim = -1)
            if i == 0:
                save_inital = x

        if self.model == 'GCN2Conv':
             x = torch.cat([x, q_Y_sample], dim = -1)
        pred_y = self.final(x)
        return pred_y


class OurGCNConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        nlabel: int = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.nlabel = nlabel

        self.lin0 = Linear(in_channels - nlabel, out_channels, bias=False,
                           weight_initializer='glorot')
        self.lin1 = Linear(nlabel, out_channels, bias=False,
                           weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, t: Tensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        xfeat = self.lin0(x[:,:self.in_channels - self.nlabel])
        lbx = self.lin1(x[:,-self.nlabel:])
        x = xfeat + t * lbx

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class OurGATConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        nlabel: int = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.nlabel = nlabel

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            #self.lin_src = Linear(in_channels, heads * out_channels,
            #                      bias=False, weight_initializer='glorot')
            #self.lin_dst = self.lin_src
            self.lin_src0 = Linear(in_channels - nlabel, heads * out_channels,
                                   bias=False, weight_initializer='glorot')
            self.lin_src1 = Linear(nlabel, heads * out_channels,
                                   bias=False, weight_initializer='glorot')
            self.lin_dst0 = self.lin_src0
            self.lin_dst1 = self.lin_src1
        else:
            # here this branch would not be used
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src0.reset_parameters()
        self.lin_src1.reset_parameters()
        self.lin_dst0.reset_parameters()
        self.lin_dst1.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None, t: Tensor = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            #x_src = x_dst = self.lin_src(x).view(-1, H, C)

            x_feat = self.lin_src0(x[:,:self.in_channels - self.nlabel])
            lbx = self.lin_src1(x[:,-self.nlabel:])
            x_dst = (x_feat + t * lbx).view(-1, H, C)
            x_src = x_dst
        else:  # Tuple of source and target node features:
            # here this branch would not be used
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class OurGATsepConv(MessagePassing):
    """Based on https://github.com/yandex-research/heterophilous-graphs/blob/a431395582e929d88271309716bea4fe24ce6318/modules.py#L120
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        nlabel: int = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.nlabel = nlabel

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            #self.lin_src = Linear(in_channels, heads * out_channels,
            #                      bias=False, weight_initializer='glorot')
            #self.lin_dst = self.lin_src
            self.lin_src0 = Linear(in_channels - nlabel, heads * out_channels,
                                   bias=False, weight_initializer='glorot')
            self.lin_src1 = Linear(nlabel, heads * out_channels,
                                   bias=False, weight_initializer='glorot')
            self.lin_dst0 = self.lin_src0
            self.lin_dst1 = self.lin_src1
        else:
            # here this branch would not be used
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # for transforming the concatenation of central node's embedding and the aggregated embeddings
        # self.lin_phi = Linear(2 * heads * out_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.lin_phi = MLP(num_layers=2, input_dim=2 * heads * out_channels, hidden_dim=heads * out_channels, output_dim=heads * out_channels,
                           use_bn=False, activate_func=F.gelu, apply_dr = True)

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src0.reset_parameters()
        self.lin_src1.reset_parameters()
        self.lin_dst0.reset_parameters()
        self.lin_dst1.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        # self.lin_phi.reset_parameters()
        for module in self.lin_phi.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None, t: Tensor = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            #x_src = x_dst = self.lin_src(x).view(-1, H, C)

            x_feat = self.lin_src0(x[:,:self.in_channels - self.nlabel])
            lbx = self.lin_src1(x[:,-self.nlabel:])
            x_dst = (x_feat + t * lbx).view(-1, H, C)
            x_src = x_dst
        else:  # Tuple of source and target node features:
            # here this branch would not be used
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def update(self, inputs, x):
        inputs = inputs.view(-1, self.heads * self.out_channels)
        central_node_emb = x[0].view(-1, self.heads * self.out_channels)
        emb = torch.cat([inputs, central_node_emb], dim=-1)
        return self.lin_phi(emb).view(-1, self.heads, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Simple_Res_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel, num_layers, num_linears, nhid, nhead=8, alpha=0.1, p=0.2):
        super(Simple_Res_Model, self).__init__()
        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        self.input_linear = nn.Linear(self.nfeat, self.nhid * self.nhead)
        self.act = nn.GELU()

        assert model == 'GATsepConv'

        for i in range(self.depth):
            self.norm_layers.append(nn.LayerNorm(self.nhid * self.nhead))
            # if i == 0:
            #     self.layers.append(GATsepConv(self.nhid * self.nhead, self.nhid, nhead, concat=True))
            # else:
            #     self.layers.append(GATsepConv(self.nhead * self.nhid, self.nhid, nhead, concat=True))

            self.layers.append(GATsepConv(self.nhead * self.nhid, self.nhid, nhead, concat=True))


        self.nhid = self.nhid * self.nhead


        self.fdim = self.nhid
        self.output_normalization = nn.LayerNorm(self.fdim)
        # self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel,
        #                use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)

        self.final = nn.Linear(self.fdim, self.nlabel)  # for roman-empire

        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=p)
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, adj):

        x = self.input_linear(x)
        x = self.dr(x)
        x = self.act(x)

        for i in range(self.depth):
            cur_x = x

            x = self.norm_layers[i](x)

            x_before_act = self.layers[i](x, edge_index = adj)

            #x = self.activation(x_before_act)
            x = self.dr(x_before_act)

            x = x + cur_x

        # x = self.output_normalization(x)  # used in roman-empire
        pred_y = self.final(x)

        return F.log_softmax(pred_y, dim=1)



class OurDenoising_Model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, nhid, nhead=8):
        super(OurDenoising_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(OurGCNConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, improved = self.improving, nlabel = self.nlabel))
                elif self.model == 'GCN2Conv':
                    self.layers.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid))
                elif model == 'GATConv':
                    self.layers.append(OurGATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True, nlabel = self.nlabel))
                elif model == 'GATsepConv':
                    self.layers.append(OurGATsepConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True, nlabel = self.nlabel))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(OurGCNConv(in_channels = self.nhid + self.nlabel, out_channels =self.nhid, improved = self.improving, nlabel = self.nlabel))
                elif self.model == 'GCN2Conv':
                    self.layers.append(GCN2Conv(channels = self.nhid, alpha = 0.1, layer = i, theta = 0.5))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid))
                elif self.model == 'GATConv':
                    self.layers.append(OurGATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True, nlabel = self.nlabel))
                elif self.model == 'GATsepConv':
                    self.layers.append(OurGATsepConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True, nlabel = self.nlabel))

        self.activation = torch.nn.ReLU()
        if self.model == 'GATConv' or self.model == 'GATsepConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()

        self.fdim = self.nhid + self.nlabel
        self.time_mlp = nn.Sequential(
            nn.Linear(128 + self.nlabel, 128),
            nn.ELU(),
            nn.Linear(128, 1) if self.model != 'SAGEConv' else nn.Linear(128, self.nhid),
            nn.Sigmoid()
        )

        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel,
                         use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)

        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=0.5)  # 0.5 for continuous version
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, q_Y_sample, adj, t, num_steps, train=False):
        t = SinusoidalPosEmb(t, num_steps, 128)
        t = self.time_mlp(torch.cat([t.repeat(q_Y_sample.shape[0], 1), q_Y_sample], dim = -1))
        x = torch.cat([x, q_Y_sample], dim = -1)

        for i in range(self.depth):

            if self.model == 'GCN2Conv':
                if i == 0:
                    x_before_act = self.layers[i](x)
                else:
                    x_before_act = self.layers[i](x, save_inital, edge_index = adj)
            elif self.model == 'SAGEConv':
                x_before_act = self.layers[i](x, adj) + t

            else:
                x_before_act = self.layers[i](x, adj, t=t)

            x = self.activation(x_before_act)
            if train:
                x = self.dr(x)

            if self.model != 'GCN2Conv':
                x = torch.cat([x, q_Y_sample], dim = -1)
            if i == 0:
                save_inital = x

        if self.model == 'GCN2Conv':
            x = torch.cat([x, q_Y_sample], dim = -1)
        pred_y = self.final(x)
        return pred_y


class Simple_Model_Large(torch.nn.Module):
    def __init__(self, nlabel, nfeat, num_layers, num_linears, nhid, dropout=0.5):
        super(Simple_Model_Large, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nlabel))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return F.log_softmax(x_all, dim=-1)


class Denoising_Model_Large(torch.nn.Module):
    def __init__(self, nlabel, nfeat, num_layers, num_linears, nhid, dropout=0.5):
        super(Denoising_Model_Large, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nlabel+nfeat, nhid))
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(nhid+nlabel, nhid))
        # self.convs.append(SAGEConv(nhid, nlabel))

        self.dropout = dropout
        self.activation = torch.nn.ReLU()

        self.fdim = nhid + nlabel
        self.time_mlp = nn.Sequential(
            # nn.Linear(128 + self.nlabel, 128),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, nhid),
            # nn.Sigmoid()
        )
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=nlabel,
                         use_bn=False, activate_func=F.elu, apply_dr = True)



    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, q_Y_sample, edge_index, t, num_steps, edge_weight=None):
        t = SinusoidalPosEmb(t, num_steps, 128)
        # t = self.time_mlp(torch.cat([t.repeat(q_Y_sample.shape[0], 1), q_Y_sample], dim = -1))
        t = self.time_mlp(t)
        x = torch.cat([x, q_Y_sample], dim = -1)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight) + t
            # x = F.relu(x)
            x = self.activation(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.cat([x, q_Y_sample], dim = -1)

        pred_y = self.final(x)
        return pred_y
        # x = self.convs[-1](x, edge_index, edge_weight)
        # return torch.log_softmax(x, dim=-1)



    # def inference(self, x_all,  q_Y_sample, subgraph_loader, t, num_steps, device):
    #     t = SinusoidalPosEmb(t, num_steps, 128)
    #     # t = self.time_mlp(torch.cat([t.repeat(q_Y_sample.shape[0], 1), q_Y_sample], dim = -1))
    #     t = self.time_mlp(t)
    #     x_all = torch.cat([x_all, q_Y_sample], dim = -1)

    #     for i, conv in enumerate(self.convs):
    #         xs = []
    #         # n_ids = []
    #         for batch_size, n_id, adj in subgraph_loader:
    #             edge_index, _, size = adj.to(device)
    #             x = x_all[n_id].to(device)
    #             x_target = x[:size[1]]
    #             x = conv((x, x_target), edge_index) + t

    #             x = self.activation(x)
    #             # print(x.shape)

    #             xs.append(x.cpu())

    #         x_all = torch.cat(xs, dim=0)
    #         x_all = torch.cat([x_all, q_Y_sample], dim = -1).cpu()

    #     y_pred = []
    #     batch_size = 1024
    #     num_batches = (x_all.size(0) + batch_size - 1) // batch_size

    #     for batch_id in range(num_batches):
    #         x_batch = x_all[batch_id * batch_size : (batch_id + 1) * batch_size].to(device)
    #         y_pred_batch = self.final(x_batch)
    #         y_pred.append(y_pred_batch.cpu())

    #     y_pred = torch.cat(y_pred, dim=0)

    #     return y_pred


    def inference(self, loader, t, num_steps, device):
        t = SinusoidalPosEmb(t, num_steps, 128)
        # t = self.time_mlp(torch.cat([t.repeat(q_Y_sample.shape[0], 1), q_Y_sample], dim = -1))
        t = self.time_mlp(t)

        y_pred = []

        for data in loader:
            data = data.to(device)
            x = torch.cat([data.x, data.y], dim = -1)

            for i in range(self.num_layers):
                x = self.convs[i](x, data.edge_index) + t
                x = self.activation(x)
                x = torch.cat([x, data.y], dim = -1)

            # x_target = x[:data.batch_size]
            # x = self.convs[-1](x, data.edge_index) + t
            # x = self.activation(x)[:data.batch_size]
            # x = torch.cat([x, data.y[:data.batch_size]], dim = -1)
            y_batch = self.final(x[:data.batch_size])
            y_pred.append(y_batch.cpu())
        y_pred = torch.cat(y_pred, dim=0).to(device)

        return y_pred


class LPA_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



class P_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CLGNN_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class G3NN_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
