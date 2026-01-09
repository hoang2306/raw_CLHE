from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# module from torch_geometric
# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv
from torch_geometric.nn.conv import (
    GATv2Conv, AntiSymmetricConv,
    GCNConv, # GCN traditional
    LGConv # LightGCN
)


class AsymMatrix(MessagePassing):
    _alpha: OptTensor

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
        extra_layer=False,
        **kwargs,
    ):
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
        self.extra_layer = extra_layer

        if self.extra_layer:
            self.lin_l = Linear(
                in_channels, 
                heads * out_channels, 
                bias=bias, 
                weight_initializer='glorot'
            )
            self.lin_r = Linear(
                in_channels, 
                heads * out_channels, 
                bias=bias, 
                weight_initializer='glorot'
            )

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.extra_layer:
            self.lin_l.reset_parameters()
            self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self, 
        x: Union[Tensor, PairTensor], 
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None
    ):
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            if self.extra_layer:
                x_l = self.lin_l(x).view(-1, H, C)
                x_r = self.lin_r(x).view(-1, H, C)
            else:
                x = x.expand(self.heads, x.shape[0], x.shape[1]).transpose(0, 1)
                x_l = x_r = x.view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        out = self.propagate(
            edge_index, 
            x=(x_l, x_r), 
            edge_attr=edge_attr, 
            size=None
        )

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(
        self, 
        x_j: Tensor, 
        x_i: Tensor, 
        edge_attr: OptTensor, 
        index: Tensor, 
        ptr: OptTensor,
        size_i: Optional[int]
    ) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    
class Amatrix(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        n_layer=3, 
        dropout=0.0, 
        heads=2, 
        concat=False, 
        self_loop=True, 
        extra_layer=False,
        type_gnn='anti_symmetric'
    ):
        super(Amatrix, self).__init__()
        self.num_layer = n_layer
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.self_loop = self_loop
        self.extra_layer = extra_layer
        # self.convs = nn.ModuleList([
        #         AsymMatrix(
        #             in_channels=self.in_dim,
        #             out_channels=self.out_dim, 
        #             dropout=self.dropout,
        #             heads=self.heads, 
        #             concat=self.concat, 
        #             add_self_loops=self.self_loop,
        #         ) 
        #         for _ in range(self.num_layer)
        # ])

        # self.convs = nn.ModuleList([
        #         GATv2Conv(
        #             in_channels=self.in_dim,
        #             out_channels=self.out_dim, 
        #             dropout=self.dropout,
        #             heads=self.heads, 
        #             concat=self.concat, 
        #             add_self_loops=self.self_loop,
        #         ) 
        #         for _ in range(self.num_layer)
        # ])

        core_gnn_model = None 
        if type_gnn == 'anti_symmetric':
            core_gnn_model = AntiSymmetricConv(
                in_channels=self.in_dim,
                # phi=GATv2Conv(
                #     in_channels=self.in_dim, 
                #     out_channels=self.out_dim,
                #     heads=self.heads,
                #     concat=self.concat,
                #     add_self_loops=self.self_loop,
                #     dropout=self.dropout
                # ),
                phi=None,
                num_iters=2,
                act="relu",
            )
        
        if type_gnn == 'gcn':
            core_gnn_model = GCNConv(
                in_channels=self.in_dim,
                out_channels=self.in_dim,
                add_self_loops=self.self_loop
            )

        if type_gnn == 'light_gcn':
            core_gnn_model = LGConv(
                normalize=True 
            )

        if type_gnn == 'gat':
            core_gnn_model = AsymMatrix(
                    in_channels=self.in_dim,
                    out_channels=self.out_dim, 
                    dropout=self.dropout,
                    heads=self.heads, 
                    concat=self.concat, 
                    add_self_loops=self.self_loop,
                    extra_layer=self.extra_layer
                ) 


        self.convs = nn.ModuleList([
            core_gnn_model for _ in range(self.num_layer)
        ])

        

    def forward(self, x, edge_index, return_attention_weights=True):
        feats = [x]
        # attns = []

        for conv in self.convs:
            # x, attn = conv(
            #     x, edge_index, 
            #     return_attention_weights=return_attention_weights
            # )
            # feats.append(x)
            # attns.append(attn)

            x  = conv(
                x, edge_index
            )
            feats.append(x)
            # x = x_conv + x 
            # attns.append(attn)

        feat = torch.stack(feats, dim=1)
        x = torch.mean(feat, dim=1)
        return x, x