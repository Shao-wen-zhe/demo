import numpy as np

from einops import rearrange, repeat
import math
# import dgl
# from dgl import ops
# import dgl.function as fn
# from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter
import einops
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.nn import GCN
from torch_geometric.utils import softmax as pyg_softmax
import torch_geometric.nn as pygnn



class gatemamba_ablations_A_bar(pyg_nn.conv.MessagePassing):
    def __init__(
        self, 
        d_model,
        conv_layes, 
        d_state, 
        pool,
        expand = 1, 
        dt_rank = "auto", 
        dt_min=0.001,
        dt_max=0.1,
        dt_init="constant", 
        dt_scale=1,
        dt_init_floor=1e-4, 
        dropout=0.1, 
        device = None, 
        dtype = None, 
        bias=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(node_dim=0,aggr='add')
        
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.dropout = dropout
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 4) if dt_rank == "auto" else dt_rank
        self.conv_layes = conv_layes
        self.pool = pool

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    # # 双路投影：in_proj 将输入映射到 2*d_inner 维度（含门控信号）对应conv之前和 silu之间
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)     # selective projection used to make dt, B and C input dependant
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
    def forward(self, x, edge_index, A_norm, edge_attr):
        row, col  = edge_index
        # e = edge_attr
        N, dim = x.shape
        self.A_norm = A_norm
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "l d -> d l"),
            "d l -> l d",
            l=N,
        )

        x, z = xz.chunk(2, dim=-1)       
        
        x = scatter_add(x[col]*A_norm.view(-1, 1), row, dim=0, dim_size=N)
               
        h  = F.silu(x)
        
        x_dbl = self.x_proj(h)
        
        B = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state]

        C = x_dbl[:, -self.d_state:] 

        Bbar_hi = torch.einsum('ln,ld->ldn', B,  h)
        
        # 计算 h_new
        new_h = self.propagate(edge_index, Bx=Bbar_hi, Ax=Bbar_hi, N=N,flow='source_to_target')
        
        Ch = torch.einsum('ln,ldn->ld', C, new_h)

        y = (Ch + x)

        if z is not None:
            out = y * F.silu(z)
            
        out = self.out_proj(out)

        e = None
        return out, e               

    def message(self,Bx):
        # atten = F.leaky_relu((h_src + h_dst), negative_slope=0.2)  # [E, out_channels]
        # Abar_ij = pyg_softmax(atten, edge_index[1],num_nodes=N)
        Abar_ij = 1
        return Abar_ij

    def aggregate(self, Abar_ij, edge_index, Bx_j, Bx):

        row, col = edge_index[0], edge_index[1]
        dim_size = Bx.shape[0]

        sum_sigma_x = Bx_j             # 消融 去除门控

        if self.pool in ['sum', 'add']:
            numerator_eta_xj = scatter(sum_sigma_x, col, 0, None, dim_size,
                                   reduce='sum')
        elif self.pool in ['max']:
            numerator_eta_xj = scatter(sum_sigma_x, col, 0, None, dim_size,
                                   reduce='max')         
        elif self.pool in ['mean']:
            numerator_eta_xj = scatter(sum_sigma_x, col, 0, None, dim_size,
                                   reduce='mean')
        elif self.pool in ['gcn']:

            numerator_eta_xj = scatter(sum_sigma_x, col, 0, None, dim_size,reduce='add')

        out = numerator_eta_xj
        return out
    
    def update(self, aggr_out):
        x = aggr_out

        return x
 

