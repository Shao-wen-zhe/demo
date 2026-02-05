import numpy as np

from einops import rearrange, repeat
import math
# import dgl
# from dgl import ops
# import dgl.function as fn
from dgl.nn.functional import edge_softmax
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

class GCNLayer(pyg_nn.conv.MessagePassing):
    def __init__(self):
        super(GCNLayer, self).__init__(aggr='add')  # 使用加法聚合
    
    def forward(self, x, edge_index):
        # 计算度矩阵
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        
        # 计算归一化系数 1/sqrt(deg(i)*deg(j))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        # 归一化后的消息传递
        return norm.view(-1, 1) * x_j
def gcn(x, edge_index):
    """
    实现公式: X' = D^(-1/2) A D^(-1/2) X Θ
    
    参数:
        x: 节点特征矩阵 [num_nodes, in_channels]
        edge_index: 边索引 [2, num_edges]
        
    返回:
        更新后的节点特征 [num_nodes, out_channels]
    """
    
    # 1. 计算度矩阵 D̂ (对角线为节点度)
    row, col = edge_index
    deg = degree(row, x.size(0), dtype=x.dtype)  # [num_nodes]
    
    # 2. 计算 D̂^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理孤立节点
    
    # 3. 对称归一化: D̂^(-1/2) Â D̂^(-1/2)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [num_edges]
    
    # 4. 消息传播与聚合 (稀疏矩阵乘法)
    # (1) 传播: X_j * norm
    out = x[col] * norm.view(-1, 1)
    # (2) 聚合: sum_{j∈N(i)} X_j * norm
    out = scatter_add(out, row, dim=0, dim_size=x.size(0))
    
    return out

class GateMambaGCN_layer1(pyg_nn.conv.MessagePassing):
    def __init__(
        self, 
        d_model,
        conv_layes, 
        d_state, 
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
        self.hidden_layers = 2
        self.dropout = dropout
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 4) if dt_rank == "auto" else dt_rank
        self.conv_layes = conv_layes
        # self.n_heads = n_heads
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    # # 双路投影：in_proj 将输入映射到 2*d_inner 维度（含门控信号）对应conv之前和 silu之间
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)     # selective projection used to make dt, B and C input dependant
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner * 2, bias=True, **factory_kwargs)        # # time step projection (discretization)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        # self.C = nn.Linear(self.d_model, self.d_inner, bias=True)
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner * 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n ->  d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state).uniform_(1,d_state))
        # self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(1, self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, x, edge_index, edge_attr):
        x, edge_index = x, edge_index
        # e = edge_attr
        N, dim = x.shape
            
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "l d -> d l"),
            "d l -> l d",
            l=N,
        )

        x, z = xz.chunk(2, dim=-1)           
        token = True
        if token:
            x = self.conv_fn(x, edge_index, self.conv_layes)
               
        h  = F.silu(x)
        delta_rank = self.dt_proj.weight.shape[1]

        
        A = -torch.exp(self.A_log.float())  # A is a matrix of shape (d_inner, d_state)

        D = self.D.float()

        x_dbl = self.x_proj(h)

        delta = self.dt_proj(x_dbl[:, :delta_rank])

        # Ce = F.softplus(self.C(e))

        B_i = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state]
        B_j = x_dbl[:, self.dt_rank + self.d_state:self.dt_rank + 2 * self.d_state]

        C = x_dbl[:, -self.d_state:] 
        
        delta = self._pre_delta(delta)
        
        delta_u, delta_v = delta.chunk(2, dim=-1)
        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        # pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        self.A_u = torch.einsum('ld,dn->ldn', delta_u, A)
        self.A_v = torch.einsum('ld,dn->ldn', delta_v, A)

        # 应用到边
        edge_index_u = edge_index[0]
        edge_index_v = edge_index[1]
        self.A_uv = self.A_u[edge_index_u] + self.A_v[edge_index_v]

        # 计算 e
        e = self.A_uv
        delta = None
        # D = None
        # 计算 Bbar_hi 和 Bbar_hj
        # h = 1
        if delta is not None:
            Bbar_hi = torch.einsum('ln,ld->ldn', B_i, (delta_v * h))
            Bbar_hj = torch.einsum('ln,ld->ldn', B_j, (delta_u * h))
        else:
            # delta_v = torch.ones_like(delta_v)
            # delta_u = torch.ones_like(delta_u)
            # Bbar_hi = torch.einsum('ln,ld->ldn', B_i, h)
            # Bbar_hj = torch.einsum('ln,ld->ldn', B_j, h)
            Bbar_hi = torch.einsum('ln,ld->ldn', B_i, h)
            Bbar_hj = torch.einsum('ln,ld->ldn', B_j, h)
        # 计算 h_new
        # new_h = Bbar_hi + sum_sigma_hj
        # new_h = self.propagate(edge_index, x=self.A_delta_u,flow='source_to_target')
        new_h = self.propagate(edge_index, Bx=Bbar_hj, Dx=self.A_v, Ex=self.A_u, Ax=Bbar_hi, flow='source_to_target')
        # new_h = self.propagate(edge_index, A_u=A_u, A_v=A_v, Bbar_hj=Bbar_hj, Bbar_hi=Bbar_hi)
        
        Ch = torch.einsum('ln,ldn->ld', C, new_h)
        # Ce = torch.einsum('ln,ldn->ld', C, new_e)
        y = Ch if D is None else (Ch + h * D)
        
        if z is not None:
            out = y * F.silu(z)
            
        out = self.out_proj(out)
        
        # batch.edge_attr = new_e

        return out, e
    def conv_fn(self, x, edge_index, conv_layes):
        for _ in range(conv_layes):
            # GCN = GCNLayer()
            x = gcn(x, edge_index).to(x.device) 

        return x              
    def _pre_delta(self, delta, delta_softplus=True):

        if self.dt_proj.bias.float() is not None:
            delta = delta + self.dt_proj.bias[..., None].float().reshape(-1, delta.shape[-1])
        if delta_softplus:
            delta = F.softplus(delta)
        return delta   

    
    def message(self, Dx_i, Ex_j):
        e_ij = Dx_i + Ex_j
        Abar_ij = torch.exp(e_ij)
        self.Abar_ij = Abar_ij
        return Abar_ij

    def aggregate(self, Abar_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        sum_sigma_x = Abar_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')
        out = numerator_eta_xj
        return out
    
    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        return x
 