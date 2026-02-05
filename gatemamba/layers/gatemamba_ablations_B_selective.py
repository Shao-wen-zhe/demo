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

def edge_index_to_dense_adj(edge_index, num_nodes=None):
    """
    将 edge_index 转换为稠密邻接矩阵
    
    参数:
        edge_index: 边索引 [2, num_edges]
        num_nodes: 节点数量 (可选)
    
    返回:
        adj_matrix: 稠密邻接矩阵 [num_nodes, num_nodes]
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # 创建全零矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    
    # 将边位置设为1
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    
    return adj_matrix


def gcn(x, edge_index):
    """
    实现归一化公式: X' = D_out^(-1/2) A D_in^(-1/2) X 
    
    参数:
        x: 节点特征矩阵 [num_nodes, in_channels]
        edge_index: 边索引 [2, num_edges]
        
    返回:
        更新后的节点特征 [num_nodes, out_channels]
    """
    
    # 1. 计算源节点度(出度)和目标节点度(入度)
    row, col = edge_index
    num_nodes = x.size(0)
    
    # adj_matrix = edge_index_to_dense_adj(edge_index)
    # 计算出度和入度（添加自环）
    deg_out = degree(row, num_nodes, dtype=x.dtype)  # 出度 + 自环
    deg_in = degree(col, num_nodes, dtype=x.dtype)   # 入度 + 自环
    
    # 2. 计算 D_out^(-1/2) 和 D_in^(-1/2)
    deg_out_inv_sqrt = deg_out.pow(-0.5)
    deg_out_inv_sqrt[deg_out_inv_sqrt == float('inf')] = 0  # 处理孤立节点
    
    deg_in_inv_sqrt = deg_in.pow(-0.5)
    deg_in_inv_sqrt[deg_in_inv_sqrt == float('inf')] = 0    # 处理孤立节点
    
    # 3. 双向归一化: D_out^(-1/2) A D_in^(-1/2)
    norm = deg_out_inv_sqrt[row] * deg_in_inv_sqrt[col]  # [num_edges]
    
    # 4. 消息传播与聚合
    # (1) 传播: X_j * norm
    out = x[col] * norm.view(-1, 1)
    # (2) 聚合: sum_{j∈N(i)} X_j * norm
    out = scatter_add(out, row, dim=0, dim_size=num_nodes)
    
    return out

class GateMambaGCN_ablations_B_selective(pyg_nn.conv.MessagePassing):
    def __init__(
        self, 
        d_model,
        conv_layes, 
        d_state, 
        pool,
        num_nodes,
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
        self.num_nodes = num_nodes
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    # # 双路投影：in_proj 将输入映射到 2*d_inner 维度（含门控信号）对应conv之前和 silu之间
        self.Wb = nn.Linear(self.d_model, self.d_state, bias=bias, **factory_kwargs)
        # self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)     # selective projection used to make dt, B and C input dependant
        # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner * 2, bias=True, **factory_kwargs)        # # time step projection (discretization)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        # dt_init_std = self.dt_rank**-0.5 * dt_scale
        # if dt_init == "constant":
        #     nn.init.constant_(self.dt_proj.weight, dt_init_std)
        # elif dt_init == "random":
        #     nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # else:
        #     raise NotImplementedError

        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # dt = torch.exp(
        #     torch.rand(self.d_inner * 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #     + math.log(dt_min)
        # ).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # with torch.no_grad():
        #     self.dt_proj.bias.copy_(inv_dt)
        # # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True
        
        # 静态参数初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n ->  d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # 2. B 矩阵 (输入投影) - 静态
        # self.B = nn.Parameter(torch.randn(self.num_nodes, self.d_state, **factory_kwargs))
        
        # 3. C 矩阵 (输出投影) - 静态  
        self.C = nn.Parameter(torch.randn(self.num_nodes, self.d_state, **factory_kwargs))
        
        # 4. Δ 参数 (时间步长) - 静态
        self.delta = nn.Parameter(torch.randn(self.num_nodes, self.d_inner * 2,**factory_kwargs) * dt_scale)   
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(1, self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        
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
        token = True
        if token:
            out = x[col] * A_norm.view(-1, 1)
            x = scatter_add(out, row, dim=0, dim_size=N)
            # x = self.conv_fn(x, edge_index, self.conv_layes)
               
        h  = F.silu(x)
        # delta_rank = self.dt_proj.weight.shape[1]
        
        A = -torch.exp(self.A_log.float())  # A is a matrix of shape (d_inner, d_state)
        
        self.A_inv = 1 / (A)

        D = self.D.float()

        # x_dbl = self.x_proj(h)

        #delta = self.dt_proj(x_dbl[:, :delta_rank])
        delta = self.delta.float()
        #delta = self.dt_proj.weight @ x_dbl[:, :delta_rank].t()

        #B = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state]
        B = self.Wb(h)

        #C = x_dbl[:, -self.d_state:] 
        C = self.C.float()
        
        # delta = self._pre_delta(delta)
        delta = F.softplus(delta)
        
        delta_u, delta_v = delta.chunk(2, dim=-1)

        A_u = torch.einsum('ld,dn->ldn', delta_u, A)
        A_v = torch.einsum('ld,dn->ldn', delta_v, A)
        
        delta_uv = None
        # D = None
        # 计算 Bbar_hi 和 Bbar_hj
        if delta_uv is not None:
            Bbar_hi = torch.einsum('ln,ld->ldn', B, (delta_v * h))
            Bbar_hj = torch.einsum('ln,ld->ldn', B, (delta_u * h))
        else:
            Bbar_hi = torch.einsum('ln,ld->ldn', B,  h)
            
            # Bbar_hj = torch.einsum('ln,ld->ldn', B,  h)
        
        
        # 计算 h_new
        # new_h = Bbar_hi + sum_sigma_hj
        # new_h = self.propagate(edge_index, x=self.A_delta_u,flow='source_to_target')
        new_h = self.propagate(edge_index, Bx=Bbar_hi, Dx=A_v, Ex=A_u, Ax=Bbar_hi, flow='source_to_target')
        # new_h = self.propagate(edge_index, A_u=A_u, A_v=A_v, Bbar_hj=Bbar_hj, Bbar_hi=Bbar_hi)
        
        Ch = torch.einsum('ln,ldn->ld', C, new_h)
        # Ce = torch.einsum('ln,ldn->ld', C, new_e)
        # D = None
        y = Ch if D is None else (Ch + x * D)
        
        if z is not None:
            out = y * F.silu(z)
            
        out = self.out_proj(out)
        
        # batch.edge_attr = new_e
        e = None
        return out, e
    def conv_fn(self, x, edge_index, conv_layes):
        for _ in range(conv_layes):
            # GCN = GCNLayer()
            x = gcn(x, edge_index).to(x.device) 

        return x              
    def _pre_delta(self, delta, delta_softplus=True):

        if self.dt_proj.bias.float() is not None:
            delta = delta + self.dt_proj.bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        return delta.t()   

    def message(self, Dx_i, Ex_j):
        # self.delta_ij = Dx_i + Ex_j
        Abar_ij = torch.exp(Dx_i + Ex_j)
        # Abar_ij = torch.exp(Ex_j)
        return Abar_ij

    def aggregate(self, Abar_ij, edge_index, Bx_j, Ax_i, Bx):
        Bbar_ij = self.A_inv.unsqueeze(0) * (Abar_ij-1)
        row, col = edge_index[0], edge_index[1]
        dim_size = Bx.shape[0]
        sum_sigma_x = Abar_ij * Bx_j + Bbar_ij * Ax_i  
        # sum_sigma_x = Abar_ij * Bx_j + Ax_i              # 消融实验   去除输入门
        # sum_sigma_x = Bx_j + self.delta_ij * Ax_i            # 消融实验   去除遗忘门
        # sum_sigma_x = Bx_j + Ax_i  
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

            norm_coeff = self.A_norm.unsqueeze(1).unsqueeze(2)
            sum_sigma_x = sum_sigma_x * norm_coeff
            numerator_eta_xj = scatter(sum_sigma_x, col, 0, None, dim_size,reduce='add')

        out = numerator_eta_xj
        return out
    
    def update(self, aggr_out, Ax):
        x = aggr_out
        # x = Ax + aggr_out
        return x
 

 