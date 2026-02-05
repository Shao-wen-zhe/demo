import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from layers import gate_mamba_layer
from einops import rearrange
import torch_geometric.nn as pygnn


from gatemamba.layers.gatemamba_layer import GateMambaGCN_layer
from gatemamba.layers.gatemamba_ablations_ssm import gatemamba_ablations_ssm
from gatemamba.layers.gatemamba_ablations_A_bar import gatemamba_ablations_A_bar
from gatemamba.layers.gatemamba_ablations_A_bar_selective import gatemamba_ablations_A_bar_selective
from gatemamba.layers.gatemamba_top_k_layer import GateMambaGCN_top_k_layer
from gatemamba.layers.gatemamba_layer_sigmoid import GateMambaGCN_layer_sigmoid
from gatemamba.layers.gatemamba_ablations_None_selective import GateMambaGCN_ablations_None_selective
from gatemamba.layers.gatemamba_ablations_B_selective import GateMambaGCN_ablations_B_selective
from gatemamba.layers.gatemamba_ablations_C_selective import GateMambaGCN_ablations_C_selective
from gatemamba.layers.gatemamba_ablations_BC_selective import GateMambaGCN_ablations_BC_selective
from gatemamba.layers.gatemamba_ablations_DeltaC_selective import GateMambaGCN_ablations_DeltaC_selective
from gatemamba.layers.gatemamba_ablations_DeltaB_selective import GateMambaGCN_ablations_DeltaB_selective
from gatemamba.layers.gatemamba_ablations_Delta_selective import GateMambaGCN_ablations_Delta_selective
from gatemamba.layers.GateMambaGCN_ablations_z_selective import GateMambaGCN_ablations_z_selective

from gatemamba.layers.experiment import GateMambaGCN_
# from gatemamba.layers.gatemamba_layer1 import GateMambaGCN_layer1
# from gatemamba.layers.gatemamba_layer2 import GateMambaGCN_layer2
from gatemamba.layers.gategcn import GatedGCNLayer
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear as Linear_pyg


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.act = nn.GELU()
        if module is GateMambaGCN:
            self.module = module(dim, **kwargs)
        elif module is GatedGCNModule:
            self.module = module(dim, kwargs['dropout'])
        elif module is GCNModule:
            self.module = module(dim, kwargs['dropout'])
        elif module is GENConvModule:
            self.module = module(dim, kwargs['dropout'])
        elif module is GATModule:
            self.module = module(dim, kwargs['dropout'], kwargs['nb_heads'])
        elif module is GINEConv:
            self.module = module(dim, kwargs['dropout'])
        elif module is SAGEModule:
            self.module = module(dim, kwargs['dropout'])
        elif module is MixHopConv:
            self.module = module(dim, kwargs['dropout'])
        elif module is GATv2Conv:
            self.module = module(dim, kwargs['dropout'])
        
        
    def forward(self, x, edge_index, A_norm, edge_attr):
        x_res = self.normalization(x)
        if isinstance(self.module, GateMambaGCN):
            x_res, e = self.module(x_res, edge_index, A_norm, edge_attr)
            x = x + x_res
        elif isinstance(self.module, GatedGCNModule):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x + x_res
        elif isinstance(self.module, GCNModule):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x + x_res
        elif isinstance(self.module, GENConvModule):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        elif isinstance(self.module, GATModule):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        elif isinstance(self.module, GINEConv):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        elif isinstance(self.module, SAGEModule):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        elif isinstance(self.module, MixHopConv):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        elif isinstance(self.module, GATv2Conv):
            x_res, e = self.module(x_res, edge_index, edge_attr)
            x = x_res
        else:
            x_res = self.module(x_res, edge_index, edge_attr)

        return x, e
    
    
class GateMambaGCN(nn.Module):
    def __init__(self, inputfeature, dt_init, conv_layes, d_state, top_k, num_heads,hidden_dim_multiplier, dropout, pool, num_nodes, bias=False, layernorm=True):
        super(GateMambaGCN, self).__init__()
        self.dropout = dropout
        self.dt_init = dt_init
        self.pool = pool
        self.nhid = inputfeature
        self.conv_layes = conv_layes
        self.expand = hidden_dim_multiplier
        self.d_state = d_state
        self.num_nodes = num_nodes
        self.top_k = top_k
        
        self.gate_attentions = GateMambaGCN_layer(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool,alpha=self.top_k)  # 共享参数
        # self.gate_attentions = gatemamba_ablations_ssm(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool)
        # self.gate_attentions = gatemamba_ablations_A_bar_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool)
        # self.gate_attentions = gatemamba_ablations_A_bar(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool)
        # self.gate_attentions = GateMambaGCN_top_k_layer(self.nhid, self.top_k, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool)  # 共享参数
        # self.gate_attentions = GateMambaGCN_(self.nhid, self.top_k, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool)
        # self.gate_attentions = GateMambaGCN_layer_sigmoid(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool) 
        # self.gate_attentions = GateMambaGCN_ablations_None_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_B_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_C_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_BC_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_DeltaC_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_Delta_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_DeltaB_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        # self.gate_attentions = GateMambaGCN_ablations_z_selective(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand, dt_init=self.dt_init, pool=self.pool, num_nodes=self.num_nodes)
        
        # self.gate_attentions = GateMambaGCN_layer1(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand) # 不同参数
        # self.gate_attentions = GateMambaGCN_layer2(self.nhid, self.conv_layes, d_state=self.d_state, expand=self.expand)

    def forward(self, x, edge_index,A_norm, edge_attr):
        r'''
        graph: like Graph(num_nodes=24492, num_edges=186100,ndata_schemes={},edata_schemes={})
        x: input feature matrix (N, F_in)
        '''
        x, edge_attr = self.gate_attentions(x, edge_index, A_norm, edge_attr)  # (F_in, F_out) @ (N, F_in) -> (N, F_out)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, edge_attr

class GatedGCNModule(nn.Module):
    
    def __init__(self, inputfeature, dropout):
        super(GatedGCNModule, self).__init__()

        self.nhid = inputfeature
        self.dropout = dropout
        self.residual = True
        self.gate_gcn = GatedGCNLayer(self.nhid, self.nhid, self.dropout, self.residual, equivstable_pe=False)
    def forward(self, x, edge_index, edge_attr):
        
        x, edge_attr = self.gate_gcn(x, edge_index, edge_attr)
        
        return x, edge_attr

# 定义 GCN 模型
class GCNModule(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.dim = hidden_channels
        self.conv = pygnn.GCNConv(in_channels=self.dim, out_channels=self.dim)  # 第一层 GCN
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, edge_index,edge_attr=None):
        x = self.conv(x, edge_index)  # 第一层卷积
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)  # Dropout

        return x, edge_attr

class GENConvModule(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.normalization = nn.LayerNorm(dim)
        self.gen_conv = pygnn.GENConv(in_channels=dim,
                                       out_channels=dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gen_conv(x, edge_index)
        x = self.normalization(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x, edge_attr

class GATModule(nn.Module):
    def __init__(self, dim, dropout, num_heads):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.GATConv = pygnn.GATConv(in_channels=dim,
                                             out_channels=dim // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.GATConv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        return x, edge_attr

class GINEConv(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        gin_nn = nn.Sequential(Linear_pyg(dim, dim),
                                nn.ReLU(),
                                Linear_pyg(dim, dim))
        self.GINEConv = pygnn.GINEConv(gin_nn,edge_dim=None)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.GINEConv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        return x, edge_attr

class PNAConv(nn.Module):
    def __init__(self, dim, dropout, pna_degrees):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        aggregators = ['mean', 'max', 'sum']
        scalers = ['identity']
        deg = torch.from_numpy(np.array(pna_degrees))
        self.PNAConv = pygnn.PNAConv(dim, dim,
                                            aggregators=aggregators,
                                            scalers=scalers,
                                            deg=deg,
                                            edge_dim=16, # dim_h,
                                            towers=1,
                                            pre_layers=1,
                                            post_layers=1,
                                            divide_input=False)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.PNAConv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        return x, edge_attr
class SAGEModule(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.sage_conv = pygnn.SAGEConv(dim, dim)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.sage_conv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        return x, edge_attr
    
class APPNPModule(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.appnp = pygnn.APPNP(dim, dim)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.appnp(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        return x, edge_attr
    
class MixHopConv(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        #num_hops = 3
        self.dropout = nn.Dropout(p=dropout)
        self.mixhop_conv = pygnn.MixHopConv(in_channels=dim,
                                            out_channels=dim)
        self.compress = nn.Linear(dim * 3 , dim)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.mixhop_conv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        x = self.compress(x)
        return x, edge_attr
class GATv2Conv():
    def __init__(self, dim, dropout):
        super().__init__()
        #num_hops = 3
        self.dropout = nn.Dropout(p=dropout)
        self.GATv2Conv = pygnn.GATv2Conv(in_channels=dim,
                                        out_channels=dim,
                                        heads=3)
        self.compress = nn.Linear(dim * 3 , dim)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.GATv2Conv(x, edge_index)
        x = F.relu(x)                  # ReLU 激活
        x = self.dropout(x)            # Dropout
        x = self.compress(x)
        return x, edge_attr