
from torch import nn
from modules import (GateMambaGCN, ResidualModuleWrapper, GatedGCNModule, GENConvModule, 
                GCNModule, GATModule, GINEConv, PNAConv,SAGEModule, MixHopConv, GATv2Conv,APPNPModule)
from einops import rearrange, repeat
import torch_geometric.nn as pygnn
import torch.nn.functional as F

MODULES = {
    'GateMambaGCN': [GateMambaGCN],
    'GatedGCNLayer': [GatedGCNModule],  # Assuming GatedGCNLayer is a variant of GateMambaGCN
    # 'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    # 'GAT-sep': [GATSepModule],
    # 'GT': [TransformerAttentionModule, FeedForwardModule],
    # 'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
    'GENConv': [GENConvModule],
    'GINE': [GINEConv],
    'PNA': [PNAConv],
    'MixHop': [MixHopConv],
    'GATv2': [GATv2Conv],
    'APPNP': [APPNPModule],
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, dt_init, num_layers, input_dim, num_nodes, hidden_dim, conv_layes, mamba_hidden_dim, output_dim, hidden_dim_multiplier, top_k,num_heads,
                 normalization, dropout1, dropout2, pool):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        # self.input_linear0 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        # self.dropout = nn.Dropout(p=dropout1)
        self.act = nn.GELU()
        self.dropout = dropout1
        self.residual_modules = nn.ModuleList()
        for num_layers in range(num_layers):
            # print(num_layers)
            for module in MODULES[model_name]:

                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        dt_init=dt_init,
                                                        conv_layes = conv_layes, 
                                                        d_state = mamba_hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        top_k=top_k,
                                                        num_heads = num_heads,
                                                        dropout=dropout2,
                                                        pool=pool,
                                                        num_nodes=num_nodes)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear1 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x, edge_index,A_norm,edge_attr=None):
        x = self.input_linear(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(x)
        for residual_module in self.residual_modules:
            x, edge_attr = residual_module(x, edge_index, A_norm, edge_attr)

        x = self.output_normalization(x)
        perd = self.output_linear1(x)

        return perd
    

