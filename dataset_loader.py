import torch
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz
from torch_geometric.datasets import (Amazon, Coauthor, GNNBenchmarkDataset, TUDataset, Actor, WebKB,
                                      WikipediaNetwork, ZINC, HeterophilousGraphDataset)
import dgl
from gatemamba.loader.split_generator import prepare_splits
from typing import Callable, Optional
from torch_geometric.utils import degree




class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, f"{self.name}.npz")
        with open(p2f, 'rb') as f:
            data = np.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)



class load_wiki_new(InMemoryDataset):
    
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')
    
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        
        self.name = name.lower().replace('-', '_')
        assert self.name in ['chameleon_filtered', 'squirrel_filtered',
                             'squirrel_filtered_directed']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def download(self) -> None:
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)
    
    def process(self) -> None:

        data = np.load(self.raw_paths[0], 'r')
        node_features = torch.FloatTensor(data['node_features'])
        node_labels = torch.LongTensor(data['node_labels'])
        edge_index = torch.LongTensor(data['edges']).t()  
        train_masks = torch.BoolTensor(data['train_masks']).t().contiguous()
        val_masks = torch.BoolTensor(data['val_masks']).t().contiguous()
        test_masks = torch.BoolTensor(data['test_masks']).t().contiguous()
        
        # 创建 PyG Data 对象
        data = Data(
            x=node_features,
            y=node_labels,
            edge_index=edge_index,
            train_mask=train_masks,
            val_mask=val_masks,
            test_mask=test_masks
        )
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
            
        self.save([data], self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
    
def A_norm(dataset, edge_index):
    x = dataset.data.x
    num_nodes = x.size(0)
    
    row, col = edge_index
    deg_out = degree(row, num_nodes, dtype=x.dtype)
    deg_in = degree(col, num_nodes, dtype=x.dtype)

    deg_out_inv_sqrt = deg_out.pow(-0.5)
    deg_out_inv_sqrt[deg_out_inv_sqrt == float('inf')] = 0  # 处理孤立节点
    
    deg_in_inv_sqrt = deg_in.pow(-0.5)
    deg_in_inv_sqrt[deg_in_inv_sqrt == float('inf')] = 0    # 处理孤立节点
    
    norm = deg_out_inv_sqrt[row] * deg_in_inv_sqrt[col] 

    return norm

def DataLoader(args):
    name = args.dataset.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())# ,split='geom-gcn'
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        if name == 'crocodile':
            raise NotImplementedError(f"crocodile not implemented yet")
        dataset = WikipediaNetwork(root='./data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures(), force_reload=True)
    elif name in ['actor']:
        dataset = Actor(root='./data/', transform=T.NormalizeFeatures(), force_reload=False)
        
    elif name in ['texas', 'cornell',"wisconsin"]:
        dataset = WebKB(root='./data/',name=name, transform=T.NormalizeFeatures(), force_reload=True)
        
    elif name in ['roman-empire','amazon-ratings','minesweeper','tolokers','questions']:
        dataset = HeterophilousGraphDataset(root='./data/', name=name, transform=T.NormalizeFeatures(), force_reload=True)
        
    elif name in ['squirrel-filtered','chameleon-filtered','squirrel-filtered-directed']:
        from torch_geometric.transforms import AddSelfLoops
        transform = T.Compose([
                              # AddSelfLoops(),  # 添加自环
                              T.NormalizeFeatures()
                             ])
        dataset = load_wiki_new(root='./data/', name=name, transform=transform, force_reload=True)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    
    # 邻接矩阵归一化系数
    dataset.data.norm_A = A_norm(dataset, dataset.data.edge_index)

    # edge_index1 = to_undirected(dataset.data.edge_index)
    
    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset,args)

    dataset.num_targets = 1 if dataset.num_classes == 2 else dataset.num_classes
    dataset.loss_fn = F.binary_cross_entropy_with_logits if dataset.num_targets == 1 else F.cross_entropy
    dataset.metric = 'ROC AUC' if dataset.num_targets == 1 else 'accuracy'
    
    return dataset
