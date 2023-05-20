import dgl
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import dgl.function as fn

from recstudio.data import dataset
from dgl.nn.pytorch.conv import GraphConv
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class GraphLearner(nn.Module):
    def __init__(self, n_nodes, embed_dim, learner_type='att') -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.learner_type = learner_type
        if learner_type == 'att':
            self.learner = torch.nn.MultiheadAttention()
        elif learner_type == 'ae':
            self.encoder = nn.Linear(n_nodes, embed_dim)
            self.decoder = nn.Linear(embed_dim, n_nodes)

    def _build_weighted_dgl_graph(self, row, col, val):
        g = dgl.graph((row, col), num_nodes=self.n_nodes, device=row.device)
        g.edata['w'] = val
        return g

    def metric(self, g : dgl.DGLGraph, emb : torch.Tensor):
        device = emb.device
        A = g.adj(ctx=device)
        if self.learner_type == 'ae':
            h = torch.sparse.mm(A, self.encoder.weight.T) + self.encoder.bias
        return

    def forward(self, item_encoder : nn.Embedding):
        
        return

class WeightedLGN(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeightedLGN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, g : dgl.DGLGraph, x):
        with g.local_scope():
            g.srcdata['h'] = x
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.dstdata['h']

class Test2(SASRec):
    # With GSL
    def _init_model(self, train_data):
        super()._init_model(train_data)
        initial_graph = self.global_graph_construction(train_data)
        gnn_list = self.get_gnn_list()
        self.augmentation_model = data_augmentation.MyAugmentation2(
            self.config['model'], train_data, initial_graph, gnn_list
        )

    def get_gnn_list(self):
        gnn_list = nn.ModuleList()
        for _ in range(self.config['gnn_layers']):
            if self.config['gnn'] == 'wlgn':
                gnn_list.append(WeightedLGN(1, 1))
            elif self.config['gnn'] == 'lgn':
                gnn_list.append(GraphConv(1, 1, weight=False, bias=False, allow_zero_in_degree=True))
            else:
                raise NotImplementedError
        return gnn_list

    def global_graph_construction(self, train_data : dataset.TripletDataset):
        history_matrix, history_len, n_items = train_data.user_hist, train_data.user_count, train_data.num_items
        history_matrix = history_matrix.tolist()
        row, col, data = [], [], []
        for idx in range(len(history_len)):
            item_list_len = history_len[idx]
            item_list = history_matrix[idx][:item_list_len]
            for item_idx in range(item_list_len - 1):
                target_num = min(self.k, item_list_len - item_idx - 1)
                row += [item_list[item_idx]] * target_num
                col += item_list[item_idx + 1: item_idx + 1 + target_num]
                data.append(1 / np.arange(1, 1 + target_num))
        data = np.concatenate(data)
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sparse_matrix.T + sp.eye(n_items)
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        g = dgl.from_scipy(norm_adj)
        g.edata['weight'] = torch.tensor(norm_adj.data)
        return g
    
    def graph_learner():
        return

    def training_step(self, batch):
        output = self.forward(batch, False)
        cl_output = self.augmentation_model(batch, self.item_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['model']['cl_weight'] * cl_output['cl_loss']
        return loss_value
