import dgl
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import dgl.function as fn

from dgl.nn.pytorch.conv import SAGEConv
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from .sasrec import SASRec, SASRecQueryEncoder

class GCL4SRQueryEncoder(SASRecQueryEncoder):
    def __init__(self, num_users, fuid, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__(fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional, training_pooling_type, eval_pooling_type)
        self.embed_dim = embed_dim
        self.linear_gate1 = nn.Linear(embed_dim, 1)
        self.linear_gate2 = nn.Linear(embed_dim, max_seq_len)
        self.user_gate_emb = nn.Embedding(num_users, embed_dim)
        self.gate_act = nn.Sigmoid()
        self.dgl_batch1, self.dgl_batch2 = None, None
        self.fuid = fuid
        self.gnn_list = self.generate_gnn_list()

    def generate_gnn_list(self):
        gnn_list = nn.ModuleList()
        gnn_list.append(WeightedGCN(self.embed_dim, self.embed_dim))
        gnn_list.append(SAGEConv(self.embed_dim, self.embed_dim, 'mean'))
        return gnn_list

    def get_gnn_embeddings(self, emb, graph_list, gnns):
        for idx, (g, layer) in enumerate(zip(graph_list, gnns)):
            emb = layer(g, emb)
        return emb
    
    def forward(self, batch, need_pooling=True):
        transformer_out = super().forward(batch, need_pooling)
        self.trm_out = transformer_out

        device = transformer_out.device

        item_seq, seq_len = batch['in_' + self.fiid], batch['seqlen']
        mask = torch.arange(item_seq.size(-1), device=device).expand_as(item_seq) < seq_len.unsqueeze(1)
        mean_pooling_helper = (1 / seq_len).unsqueeze(-1)

        max_len = batch['seqlen'].max()

        user_specific_gate = self.user_gate_emb(batch[self.fuid])

        input_nodes1, output_nodes1, blocks1 = self.dgl_batch1
        item_all_vec1 = self.get_gnn_embeddings(self.item_encoder.weight[input_nodes1], blocks1, self.gnn_list)
        graph_emb1 = torch.zeros_like(item_all_vec1, device=device)
        graph_emb1[output_nodes1] = item_all_vec1
        graph_emb1 = graph_emb1[item_seq] * mask.unsqueeze(-1)
        self.graph_emb1 = graph_emb1.sum(-2) * mean_pooling_helper
        self.gated_graph_emb1 = (graph_emb1 * self.gate_act(
            self.linear_gate1(graph_emb1) + self.linear_gate2(user_specific_gate)[:, :max_len].unsqueeze(-1)
        )).sum(-2) * mean_pooling_helper

        input_nodes2, output_nodes2, blocks2 = self.dgl_batch2
        item_all_vec2 = self.get_gnn_embeddings(self.item_encoder.weight[input_nodes2], blocks2, self.gnn_list)
        graph_emb2 = torch.zeros_like(item_all_vec2, device=device)
        graph_emb2[output_nodes2] = item_all_vec2
        graph_emb2 = graph_emb2[item_seq] * mask.unsqueeze(-1)
        self.graph_emb2 = graph_emb2.sum(-2) * mean_pooling_helper
        self.gated_graph_emb2 = (graph_emb2 * self.gate_act(
            self.linear_gate1(graph_emb2) + self.linear_gate2(user_specific_gate)[:, :max_len].unsqueeze(-1)
        )).sum(-2) * mean_pooling_helper

        final_out = self.gated_graph_emb1 + self.gated_graph_emb2 + transformer_out
        return final_out

class WeightedGCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeightedGCN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, g : dgl.DGLGraph, x):
        with g.local_scope():
            g.srcdata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.dstdata['h']

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class GCL4SR(SASRec):
    r"""
    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.GCL4SRAugmentation(self.config['model'], train_data)
        self.g = self.augmentation_model.g
        self.mmd_loss = MMD_loss()
        
    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return GCL4SRQueryEncoder(
            train_data.num_users, fuid=self.fuid, fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            item_encoder=self.item_encoder
        )

    def generate_dgl_blocks(self):
        sampler = dgl.dataloading.NeighborSampler([20, 20])
        dataloader = dgl.dataloading.DataLoader(
            self.g, self.g.nodes(), sampler,
            batch_size=1000000000, shuffle=True, drop_last=False, device='cuda'
        )
        input_nodes, output_nodes, blocks = next(iter(dataloader))
        blocks = [block.to(self.device) for block in blocks]
        return input_nodes, output_nodes, blocks

    def training_step(self, batch):
        # Graph emb and graph contrastive learning loss
        self.g = self.g.to(self.device)
        if self.query_encoder.dgl_batch1 == None or self.query_encoder.dgl_batch2 == None:
            # Initialization
            self.query_encoder.dgl_batch1 = self.generate_dgl_blocks()
            self.query_encoder.dgl_batch2 = self.generate_dgl_blocks()

        # Recommendation loss
        output = self.forward(batch, False)

        # Contrastive loss
        cl_output = self.augmentation_model(self.query_encoder)

        # MMD loss
        MMD_loss1 = self.mmd_loss(self.query_encoder.trm_out, self.query_encoder.gated_graph_emb1)
        MMD_loss2 = self.mmd_loss(self.query_encoder.trm_out, self.query_encoder.gated_graph_emb2)

        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
            + self.config['model']['cl_weight'] * cl_output['cl_loss'] \
            + self.config['model']['mmd_weight'] * (MMD_loss1 + MMD_loss2)
        return loss_value
    
    def training_epoch_end(self, output_list):
        # Refresh sampled graph per epoch
        self.query_encoder.dgl_batch1 = self.generate_dgl_blocks()
        self.query_encoder.dgl_batch2 = self.generate_dgl_blocks()
        return super().training_epoch_end(output_list)


