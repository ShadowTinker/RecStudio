import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from recstudio.model.module import SeqPoolingLayer
from .sasrec import SASRec
from dgl.nn.pytorch.conv import GraphConv

class Global_ATT(nn.Module):
    def __init__(self, hidden_size, num_heads, dp_att, dp_ffn):
        super(Global_ATT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_att = dp_att
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(self.dropout_att)
        self.ffn = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout_ffn = dp_ffn
        self.ffn_dropout = nn.Dropout(self.dropout_ffn)

    def transpose_qkv(self, hidden):
        hidden = hidden.reshape(hidden.shape[0], self.num_heads, -1)
        hidden = hidden.permute(1, 0, 2)
        return hidden

    def transpose_output(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        inputs = inputs.contiguous().view(inputs.size()[0], 1, -1).squeeze(1)
        return inputs

    def forward(self, inputs):
        query_h = self.transpose_qkv(self.q(inputs))
        key_h = self.transpose_qkv(self.k(inputs))
        value_h = self.transpose_qkv(self.v(inputs))
        softmax_score = torch.tanh(torch.matmul(query_h, key_h.transpose(dim0=1, dim1=-1)))
        att_hidden = self.transpose_output(torch.matmul(self.attention_dropout(softmax_score), value_h))
        att_hidden = self.ffn_dropout(torch.relu(self.ffn(att_hidden)))
        return inputs, att_hidden

class AGNN(nn.Module):
    def __init__(self, block_nums_global, hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn):
        super(AGNN, self).__init__()
        self.block_nums = block_nums_global
        self.att_layer = Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn)
        # self.multi_block_att = [self.att_layer for _ in range(self.block_nums)]
        self.multi_block_att = [Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn) for _ in range(self.block_nums)]
        for i, global_attention in enumerate(self.multi_block_att):
            self.add_module('global_attention_{}'.format(i), global_attention)

    def forward(self, inputs):
        for global_att_temp in self.multi_block_att:
            inputs, att_hidden = global_att_temp(inputs)
            inputs = inputs + att_hidden
        return inputs

class SGGNN(nn.Module):
    def __init__(self, embed_dim, step=1) -> None:
        super(SGGNN, self).__init__()
        self.step = step
        self.embed_dim = embed_dim
        self.w_h = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.w_hf = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.simple_gnn_layer = GraphConv(1, 1, weight=False, bias=False, allow_zero_in_degree=True) # A * hidden

    def gnn_conv(self, g, hidden): 
        hidden = self.w_h(hidden)
        hidden1, hidden2, hidden3  = torch.chunk(hidden, 3, -1)
        hidden_fuse1, hidden_fuse2 = self.w_hf(self.simple_gnn_layer(g, hidden1)).chunk(2, -1)
        gate = F.relu(hidden_fuse1 + hidden2)
        return hidden3 + gate * hidden_fuse2

    def forward(self, g, hidden):
        seqs = []
        for i in range(self.step):
            hidden = self.gnn_conv(g, hidden)
            seqs.append(hidden)
        return hidden, torch.stack(seqs, dim=1).mean(1)

class DGNNQueryEncoder(nn.Module):
    def __init__(self, fiid, embed_dim, n_head, dropout, n_layer, item_encoder) -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.n_items = len(self.item_encoder.weight)
        self.agnn = AGNN(n_layer, embed_dim, n_head, dropout, dropout)
        self.sggnn = SGGNN(embed_dim, n_layer)
        self.global_fuse = nn.Linear(embed_dim * 2, embed_dim)
        self.pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.att_linear1 = nn.Linear(embed_dim, embed_dim)
        self.att_linear2 = nn.Linear(embed_dim, embed_dim)
        self.att_linear3 = nn.Linear(embed_dim, 1, bias=False)
        self.att_linear4 = nn.Linear(embed_dim * 2, embed_dim)

    def build_batch_graph(self, item_seq : torch.Tensor, seq_len):
        device = item_seq.device
        batch_item, inverse_mapping = item_seq.unique(sorted=True, return_inverse=True)
        
        # Generate position bias for row and col
        bias = [[idx * item_seq.size(1)] * (_ - 1) for idx, _ in enumerate(seq_len)]
        bias = [item for sublist in bias for item in sublist]
        row = [list(range(_ - 1)) for _ in seq_len]
        row = [item for sublist in row for item in sublist]
        row = torch.tensor(row + bias, device=device, dtype=torch.long)
        col = row + 1
        row, col = item_seq.flatten()[row], item_seq.flatten()[col]

        g = dgl.graph((row, col), num_nodes=self.n_items, device=device)
        return g, batch_item, inverse_mapping
        
    def att_pooling(self, hidden, seq_len, mask): # [B, L, d]
        ht = self.pooling_layer(hidden, seq_len)
        q1 = self.att_linear1(ht).unsqueeze(-2)
        q2 = self.att_linear2(hidden)
        alpha = self.att_linear3(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.unsqueeze(-1), 1)
        a = self.att_linear4(torch.cat([a, ht], dim=-1))
        return a

    def forward(self, batch):
        item_seq, seq_len = batch['in_'+self.fiid], batch['seqlen']

        batch_graph, batch_item, inverse_mapping = self.build_batch_graph(item_seq, seq_len)

        global_item = self.item_encoder(batch_item)

        hidden_last_gnn, hidden_fuse_gnn = self.sggnn(batch_graph, self.item_encoder.weight) # [B, d]
        hidden_last_gnn = hidden_last_gnn[item_seq] # [B, L, d]
        hidden_att = self.agnn(global_item)[inverse_mapping] # [B, L, d]

        hidden_last = self.global_fuse(torch.cat([hidden_last_gnn, hidden_att], dim=-1))
        hidden_fuse = self.att_pooling(hidden_last, seq_len, item_seq != 0)

        return hidden_fuse

class DGNN(SASRec):
    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return DGNNQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim, n_head=model_config['head_num'],
            dropout=model_config['dropout_rate'], n_layer=model_config['layer_num'], item_encoder=self.item_encoder
        )

