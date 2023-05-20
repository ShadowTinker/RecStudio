import dgl
import torch
import numpy as np
from recstudio.data import dataset
from torch.nn.utils.rnn import pad_sequence
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

# class Test12QueryEncoder(SASRecQueryEncoder):
#     def __init__(self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
#         super().__init__(fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional, training_pooling_type, eval_pooling_type)
#         self.max_seq_len = max_seq_len
#         self.g = None
#         self.n_head = n_head

#     def update_graph(self, g):
#         self.g = g.to('cpu')

#     def sampling_with_graph(self, batch):
#         # Generate extra items and corresponding mask
#         sequences, seq_lens = batch['in_'+self.fiid].cpu(), batch['seqlen'].cpu()
#         batch_size = sequences.size(0)
#         device = batch['in_'+self.fiid].device

#         augmented_sequences = []
#         mask = torch.zeros(batch_size, seq_lens.max(), seq_lens.max(), dtype=torch.bool, device=device)
#         augmented_seq_lens = torch.zeros(batch_size, dtype=seq_lens.dtype, device=seq_lens.device)
#         for i in range(batch_size):
#             cur_seq_len = seq_lens[i]
#             cur_seq = sequences[i][:cur_seq_len]
#             # to_be_sampled = self.max_seq_len - cur_seq_len
#             if cur_seq_len * 2 <= self.max_seq_len:
#                 to_be_sampled = cur_seq_len
#             else:
#                 to_be_sampled = self.max_seq_len - cur_seq_len
#             sg = dgl.sampling.sample_neighbors(self.g, cur_seq, -1)
#             prob = sg.edata['weight']
#             sampled_eid = np.random.choice(sg.edges(form='eid'), to_be_sampled.item(), p=prob / prob.sum())
#             row, col = sg.edges(form='uv')
#             row, col = row[sampled_eid], col[sampled_eid]
#             mask[i][:to_be_sampled] = True
#             mask[i] = mask[i] * (~torch.eye(seq_lens.max(), device=device, dtype=torch.bool))
#             augmented_seq = torch.cat([row, cur_seq])
#             augmented_seq_lens[i] = augmented_seq.shape[0]
#             augmented_sequences.append(augmented_seq)
#         mask = mask.repeat(self.n_head, 1, 1)
#         return pad_sequence(augmented_sequences, batch_first=True).to(device), augmented_seq_lens.to(device), mask
    
#     def forward(self, batch, need_pooling=True):
#         if need_pooling: # False when calling in another augmentation
#             real_seq_len = batch['seqlen']
#             aug_seq, aug_seq_len, aug_mask = self.sampling_with_graph(batch)
#             batch['in_'+self.fiid], batch['seqlen'] = aug_seq, aug_seq_len
#             self.aug_seq = aug_seq
#             self.aug_seq_len = aug_seq_len
#             self.aug_mask = aug_mask

#         user_hist = batch['in_'+self.fiid]
#         positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
#         positions = positions.unsqueeze(0).expand_as(user_hist)
#         position_embs = self.position_emb(positions)
#         seq_embs = self.item_encoder(user_hist)
#         mask4padding = user_hist == 0  # BxL
#         L = user_hist.size(-1)
#         if not self.bidirectional:
#             attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
#             attention_mask = attention_mask.unsqueeze(0) + self.aug_mask
#         else:
#             attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
#         transformer_out = self.transformer_layer(
#             src=self.dropout(seq_embs+position_embs),
#             mask=attention_mask,
#             src_key_padding_mask=mask4padding)  # BxLxD
        
#         if need_pooling:
#             if self.training:
#                 if self.training_pooling_type == 'mask':
#                     trm_out = self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
#                 else:
#                     trm_out = self.training_pooling_layer(transformer_out, batch['seqlen'])
#             else:
#                 if self.eval_pooling_type == 'mask':
#                     trm_out = self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
#                 else:
#                     trm_out = self.eval_pooling_layer(transformer_out, batch['seqlen'])
#         else:
#             trm_out = transformer_out
        
#         if need_pooling and self.training:
#             real_seq = []
#             for seq_len, aug_len, seq_out in zip(real_seq_len, self.aug_seq_len, trm_out):
#                 real_seq.append(seq_out[aug_len - seq_len: aug_len])
#             real_trm_out = pad_sequence(real_seq, batch_first=True)
#             return real_trm_out
#         else:
#             return trm_out

class Test12QueryEncoder(SASRecQueryEncoder):
    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)
        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
            if batch.get('mask') != None:
                attention_mask = attention_mask.unsqueeze(0) + batch['mask']
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD
        
        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                if self.eval_pooling_type == 'mask':
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return transformer_out

class Test12(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.GSLAugmentation3(self.config['model'], train_data)
        self.augmentation_model2 = data_augmentation.GSLCL4SRecAugmentation(self.config['model'], train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last item is mask
    
    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return Test12QueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            training_pooling_type='origin',
            item_encoder=self.item_encoder
        )

    def training_step(self, batch):
        cl_output = self.augmentation_model(batch, self.item_encoder.weight[:-1])
        # self.query_encoder.update_graph(self.augmentation_model.g)
        output = self.forward(batch, False)
        cl_output2 = self.augmentation_model2(batch, self.augmentation_model.g.to('cpu'), self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['model']['gcl_weight'] * cl_output['cl_loss'] \
           + self.config['model']['cl_weight'] * cl_output2['cl_loss'] \
           + self.config['model']['kl_weight'] * cl_output['kl_loss']
        return loss_value
