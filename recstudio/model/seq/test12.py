import dgl
import torch
import numpy as np
from recstudio.data import dataset
from torch.nn.utils.rnn import pad_sequence
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

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
