from typing import Dict, List
import torch
import torch.nn as nn
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class Test4(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.MyAugmentation(self.config['model'], train_data)
        self.augmentation_model2 = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)
        self.projection_head_train = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_head_test = nn.Linear(self.embed_dim, self.embed_dim)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last item is mask
    
    def _get_train_loaders(self, train_data : dataset.SeqToSeqDataset, ddp=False) -> List:
        train_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'],
                                                   shuffle = True, ddp=ddp)
        adaption_train_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'],
                                                      shuffle = True, ddp=ddp)
        return [train_loader, adaption_train_loader]

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            training_pooling_type='origin',
            item_encoder=self.item_encoder
        )

    def training_step(self, batch, loader_idx):
        if loader_idx == 0:
            output = self.forward(batch, False)
            cl_output = self.augmentation_model(batch, self.item_encoder.weight[:-1], self.projection_head_train)
            cl_output2 = self.augmentation_model2(batch, self.query_encoder, self.projection_head_train)
            loss_value = self.loss_fn(batch[self.frating], **output['score']) \
            + self.config['model']['gcl_weight'] * cl_output['cl_loss'] \
            + self.config['model']['cl_weight'] * cl_output2['cl_loss']
        elif loader_idx == 1:
            cl_output2 = self.augmentation_model2(batch, self.query_encoder, self.projection_head_test)
            loss_value = self.config['model']['cl_weight'] * cl_output2['cl_loss']
        return loss_value

    def training_epoch(self, nepoch):
        return super().training_epoch(nepoch)
