import torch
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class Test10(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.GSLAugmentation4(self.config['model'], train_data)
        self.augmentation_model2 = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last item is mask
    
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

    def training_step(self, batch):
        output = self.forward(batch, False)
        cl_output = self.augmentation_model(batch, self.item_encoder.weight[:-1])
        cl_output2 = self.augmentation_model2(batch, self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['model']['gcl_weight'] * cl_output['cl_loss'] \
           + self.config['model']['cl_weight'] * cl_output2['cl_loss']
        return loss_value