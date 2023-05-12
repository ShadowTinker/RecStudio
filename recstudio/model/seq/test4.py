import torch
import torch.nn as nn
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class Test4QueryEncoder(SASRecQueryEncoder):
    def __init__(self, n_user, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__(fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional, training_pooling_type, eval_pooling_type)
        self.user_embed = nn.Embedding(n_user, embed_dim)
        self.finetune = False

    def forward(self, batch, need_pooling=True):
        trm_out = super().forward(batch, need_pooling)
        if self.finetune:
            user = batch['user_id']
            if len(trm_out.shape) == 3:
                trm_out = trm_out + self.user_embed.weight[user].unsqueeze(1) * trm_out
            else:
                trm_out = trm_out + self.user_embed.weight[user] * trm_out
        return trm_out

class Test4(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.finetune = False
        self.augmentation_model = data_augmentation.MyAugmentation(self.config['model'], train_data)
        self.augmentation_model2 = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last item is mask
    
    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return Test4QueryEncoder(
            train_data.num_users, fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            training_pooling_type='origin',
            item_encoder=self.item_encoder
        )

    def training_step(self, batch, adaption=False):
        if not adaption:
            output = self.forward(batch, False)
            cl_output = self.augmentation_model(batch, self.item_encoder.weight[:-1])
            cl_output2 = self.augmentation_model2(batch, self.query_encoder)
            loss_value = self.loss_fn(batch[self.frating], **output['score']) \
                + self.config['model']['gcl_weight'] * cl_output['cl_loss'] \
                + self.config['model']['cl_weight'] * cl_output2['cl_loss']
        elif adaption:
            output = self.forward(batch, False)
            cl_output2 = self.augmentation_model2(batch, self.query_encoder)
            loss_value = self.loss_fn(batch[self.frating], **output['score']) \
                + self.config['model']['cl_weight'] * cl_output2['cl_loss']
        return loss_value

    def test_epoch(self, dataloader):
        if self.finetune:
            self.train()
            trn_loader = self.trainloaders[0]
            for _ in range(1):
                print('test')
                # TTA for 10 epochs
                for batch_idx, batch in enumerate(trn_loader):
                    # data to device
                    batch = self._to_device(batch, self._parameter_device)
                    self.optimizers[0]['optimizer'].zero_grad()
                    # model loss
                    training_step_args = {'batch': batch, 'adaption': True}
                    loss = self.training_step(**training_step_args)
                    loss.backward()
                    self.optimizers[0]['optimizer'].step()
            self.eval()
        return super().test_epoch(dataloader)
