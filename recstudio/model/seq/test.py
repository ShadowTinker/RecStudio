import torch
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class Test(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.MyAugmentation(self.config['model'], train_data)

    def training_step(self, batch):
        output = self.forward(batch, False)
        cl_output = self.augmentation_model(batch, self.item_encoder.weight)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['model']['cl_weight'] * cl_output['cl_loss']
        return loss_value
