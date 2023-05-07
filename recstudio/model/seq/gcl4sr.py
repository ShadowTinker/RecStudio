import dgl
import torch
import numpy as np
import scipy.sparse as sp

from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from .sasrec import SASRec, SASRecQueryEncoder

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

    def training_step(self, batch):
        output = self.forward(batch, False)
        cl_output = self.augmentation_model(batch, self.item_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
            + self.config['model']['cl_weight'] * cl_output['cl_loss']
        return loss_value

