from operator import mod
import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer
from recstudio.model.module import data_augmentation

r"""
GRU4Rec
############

Paper Reference:
    Balazs Hidasi, et al. "Session-Based Recommendations with Recurrent Neural Networks" in ICLR2016.
    https://arxiv.org/abs/1511.06939
"""


class GRU4Rec_G(basemodel.BaseRetriever):
    r"""
    GRU4Rec apply RNN in Recommendation System, where sequential behavior of user is regarded as input
    of the RNN.
    """

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.GSLAugmentation3(self.config['model'], train_data)

    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return (
            module.VStackLayer(
                module.HStackLayer(
                    torch.nn.Sequential(
                        module.LambdaLayer(lambda x: x['in_'+self.fiid]),
                        self.item_encoder,
                        torch.nn.Dropout(model_config['dropout_rate']),
                        module.GRULayer(self.embed_dim, model_config['hidden_size'], model_config['layer_num']),
                    ),
                    module.LambdaLayer(lambda_func=lambda x: x['seqlen']),
                ),
                module.SeqPoolingLayer(pooling_type='last'),
                torch.nn.Linear(model_config['hidden_size'], self.embed_dim)
            )
        )

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
    
    def training_step(self, batch):
        output = self.forward(batch, False) 
        cl_output = self.augmentation_model(batch, self.item_encoder.weight)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) + \
            self.config['model']['cl_weight'] * cl_output['cl_loss']
        return loss_value
