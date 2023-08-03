from typing import Dict
import torch
from recstudio.ann import sampler
from recstudio.ann.sampler import Dict
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer

class BPR_CDR(basemodel.CrossRetriever):

    def _get_dataset_class():
        return dataset.CrossDomainDataset

    def _get_item_encoder(self, meta_dataset):
        return torch.nn.Embedding(meta_dataset.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, meta_dataset):
        return torch.nn.Embedding(meta_dataset.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
