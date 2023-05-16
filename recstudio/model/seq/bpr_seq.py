import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer

class BPR_SeqQueryEncoder(torch.nn.Module):
    def __init__(self, num_users, embed_dim, fuid) -> None:
        super().__init__()
        self.fuid = fuid
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, padding_idx=0)

    def forward(self, batch):
        return self.user_embedding(batch[self.fuid])

class BPR_Seq(basemodel.BaseRetriever):

    def _get_dataset_class():
        return dataset.SeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return BPR_SeqQueryEncoder(train_data.num_users, self.embed_dim, self.fuid)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
