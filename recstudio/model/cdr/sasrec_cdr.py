from typing import Dict
import torch
from recstudio.ann import sampler
from recstudio.ann.sampler import Dict
from recstudio.data import dataset
from recstudio.model.module import functional as recfn
from recstudio.model import basemodel, loss_func, module, scorer
import torch.nn.functional as F
from recstudio.model.seq.sasrec import SASRecQueryEncoder

class SASRec_CDR(basemodel.CrossRetriever):
    r"""
    SASRec models user's sequence with a Transformer.

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

    @staticmethod
    def _get_dataset_class():
        r"""SeqDataset is used for SASRec."""
        return dataset.CrossDomainSeqToSeqDataset

    def _get_query_encoder(self, meta_dataset):
        model_config = self.config['model']
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=meta_dataset.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            training_pooling_type='origin',
            item_encoder=self.item_encoder
        )

    def _get_item_encoder(self, meta_dataset):
        return torch.nn.Embedding(meta_dataset.num_items, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        r"""InnerProduct is used as the score function."""
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""Binary Cross Entropy is used as the loss function."""
        return loss_func.BinaryCrossEntropyLoss()

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=-1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch):
        outputs = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), True, True)
        if 'score' in outputs:
            # domain sampling is set to 'togather'
            if self.config['model']['directau_loss']:
                if self._get_dataset_class() is dataset.CrossDomainSeqToSeqDataset:
                    alignment = self.alignment(outputs['query'], outputs['item'])
                    mask = (batch['item_id'] != 0).flatten()
                    uniformity = (
                        self.uniformity(outputs['query'].flatten(0, 1)[mask]) +
                        self.uniformity(outputs['item'].flatten(0, 1)[mask])
                    )
                    loss_value = alignment + self.config['model']['gamma'] * uniformity
                elif self._get_dataset_class() is dataset.CrossDomainSeqDataset:
                    alignment = self.alignment(outputs['query'].squeeze(), outputs['item'])
                    uniformity = (
                        self.uniformity(outputs['query'].squeeze()) +
                        self.uniformity(outputs['item'])
                    )
                    loss_value = alignment + self.config['model']['gamma'] * uniformity
            else:
                loss_value = self.loss_fn(batch[self.frating], **outputs['score'])
        else:
            loss_value = 0
            if self.config['model']['directau_loss']:
                for idx, domain in enumerate(self.SOURCE_DOMAINS):
                    if outputs.get(domain, None) == None:
                        continue
                    domain_user, domain_item = outputs[domain]['query'], outputs[domain]['item']
                    domain_align_loss = self.alignment(domain_user, domain_item)
                    domain_uniformity_loss = self.uniformity(domain_user) + self.uniformity(domain_item)
                    domain_loss = domain_align_loss + self.config['model']['gamma'] * domain_uniformity_loss
                    loss_value += domain_loss
            else:
                for idx, domain in enumerate(self.SOURCE_DOMAINS):
                    if outputs.get(domain, None) == None:
                        continue
                    loss_value += self.loss_fn(batch[domain][self.frating], **outputs[domain]['score'])
        return loss_value

    def _get_sampler(self, meta_dataset):
        r"""Uniform sampler is used as negative sampler."""
        return sampler.CrossDomainUniformSampler(meta_dataset)
