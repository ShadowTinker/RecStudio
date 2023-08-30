from recstudio.data.dataset import CrossDomainDataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.module import graphmodule
from recstudio.ann import sampler
import torch

r"""
LightGCN
#############
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR'20)
    Reference:
        https://dl.acm.org/doi/10.1145/3397271.3401063
"""
class LightGCN_CDR(basemodel.CrossRetriever):
    r"""
    LightGCN simplifies the design of GCN to make it more concise and appropriate for recommendation.
    LightGCN learns user and item embeddings by linearly propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embedding.
    """
    def _init_model(self, meta_dataset: CrossDomainDataset, train_data):
        super()._init_model(meta_dataset, train_data)
        self.num_users = meta_dataset.num_users
        self.num_items = meta_dataset.num_items
        self.user_emb = torch.nn.Embedding(meta_dataset.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = torch.nn.Embedding(meta_dataset.num_items, self.embed_dim, padding_idx=0)
        self.combiners = torch.nn.ModuleList()
        for i in range(self.config['model']['n_layers']):
            self.combiners.append(graphmodule.LightGCNCombiner(self.embed_dim, self.embed_dim))
        self.LightGCNNet = graphmodule.LightGCNNet_dglnn(self.combiners)
        adj_size = meta_dataset.num_users + meta_dataset.num_items
        self.adj_mat, _ = meta_dataset.get_graph([0], form='dgl', value_fields='inter', \
            col_offset=[meta_dataset.num_users], bidirectional=[True], shape=(adj_size, adj_size))

    def _get_dataset_class():
        return CrossDomainDataset

    def _get_loss_func(self):
        self.domain_loss_ratio = self.config['train']['domain_loss_ratio']
        return loss_func.BPRLoss()

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_sampler(self, meta_dataset):
        r"""Uniform sampler is used as negative sampler."""
        return sampler.CrossDomainUniformSampler(meta_dataset)

    def _get_query_encoder(self, train_data):
        return graphmodule.GraphUserEncoder()

    def _get_item_encoder(self, train_data):
        return graphmodule.GraphItemEncoder()

    def update_encoders(self):
        self.adj_mat = self.adj_mat.to(self._parameter_device)
        # [num_users + num_items, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        all_embeddings = self.LightGCNNet(self.adj_mat, embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        self.query_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        # TODO: make sure that padding embedding is all 0.

    def forward(self, batch_data, full_score, return_query=True, return_item=True, return_neg_id=True):
        self.update_encoders()
        return super().forward(batch_data, full_score, return_query=return_query, return_item=return_item, return_neg_id=return_neg_id)

    def training_step(self, batch):
        outputs = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), True, True)
        loss_value = 0
        for idx, domain in enumerate(self.SOURCE_DOMAINS):
            if outputs.get(domain, None) == None:
                continue
            score = outputs[domain]['score']
            score['label'] = batch[domain][self.frating]
            domain_loss = self.loss_fn(**score)
            domain_loss += self.config['model']['l2_reg_weight'] * loss_func.l2_reg_loss_fn(
                self.user_emb(batch[domain][self.fuid]),
                self.item_emb(batch[domain][self.fiid]),
                self.item_emb(outputs[domain]['neg_id'].reshape(-1))
            )
            loss_value += self.domain_loss_ratio[idx] * domain_loss
        return loss_value

    def _get_item_vector(self, domain=None):        
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[self.DOMAIN_ITEM_ID[domain][1:]].detach().clone()
        else:
            return self.item_encoder.item_embeddings[self.DOMAIN_ITEM_ID[domain][1:]].detach().clone()

    def _update_item_vector(self, domain=None):
        self.update_encoders()
        super()._update_item_vector(domain)

