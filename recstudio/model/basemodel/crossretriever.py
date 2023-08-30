import abc
import os
import inspect
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

import time
import nni
from recstudio.ann.sampler import Dict
import recstudio.eval as eval
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.utils.clip_grad import clip_grad_norm_
from recstudio.model import init, basemodel, loss_func
from recstudio.utils import callbacks
from recstudio.utils.utils import *
from recstudio.utils.data_parallel import data_parallel
from recstudio.data.dataset import (CrossDomainLoaders, CombinedLoaders)
from recstudio.data import UserDataset, SeqDataset, CrossDomainDataset, CrossDomainSeqDataset


class CrossRetriever(basemodel.BaseRetriever):
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)


    def fit(
        self,
        meta_dataset : CrossDomainDataset,
        train_data: List,
        val_data: Optional[List] = None,
        run_mode='light',
        config: Dict = None,
        **kwargs
    ) -> None:
        r"""
        Fit the model with train data.
        """
        # self.set_device(self.config['gpu'])
        self.logger = logging.getLogger('recstudio')
        if config is not None:
            self.config.update(config)

        if kwargs is not None:
            self.config.update(kwargs)

        # set tensorboard
        tb_log_name = None
        for handler in self.logger.handlers:
            if type(handler) == logging.FileHandler:
                tb_log_name = os.path.basename(handler.baseFilename).split('.')[0]

        if tb_log_name is None:
            import time
            tb_log_name = time.strftime(f"%Y-%m-%d-%H-%M-%S", time.localtime())
        if self.config['train']['tensorboard_path'] is not None:
            self.tensorboard_logger = SummaryWriter(os.path.join(self.config['train']['tensorboard_path'], tb_log_name))
        else:
            self.tensorboard_logger = SummaryWriter(os.path.join(f'./tensorboard/{self.__class__.__name__}/{meta_dataset.name}/', tb_log_name))

        self.tensorboard_logger.add_text('Configuration/model', dict2markdown_table(self.config, nested=True))
        for domain_train_data in train_data:
            self.tensorboard_logger.add_text('Configuration/data_{}'.format(domain_train_data.name), dict2markdown_table(domain_train_data.config))

        self._init_model(meta_dataset, train_data)

        self._init_parameter()

        # config callback
        self.run_mode = run_mode
        val_metrics = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff']
        self.val_check = val_data is not None and val_metrics is not None
        if val_data is not None:
            for _ in val_data:
                _.use_field = train_data[0].use_field
        if self.val_check:
            self.val_metric = next(iter(val_metrics)) if isinstance(val_metrics, list) else val_metrics
            cutoffs = cutoff if isinstance(cutoff, list) else [cutoff]
            if len(eval.get_rank_metrics(self.val_metric)) > 0:
                self.val_metric += '@' + str(cutoffs[0])
        self.callback = self._get_callback(meta_dataset.name)
        self.logger.info('save_dir:' + self.callback.save_dir)
        # refresh_rate = 0 if run_mode in ['light', 'tune'] else 1

        self.logger.info(self)

        self._accelerate()

        if self.config['train']['accelerator'] == 'ddp':
            mp.spawn(self.parallel_training, args=(self.world_size, train_data, val_data),
                     nprocs=self.world_size, join=True)
        else:
            self.trainloaders = self._get_train_loaders(train_data)

            if val_data:
                val_loader = [_.eval_loader(batch_size=self.config['eval']['batch_size']) for _ in val_data]
            else:
                val_loader = None
            self.optimizers = self._get_optimizers()
            self.fit_loop(val_loader)
        return self.callback.best_ckpt['metric']

    def training_epoch(self, nepoch):
        # if hasattr(self, "_update_item_vector"):
        #     for domain in self.UNIQUE_DOMAINS:
        #         self._update_item_vector()

        if hasattr(self, "sampler"):
            if hasattr(self.sampler, "update"):
                if hasattr(self, 'item_vector'):
                    # TODO: add frequency
                    self.sampler.update(item_embs=self.item_vector)
                else:
                    self.sampler.update(item_embs=None)
        output_list = []
        optimizers = self.current_epoch_optimizers(nepoch)

        trn_dataloaders, combine = self.current_epoch_trainloaders(nepoch)
        trn_dataloaders = [CrossDomainLoaders(trn_dataloaders, False)]

        if not (isinstance(optimizers, List) or isinstance(optimizers, Tuple)):
            optimizers = [optimizers]

        for loader_idx, loader in enumerate(trn_dataloaders):
            outputs = []
            loader = tqdm(
                loader,
                total=len(loader),
                ncols=75,
                desc=set_color(f"Training {nepoch:>5}", "green"),
                leave=False,
                disable=self.run_mode == 'tune', # Mute the progressbar when tuning
            )
            for batch_idx, batch in enumerate(loader):
                # data to device
                batch = self._to_device(batch, self._parameter_device)

                for opt in optimizers:
                    if opt is not None:
                        opt['optimizer'].zero_grad()
                # model loss
                training_step_args = {'batch': batch}
                if 'nepoch' in inspect.getargspec(self.training_step).args:
                    training_step_args['nepoch'] = nepoch
                if 'loader_idx' in inspect.getargspec(self.training_step).args:
                    training_step_args['loader_idx'] = loader_idx
                if 'batch_idx' in inspect.getargspec(self.training_step).args:
                    training_step_args['batch_idx'] = batch_idx

                if getattr(self, '_dp', False):     # there are perfermance degrades in DP mode
                    loss = data_parallel(self, 'training_step', inputs=None, module_kwargs=training_step_args, device_ids=self._gpu_list, output_device=self.device)
                else:
                    loss = self.training_step(**training_step_args)

                if isinstance(loss, dict):
                    if loss['loss'].requires_grad:
                        if isinstance(loss['loss'], torch.Tensor):
                            loss['loss'].backward()
                        elif isinstance(loss['loss'], List):
                            for l in loss['loss']:
                                l.backward()
                        else:
                            raise TypeError("loss must be Tensor or List of Tensor")
                    loss_ = {}
                    for k, v in loss.items():
                        if k == 'loss':
                            if isinstance(v, torch.Tensor):
                                v = v.detach()
                            elif isinstance(v, List):
                                v = [_ for _ in v]
                        loss_[f'{k}_{loader_idx}'] = v
                    outputs.append(loss_)
                elif isinstance(loss, torch.Tensor):
                    if loss.requires_grad:
                        loss.backward()
                    outputs.append({f"loss_{loader_idx}": loss.detach()})
                #
                for opt in optimizers:
                    if self.config['train']['grad_clip_norm'] is not None:
                        clip_grad_norm_(opt['optimizer'].params, self.config['train']['grad_clip_norm'])
                    #
                    if opt is not None:
                        opt['optimizer'].step()
            if len(outputs) > 0:
                output_list.append(outputs)
        return output_list

    def _token_id_remap(self, batch, dataset):
        mapping_dict = dataset.reverse_field2token2idx
        for field in mapping_dict:
            # original value for 0 is '[PAD]'
            mapping_dict[field][0] = 0
        def remap(x, *args):
            return field_mapping[x]
        for field, batch_data in batch.items():
            if field in mapping_dict:
                if field not in self.query_fields:
                    continue
                field_mapping = mapping_dict[field]
                batch[field] = batch_data.map_(batch_data, remap)
            elif field in self.query_fields and field[3:] in mapping_dict:
                # field[3:] means remove potential 'in_' prefix for Seq dataset 
                field_mapping = mapping_dict[field[3:]]
                batch[field] = batch_data.map_(batch_data, remap)
        return batch

    @torch.no_grad()
    def validation_epoch(self, nepoch, dataloaders):
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
        output_list = []
        self.eval_batch_splits = {}
        left_idx, right_idx = 0, 0
        for loader in dataloaders:
            dataset = loader.dataset
            domain = dataset.name
            self._update_item_vector(domain)
            loader = tqdm(
                loader,
                total=len(loader),
                ncols=75,
                desc=set_color(f"Evaluating {domain:>5} {nepoch:>5}", "green"),
                leave=False,
                disable=self.run_mode == 'tune', # Mute the progressbar when tuning
            )
            for domain_batch_num, batch in enumerate(loader):
                # remap local query related token id to meta token id, ensuring correct queries.
                # 'item_id' will not be remapped, as the evaluation is performed in domain context.
                batch = self._token_id_remap(batch, dataset)
                # data to device
                batch = self._to_device(batch, self._parameter_device)

                # model validation results
                output = self.validation_step(batch)

                # add domain information
                metrics = list(output[0].keys()) # TODO: this line should only be called once.
                for metric in metrics:
                    output[0][domain + '_' + metric] = output[0].pop(metric)

                output_list.append(output)
            right_idx = left_idx + domain_batch_num + 1
            self.eval_batch_splits[domain] = slice(left_idx, right_idx)
            left_idx = right_idx
            
        return output_list
    
    @torch.no_grad()
    def test_epoch(self, dataloaders):
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
        output_list = []
        self.eval_batch_splits = {}
        left_idx, right_idx = 0, 0
        for loader in dataloaders:
            dataset = loader.dataset
            domain = dataset.name
            self._update_item_vector(domain)
            for domain_batch_num, batch in enumerate(loader):
                # remap local token id to meta token id
                batch = self._token_id_remap(batch, dataset)
                # data to device
                batch = self._to_device(batch, self._parameter_device)

                # model validation results
                output = self.test_step(batch)

                # add domain information
                metrics = list(output[0].keys()) # TODO: this line should only be called once.
                for metric in metrics:
                    output[0][domain + '_' + metric] = output[0].pop(metric)

                output_list.append(output)
            right_idx = left_idx + domain_batch_num + 1
            self.eval_batch_splits[domain] = slice(left_idx, right_idx)
            left_idx = right_idx
            
        return output_list
    
    def _test_epoch_end(self, outputs, metrics):
        if isinstance(outputs[0][0], List):
            metric, bs = zip(*outputs)
            metric = torch.tensor(metric)
            bs = torch.tensor(bs)
            out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        elif isinstance(outputs[0][0], Dict):
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                for domain in self.eval_batch_splits:
                    if domain in k:
                        domain_bs = bs[self.eval_batch_splits[domain]]
                        out[k] = (metric * domain_bs).sum() / domain_bs.sum()
                        break
        return out

    def log_dict(self, metrics: Dict, tensorboard: bool=True):
        if tensorboard:
            for k, v in metrics.items():
                if 'train' in k:
                    self.tensorboard_logger.add_scalar(f"train/{k}", v, self.logged_metrics['epoch']+1)
                else:
                    self.tensorboard_logger.add_scalar(f"valid/{k}", v, self.logged_metrics['epoch']+1)
        self.logged_metrics.update(metrics)

    def forward(
            self,
            batch: Dict,
            full_score: bool = False,
            return_query: bool = False,
            return_item: bool = False,
            return_neg_item: bool = False,
            return_neg_id: bool = False
        ):
        output_list = {}
        for domain_name in self.SOURCE_DOMAINS:
            if batch.get(domain_name, None) == None:
                continue
            domain_batch = batch[domain_name]
            if hasattr(self.query_encoder, 'domain_user_embeddings'):
                # Update domain specific embedding
                self.query_encoder.user_embeddings = self.query_encoder.domain_user_embeddings[domain_name]
                self.item_encoder.item_embeddings = self.item_encoder.domain_item_embeddings[domain_name]
            self.sampler.set_domain(domain_name)
            domain_output = super().forward(domain_batch, full_score, return_query, return_item, return_neg_item, return_neg_id)
            output_list[domain_name] = domain_output
        return output_list

    def training_step(self, batch):
        outputs = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        loss_value = 0
        for domain in self.SOURCE_DOMAINS:
            if outputs.get(domain, None) == None:
                continue
            score = outputs[domain]['score']
            score['label'] = batch[domain][self.frating]
            loss_value += self.loss_fn(**score)
        return loss_value

    def _update_cross_domain_metrics(self, method='sum'):
        # Suppose all validation metrics are in test metrics
        domain_names = self.TARGET_DOMAINS
        domain_metrics = defaultdict(float)
        for metric_name, metric_value in self.logged_metrics.items():
            for domain in domain_names:
                if domain in metric_name:
                    domain_metrics[metric_name[len(domain) + 1:]] += metric_value
                    break
        self.logged_metrics.update(domain_metrics)

    def evaluate(self, test_data : List, verbose=True, **kwargs) -> Dict:
        r""" Predict for test data.

        Args:
            test_data(recstudio.data.Dataset): The dataset of test data, which is generated by RecStudio.

            verbose(bool, optimal): whether to show the detailed information.

        Returns:
            dict: dict of metrics. The key is the name of metrics.
        """
        for _ in test_data:
            _.drop_feat(self.fields)
        test_loaders = [_.eval_loader(batch_size=self.config['eval']['batch_size']) for _ in test_data]
        output = {}
        self.load_checkpoint(os.path.join(self.config['eval']['save_path'], self.ckpt_path))
        if 'config' in kwargs:
            self.config.update(kwargs['config'])
        self.eval()
        output_list = self.test_epoch(test_loaders)
        output.update(self.test_epoch_end(output_list))
        self._update_cross_domain_metrics()
        if verbose:
            self.logger.info(color_dict(output, self.run_mode == 'tune'))
        return output

    def fit_loop(self, val_dataloader=None):
        try:
            nepoch = 0
            for e in range(self.config['train']['epochs']):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch

                # training procedure
                tik_train = time.time()
                self.train()
                training_output_list = self.training_epoch(nepoch)
                tok_train = time.time()

                # validation procedure
                tik_valid = time.time()
                if self.val_check:
                    self.eval()
                    if nepoch % self.config['eval']['val_n_epoch'] == 0:
                        validation_output_list = self.validation_epoch(nepoch, val_dataloader)
                        self.validation_epoch_end(validation_output_list)
                        self._update_cross_domain_metrics()
                tok_valid = time.time()

                self.training_epoch_end(training_output_list)
                if self.config['train']['gpu'] is not None:
                    mem_reversed = torch.cuda.max_memory_reserved(self._parameter_device) / (1024**3)
                    mem_total = torch.cuda.mem_get_info(self._parameter_device)[1] / (1024**3)
                else:
                    mem_reversed = mem_total = 0
                self.logger.info("{} {:.5f}s. {} {:.5f}s. {} {:.2f}/{:.2f} GB".format(
                    set_color('Train time:', 'white', False), (tok_train-tik_train),
                    set_color('Valid time:', 'white', False), (tok_valid-tik_valid),
                    set_color('GPU RAM:', 'white', False), mem_reversed, mem_total
                ))
                # learning rate scheduler step
                optimizers = self.current_epoch_optimizers(e)
                if optimizers is not None:
                    for opt in optimizers:
                        if 'scheduler' in opt:
                            opt['scheduler'].step()

                # model is saved in callback when the callback return True.
                if nepoch % self.config['eval']['val_n_epoch'] == 0:
                    stop_training = self.callback(self, nepoch, self.logged_metrics)
                    if stop_training:
                        break

                nepoch += 1

            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        except KeyboardInterrupt:
            # if catch keyboardinterrupt in training, save the best model.
            if (self.config['train']['accelerator']=='ddp'):
                if (dist.get_rank()==0):
                    self.callback.save_checkpoint(nepoch)
                    self.ckpt_path = self.callback.get_checkpoint_path()
            else:
                self.callback.save_checkpoint(nepoch)
                self.ckpt_path = self.callback.get_checkpoint_path()

    def _init_model(self, meta_dataset : CrossDomainDataset, train_data, drop_unused_field=True):
        # ============= Register as a basic Recommender ===============
        self._set_data_field(meta_dataset)
        self.fields = meta_dataset.use_field
        self.frating = meta_dataset.frating
        assert self.frating in self.fields, 'rating field is required.'
        if drop_unused_field:
            for _ in train_data:
                _.drop_feat(self.fields)
        self.item_feat = meta_dataset.item_feat
        self.item_fields = set(meta_dataset.item_feat.fields).intersection(self.fields)
        self.neg_count = self.config['train']['negative_count']
        if self.loss_fn is None:
            if 'train_data' in inspect.signature(self._get_loss_func).parameters:
                self.loss_fn = self._get_loss_func(train_data)
            else:
                self.loss_fn = self._get_loss_func()
        # ============== Register as a basic Retriever ================
        self.query_fields = set(meta_dataset.user_feat.fields).intersection(meta_dataset.use_field)
        if isinstance(meta_dataset, UserDataset) or isinstance(meta_dataset, CrossDomainSeqDataset):
            self.query_fields = self.query_fields | set(["in_"+f for f in self.item_fields])
            if isinstance(meta_dataset, CrossDomainSeqDataset):
                self.query_fields = self.query_fields | set(['seqlen'])

        self.fiid = meta_dataset.fiid
        self.fuid = meta_dataset.fuid
        assert self.fiid in self.item_fields, 'item id is required to use.'

        # =============== Register as a CDR Retriever ================
        self.datasets = meta_dataset
        self.SOURCE_DOMAINS = meta_dataset.source_dataset_names
        self.TARGET_DOMAINS = meta_dataset.target_dataset_names
        self.UNIQUE_DOMAINS = meta_dataset.unique_dataset_names
        self.DOMAIN_USER_ID = {}
        self.DOMAIN_ITEM_ID = {}
        for domain_name in meta_dataset.unique_dataset_names:
            dataset = meta_dataset.unique_datasets(domain_name)

            domain_user_id = list(dataset.field2token2idx[self.fuid].keys())
            domain_user_id[0] = 0 # change [PAD] to 0
            self.DOMAIN_USER_ID[domain_name] = domain_user_id

            domain_item_id = list(dataset.field2token2idx[self.fiid].keys())
            domain_item_id[0] = 0 # change [PAD] to 0
            self.DOMAIN_ITEM_ID[domain_name] = domain_item_id

        # Register encoder and sampler
        self.item_encoder = self._get_item_encoder(meta_dataset) if not self.item_encoder else self.item_encoder
        self.query_encoder = self._get_query_encoder(meta_dataset) if not self.query_encoder else self.query_encoder
        self.sampler = self._get_sampler(meta_dataset) if not self.sampler else self.sampler

    def _get_item_vector(self, domain=None):
        if len(self.item_fields) == 1 and isinstance(self.item_encoder, torch.nn.Embedding):
            return self.item_encoder.weight[self.DOMAIN_ITEM_ID[domain][1:]]
        else:
            device = next(self.parameters()).device
            output = [self.item_encoder(self._get_item_feat(self._to_device(batch, device)))
                      for batch in self.item_feat.loader(batch_size=self.config['train'].get('item_batch_size', 1024))]
            output = torch.cat(output, dim=0)
            return output[self.DOMAIN_ITEM_ID[domain][1:]]

    def _update_item_vector(self, domain=None):
        if domain != None:
            # Called when valid/test, the item_vector belongs to only one domain
            item_vector = self._get_item_vector(domain)
        else:
            # Called when training, the item_vector belongs to all domains
            item_vector = super()._get_item_vector()
        if not hasattr(self, "item_vector"):
            self.register_buffer('item_vector', item_vector.detach().clone() if isinstance(item_vector, torch.Tensor) \
                else item_vector.copy())
        else:
            self.item_vector = item_vector

        if self.use_index:
            self.ann_index = self.build_ann_index()

    def _get_train_loaders(self, train_data : List, ddp=False) -> List:
        return [
            _.train_loader(
                batch_size=self.config['train']['batch_size'],
                shuffle=True,
                drop_last=False,
                ddp=ddp
            ) for _ in train_data
        ]