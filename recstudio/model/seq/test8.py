from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

class NoiseTransformerLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src:Tensor, src_mask=None, src_key_padding_mask=None, is_causal=False, noise=0) -> Tensor:
        out = super().forward(src, src_mask, src_key_padding_mask, is_causal)
        if noise > 0:
            random_noise = torch.rand_like(out, device=out.device)
            out = out + torch.sign(out) * torch.nn.functional.normalize(random_noise, dim=-1) * noise
        return out

class NoiseTransformer(torch.nn.TransformerEncoder):
    def forward(self, src: Tensor, mask = None, src_key_padding_mask = None, is_causal = None, noise=0) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, noise=noise)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output

class NoiseSASRecQueryEncoder(SASRecQueryEncoder):
    def __init__(self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__(fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional, training_pooling_type, eval_pooling_type)
        transformer_encoder = NoiseTransformerLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = NoiseTransformer(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )

class Test8(SASRec):
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.MyAugmentation(self.config['model'], train_data)
        self.augmentation_model2 = data_augmentation.SpectralAugmentation(self.config['model'], train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return NoiseSASRecQueryEncoder(
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
        cl_output = self.augmentation_model(batch, self.item_encoder.weight)
        cl_output2 = self.augmentation_model2(batch, self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['model']['gcl_weight'] * cl_output['cl_loss'] \
           + self.config['model']['cl_weight'] * cl_output2['cl_loss']
        return loss_value
