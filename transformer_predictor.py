from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn, optim
import pytorch_lightning as pl

from cosine_warmup_scheduler import CosineWarmupScheduler
from position_encoding import PositionalEncoding
from transformer import TransformerEncoder


class TransformerPredictor(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0,
                 input_dropout=0.0, use_pytorch_transformer=True):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        if self.hparams.use_pytorch_transformer:
            print('Using pytorch transformer')
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hparams.model_dim,
                                                                                nhead=self.hparams.num_heads,
                                                                                dim_feedforward=2 * self.hparams.model_dim,
                                                                                dropout=self.hparams.dropout,
                                                                                batch_first=True),
                                                     num_layers=self.hparams.num_layers)
        else:
            self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                                  input_dim=self.hparams.model_dim,
                                                  dim_feedforward=2 * self.hparams.model_dim,
                                                  num_heads=self.hparams.num_heads,
                                                  dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps_pytorch(self, x, mask=None):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        This is when using the pytorch transformers library
        """

        print('Getting pytorch transformer attention maps')

        # Attention maps for each layer
        attention_maps = []

        class SelfAttentionWrapper(nn.MultiheadAttention):
            """
            Wrapper for the self attention layer that captures the attention matrices into the attention_maps array
            """
            def __init__(self, _self_attn):
                super().__init__(embed_dim=1, num_heads=1)  # Not going to be used
                self._self_attn = _self_attn

            def forward(self,
                        query: Tensor,
                        key: Tensor,
                        value: Tensor,
                        key_padding_mask: Optional[Tensor] = None,
                        need_weights: bool = True,
                        attn_mask: Optional[Tensor] = None,
                        average_attn_weights: bool = True,
                        is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                # Put it through the attention layer that we got in construction
                output = self._self_attn(query, key, value, key_padding_mask=key_padding_mask,
                                                need_weights=True, attn_mask=attn_mask, average_attn_weights=False,
                                                is_causal=is_causal)
                attention_maps.append(output[1])  # output is (x, attention_weights)
                return output

            @property
            def self_attn(self):
                return self._self_attn

        # Iterate over the layers and put our wrapper to capture the attention matrices
        for layer in self.transformer.layers:
            # Wrap the self attention and assign it
            self_attention_wrapper = SelfAttentionWrapper(layer.self_attn)
            layer.self_attn = self_attention_wrapper
            x = layer(x)
            layer.self_attn = self_attention_wrapper.self_attn  # Restore the original self attention

        return attention_maps

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)

        if self.hparams.use_pytorch_transformer:
            return self.get_attention_maps_pytorch(x, mask=mask)

        # Get the attention maps from the transformer.py implementation
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
