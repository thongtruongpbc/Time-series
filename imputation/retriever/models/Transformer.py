import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, TokenEmbedding
from layers.RevIN import RevIN
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))

        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        # Embedding
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.enc_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.enc_embedding = DataEmbedding(1, configs.d_model)  # univariate

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_enc, x_mark_enc, mask=None):
        # Embedding
        B = x_enc.size(0)
        # x_enc = self.revin(x_enc, mode="norm")
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # univariate
        B, T, C = x_enc.shape
        x_enc = x_enc.transpose(1, 2).reshape(B * C, T, 1)  # [B*C, T, 1]
        x_mark_enc = x_mark_enc.repeat_interleave(C, dim=0)
        if mask is not None:
            mask = mask.transpose(1, 2).reshape(B * C, 1, 1, T)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=mask)
        enc_out, attns = self.encoder(enc_out, attn_mask=mask)  # [B*C, T, d_model]
        enc_out = enc_out.reshape(B, C, T, -1).transpose(1, 2)  # [B, T, C, d_model]
        enc_out_reshaped = enc_out.reshape(B, T, C * self.d_model)
        return enc_out, enc_out_reshaped
