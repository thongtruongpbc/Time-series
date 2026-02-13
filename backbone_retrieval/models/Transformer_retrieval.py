import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
    CrossEncoderLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer, RoPEAttention
from layers.Embed import DataEmbedding, TokenEmbedding, DataEmbedding_wo_pos
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))

        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        self.multihead_attn = nn.MultiheadAttention(
            configs.d_model, num_heads=1, dropout=0.2
        )
        self.attn_norm = nn.LayerNorm(configs.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
            nn.Dropout(configs.dropout),
        )

        self.ffn_norm = nn.LayerNorm(configs.d_model)
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.cross_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        Encoder

        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 RoPEAttention(configs.seq_len, configs.d_model, configs.n_heads, False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

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

        # ref_encoder
        self.ref_encoder = Encoder(
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

        # cross encoder
        self.cross_encoder = CrossEncoderLayer(
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

        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
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
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            )
        if self.task_name == "imputation" or self.task_name == "imputation_retrieval":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        B = x_enc.size(0)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        enc_out = torch.cat([cls_token, enc_out], dim=1)  # [B, L+1, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        e_out, cls_out = (
            enc_out[:, 1:, :],
            enc_out[:, 0, :],
        )  # [B, L, d_model], [B, 1, d_model]

        dec_out = self.projection(e_out)
        return dec_out, cls_out

    def imputation_retrieval(
        self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=1
    ):
        # reference
        # dec_out = self.projection(enc_out)
        B, top_k, T, C = reference.shape
        reference = reference.view(B * top_k, T, C)
        reference = self.ref_revin(reference, mode="norm", mask=None)
        reference = self.enc_embedding(reference, x_mark_enc)
        reference, ref_attns = self.ref_encoder(reference, attn_mask=None)

        # input
        x_enc = self.revin(x_enc, mode="norm", mask=None)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        x_inp, cross_attns = self.cross_encoder(enc_out, reference, attn_mask=None)

        # x_inp = torch.cat([enc_out, reference], dim=1)

        # x_inp = self.cross_embedding(x_inp)
        # attn_output, attn_output_weights = self.multihead_attn(query=x_inp, key=x_inp, value=x_inp)
        # attn_output, attn_output_weights = self.encoder(x_inp, attn_mask=None)
        # x_inp = self.attn_norm(x_inp + attn_output)
        x_inp = self.ffn_norm(x_inp + self.ffn(x_inp))

        output = self.projection(x_inp[:, :T, :])
        output = self.revin(output, mode="denorm", mask=mask)
        return output

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(
        self,
        x_enc,
        x_mark_enc,
        reference=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        training=1,
    ):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out, cls_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask
            )
            return dec_out, cls_out  # [B, L, D], [B, 1, D]
        if self.task_name == "imputation_retrieval":
            dec_out = self.imputation_retrieval(
                x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=training
            )
            return dec_out
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
