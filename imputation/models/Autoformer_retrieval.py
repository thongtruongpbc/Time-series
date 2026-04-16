import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, TokenEmbedding
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)
from layers.RevIN import RevIN
from layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    SampleWiseAttention,
)

from models.freeze_backbone import Freeze_Backbone
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


class TopKTimestepMask:
    def __init__(self, B, T, top_k, device):
        q_idx = torch.arange(T, device=device).view(1, 1, T, 1)
        k_idx = torch.arange(T, device=device).repeat(top_k).view(1, 1, 1, top_k * T)

        mask = q_idx != k_idx

        self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.top_k = configs.k
        self.d_cond = 16
        self.null_ref_feature = nn.Parameter(torch.randn(1, 1, 1, self.top_k))

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.ref_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.ref_projection = nn.Linear(configs.enc_in, configs.d_model, bias=False)
        self.concat_projection = nn.Linear(
            self.top_k * configs.d_model, configs.d_model, bias=False
        )

        # multihead self-attention
        self.multihead_attn = torch.nn.MultiheadAttention(
            configs.d_model, configs.n_heads, dropout=0.1
        )
        # cross attention
        self.cross_attn = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False,
            ),
            configs.d_model,
            configs.n_heads,
        )
        # samplewise attention
        self.samplewise_attention = SampleWiseAttention(
            T=self.seq_len, D=configs.d_model, d_model=configs.d_model
        )
        self.attn_norm = nn.LayerNorm(configs.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
            nn.Dropout(configs.dropout),
        )
        self.norm2 = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
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
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # Freeze backbone
        self.freeze_model = Freeze_Backbone(configs)
        if (
            self.configs.ablation_arch == "freeze-backbone-retrieval"
            or self.configs.ablation_arch
            == "freeze-backbone-retrieval + learnable fusing"
            or self.configs.ablation_arch == "backbone-retrieval + learnable fusing"
        ):
            self.gate = nn.Sequential(
                nn.Linear(configs.c_out, configs.c_out), nn.Sigmoid()
            )

            self.ffn = nn.Sequential(
                nn.Linear(configs.c_out, 4 * configs.c_out),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(4 * configs.c_out, configs.c_out),
                nn.Dropout(configs.dropout),
            )
            self.projection_refine = nn.Sequential(
                nn.LayerNorm(configs.enc_in),
                nn.GELU(),
                nn.Linear(configs.enc_in, configs.enc_in),
                nn.Dropout(configs.dropout),
                nn.GELU(),
                nn.Linear(configs.enc_in, configs.enc_in),
                nn.Dropout(configs.dropout),
            )

            self.layer_norm = nn.LayerNorm(configs.c_out)
            self.layer_norm_1 = nn.LayerNorm(configs.d_model)
            self.ffn_norm = nn.LayerNorm(configs.c_out)
            self.fuse_projection = nn.Linear(configs.c_out, configs.c_out)

        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            )
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "imputation_retrieval":
            self.projection = nn.LazyLinear(configs.c_out)

        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation_retrieval(
        self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=1
    ):
        if self.configs.ablation_arch == "Linear fuse":
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.view(B * top_k, T, C)
            reference = self.ref_revin(reference, mode="norm", mask=mask)
            # reference = self.ref_embedding(reference, x_mark_enc.repeat_interleave(top_k, dim=0))
            reference = reference.reshape(B, top_k, T, reference.shape[-1])
            reference = reference.reshape(B, top_k * T, reference.shape[-1])
            # reference, ref_attns = self.ref_encoder(reference, attn_mask=None)
            # reference = reference.reshape(B, top_k * T, reference.shape[-1])
            ref_proj = self.ref_projection(reference)
            ref_proj = ref_proj.reshape(B, top_k, T, ref_proj.shape[-1])
            ref_proj = ref_proj.reshape(B, T, top_k * ref_proj.shape[-1])

            concat_proj = self.concat_projection(ref_proj)

            # input
            x_enc = self.revin(x_enc, mode="norm", mask=mask)
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            # enc_out = self.enc_embedding(x_enc)

            enc_out, attns = self.encoder(enc_out)

            x_inp = torch.cat([enc_out, concat_proj], dim=2)

            output = self.projection(x_inp)
            output = self.revin(output, mode="denorm", mask=mask)
            return output

        elif (
            self.configs.ablation_arch == "freeze-backbone-retrieval + learnable fusing"
        ):
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.mean(dim=1)
            enc_out = self.freeze_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            enc_out = x_enc * mask + enc_out * (1 - mask)

            enc_out = self.revin(enc_out, mode="norm")
            reference = self.ref_revin(reference, mode="norm")
            gate_weight = self.gate(enc_out)
            fuse_x = gate_weight * enc_out + (1 - gate_weight) * reference
            # fuse_x = self.layer_norm(fuse_x)

            # ffn_x = self.ffn(fuse_x)
            # ffn_output = self.ffn_norm(ffn_x + fuse_x)
            output = fuse_x + self.projection_refine(fuse_x)
            output = self.revin(output, mode="denorm")

            return output

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
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
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "imputation_retrieval":
            dec_out = self.imputation_retrieval(
                x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=training
            )
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
