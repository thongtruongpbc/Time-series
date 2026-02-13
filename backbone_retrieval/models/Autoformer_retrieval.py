import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.RevIN import RevIN
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        self.d_cond = 16
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
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
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=False)
            self.ref_projection = nn.Linear(configs.c_out, configs.c_out, bias=False)
            self.ref_concat = nn.Linear(
                self.top_k * configs.c_out, configs.c_out, bias=False
            )
            self.cond_embed = nn.Embedding(2, self.d_cond)

            self.both = nn.Linear(
                2 * configs.c_out + self.d_cond, configs.c_out, bias=False
            )

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

    # def imputation_retrieval(self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask):
    ### Visualization ###
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import sys

    # b_idx, c_idx = 0, 0
    # K = reference.shape[1]

    # orig_val = x_enc[b_idx, :, c_idx].detach().cpu().numpy()
    # mask_val = mask[b_idx, :, c_idx].detach().cpu().numpy()

    # masked_val = orig_val.copy()
    # masked_val[mask_val == 0] = np.nan

    # plt.figure(figsize=(15, 7))

    # colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # for k in range(min(K, 2)):
    #     ref_k = reference[b_idx, k, :, c_idx].detach().cpu().numpy()
    #     plt.plot(ref_k, color=colors[k], alpha=0.5, linewidth=1.2, label=f'Ref K={k}')

    # plt.plot(orig_val, color='#1f77b4', alpha=0.2, linewidth=1, label='Ground Truth')
    # plt.plot(masked_val, color='#1f77b4', alpha=1.0, linewidth=2, label='Input')

    # diff = np.diff(np.concatenate([[1], mask_val, [1]]))
    # starts = np.where(diff == -1)[0]
    # ends = np.where(diff == 1)[0]
    # for s, e in zip(starts, ends):
    #     plt.axvspan(s-0.5, e-0.5, color='gray', alpha=0.1)

    # plt.title(f"Check B:{b_idx} C:{c_idx} K:{K}")
    # plt.legend()
    # plt.savefig("imputation_debug.png")
    # plt.close()

    # sys.exit()
    ### Kết thúc Visualization ###

    def imputation_retrieval(
        self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=1
    ):
        # reference
        # dec_out = self.projection(enc_out)
        B, top_k, T, C = reference.shape
        print(reference.shape)
        reference = reference.view(B * top_k, T, C)
        reference = self.ref_revin(reference, mode="norm", mask=None)
        reference = self.enc_embedding(reference, x_mark_enc)
        # reference, ref_attns = self.ref_encoder(reference, attn_mask=None)

        # input
        x_enc = self.revin(x_enc, mode="norm", mask=None)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.projection(enc_out)

        x_inp, cross_attns = self.cross_encoder(enc_out, reference, attn_mask=None)

        # x_inp = torch.cat([enc_out, reference], dim=1)

        # x_inp = self.cross_embedding(x_inp)
        # attn_output, attn_output_weights = self.multihead_attn(query=x_inp, key=x_inp, value=x_inp)
        # attn_output, attn_output_weights = self.encoder(x_inp, attn_mask=None)
        # x_inp = self.attn_norm(x_inp + attn_output)
        x_inp = self.ffn_norm(x_inp + self.ffn(x_inp))

        output = self.revin(output, mode="denorm", mask=mask)
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
