import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import (
    TSEncoder,
    IntAttention,
    PatchSampling,
    CointAttention,
)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False),
        )

        if self.task_name == "imputation" or self.task_name == "imputation_retrieval":
            self.projection = nn.Linear(
                configs.num_p * configs.d_model, configs.seq_len, bias=True
            )
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def layers_init(self, configs):
        integrated_attention = [
            IntAttention(
                TSMixer(
                    ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                stable_len=configs.stable_len,
                activation=configs.activation,
                stable=True,
                enc_in=self.c_in,
            )
            for i in range(configs.ia_layers)
        ]

        patch_sampling = [
            PatchSampling(
                TSMixer(
                    ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                stable=False,
                stable_len=configs.stable_len,
                in_p=self.num_p if i == 0 else configs.num_p,
                out_p=configs.num_p,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for i in range(configs.pd_layers)
        ]

        cointegrated_attention = [
            CointAttention(
                TSMixer(
                    ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                stable=False,
                enc_in=self.c_in,
                stable_len=configs.stable_len,
            )
            for i in range(configs.ca_layers)
        ]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (
            x_enc.mean(1, keepdim=True).detach(),
            x_enc.std(1, keepdim=True).detach(),
        )
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, : self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)

        return dec_out * std + mean

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)[0][:, : self.c_in, :, :]
        dec_out = enc_out.flatten(start_dim=-2)
        # final
        dec_out = self.projection(dec_out)
        dec_out = dec_out.transpose(-1, -2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation" or self.task_name == "imputation_retrieval":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
