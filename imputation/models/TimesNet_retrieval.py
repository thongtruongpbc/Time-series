import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding, TokenEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.RevIN import RevIN
from layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    SampleWiseAttention,
)
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    CrossEncoder,
    EncoderLayer,
    ConvLayer,
    CrossEncoderLayer,
)
from models.freeze_backbone import Freeze_Backbone
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


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, 3)  # self.k

        res = []
        for i in range(3):  # self.k
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        self.d_cond = 16
        self.fuse_rate = configs.fuse_rate

        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )

        self.ref_model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        # self.enc_embedding = DataEmbedding(
        #     configs.enc_in,
        #     configs.d_model,
        #     configs.embed,
        #     configs.freq,
        #     configs.dropout,
        # )

        # ref embedding
        # self.cross_embedding = DataEmbedding(
        #     configs.enc_in,
        #     configs.d_model,
        #     configs.embed,
        #     configs.freq,
        #     configs.dropout,
        # )
        self.enc_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.cross_embedding = TokenEmbedding(configs.enc_in, configs.d_model)

        self.ref_projection = nn.Linear(configs.enc_in, configs.d_model, bias=False)
        self.concat_projection = nn.Linear(
            configs.top_k * configs.d_model, configs.d_model, bias=False
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
        self.attn_norm = nn.ModuleList(
            [nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)]
        )  # nn.LayerNorm(configs.d_model)

        # cross attention in each layer
        self.cross_attn = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    configs.d_model,
                    configs.n_heads,
                )
                for _ in range(configs.e_layers)
            ]
        )

        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_ff),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_ff, configs.d_model),
                    nn.Dropout(configs.dropout),
                )
                for _ in range(configs.e_layers)
            ]
        )

        self.ffn_norm = nn.ModuleList(
            [nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)]
        )  # nn.LayerNorm(configs.d_model)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Freeze backbone
        Freeze = Freeze_Backbone(configs)
        self.freeze_model = Freeze
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

        self.ref_layer_norm = nn.ModuleList(
            [nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)]
        )  # nn.LayerNorm(configs.d_model)

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "imputation_retrieval":
            self.projection = nn.LazyLinear(configs.c_out)

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def imputation_retrieval(
        self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=1
    ):
        if self.configs.ablation_arch == "Linear fuse":
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.view(B * top_k, T, C)
            reference = self.ref_revin(reference, mode="norm")
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
            x_enc = self.revin(x_enc, mode="norm", mask=None)
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            # enc_out = self.enc_embedding(x_enc)
            for i in range(self.layer):
                enc_out = self.layer_norm(self.model[i](enc_out))

            x_inp = torch.cat([enc_out, concat_proj], dim=2)

            output = self.projection(x_inp)
            output = self.revin(output, mode="denorm", mask=mask)
            return output
        elif self.configs.ablation_arch == "Linear fuse + Cross-attention":
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.view(B * top_k, T, C)
            reference = self.ref_revin(reference, mode="norm")
            reference = reference.reshape(
                B, top_k, T, reference.shape[-1]
            )  # reference.shape[-1] = d_model
            reference = reference.reshape(B, top_k * T, reference.shape[-1])
            ref_proj = self.ref_projection(reference)

            # input
            x_enc = self.revin(x_enc, mode="norm", mask=None)
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            # enc_out = self.enc_embedding(x_enc)
            for i in range(self.layer):
                enc_out = self.layer_norm(self.model[i](enc_out))

            # cross attn
            attn_output, attn_output_weights = self.cross_attn(
                queries=enc_out, keys=ref_proj, values=ref_proj, attn_mask=None
            )

            # add & norm
            x_out = self.attn_norm(enc_out + attn_output)

            # ffn + norm
            x_out = self.ffn_norm(x_out + self.ffn(x_out))

            output = self.projection(x_out)
            output = self.revin(output, mode="denorm", mask=mask)
            return output
        elif self.configs.ablation_arch == "Samplewise-attention":
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.view(B * top_k, T, C)
            reference = self.ref_revin(reference, mode="norm")
            # reference = self.ref_embedding(reference, x_mark_enc.repeat_interleave(top_k, dim=0))
            reference = reference.reshape(B, top_k, T, reference.shape[-1])
            reference = reference.reshape(B, top_k * T, reference.shape[-1])
            # reference, ref_attns = self.ref_encoder(reference, attn_mask=None)
            # reference = reference.reshape(B, top_k * T, reference.shape[-1])
            ref_proj = self.ref_projection(reference)
            ref_proj = ref_proj.reshape(B, top_k, T, ref_proj.shape[-1])

            # input
            x_enc = self.revin(x_enc, mode="norm", mask=None)
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            # enc_out = self.enc_embedding(x_enc)
            for i in range(self.layer):
                enc_out = self.layer_norm(self.model[i](enc_out))

            attn_output, attn_output_weights = self.samplewise_attention(
                enc_out, ref_proj
            )
            x_inp = self.attn_norm(enc_out + attn_output)
            output = self.projection(x_inp)
            output = self.revin(output, mode="denorm", mask=mask)
            return output

        elif (
            self.configs.ablation_arch
            == "(Token embedding) Two-branch backbone + Cross-attention"
        ):
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.view(B * top_k, T, C)
            reference = self.ref_revin(reference, mode="norm")
            # reference = reference.reshape(B, top_k, T, reference.shape[-1]) # reference.shape[-1] = d_model
            ref_out = self.cross_embedding(reference)
            # encoder
            for i in range(self.layer):
                ref_out = self.layer_norm[0](self.model[i](ref_out))

            ref_out = ref_out.reshape(B, top_k, T, ref_out.shape[-1])
            ref_out = ref_out.reshape(B, top_k * T, ref_out.shape[-1])

            # input
            x_enc = self.revin(x_enc, mode="norm", mask=None)
            enc_out = self.enc_embedding(x_enc)
            # enc_out = self.enc_embedding(x_enc)

            custom_mask = TopKTimestepMask(B, T, top_k, device=enc_out.device)
            for i in range(self.layer):
                enc_out = self.layer_norm[0](self.model[i](enc_out))
                attn_output, attn_output_weights = self.cross_attn[i](
                    queries=enc_out, keys=ref_out, values=ref_out, attn_mask=custom_mask
                )
                enc_out = self.attn_norm[0](enc_out + attn_output)
                enc_out = self.ffn_norm[0](enc_out + self.ffn[0](enc_out))

            # #cross attn
            # attn_output, attn_output_weights = self.cross_attn(queries=enc_out, keys=ref_out, values=ref_out, attn_mask=None)

            # #add & norm
            # x_out = self.attn_norm(enc_out + attn_output)

            # # ffn + norm
            # x_out = self.ffn_norm(x_out + self.ffn(x_out))

            output = self.projection(enc_out)
            output = self.revin(output, mode="denorm", mask=mask)
            return output

        elif self.configs.ablation_arch == "freeze-backbone-retrieval":
            # reference
            # dec_out = self.projection(enc_out)
            B, top_k, T, C = reference.shape
            reference = reference.mean(dim=1)
            enc_out = self.freeze_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            enc_out = x_enc * mask + enc_out * (1 - mask)

            enc_out = self.revin(enc_out, mode="norm", mask=None)
            reference = self.ref_revin(reference, mode="norm")
            fuse_x = self.fuse_rate * enc_out + (1 - self.fuse_rate) * reference
            fuse_x = self.layer_norm(fuse_x)

            # ffn_x = self.ffn(fuse_x)
            # ffn_output = self.ffn_norm(ffn_x + fuse_x)
            output = fuse_x + self.projection_refine(fuse_x)
            output = self.revin(output, mode="denorm", mask=None)

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

            enc_out = self.revin(enc_out, mode="norm", mask=mask)
            reference = self.ref_revin(reference, mode="norm")
            gate_weight = self.gate(enc_out)
            fuse_x = gate_weight * enc_out + (1 - gate_weight) * reference

            output = fuse_x + self.projection_refine(fuse_x)
            output = self.revin(output, mode="denorm", mask=mask)

            return output

        # elif self.configs.ablation_arch == "backbone-retrieval + learnable fusing":
        #     # reference
        #     # dec_out = self.projection(enc_out)
        #     B, top_k, T, C = reference.shape
        #     reference = reference.mean(dim=1)
        #     # input
        #     x_enc = self.revin(x_enc, mode="norm", mask=None)
        #     enc_out = self.enc_embedding(x_enc)
        #     # enc_out = self.enc_embedding(x_enc)
        #     for i in range(self.layer):
        #         enc_out = self.layer_norm_1(self.model[i](enc_out))

        #     enc_out = self.projection(enc_out)
        #     enc_out = x_enc * mask + enc_out * (1 - mask)

        #     enc_out = self.revin(enc_out, mode="norm", mask=None)
        #     reference = self.ref_revin(reference, mode="norm")
        #     gate_weight = self.gate(enc_out)
        #     fuse_x = gate_weight * enc_out + (1 - gate_weight) * reference
        #     # fuse_x = self.layer_norm(fuse_x)

        #     # ffn_x = self.ffn(fuse_x)
        #     # ffn_output = self.ffn_norm(ffn_x + fuse_x)
        #     output = fuse_x + self.projection_refine(fuse_x)
        #     output = self.revin(output, mode="denorm", mask=None)

        #     return output

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

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
            return dec_out
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
