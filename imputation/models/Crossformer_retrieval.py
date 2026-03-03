import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    TwoStageAttentionLayer,
)
from models.PatchTST import FlattenHead
from layers.RevIN import RevIN
from models.freeze_backbone import Freeze_Backbone
from math import ceil


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
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (configs.e_layers - 1))
        )
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        self.enc_value_embedding = PatchEmbedding(
            configs.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - configs.seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    configs,
                    1 if l == 0 else self.win_size,
                    configs.d_model,
                    configs.n_heads,
                    configs.d_ff,
                    1,
                    configs.dropout,
                    (
                        self.in_seg_num
                        if l == 0
                        else ceil(self.in_seg_num / self.win_size**l)
                    ),
                    configs.factor,
                )
                for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model
            )
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        configs,
                        (self.pad_out_len // self.seg_len),
                        configs.factor,
                        configs.d_model,
                        configs.n_heads,
                        configs.d_ff,
                        configs.dropout,
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
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
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

        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc.shape[0],
        )
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

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

            enc_out = self.revin(enc_out, mode="norm", mask=None)
            reference = self.ref_revin(reference, mode="norm")
            gate_weight = self.gate(enc_out)
            fuse_x = gate_weight * enc_out + (1 - gate_weight) * reference
            # fuse_x = self.layer_norm(fuse_x)

            # ffn_x = self.ffn(fuse_x)
            # ffn_output = self.ffn_norm(ffn_x + fuse_x)
            output = fuse_x + self.projection_refine(fuse_x)
            output = self.revin(output, mode="denorm", mask=None)

            return output

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
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
