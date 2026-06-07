import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN
from models.freeze_backbone import Freeze_Backbone


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # RevIN
        self.revin = RevIN(num_features=configs.enc_in)
        self.ref_revin = RevIN(num_features=configs.enc_in)

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        # Frezee backbone
        self.freeze_model = Freeze_Backbone(configs)

        # Adapter
        self.gate = nn.Sequential(nn.Linear(configs.c_out, configs.c_out), nn.Sigmoid())

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

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        if self.task_name == "classification":
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class
            )

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation_retrieval(
        self, x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask, training=1
    ):
        # Encoder
        # return self.encoder(x_enc)
        B, top_k, T, C = reference.shape
        reference = reference.mean(dim=1)
        enc_out = self.freeze_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        # print(f"x_enc type: {type(x_enc)}, mask type: {type(mask)}")
        enc_out = x_enc * mask + enc_out * (1 - mask)

        enc_out = self.revin(enc_out, mode="norm", mask=mask)
        reference = self.ref_revin(reference, mode="norm")
        gate_weight = self.gate(enc_out)
        fuse_x = gate_weight * enc_out + (1 - gate_weight) * reference
        # fuse_x = self.layer_norm(fuse_x)

        # ffn_x = self.ffn(fuse_x)
        # ffn_output = self.ffn_norm(ffn_x + fuse_x)
        output = fuse_x + self.projection_refine(fuse_x)
        output = self.revin(output, mode="denorm", mask=mask)

        return output

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
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
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation_retrieval":
            dec_out = self.imputation_retrieval(
                x_enc, x_mark_enc, reference, x_dec, x_mark_dec, mask=mask
            )
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
