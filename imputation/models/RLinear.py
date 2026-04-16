import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = (
            nn.ModuleList(
                [
                    nn.Linear(configs.seq_len, configs.seq_len)
                    for _ in range(configs.channel)
                ]
            )
            if configs.individual
            else nn.Linear(configs.seq_len, configs.seq_len)
        )

        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in)
        self.individual = configs.individual

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [B, L, D]
        x = self.rev(x_enc, "norm", mask=mask)
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(x)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, "denorm", mask=mask)

        return pred
