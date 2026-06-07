import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class RetrievalTool:
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=1,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period :]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key

    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[2])

            if self.with_dec:
                y_data_all.append(
                    td[2][-(train_data.pred_len + train_data.label_len) :]
                )
            else:
                y_data_all.append(td[2][-train_data.pred_len :])

        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)

        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all)  # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)

            mg.append(cur)
        #             data_all = data_all - cur

        mg = torch.stack(mg, dim=0)  # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:, -1:, :]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None

        offset = torch.stack(offset, dim=0)

        return mg, offset

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            cur_sim = torch.bmm(
                F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2)
            )
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)

        return sim

    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels, "Mismatched shape"

        x_mg, _ = self.decompose_mg(x)  # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),  # G, T, S*C
            x_mg.flatten(start_dim=2),  # G, B, S*C
        )  # G, B, T

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(
                x.device
            )
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (
                index - self.seq_len - self.pred_len + 1
            ).unsqueeze(dim=1)
            sliding_index = sliding_index.clamp(0, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train), device=x.device).scatter_(
                1, sliding_index, 1.0
            )
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            sim = sim.masked_fill_(self_mask.bool(), float("-inf"))  # G, B, T

        sim_reshaped = sim.reshape(self.n_period * bsz, self.n_train)  # G*B, T
        topm_index = torch.topk(sim_reshaped, self.topm, dim=1).indices  # G*B, M
        topm_index = topm_index.reshape(self.n_period, bsz, self.topm)  # G, B, M

        return topm_index

    def retrieve_all(self, data, train=False, device=torch.device("cpu")):
        assert self.train_data_all_mg is not None

        rt_loader = DataLoader(
            data, batch_size=1024, shuffle=False, num_workers=8, drop_last=False
        )

        all_indices = []
        with torch.no_grad():
            for index, _, batch_x, _, _, _ in tqdm(rt_loader):
                batch_indices = self.retrieve(
                    batch_x.float().to(device), index, train=train
                )  # G, B, M
                all_indices.append(batch_indices.cpu())

        all_indices = torch.cat(all_indices, dim=1)  # G, N_total, M
        return all_indices
