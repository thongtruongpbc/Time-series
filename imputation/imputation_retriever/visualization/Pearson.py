import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from imputation_retriever.data_provider.data_factory import data_provider
from imputation_retriever.layers.Retrieval import RetrievalTool

warnings.filterwarnings("ignore")


class Retriever_visualization:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retrieval_indices = {}
        self.window_bank = None

    def prepare_retrieval_indices(self, train_data, vali_data, test_data):
        rt = RetrievalTool(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            channels=self.args.enc_in,
            n_period=1,
            topm=self.args.topm,
        )
        rt.prepare_dataset(train_data)
        self.retrieval_indices["train"] = rt.retrieve_all(
            train_data, train=True, device=self.device
        ).cpu()
        self.retrieval_indices["val"] = rt.retrieve_all(
            vali_data, train=False, device=self.device
        ).cpu()
        if test_data is not None:
            self.retrieval_indices["test"] = rt.retrieve_all(
                test_data, train=False, device=self.device
            ).cpu()

        raw_x = torch.from_numpy(train_data.data_x).float()
        S = self.args.seq_len
        self.window_bank = raw_x.unfold(0, S, 1).transpose(1, 2)
        del rt
        torch.cuda.empty_cache()

    def _get_data(self, flag, shuffle_override=None):
        data_set, data_loader = data_provider(
            self.args, flag, shuffle_override=shuffle_override
        )
        return data_set, data_loader

    def visualize(
        self,
        batch_x,
        inp,
        batch_neg_candidates,
        mask,
        batch_idx,
        sample_idx,
        channel=0,
        save_dir="plots",
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gt = batch_x[batch_idx, :, channel].detach().cpu().numpy()
        negatives = (
            batch_neg_candidates[batch_idx, :, :, channel].detach().cpu().numpy()
        )
        m = mask[batch_idx, :, channel].detach().cpu().numpy()

        S = gt.shape[0]
        M = negatives.shape[0]
        x = np.arange(S)

        fig, axes = plt.subplots(M + 1, 1, figsize=(14, 3 * (M + 1)), sharex=True)

        axes[0].plot(x, gt, color="gray", alpha=0.3, label="Original")
        q_visible = np.ma.masked_where(m == 0, gt)
        axes[0].plot(x, q_visible, color="#0000FF", linewidth=2, label="Query")
        axes[0].set_title(
            f"Sample {sample_idx} - Channel {channel}", fontsize=14, fontweight="bold"
        )
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        for i in range(M):
            ax = axes[i + 1]
            neg_series = negatives[i]

            ax.plot(x, neg_series, color="gray", alpha=0.3)

            neg_match = np.ma.masked_where(m == 0, neg_series)
            ax.plot(x, neg_match, color="#00AA00", linewidth=1.5, label="Match")

            neg_diff = np.ma.masked_where(m == 1, neg_series)
            ax.plot(
                x,
                neg_diff,
                color="#FF0000",
                linestyle="--",
                linewidth=1.5,
                label="Mask Zone",
            )

            ax.set_ylabel(f"Neg {i+1}", fontweight="bold")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"sample_{sample_idx}_c{channel}.png"), dpi=150
        )
        plt.close(fig)

    def train(self, setting=None):
        train_data, train_loader = self._get_data(flag="train", shuffle_override=False)
        vali_data, vali_loader = self._get_data(flag="val", shuffle_override=False)
        test_data, test_loader = self._get_data(flag="test", shuffle_override=False)

        self.prepare_retrieval_indices(train_data, vali_data, test_data)

        target_indices = [0, 50, 100, 150, 200]

        for i, (batch_index, batch_context, _, _, _) in enumerate(train_loader):
            for idx in target_indices:
                if (batch_index == idx).any():
                    local_idx = (batch_index == idx).nonzero(as_tuple=True)[0][0].item()

                    indices = self.retrieval_indices["train"][0, batch_index, :]
                    batch_neg_candidates = self.window_bank[indices].to(self.device)
                    batch_x = batch_context.float().to(self.device)

                    B, T, C = batch_x.shape
                    mask = torch.rand((B, T, C)).to(self.device)
                    mask = (mask > self.args.mask_rate).float()

                    inp = batch_x.clone()
                    inp[mask == 0] = 0

                    f_dim = -1 if self.args.features == "MS" else 0

                    self.visualize(
                        batch_x[:, :, f_dim:],
                        inp[:, :, f_dim:],
                        batch_neg_candidates[:, :, f_dim:],
                        mask[:, :, f_dim:],
                        local_idx,
                        idx,
                    )

            if i > max(target_indices) + 10:
                break
