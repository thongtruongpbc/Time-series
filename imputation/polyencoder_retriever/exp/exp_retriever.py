from polyencoder_retriever.data_provider.data_factory import data_provider
from polyencoder_retriever.exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.experiments import save_experiment_to_gsheet_oauth
import torch
import torch.nn as nn
from torch import optim
from transformers import get_linear_schedule_with_warmup
import os
import time
import warnings
import numpy as np
from polyencoder_retriever.models.Encoder import TransformerPolyModel
from polyencoder_retriever.layers.Retrieval import RetrievalTool

warnings.filterwarnings("ignore")


class Exp_Retriever(Exp_Basic):
    def __init__(self, args):
        super(Exp_Retriever, self).__init__(args)
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

        print("Pre-computing Retrieval Indices (Multi-Positive)")
        dev = self.device

        #  shape: [1, N, topm]
        self.retrieval_indices["train"] = rt.retrieve_all(
            train_data, train=True, device=dev
        ).cpu()
        self.retrieval_indices["val"] = rt.retrieve_all(
            vali_data, train=False, device=dev
        ).cpu()
        if test_data is not None:
            self.retrieval_indices["test"] = rt.retrieve_all(
                test_data, train=False, device=dev
            ).cpu()

        raw_x = torch.from_numpy(train_data.data_x).float()
        S = self.args.seq_len
        self.window_bank = raw_x.unfold(0, S, 1).transpose(1, 2)

        del rt
        torch.cuda.empty_cache()
        print("Indices Ready. Memory Cleared")

    def _build_model(self):
        model = TransformerPolyModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

        return optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, flag="val"):
        total_loss = []
        total_acc = []
        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_index,
                batch_context,
                _,
                batch_pos_candidates,
                batch_seq_x_mark,
                batch_mask,
            ) in enumerate(vali_loader):
                #         batch_context = batch_context.float().to(self.device)
                #         batch_pos_candidate = batch_pos_candidate.float().to(self.device)
                #         batch_seq_x_mark = batch_seq_x_mark.float().to(self.device)

                #         loss, logits = self.model(batch_context, batch_pos_candidate, batch_seq_x_mark)
                #         total_loss.append(loss.item())

                #         preds = torch.argmax(logits, dim=-1)
                #         labels = torch.arange(preds.size(0)).to(preds.device)
                #         accuracy = (preds == labels).float().mean()
                #         total_acc.append(accuracy.item())

                # avg_loss = np.average(total_loss)

                # avg_acc = np.average(total_acc)
                # self.model.train()
                # return avg_loss
                indices = self.retrieval_indices[flag][0, batch_index, :]  # [B, M]
                batch_neg_candidates = self.window_bank[indices].to(
                    self.device
                )  # [B, M, S, C]

                batch_context = batch_context.float().to(self.device)
                batch_seq_x_mark = batch_seq_x_mark.float().to(self.device)
                loss, _ = self.model(
                    batch_context,
                    batch_pos_candidates,
                    batch_neg_candidates,
                    batch_seq_x_mark,
                    batch_mask,
                )
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        self.prepare_retrieval_indices(
            train_data=train_data, vali_data=vali_data, test_data=test_data
        )

        t_total = len(train_loader) * self.args.train_epochs

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            global_step = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (
                batch_index,
                batch_context,
                _,
                batch_pos_candidates,
                batch_seq_x_mark,
                batch_mask,
            ) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                # batch_context = batch_context.float().to(self.device)
                # batch_pos_candidate = batch_pos_candidate.float().to(self.device)
                # batch_seq_x_mark = batch_seq_x_mark.float().to(self.device)

                indices = self.retrieval_indices["train"][0, batch_index, :]  # [B, M]
                batch_neg_candidates = self.window_bank[indices].to(
                    self.device
                )  # [B, M, S, C]

                batch_context = batch_context.float().to(self.device)  # [B, S, C]
                batch_seq_x_mark = batch_seq_x_mark.float().to(
                    self.device
                )  # [B, S, C_mark]

                loss, _ = self.model(
                    batch_context,
                    batch_pos_candidates,
                    batch_neg_candidates,
                    batch_seq_x_mark,
                    batch_mask,
                )
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()

                global_step += 1

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, flag="test")

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        checkpoint_to_save = {
            "model_state_dict": self.model.state_dict(),
            "args": self.args,
        }
        torch.save(checkpoint_to_save, best_model_path)
        self.model.load_state_dict(
            torch.load(best_model_path, weights_only=False)["model_state_dict"]
        )

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.args.checkpoints + setting,
                        "checkpoint.pth",
                        weights_only=False,
                    )
                )["model_state_dict"]
            )

        total_accuracy = []
        total_cossim = []
        total_loss = []

        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        r1_list, r3_list, r5_list = [], [], []
        with torch.no_grad():
            for i, (
                batch_index,
                batch_context,
                batch_seq_x,
                batch_pos_candidates,
                batch_seq_x_mark,
                batch_mask,
            ) in enumerate(test_loader):
                B = batch_context.size(0)

                indices = self.retrieval_indices["test"][0, batch_index, :]
                batch_neg_candidates = self.window_bank[indices].to(self.device)

                batch_context = batch_context.float().to(self.device)
                batch_seq_x_mark = batch_seq_x_mark.float().to(self.device)

                # loss, logits (B, B)
                loss, logits = self.model(
                    batch_context,
                    batch_pos_candidates,
                    batch_neg_candidates,
                    batch_seq_x_mark,
                    batch_mask,
                )
                total_loss.append(loss.item())

                labels = torch.arange(B).to(self.device).view(-1, 1)
                _, topk_indices = torch.topk(logits, k=min(5, B), dim=-1)

                r1 = (topk_indices[:, :1] == labels).any(dim=-1).float().mean().item()
                r3 = (topk_indices[:, :3] == labels).any(dim=-1).float().mean().item()
                r5 = (topk_indices[:, :5] == labels).any(dim=-1).float().mean().item()

                r1_list.append(r1)
                r3_list.append(r3)
                r5_list.append(r5)

        avg_loss = np.average(total_loss)
        avg_r1, avg_r3, avg_r5 = (
            np.average(r1_list),
            np.average(r3_list),
            np.average(r5_list),
        )

        print(
            f"Test Loss: {avg_loss:.4f} | R@1: {avg_r1:.4f} | R@3: {avg_r3:.4f} | R@5: {avg_r5:.4f}"
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Setting: {setting}\nR@1: {avg_r1:.4f}, R@3: {avg_r3:.4f}, R@5: {avg_r5:.4f}, Loss: {avg_loss:.4f}\n---\n"
        )
        with open("result_contrastive_final.txt", "a") as f:
            f.write(
                f"Setting: {setting}\nR@1: {avg_r1:.4f}, R@3: {avg_r3:.4f}, R@5: {avg_r5:.4f}, Loss: {avg_loss:.4f}\n---\n"
            )

        return
