from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.experiments import save_experiment_to_gsheet_oauth, save_experiment_to_excel
from utils.push2huggingface import _push_to_huggingface
import torch

torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import faiss
import torch.nn.functional as F
from tracking.tracking import MemoryCallback
from torchinfo import summary

# import mlflow
# from mlflow.models import infer_signature

import importlib.util
import sys


def load_poly_model_class(model_path):
    import os
    import sys
    import importlib

    folder_models = os.path.dirname(model_path)
    retriever_root = os.path.dirname(folder_models)
    parent_of_root = os.path.dirname(retriever_root)

    if parent_of_root not in sys.path:
        sys.path.insert(0, parent_of_root)

    for mod in list(sys.modules.keys()):
        if mod.startswith("polyencoder_retriever"):
            del sys.modules[mod]

    try:
        module = importlib.import_module("polyencoder_retriever.models.Encoder")
        importlib.reload(module)
        return module.TransformerPolyModel
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


warnings.filterwarnings("ignore")


class Exp_Imputation_retrieval(Exp_Basic):
    def __init__(self, args):
        retrieval_checkpoint_path = args.retrieval_checkpoint_path
        self.args_emb = torch.load(
            f=retrieval_checkpoint_path,
            weights_only=False,
            map_location=args.device,
        )["args"]

        self.push2hf = True
        self.repo_id = f"thongtx/embeddings"

        super(Exp_Imputation_retrieval, self).__init__(args)
        # build embedding model
        self.model_emb = self._build_model_emb()
        state = torch.load(
            f=retrieval_checkpoint_path,
            weights_only=False,
            map_location=self.args.device,
        )["model_state_dict"]

        self.model_emb.load_state_dict(state, strict=False)
        self.model_emb.to(self.device)
        self.model_emb.eval()

        db_path = os.path.join(
            "./vector_db_poly", args.setting.split("_", 1)[1], "cached_states.pt"
        )
        if not os.path.exists(db_path):
            self.embedding(setting=args.setting)

        self._load_vector_db(setting=args.setting)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model: nn.DataParallel[os.Any] = nn.DataParallel(
                model, device_ids=self.args.device_ids
            )
        return model

    def _build_model_emb(self):
        poly_file_path = "retriever/models/Encoder.py"
        PolyModelClass = load_poly_model_class(poly_file_path)
        model_emb = PolyModelClass(self.args_emb).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model_emb = nn.DataParallel(model_emb, device_ids=self.args.device_ids)
        return model_emb

    def _get_data(self, flag, shuffle_override=None):
        data_set, data_loader = data_provider(
            self.args, flag, shuffle_override=shuffle_override
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self) -> nn.MSELoss:
        criterion = nn.MSELoss()
        return criterion

    # Retrieval phrase
    def clean_params(args):  # -> dict:
        """
        Docstring for clean_params
         serve for infer_signature in mlflow
        :param args
        """
        clean = {}
        for k, v in vars(args).items():
            if v is None:
                continue
            if isinstance(v, torch.device):
                clean[k] = str(v)
            elif isinstance(v, (int, float, str, bool)):
                clean[k] = v
        return clean

    def _cache_dataset(self, dataset):
        """
        Cache dataset into tensor to avoid dataset[idx] in retrieval
        """
        all_x = []
        for i in range(len(dataset)):
            x, _, _, _, _ = dataset[i]  # (T, C)
            x = torch.as_tensor(x, dtype=torch.float32)
            all_x.append(x)

        all_x = torch.stack(all_x)  # (N, T, C)
        all_x = all_x.to(self.device)

        self.cached_x = all_x  # (N, T, C)

        print(f"Cached dataset: {all_x.shape}")

    def embedding(self, setting):
        _, train_loader = self._get_data(flag="train", shuffle_override=False)

        save_dir = os.path.join("./vector_db_poly", setting.split("_", 1)[1])
        save_path = os.path.join(save_dir, "cached_states.npy")
        num_samples = len(train_loader.dataset)
        final_shape = (
            num_samples,
            self.args_emb.seq_len,
            self.args_emb.enc_in,
            self.args_emb.d_model,
        )

        if os.path.exists(save_path):
            all_outputs_mmap = np.memmap(
                save_path,
                dtype="float16",
                mode="r",
                shape=(
                    num_samples,
                    self.args_emb.seq_len,
                    self.args_emb.enc_in,
                    self.args_emb.d_model,
                ),
            )

        else:
            os.makedirs(save_dir, exist_ok=True)

            all_outputs_mmap = np.memmap(
                save_path, dtype="float16", mode="w+", shape=final_shape
            )

            self.model_emb.eval()
            with torch.no_grad():
                for i, (batch_x, _, batch_x_mark, _, index) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    outputs = self.model_emb.encode_candidate(batch_x, batch_x_mark)
                    outputs_np = outputs.detach().half().cpu().numpy()

                    all_outputs_mmap[index.numpy()] = outputs_np

                    if i % 30 == 0:
                        all_outputs_mmap.flush()

            all_outputs_mmap.flush()

        if self.push2hf:
            if self.repo_id:
                _push_to_huggingface(save_path, self.repo_id, setting.split("_", 1)[1])
            else:
                print("Warning: push2hf is True but repo_id is missing.")

        print(
            "Completed embedding:", all_outputs_mmap.shape
        )  # (#samples, T, C, d_model)

    def stride_filter_fast(self, indices, query_idx, k, stride):
        """
        indices: Tensor (num_candidates,) - FAISS results
        query_idx: int - index of query sample in cached dataset
        """
        selected = []

        for idx in indices:
            idx = int(idx)

            if abs(idx - query_idx) >= stride:
                selected.append(idx)

            if len(selected) == k:
                break

        return torch.tensor(selected, device=indices.device, dtype=torch.long)

    def batch_retrieval_fast(
        self, batch_x, batch_x_mark, query_indices, mask, top_k, stride, chunk_size=32
    ):
        B = batch_x.shape[0]
        N_train = self.cached_states.shape[0]

        context_vecs = self.model_emb.encode_context(batch_x, batch_x_mark, mask)
        # context_vecs: [B, poly_m, vec_dim]

        all_scores = []
        for i in range(0, N_train, chunk_size):
            end_idx = min(i + chunk_size, N_train)
            N_chunk = end_idx - i

            chunk_states = self.cached_states[i:end_idx].to(self.device).float()
            # chunk_states: [N_chunk, T, C, D]

            chunk_states = (
                chunk_states.unsqueeze(0)
                .expand(B, N_chunk, -1, -1, -1)
                .reshape(B * N_chunk, *chunk_states.shape[1:])
            )
            # chunk_states: [B*N_chunk, T, C, D]

            scores_chunk = self.model_emb.compute_similarity(
                context_vecs, chunk_states, N=N_chunk, batch_mask=mask
            )
            # scores_chunk: [B, N_chunk]
            del chunk_states

            all_scores.append(scores_chunk.cpu())

        all_scores = torch.cat(all_scores, dim=1)
        # all_scores: [B, N_train]

        if stride > 0:
            for b in range(B):
                q_idx = query_indices[b].item()
                low = max(0, q_idx - stride)
                high = min(N_train, q_idx + stride)
                all_scores[b, low:high] = -1e9

        top_val, top_idx = torch.topk(all_scores, top_k, dim=-1)
        # top_idx: [B, top_k]

        support = self.cached_x[top_idx.to(self.device)]
        # support: [B, top_k, T, C]

        return support

    def _load_vector_db(self, setting):
        train_set, _ = self._get_data(flag="train", shuffle_override=False)
        self._cache_dataset(train_set)
        num_samples = len(train_set)

        save_path = os.path.join(
            "./vector_db_poly", setting.split("_", 1)[1], "cached_states.npy"
        )
        final_shape = (
            num_samples,
            self.args_emb.seq_len,
            self.args_emb.enc_in,
            self.args_emb.d_model,
        )

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Missing cache file at {save_path}")

        mmap_array = np.memmap(save_path, dtype="float16", mode="r", shape=final_shape)

        self.cached_states = torch.from_numpy(mmap_array)
        print(f"Loaded vector DB with shape {final_shape}")

    ### Main model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, C = batch_x.shape
                mask = torch.rand((B, T, C)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained

                reference_x = self.batch_retrieval_fast(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    query_indices=index,
                    mask=mask,
                    top_k=self.args.k,
                    stride=0,
                )

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(
                    inp, batch_x_mark, reference_x, None, None, mask, training=0
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach()
                true = batch_x.detach()
                mask = mask.detach()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        self._cache_dataset(train_data)
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        mem_monitor = MemoryCallback(
            ram_threshold=95.0,
            vram_min_free_gb=2.0,
            checkpoint_path=os.path.join(path, "emergency_checkpoint.pth"),
        )

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        global_step_count = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(
                train_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, C = batch_x.shape
                mask = torch.rand((B, T, C)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained

                global_step_count += 1
                # with torch.no_grad():
                #     query_emb = self.model_emb.encode_context(
                #         batch_x, batch_x_mark.to(batch_x.device), batch_mask=mask
                #     )

                #     f_dim = -1 if self.args.features == "MS" else 0
                #     query_emb = query_emb.mean(dim=1)
                #     query_emb = query_emb.to(self.device)
                # print(query_emb.shape)
                reference_x = self.batch_retrieval_fast(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    query_indices=index,
                    mask=mask,
                    top_k=self.args.k,
                    stride=self.args.seq_len,
                )
                # print(reference_x.shape)

                if i % 200 == 0:
                    mem_monitor.check_and_safe_exit(
                        self.model, optimizer=model_optim, epoch=epoch, batch_idx=i
                    )

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(
                    inp, batch_x_mark, reference_x, None, None, mask, training=1
                )

                # signature mlflow
                args = vars(self.args)
                # args["device"] = str(args["device"])
                # args["down_sampling_method"] = args["down_sampling_method"] or "none"

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
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
                model_optim.step()

                # mlflow.log_metric("batch_loss", loss.item(), step=global_step_count)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # summary
            # model_stats = summary(self.model, input_size=(B, T, C))
            # with open("model_summary.txt", "w") as f:
            #     f.write(str(model_stats))
            # mlflow.log_artifact("model_summary.txt")

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # signature = infer_signature(
            #     batch_x.cpu().numpy(),
            #     outputs.cpu().detach().numpy(),
            #     # params=vars(self.args),
            #     params=args,
            # )

            # mlflow.log_metrics(
            #     {
            #         "train_loss": float(train_loss),
            #         "val_loss": float(vali_loss),
            #         "test_loss": float(test_loss),
            #     },
            #     step=epoch,
            # )
            # mlflow.log_artifact(os.path.join(path, "checkpoint.pth"))

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
        # Log the final trained model
        # mlflow.pytorch.log_model(
        #     pytorch_model=self.model,
        #     signature=signature,
        #     input_example=batch_x.cpu().numpy(),
        #     name="model",  # folder of MLflow Artifacts
        #     registered_model_name=f"Imputation_{self.args.model}_{self.args}",  #  Model Registry
        # )

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        # cache
        train_data, train_loader = self._get_data(flag="train")
        if self.args.is_training == 0:
            self._cache_dataset(train_data)
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        "./checkpoints_imputation_retrieval/" + setting,
                        "checkpoint.pth",
                    )
                )
            )

        preds = []
        trues = []
        masks = []
        folder_path = "./test_results_retrieval/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained

                reference_x = self.batch_retrieval_fast(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    query_indices=index,
                    mask=mask,
                    top_k=self.args.k,
                    stride=0,
                )

                # in mask position, temporal fill == 0
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(
                    inp, batch_x_mark, reference_x, None, None, mask, training=0
                )

                # eval
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + pred[
                        0, :, -1
                    ] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(
                        true[0, :, -1],
                        filled,
                        os.path.join(folder_path, str(i) + ".pdf"),
                    )

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        # mlflow.log_metrics(
        #     {
        #         "final_mse": float(mse),
        #         "final_mae": float(mae),
        #         "final_rmse": float(rmse),
        #         "final_mape": float(mape),
        #         "final_mspe": float(mspe),
        #     }
        # )

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }

        # save experiments
        save_experiment_to_gsheet_oauth(
            args=self.args, metrics=metrics, sheet_name=self.args.sheet_name
        )

        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result_imputation_retrieval.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        # mlflow.log_artifacts(folder_path)

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        return
