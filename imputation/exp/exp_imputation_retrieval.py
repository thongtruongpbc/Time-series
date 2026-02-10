from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import faiss
import torch.nn.functional as F
warnings.filterwarnings('ignore')


class Exp_Imputation_retrieval(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_retrieval, self).__init__(args)
        # build embedding model
        self.model_emb = self._build_model_emb()
        state = torch.load(
            os.path.join(
                    '/mnt/time-series/thongtx/imputation/checkpoints_imputation',
                    args.setting,
                    'checkpoint.pth'
                ),
                map_location=self.device
            )

        self.model_emb.load_state_dict(state, strict=False)
        self.model_emb.to(self.device)
        self.model_emb.eval()


        # load vector DB + FAISS

        self.embedding(args.setting)
        self._load_vector_db(args.setting)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _build_model_emb(self):
        model_emb = self.model_dict[self.args.model_emb].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model_emb = nn.DataParallel(model_emb, device_ids=self.args.device_ids)
        return model_emb

    def _get_data(self, flag, shuffle_override=None):
        data_set, data_loader = data_provider(self.args, flag, shuffle_override=shuffle_override)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # Retrieval phrase

    def _cache_dataset(self, dataset):
        """
        Cache dataset into tensor to avoid dataset[idx] in retrieval
        """
        all_x = []
        for i in range(len(dataset)):
            x, _, _, _, _ = dataset[i]   # (T, C)
            x = torch.as_tensor(x, dtype=torch.float32)
            all_x.append(x)

        all_x = torch.stack(all_x)        # (N, T, C)
        all_x = all_x.to(self.device)

        self.cached_x = all_x             # (N, T, C)

        print(f"Cached dataset: {all_x.shape}")

    def embedding(self, setting):
        _, train_loader = self._get_data(flag='train', shuffle_override=False)

        save_dir = os.path.join('./vector_db', setting)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, 'vector_db.pt')
        if os.path.isfile(save_path):
            print(f"Found existing vector DB: {save_path}")
            return
        all_outputs = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """

                outputs = self.model_emb.get_representation(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:]
                all_outputs.append(outputs.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        
        torch.save(all_outputs, save_path)
        print("Saved embedding:", all_outputs.shape)

    def stride_filter(self, indices, k, stride):
        """
        indices: 1D array-like, sorted by distance (nearest → farthest)
                dataset indices
        k: number of neighbors to keep
        stride: minimal distance between indices

        return: list of selected indices (len <= k)
        """
        selected = []
        for idx in indices:
            if all(abs(idx - s) >= stride for s in selected):
                selected.append(int(idx))
            if len(selected) == k:
                break
        return selected

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


    def batch_retrieval(
        self,
        query_emb,          # (B, D)
        faiss_index,        # FAISS index built on vector_db
        dataset,            # train_dataset
        top_k,
        stride,
        device
    ):
        """
        Perform batch retrieval with stride constraint.

        Returns:
            support_x: Tensor (B, k, T, N)
            support_indices: LongTensor (B, k)
        """

        B = query_emb.shape[0]

        #FAISS search
        query_np = query_emb.detach().cpu().numpy().astype("float32")
        _, I = faiss_index.search(query_np, top_k + 2 * self.args.seq_len)

        support_x = []
        support_indices = []

        for b in range(B):
            # stride filter
            selected = self.stride_filter(I[b], top_k, stride)

            if len(selected) < top_k:
                selected += list(I[b][:top_k - len(selected)])

            selected = selected[:top_k]
            support_indices.append(selected)

            samples = []
            for idx in selected:
                x, _, _, _ = dataset[idx]
                samples.append(x)

            support_x.append(torch.stack(samples))  # (k, T, N)

        support_x = torch.stack(support_x).to(device)        # (B, k, T, N)
        support_indices = torch.tensor(support_indices)      # (B, k)

        return support_x, support_indices

    def batch_retrieval_fast(
        self,
        query_emb,      # (B, d_model)
        query_indices, # (B,)
        top_k,
        stride,
    ):
        B, d_model = query_emb.shape
        N, T_raw, C = self.cached_x.shape

        # (B, T, d_model)
        query = query_emb.reshape(-1, d_model)
        query = F.normalize(query, dim=-1)
        query_np = query.detach().cpu().numpy().astype("float32")

        # FAISS search 
        _, I = self.faiss_index.search(query_np, top_k + 2 * stride)
        I = torch.from_numpy(I).to(self.device)

        support = torch.empty(
            (B, top_k, T_raw, C),
            device=self.device
        )

        for b in range(B):
            selected = self.stride_filter_fast(
                I[b], query_indices[b], top_k, stride
            )  # (top_k,)

            support[b] = self.cached_x[selected]   # (top_k, T, C)
        return support   # (B, top_k, T, C)


    def _load_vector_db(self, setting):
        db_path = os.path.join('./vector_db', setting, 'vector_db.pt')
        vector_db = torch.load(db_path)  # (N, D_MODEL)

        N,  d_model = vector_db.shape
        self.db_channels = d_model

        vector_db = vector_db.reshape(N, d_model)
        vector_db = F.normalize(vector_db, dim=-1)

        vector_np = vector_db.cpu().numpy().astype('float32')

        index = faiss.IndexFlatL2( d_model)
        index.add(vector_np)

        self.faiss_index = index
        print(f"Loaded vector DB: {vector_np.shape}")


    ### Main model


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                with torch.no_grad():
                    query_emb = self.model_emb.get_representation(
                        batch_x, batch_x_mark
                    )
                    f_dim = -1 if self.args.features == 'MS' else 0
                    #query_emb = query_emb[:, :, f_dim:]   # (B, C, T)
                    query_emb = query_emb.to(self.device)
                
                reference_x = self.batch_retrieval_fast(
                    query_emb=query_emb,
                    query_indices=index,
                    top_k=self.args.top_k,
                    stride=0
                )


                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=0)

                f_dim = -1 if self.args.features == 'MS' else 0
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
        train_data, train_loader = self._get_data(flag='train')
        self._cache_dataset(train_data)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                with torch.no_grad():
                    query_emb = self.model_emb.get_representation(
                        batch_x, batch_x_mark.to(batch_x.device)
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # query_emb = query_emb[:, :, f_dim:].to(self.device)   # (B, C, T)
                    query_emb = query_emb.to(self.device)
                #print(query_emb.shape)
                reference_x = self.batch_retrieval_fast(
                    query_emb=query_emb,
                    query_indices=index,
                    top_k=self.args.top_k,
                    stride=self.args.seq_len
                )
                #print(reference_x.shape)

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints_imputation/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                with torch.no_grad():
                    query_emb = self.model_emb.get_representation(
                        batch_x, batch_x_mark
                    )
                    f_dim = -1 if self.args.features == 'MS' else 0
                    #query_emb = query_emb[:, :, f_dim:]   # (B, C, T)
                    query_emb = query_emb.to(self.device)
                
                reference_x = self.batch_retrieval_fast(
                    query_emb=query_emb,
                    query_indices=index,
                    top_k=self.args.top_k,
                    stride=0
                )

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=0)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
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
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_imputation_retrieval.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return





# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import numpy as np
# import faiss
# import torch.nn.functional as F
# warnings.filterwarnings('ignore')


# class Exp_Imputation_retrieval(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Imputation_retrieval, self).__init__(args)
#         # build embedding model
#         self.model_emb = self._build_model_emb()
#         state = torch.load(
#             os.path.join(
#                     '/mnt/time-series/thongtx/imputation/checkpoints_imputation',
#                     args.setting,
#                     'checkpoint.pth'
#                 ),
#                 map_location=self.device
#             )

#         self.model_emb.load_state_dict(state, strict=False)
#         self.model_emb.to(self.device)
#         self.model_emb.eval()


#         # load vector DB + FAISS

#         self.embedding(args.setting)
#         self._load_vector_db(args.setting)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model
    
#     def _build_model_emb(self):
#         model_emb = self.model_dict[self.args.model_emb].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model_emb = nn.DataParallel(model_emb, device_ids=self.args.device_ids)
#         return model_emb

#     def _get_data(self, flag, shuffle_override=None):
#         data_set, data_loader = data_provider(self.args, flag, shuffle_override=shuffle_override)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     # Retrieval phrase

#     def _cache_dataset(self, dataset):
#         """
#         Cache dataset into tensor to avoid dataset[idx] in retrieval
#         """
#         all_x = []
#         for i in range(len(dataset)):
#             x, _, _, _, _ = dataset[i]   # (T, C)
#             x = torch.as_tensor(x, dtype=torch.float32)
#             all_x.append(x)

#         all_x = torch.stack(all_x)        # (N, T, C)
#         all_x = all_x.to(self.device)

#         self.cached_x = all_x             # (N, T, C)

#         print(f"Cached dataset: {all_x.shape}")

#     def embedding(self, setting):
#         _, train_loader = self._get_data(flag='train', shuffle_override=False)

#         save_dir = os.path.join('./vector_db', setting)
#         os.makedirs(save_dir, exist_ok=True)

#         save_path = os.path.join(save_dir, 'vector_db.pt')
#         if os.path.isfile(save_path):
#             print(f"Found existing vector DB: {save_path}")
#             return
#         all_outputs = []
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)

#                 # random mask
#                 B, T, N = batch_x.shape
#                 """
#                 B = batch size
#                 T = seq len
#                 N = number of features
#                 """

#                 outputs = self.model_emb.get_representation(batch_x, batch_x_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 # outputs = outputs[:, :, f_dim:]
#                 all_outputs.append(outputs.cpu())
        
#         all_outputs = torch.cat(all_outputs, dim=0)
        
#         torch.save(all_outputs, save_path)
#         print("Saved embedding:", all_outputs.shape)

#     def stride_filter(self, indices, k, stride):
#         """
#         indices: 1D array-like, sorted by distance (nearest → farthest)
#                 dataset indices
#         k: number of neighbors to keep
#         stride: minimal distance between indices

#         return: list of selected indices (len <= k)
#         """
#         selected = []
#         for idx in indices:
#             if all(abs(idx - s) >= stride for s in selected):
#                 selected.append(int(idx))
#             if len(selected) == k:
#                 break
#         return selected

#     def stride_filter_fast(self, indices, query_idx, k, stride):
#         """
#         indices: Tensor (num_candidates,) - FAISS results
#         query_idx: int - index of query sample in cached dataset
#         """
#         selected = []

#         for idx in indices:
#             idx = int(idx)

#             if abs(idx - query_idx) >= stride:
#                 selected.append(idx)

#             if len(selected) == k:
#                 break

#         return torch.tensor(selected, device=indices.device, dtype=torch.long)


#     def batch_retrieval(
#         self,
#         query_emb,          # (B, D)
#         faiss_index,        # FAISS index built on vector_db
#         dataset,            # train_dataset
#         top_k,
#         stride,
#         device
#     ):
#         """
#         Perform batch retrieval with stride constraint.

#         Returns:
#             support_x: Tensor (B, k, T, N)
#             support_indices: LongTensor (B, k)
#         """

#         B = query_emb.shape[0]

#         #FAISS search
#         query_np = query_emb.detach().cpu().numpy().astype("float32")
#         _, I = faiss_index.search(query_np, top_k + 2 * self.args.seq_len)

#         support_x = []
#         support_indices = []

#         for b in range(B):
#             # stride filter
#             selected = self.stride_filter(I[b], top_k, stride)

#             if len(selected) < top_k:
#                 selected += list(I[b][:top_k - len(selected)])

#             selected = selected[:top_k]
#             support_indices.append(selected)

#             samples = []
#             for idx in selected:
#                 x, _, _, _ = dataset[idx]
#                 samples.append(x)

#             support_x.append(torch.stack(samples))  # (k, T, N)

#         support_x = torch.stack(support_x).to(device)        # (B, k, T, N)
#         support_indices = torch.tensor(support_indices)      # (B, k)

#         return support_x, support_indices

#     def batch_retrieval_fast(
#         self,
#         query_emb,      # (B, d_model)
#         query_indices, # (B,)
#         top_k,
#         stride,
#     ):
#         B, d_model = query_emb.shape
#         N, T_raw, C = self.cached_x.shape

#         # (B, T, d_model)
#         query = query_emb.reshape(-1, d_model)
#         query = F.normalize(query, dim=-1)
#         query_np = query.detach().cpu().numpy().astype("float32")

#         # FAISS search 
#         _, I = self.faiss_index.search(query_np, top_k + 2 * stride)
#         I = torch.from_numpy(I).to(self.device)

#         support = torch.empty(
#             (B, top_k, T_raw, C),
#             device=self.device
#         )

#         for b in range(B):
#             selected = self.stride_filter_fast(
#                 I[b], query_indices[b], top_k, stride
#             )  # (top_k,)

#             support[b] = self.cached_x[selected]   # (top_k, T, C)
#         return support   # (B, top_k, T, C)


#     def _load_vector_db(self, setting):
#         db_path = os.path.join('./vector_db', setting, 'vector_db.pt')
#         vector_db = torch.load(db_path)  # (N, D_MODEL)

#         N,  d_model = vector_db.shape
#         self.db_channels = d_model

#         vector_db = vector_db.reshape(N, d_model)
#         vector_db = F.normalize(vector_db, dim=-1)

#         vector_np = vector_db.cpu().numpy().astype('float32')

#         index = faiss.IndexFlatL2( d_model)
#         index.add(vector_np)

#         self.faiss_index = index
#         print(f"Loaded vector DB: {vector_np.shape}")


#     ### Main model


#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 with torch.no_grad():
#                     query_emb = self.model_emb.get_representation(
#                         batch_x, batch_x_mark
#                     )
#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     #query_emb = query_emb[:, :, f_dim:]   # (B, C, T)
#                     query_emb = query_emb.to(self.device)
                
#                 reference_x = self.batch_retrieval_fast(
#                     query_emb=query_emb,
#                     query_indices=index,
#                     top_k=self.args.top_k,
#                     stride=0
#                 )


#                 # random mask
#                 B, T, N = batch_x.shape
#                 """
#                 B = batch size
#                 T = seq len
#                 N = number of features
#                 """
#                 mask = torch.rand((B, T, N)).to(self.device)
#                 mask[mask <= self.args.mask_rate] = 0  # masked
#                 mask[mask > self.args.mask_rate] = 1  # remained
#                 inp = batch_x.masked_fill(mask == 0, 0)

#                 outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=0)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, :, f_dim:]

#                 # add support for MS
#                 batch_x = batch_x[:, :, f_dim:]
#                 mask = mask[:, :, f_dim:]

#                 pred = outputs.detach()
#                 true = batch_x.detach()
#                 mask = mask.detach()

#                 loss = criterion(pred[mask == 0], true[mask == 0])
#                 total_loss.append(loss.item())
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         self._cache_dataset(train_data)
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
#                 with torch.no_grad():
#                     query_emb = self.model_emb.get_representation(
#                         batch_x, batch_x_mark.to(batch_x.device)
#                     )

#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     # query_emb = query_emb[:, :, f_dim:].to(self.device)   # (B, C, T)
#                     query_emb = query_emb.to(self.device)
#                 #print(query_emb.shape)
#                 reference_x = self.batch_retrieval_fast(
#                     query_emb=query_emb,
#                     query_indices=index,
#                     top_k=self.args.top_k,
#                     stride=self.args.seq_len
#                 )
#                 #print(reference_x.shape)

#                 iter_count += 1
#                 model_optim.zero_grad()

#                 batch_x = batch_x.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)

#                 # random mask
#                 B, T, N = batch_x.shape
#                 mask = torch.rand((B, T, N)).to(self.device)
#                 mask[mask <= self.args.mask_rate] = 0  # masked
#                 mask[mask > self.args.mask_rate] = 1  # remained
#                 inp = batch_x.masked_fill(mask == 0, 0)

#                 outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=1)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, :, f_dim:]

#                 # add support for MS
#                 batch_x = batch_x[:, :, f_dim:]
#                 mask = mask[:, :, f_dim:]

#                 loss = criterion(outputs[mask == 0], batch_x[mask == 0])
#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 loss.backward()
#                 model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             test_loss = self.vali(test_data, test_loader, criterion)

#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints_imputation/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         masks = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 with torch.no_grad():
#                     query_emb = self.model_emb.get_representation(
#                         batch_x, batch_x_mark
#                     )
#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     #query_emb = query_emb[:, :, f_dim:]   # (B, C, T)
#                     query_emb = query_emb.to(self.device)
                
#                 reference_x = self.batch_retrieval_fast(
#                     query_emb=query_emb,
#                     query_indices=index,
#                     top_k=self.args.top_k,
#                     stride=0
#                 )

#                 # random mask
#                 B, T, N = batch_x.shape
#                 mask = torch.rand((B, T, N)).to(self.device)
#                 mask[mask <= self.args.mask_rate] = 0  # masked
#                 mask[mask > self.args.mask_rate] = 1  # remained
#                 inp = batch_x.masked_fill(mask == 0, 0)

#                 # imputation
#                 outputs = self.model(inp, batch_x_mark, reference_x, None, None, mask, training=0)

#                 # eval
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, :, f_dim:]

#                 # add support for MS 
#                 batch_x = batch_x[:, :, f_dim:]
#                 mask = mask[:, :, f_dim:]

#                 outputs = outputs.detach().cpu().numpy()
#                 pred = outputs
#                 true = batch_x.detach().cpu().numpy()
#                 preds.append(pred)
#                 trues.append(true)
#                 masks.append(mask.detach().cpu())

#                 if i % 20 == 0:
#                     filled = true[0, :, -1].copy()
#                     filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
#                              pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
#                     visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

#         preds = np.concatenate(preds, 0)
#         trues = np.concatenate(trues, 0)
#         masks = np.concatenate(masks, 0)
#         print('test shape:', preds.shape, trues.shape)

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result_imputation_retrieval.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)
#         return
