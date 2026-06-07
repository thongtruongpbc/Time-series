### Reference: https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/CachedMultipleNegativesRankingLoss.py
"""
Adaptation of CachedMultipleNegativesRankingLoss (GradCache) for TransformerPolyModel.

Key idea (from https://arxiv.org/pdf/2101.06983.pdf):
  Stage 1 — Encode ALL context + candidates with torch.no_grad() → cache embeddings.
  Stage 2 — Run the full contrastive loss on cached embeddings → backward up to embeddings
             only, cache ∂L/∂emb.
  Stage 3 — Re-encode each mini-batch WITH grad, attach cached gradients via surrogate
             backward → true parameter gradients flow through the encoder.

Complexity specific to PolyEncoder:
  • Context  →  encode_context_raw()  returns state_vecs  (B, T, C, d_model)
                *before* poly-attention, so we can cache the heavy transformer output.
  • Candidate →  encode_candidate_raw() returns state_vecs (BN, T, C, d_model).
  • The lightweight poly-attention + FC are re-run inside compute_scores()
    because they depend on all (ctx, cand) pairs at once and are cheap.
"""

from .Transformer import Model as TransformerModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from retriever.layers import dot_attention
from contextlib import nullcontext
from torch import Tensor
from typing import Literal, Optional
import tqdm


def _get_device_states(*args):
    unique_devices = {t.device for t in args if isinstance(t, Tensor) and t.is_cuda}
    return list(unique_devices), [torch.cuda.get_rng_state(d) for d in unique_devices]


def _set_device_states(devices, states):
    for dev, state in zip(devices, states):
        torch.cuda.set_rng_state(state, dev)


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = _get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        _set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class TransformerPolyModel(nn.Module):
    def __init__(
        self, configs, mini_batch_size: int = 8, show_progress_bar: bool = False
    ):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.vec_dim = 64
        self.poly_m = configs.poly_m
        self.topm = configs.topm
        self.n_period = 1
        self.channels = configs.enc_in
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.poly_code_embeddings = nn.Embedding(self.poly_m + 2, configs.d_model)

        self.transformer_ctx = TransformerModel(configs=configs)
        self.transformer_pos = TransformerModel(configs=configs)

        self.dropout = nn.Dropout(configs.dropout)
        self.context_fc = nn.Linear(configs.d_model, self.vec_dim)
        self.pos_candidate_fc = nn.Linear(configs.d_model, self.vec_dim)

        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar
        self.cache: Optional[list[list[Tensor]]] = None

    def _encode_context_raw(self, context, seq_x_mark=None, batch_mask=None):
        state_vecs_ctx, _ = self.transformer_ctx(context, seq_x_mark, mask=batch_mask)
        return state_vecs_ctx

    def _encode_candidate_raw(self, candidates_mb, seq_x_mark_mb=None):
        state_vecs, _ = self.transformer_pos(candidates_mb, seq_x_mark_mb)
        return state_vecs

    def _poly_context_vecs(self, state_vecs_ctx, batch_mask=None):
        device = self.poly_code_embeddings.weight.device
        B, T, C, d_model = state_vecs_ctx.shape

        poly_code_ids = (
            torch.arange(self.poly_m, device=device).unsqueeze(0).expand(B, self.poly_m)
        )
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        poly_codes = poly_codes.unsqueeze(2).expand(B, self.poly_m, C, d_model)

        sv = state_vecs_ctx.transpose(1, 2)
        pc = poly_codes.transpose(1, 2)
        v_mask = batch_mask.transpose(1, 2) if batch_mask is not None else None

        ctx_vecs = dot_attention(q=pc, k=sv, v=sv, v_mask=v_mask, dropout=self.dropout)
        ctx_vecs = ctx_vecs.transpose(1, 2).mean(dim=2)
        ctx_vecs = self.context_fc(self.dropout(ctx_vecs))
        return F.normalize(ctx_vecs, p=2, dim=-1)

    def _poly_candidate_vecs(self, state_vecs, B, N, batch_mask=None):
        device = self.poly_code_embeddings.weight.device
        BN, T, C, D = state_vecs.shape

        poly_code_ids = torch.full(
            (BN, 1), self.poly_m + 1, dtype=torch.long, device=device
        )
        poly_codes = (
            self.poly_code_embeddings(poly_code_ids).unsqueeze(2).expand(BN, 1, C, D)
        )

        sv = state_vecs.transpose(1, 2)
        pc = poly_codes.transpose(1, 2)
        v_mask = None
        if batch_mask is not None:
            v_mask = (
                batch_mask.transpose(1, 2)
                .unsqueeze(1)
                .repeat(1, N, 1, 1)
                .reshape(BN, C, T)
            )

        cand_vec = dot_attention(q=pc, k=sv, v=sv, v_mask=v_mask, dropout=self.dropout)
        cand_vec = cand_vec.transpose(1, 2).squeeze(1).mean(dim=1)
        cand_vec = self.pos_candidate_fc(self.dropout(cand_vec))
        return F.normalize(cand_vec, p=2, dim=-1).reshape(B, N, self.vec_dim)

    def _compute_scores(self, context_vecs, cand_vecs):
        B = context_vecs.size(0)
        N_total = cand_vecs.size(1)

        q = cand_vecs.reshape(-1, 1, self.vec_dim)
        k = (
            context_vecs.unsqueeze(1)
            .expand(B, N_total, self.poly_m, self.vec_dim)
            .reshape(-1, self.poly_m, self.vec_dim)
        )

        agg = dot_attention(q, k, k, dropout=self.dropout)
        agg = F.normalize(agg.squeeze(1), p=2, dim=-1)
        scores = torch.sum(agg * q.squeeze(1), dim=-1).reshape(B, N_total)
        return scores * torch.exp(self.temperature)

    def _build_candidate_pool(
        self,
        pos_candidates: Tensor,
        neg_candidates: Tensor,
        hardness_mode: (
            Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None
        ),
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (all_cands, labels).

        "in_batch_negatives":
            Each sample's positive is encoded once; all B positives form a (B, B)
            score matrix. Label i = i (diagonal).  neg_candidates ignored.

        None / "hard_negatives":
            Score matrix is (B, 1+M). Label i = 0.  neg_candidates required.

        "all_negatives":
            Score matrix is (B, B+M). Label i = i (own positive on diagonal,
            then M hard negatives appended). neg_candidates required.
        """
        B, S, C = pos_candidates.shape
        pos_exp = pos_candidates.unsqueeze(1)  # (B, 1, S, C)

        if hardness_mode == "in_batch_negatives":
            all_cands = pos_candidates.unsqueeze(0).expand(B, B, S, C)
            labels = torch.arange(B, dtype=torch.long, device=pos_candidates.device)
            return all_cands, labels

        if hardness_mode is None or hardness_mode == "hard_negatives":
            all_cands = torch.cat([pos_exp, neg_candidates], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=pos_candidates.device)
            return all_cands, labels

        if hardness_mode == "all_negatives":
            in_batch = pos_candidates.unsqueeze(0).expand(B, B, S, C)
            all_cands = torch.cat([in_batch, neg_candidates], dim=1)
            labels = torch.arange(B, dtype=torch.long, device=pos_candidates.device)
            return all_cands, labels

        raise ValueError(f"Unknown hardness_mode: {hardness_mode!r}")

    def _iter_context_minibatches(
        self, context, seq_x_mark, batch_mask, with_grad, random_states=None
    ):
        B = context.size(0)
        mb = self.mini_batch_size
        for i, b in enumerate(range(0, B, mb)):
            e = min(b + mb, B)
            ctx_mb = context[b:e]
            mark_mb = seq_x_mark[b:e] if seq_x_mark is not None else None
            mask_mb = batch_mask[b:e] if batch_mask is not None else None
            rand_ctx = (
                random_states[i] if random_states is not None else RandContext(ctx_mb)
            )
            ctx_mgr = nullcontext if with_grad else torch.no_grad
            with ctx_mgr():
                with rand_ctx:
                    sv = self._encode_context_raw(ctx_mb, mark_mb, mask_mb)
            yield sv, rand_ctx

    def _iter_candidate_minibatches(
        self, candidates_flat, seq_x_mark_flat, with_grad, random_states=None
    ):
        BN = candidates_flat.size(0)
        mb = self.mini_batch_size
        for i, b in enumerate(range(0, BN, mb)):
            e = min(b + mb, BN)
            cand_mb = candidates_flat[b:e]
            mark_mb = seq_x_mark_flat[b:e] if seq_x_mark_flat is not None else None
            rand_ctx = (
                random_states[i] if random_states is not None else RandContext(cand_mb)
            )
            ctx_mgr = nullcontext if with_grad else torch.no_grad
            with ctx_mgr():
                with rand_ctx:
                    sv = self._encode_candidate_raw(cand_mb, mark_mb)
            yield sv, rand_ctx

    def _loss_and_cache_grads(
        self,
        ctx_state_vecs_list: list[Tensor],
        cand_state_vecs_list: list[Tensor],
        B: int,
        N: int,
        batch_mask,
        labels: Tensor,
    ) -> Tensor:
        ctx_states = torch.cat(ctx_state_vecs_list, dim=0).detach().requires_grad_(True)
        cand_states = (
            torch.cat(cand_state_vecs_list, dim=0).detach().requires_grad_(True)
        )

        context_vecs = self._poly_context_vecs(ctx_states, batch_mask)
        cand_vecs = self._poly_candidate_vecs(cand_states, B, N, batch_mask)
        scores = self._compute_scores(context_vecs, cand_vecs)
        loss = F.cross_entropy(scores, labels)
        loss.backward()

        mb = self.mini_batch_size
        ctx_grads = [ctx_states.grad[b : b + mb].clone() for b in range(0, B, mb)]
        cand_grads = [cand_states.grad[b : b + mb].clone() for b in range(0, B * N, mb)]
        self.cache = [ctx_grads, cand_grads]
        return loss.detach()

    def _backward_hook(
        self,
        grad_output: Tensor,
        context: Tensor,
        candidates_flat: Tensor,
        seq_x_mark_ctx,
        seq_x_mark_cand_flat,
        batch_mask,
        ctx_rand_states: list[RandContext],
        cand_rand_states: list[RandContext],
    ):
        assert self.cache is not None
        ctx_grads, cand_grads = self.cache
        B = context.size(0)
        BN = candidates_flat.size(0)
        mb = self.mini_batch_size

        with torch.enable_grad():
            for i, (b, grad_mb) in enumerate(zip(range(0, B, mb), ctx_grads)):
                e = min(b + mb, B)
                mark_mb = seq_x_mark_ctx[b:e] if seq_x_mark_ctx is not None else None
                mask_mb = batch_mask[b:e] if batch_mask is not None else None
                with ctx_rand_states[i]:
                    sv = self._encode_context_raw(context[b:e], mark_mb, mask_mb)
                (torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output).backward()

            for i, (b, grad_mb) in enumerate(zip(range(0, BN, mb), cand_grads)):
                e = min(b + mb, BN)
                mark_mb = (
                    seq_x_mark_cand_flat[b:e]
                    if seq_x_mark_cand_flat is not None
                    else None
                )
                with cand_rand_states[i]:
                    sv = self._encode_candidate_raw(candidates_flat[b:e], mark_mb)
                (torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output).backward()

        self.cache = None

    @torch.no_grad()
    def encode_context(self, context, seq_x_mark=None, batch_mask=None):
        sv = self._encode_context_raw(context, seq_x_mark, batch_mask)
        return self._poly_context_vecs(sv, batch_mask)

    @torch.no_grad()
    def encode_candidate(self, candidates, seq_x_mark=None, batch_mask=None):
        B, N, S, C = candidates.shape
        flat = candidates.reshape(B * N, S, C)
        if seq_x_mark is not None:
            seq_x_mark = (
                seq_x_mark.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, S, -1)
            )
        sv = self._encode_candidate_raw(flat, seq_x_mark)
        return self._poly_candidate_vecs(sv, B, N, batch_mask)

    @torch.no_grad()
    def compute_similarity(self, context_vecs, cand_vecs):
        return self._compute_scores(context_vecs, cand_vecs)

    def forward(
        self,
        context,
        pos_candidates,
        neg_candidates,
        seq_x_mark=None,
        batch_mask=None,
        hardness_mode: (
            Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None
        ) = None,
    ):
        B = context.size(0)
        device = context.device
        if batch_mask is not None:
            batch_mask = batch_mask.to(device)

        all_cands, labels = self._build_candidate_pool(
            pos_candidates, neg_candidates, hardness_mode
        )
        N = all_cands.size(1)
        cands_flat = all_cands.reshape(B * N, *all_cands.shape[2:])

        seq_x_mark_cand = None
        if seq_x_mark is not None:
            seq_x_mark_cand = (
                seq_x_mark.unsqueeze(1)
                .repeat(1, N, 1, 1)
                .reshape(B * N, seq_x_mark.size(1), -1)
            )

        ctx_state_vecs, ctx_rand_states = [], []
        cand_state_vecs, cand_rand_states = [], []

        for sv, rng in tqdm.tqdm(
            self._iter_context_minibatches(
                context, seq_x_mark, batch_mask, with_grad=False
            ),
            total=(B + self.mini_batch_size - 1) // self.mini_batch_size,
            desc="Encoding context",
            disable=not self.show_progress_bar,
        ):
            ctx_state_vecs.append(sv)
            ctx_rand_states.append(rng)

        for sv, rng in tqdm.tqdm(
            self._iter_candidate_minibatches(
                cands_flat, seq_x_mark_cand, with_grad=False
            ),
            total=(B * N + self.mini_batch_size - 1) // self.mini_batch_size,
            desc="Encoding candidates",
            disable=not self.show_progress_bar,
        ):
            cand_state_vecs.append(sv)
            cand_rand_states.append(rng)

        loss = self._loss_and_cache_grads(
            ctx_state_vecs, cand_state_vecs, B, N, batch_mask, labels
        )

        surrogate_anchor = sum(p.sum() * 0 for p in self.parameters())

        def _hook(grad):
            self._backward_hook(
                grad,
                context,
                cands_flat,
                seq_x_mark,
                seq_x_mark_cand,
                batch_mask,
                ctx_rand_states,
                cand_rand_states,
            )

        surrogate_anchor.register_hook(_hook)
        return loss + surrogate_anchor


# from .Transformer import Model as TransformerModel
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from retriever.layers import dot_attention
# from contextlib import nullcontext
# from torch import Tensor
# from typing import Optional
# import tqdm


# def _get_device_states(*args):
#     unique_devices = {t.device for t in args if isinstance(t, Tensor) and t.is_cuda}
#     return list(unique_devices), [torch.cuda.get_rng_state(d) for d in unique_devices]


# def _set_device_states(devices, states):
#     for dev, state in zip(devices, states):
#         torch.cuda.set_rng_state(state, dev)


# class RandContext:
#     """Captures & restores RNG state so mini-batch re-encoding is deterministic."""

#     def __init__(self, *tensors):
#         self.fwd_cpu_state = torch.get_rng_state()
#         self.fwd_gpu_devices, self.fwd_gpu_states = _get_device_states(*tensors)

#     def __enter__(self):
#         self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
#         self._fork.__enter__()
#         torch.set_rng_state(self.fwd_cpu_state)
#         _set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self._fork.__exit__(exc_type, exc_val, exc_tb)
#         self._fork = None


# # TransformerPolyModel

# class TransformerPolyModel(nn.Module):
#     """
#      Uses the GradCache trick to
#     decouple encoder memory from batch size.

#     Parameters
#    ------
#     configs          : model config (same as before)
#     mini_batch_size  : how many samples to encode at once in stage 1 & 3.
#                        Tune this to fit GPU memory. Does NOT affect the
#                        effective batch size or training signal.
#     show_progress_bar: show tqdm bar during stage-1 encoding.
#     """

#     def __init__(self, configs, mini_batch_size: int = 8, show_progress_bar: bool = False):
#         super().__init__()
#         self.task_name   = configs.task_name
#         self.pred_len    = configs.pred_len
#         self.vec_dim     = 64
#         self.poly_m      = configs.poly_m
#         self.topm        = configs.topm
#         self.n_period    = 1
#         self.channels    = configs.enc_in
#         self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.poly_code_embeddings = nn.Embedding(self.poly_m + 2, configs.d_model)

#         self.transformer_ctx = TransformerModel(configs=configs)
#         self.transformer_pos = TransformerModel(configs=configs)

#         self.dropout          = nn.Dropout(configs.dropout)
#         self.context_fc       = nn.Linear(configs.d_model, self.vec_dim)
#         self.pos_candidate_fc = nn.Linear(configs.d_model, self.vec_dim)

#         self.mini_batch_size   = mini_batch_size
#         self.show_progress_bar = show_progress_bar


#         self.cache: Optional[list[list[Tensor]]] = None
#         self.random_states: Optional[list[list[RandContext]]] = None

#     # Low-level encoders (return raw transformer state vecs, no poly-attn)

#     def _encode_context_raw(self, context, seq_x_mark=None, batch_mask=None):
#         """
#         Returns
#         state_vecs_ctx : (B, T, C, d_model)  — raw transformer output for context
#         """
#         state_vecs_ctx, _ = self.transformer_ctx(context, seq_x_mark, mask=batch_mask)
#         return state_vecs_ctx  # (B, T, C, d_model)

#     def _encode_candidate_raw(self, candidates_mb, seq_x_mark_mb=None):
#         """
#         Parameters
#         candidates_mb : (mb, S, C)  — reshaped mini-batch of candidates.

#         Returns
#         state_vecs : (mb, T, C, d_model)
#         """
#         state_vecs, _ = self.transformer_pos(candidates_mb, seq_x_mark_mb)
#         return state_vecs  # (mb, T, C, d_model)

#     # Poly-attention helpers (cheap, run on cached vecs)

#     def _poly_context_vecs(self, state_vecs_ctx, batch_mask=None):
#         """
#         Applies poly-attention on top of cached context state vecs.

#         Parameters
#         state_vecs_ctx : (B, T, C, d_model)

#         Returns
#         context_vecs : (B, poly_m, vec_dim)  normalised
#         """
#         device = self.poly_code_embeddings.weight.device
#         B, T, C, d_model = state_vecs_ctx.shape

#         poly_code_ids = torch.arange(self.poly_m, device=device).unsqueeze(0).expand(B, self.poly_m)
#         poly_codes = self.poly_code_embeddings(poly_code_ids)            # (B, poly_m, d_model)
#         poly_codes = poly_codes.unsqueeze(2).expand(B, self.poly_m, C, d_model)

#         # Transpose for 4-D attention: [B, C, *, D]
#         sv  = state_vecs_ctx.transpose(1, 2)                             # (B, C, T, D)
#         pc  = poly_codes.transpose(1, 2)                                 # (B, C, poly_m, D)
#         v_mask = batch_mask.transpose(1, 2) if batch_mask is not None else None

#         ctx_vecs = dot_attention(q=pc, k=sv, v=sv, v_mask=v_mask, dropout=self.dropout)
#         ctx_vecs = ctx_vecs.transpose(1, 2)          # (B, poly_m, C, D)
#         ctx_vecs = ctx_vecs.mean(dim=2)              # (B, poly_m, D)
#         ctx_vecs = self.context_fc(self.dropout(ctx_vecs))               # (B, poly_m, vec_dim)
#         return F.normalize(ctx_vecs, p=2, dim=-1)

#     def _poly_candidate_vecs(self, state_vecs, B, N, batch_mask=None, T_orig=None):
#         """
#         Applies poly-attention on top of cached candidate state vecs.

#         Parameters
#         state_vecs : (B*N, T, C, d_model)
#         B, N       : original batch and candidate counts
#         T_orig     : T dimension from the context (for mask expand)

#         Returns
#         cand_vecs  : (B, N, vec_dim)  normalised
#         """
#         device = self.poly_code_embeddings.weight.device
#         BN, T, C, D = state_vecs.shape

#         poly_code_ids = torch.full((BN, 1), self.poly_m + 1, dtype=torch.long, device=device)
#         poly_codes = self.poly_code_embeddings(poly_code_ids)            # (BN, 1, D)
#         poly_codes = poly_codes.unsqueeze(2).expand(BN, 1, C, D)        # (BN, 1, C, D)

#         sv = state_vecs.transpose(1, 2)                                  # (BN, C, T, D)
#         pc = poly_codes.transpose(1, 2)                                  # (BN, C, 1, D)

#         v_mask = None
#         if batch_mask is not None:
#             v_mask = batch_mask.transpose(1, 2)                          # (B, C, T)
#             v_mask = v_mask.unsqueeze(1).repeat(1, N, 1, 1).reshape(BN, C, T)

#         cand_vec = dot_attention(q=pc, k=sv, v=sv, v_mask=v_mask, dropout=self.dropout)
#         cand_vec = cand_vec.transpose(1, 2).squeeze(1).mean(dim=1)      # (BN, D)
#         cand_vec = self.pos_candidate_fc(self.dropout(cand_vec))
#         return F.normalize(cand_vec, p=2, dim=-1).reshape(B, N, self.vec_dim)

#     # Contrastive scoring  (poly cross-attention ctx ↔ cand, then dot)

#     def _compute_scores(self, context_vecs, cand_vecs):
#         """
#         Parameters
#         context_vecs : (B, poly_m, vec_dim)
#         cand_vecs    : (B, N_total, vec_dim)   N_total = 1 + num_neg

#         Returns
#         scores : (B, N_total)
#         """
#         B        = context_vecs.size(0)
#         N_total  = cand_vecs.size(1)

#         q = cand_vecs.reshape(-1, 1, self.vec_dim)                       # (B*N_total, 1, vec_dim)
#         k = (context_vecs
#              .unsqueeze(1)
#              .expand(B, N_total, self.poly_m, self.vec_dim)
#              .reshape(-1, self.poly_m, self.vec_dim))                    # (B*N_total, poly_m, vec_dim)

#         agg = dot_attention(q, k, k, dropout=self.dropout)              # (B*N_total, 1, vec_dim)
#         agg = F.normalize(agg.squeeze(1), p=2, dim=-1)                  # (B*N_total, vec_dim)

#         scores = torch.sum(agg * q.squeeze(1), dim=-1).reshape(B, N_total)
#         return scores * torch.exp(self.temperature)

#     # Mini-batch iterators  (Stage 1 & Stage 3)

#     def _iter_context_minibatches(self, context, seq_x_mark, batch_mask, with_grad, random_states=None):
#         """Yield (raw_state_vecs, rand_ctx) for each mini-batch of context."""
#         B = context.size(0)
#         for i, b in enumerate(range(0, B, self.mini_batch_size)):
#             e = min(b + self.mini_batch_size, B)
#             ctx_mb  = context[b:e]
#             mark_mb = seq_x_mark[b:e] if seq_x_mark is not None else None
#             mask_mb = batch_mask[b:e] if batch_mask is not None else None

#             grad_ctx  = nullcontext if with_grad else torch.no_grad
#             rand_ctx  = random_states[i] if random_states is not None else RandContext(ctx_mb)

#             with grad_ctx():
#                 with rand_ctx if not with_grad else rand_ctx:
#                     sv = self._encode_context_raw(ctx_mb, mark_mb, mask_mb)
#             yield sv, rand_ctx

#     def _iter_candidate_minibatches(self, candidates_flat, seq_x_mark_flat, with_grad, random_states=None):
#         """
#         Parameters
#        ------
#         candidates_flat  : (B*N, S, C)  flattened candidates
#         seq_x_mark_flat  : (B*N, S, F) or None
#         """
#         BN = candidates_flat.size(0)
#         for i, b in enumerate(range(0, BN, self.mini_batch_size)):
#             e = min(b + self.mini_batch_size, BN)
#             cand_mb = candidates_flat[b:e]
#             mark_mb = seq_x_mark_flat[b:e] if seq_x_mark_flat is not None else None

#             grad_ctx = nullcontext if with_grad else torch.no_grad
#             rand_ctx = random_states[i] if random_states is not None else RandContext(cand_mb)

#             with grad_ctx():
#                 with rand_ctx if not with_grad else rand_ctx:
#                     sv = self._encode_candidate_raw(cand_mb, mark_mb)
#             yield sv, rand_ctx

#     # Stage 2: loss + gradient caching

#     def _loss_and_cache_grads(
#         self,
#         ctx_state_vecs_list: list[Tensor],   # list of (mb, T, C, D) per ctx mini-batch
#         cand_state_vecs_list: list[Tensor],  # list of (mb, T, C, D) per cand mini-batch
#         B: int, N: int,
#         batch_mask,
#     ) -> Tensor:
#         """
#         Stage 2:
#           1. Assemble full tensors from mini-batch list (detach so no graph is stored).
#           2. Apply poly-attention → context_vecs, cand_vecs.
#           3. Compute contrastive loss, call loss.backward() to get ∂L/∂state_vecs.
#           4. Cache those gradients for Stage 3.

#         Returns the loss scalar (detached from transformer graph, used for logging).
#         """
#         #Concatenate cached raw encoder outputs
#         ctx_states  = torch.cat(ctx_state_vecs_list,  dim=0)   # (B, T, C, D)
#         cand_states = torch.cat(cand_state_vecs_list, dim=0)   # (B*N, T, C, D)

#         # Detach so we don't hold the encoder graph (memory saving)
#         ctx_states  = ctx_states.detach().requires_grad_(True)
#         cand_states = cand_states.detach().requires_grad_(True)

#         #Lightweight poly-attention (runs on detached vecs)
#         context_vecs = self._poly_context_vecs(ctx_states, batch_mask)          # (B, poly_m, vec_dim)
#         cand_vecs    = self._poly_candidate_vecs(cand_states, B, N, batch_mask) # (B, N, vec_dim)

#         #Contrastive loss
#         scores = self._compute_scores(context_vecs, cand_vecs)   # (B, N)
#         labels = torch.zeros(B, dtype=torch.long, device=scores.device)
#         loss   = F.cross_entropy(scores, labels)

#         #Backward up to detached inputs → cache ∂L/∂raw_states
#         loss.backward()

#         # Cache gradients split by original mini-batches
#         ctx_grads  = []
#         cand_grads = []
#         mb = self.mini_batch_size
#         for b in range(0, B,    mb):
#             ctx_grads.append(ctx_states.grad[b : b + mb].clone())
#         for b in range(0, B * N, mb):
#             cand_grads.append(cand_states.grad[b : b + mb].clone())

#         self.cache = [ctx_grads, cand_grads]
#         return loss.detach()

#     # Stage 3: backward hook re-encoder

#     def _backward_hook(
#         self,
#         grad_output: Tensor,
#         context: Tensor,
#         candidates_flat: Tensor,
#         seq_x_mark_ctx,
#         seq_x_mark_cand_flat,
#         batch_mask,
#         ctx_rand_states: list[RandContext],
#         cand_rand_states: list[RandContext],
#     ):
#         """
#         Re-encodes each mini-batch WITH grad, then calls surrogate.backward()
#         to route cached gradients into the actual model parameters.
#         """
#         assert self.cache is not None
#         ctx_grads, cand_grads = self.cache

#         B  = context.size(0)
#         BN = candidates_flat.size(0)
#         mb = self.mini_batch_size

#         with torch.enable_grad():
#             # -- context mini-batches --
#             for i, (b, grad_mb) in enumerate(zip(range(0, B, mb), ctx_grads)):
#                 e        = min(b + mb, B)
#                 ctx_mb   = context[b:e]
#                 mark_mb  = seq_x_mark_ctx[b:e] if seq_x_mark_ctx is not None else None
#                 rng      = ctx_rand_states[i]
#                 with rng:
#                     sv = self._encode_context_raw(ctx_mb, mark_mb,
#                                                    batch_mask[b:e] if batch_mask is not None else None)
#                 surrogate = torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output
#                 surrogate.backward()

#             # -- candidate mini-batches --
#             for i, (b, grad_mb) in enumerate(zip(range(0, BN, mb), cand_grads)):
#                 e        = min(b + mb, BN)
#                 cand_mb  = candidates_flat[b:e]
#                 mark_mb  = seq_x_mark_cand_flat[b:e] if seq_x_mark_cand_flat is not None else None
#                 rng      = cand_rand_states[i]
#                 with rng:
#                     sv = self._encode_candidate_raw(cand_mb, mark_mb)
#                 surrogate = torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output
#                 surrogate.backward()

#         self.cache = None


#     # Inference: encode_context / encode_candidate

#     @torch.no_grad()
#     def encode_context(self, context, seq_x_mark=None, batch_mask=None):
#         """Full encode for inference."""
#         sv = self._encode_context_raw(context, seq_x_mark, batch_mask)
#         return self._poly_context_vecs(sv, batch_mask)

#     @torch.no_grad()
#     def encode_candidate(self, candidates, seq_x_mark=None, batch_mask=None):
#         """
#         candidates : (B, N, S, C) or (B, 1, S, C)
#         Returns    : (B, N, vec_dim)
#         """
#         B, N, S, C = candidates.shape
#         flat = candidates.reshape(B * N, S, C)
#         if seq_x_mark is not None:
#             seq_x_mark = seq_x_mark.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, S, -1)
#         sv = self._encode_candidate_raw(flat, seq_x_mark)
#         return self._poly_candidate_vecs(sv, B, N, batch_mask)

#     @torch.no_grad()
#     def compute_similarity(self, context_vecs, cand_vecs):
#         """Wrapper around _compute_scores for eval."""
#         return self._compute_scores(context_vecs, cand_vecs)

#     # Forward  (GradCache 3-stage)

#     def forward(
#         self,
#         context,          # (B, T_ctx, C)
#         pos_candidates,   # (B, S, C)
#         neg_candidates,   # (B, M, S, C)
#         seq_x_mark=None,  # (B, T, F) or None
#         batch_mask=None,  # (B, T, C) or None
#     ):
#         B   = context.size(0)
#         device = context.device
#         if batch_mask is not None:
#             batch_mask = batch_mask.to(device)

#         # Prepare flat candidate tensor  (B*N, S, C)
#         pos_exp = pos_candidates.unsqueeze(1)                   # (B, 1, S, C)
#         all_cands = torch.cat([pos_exp, neg_candidates], dim=1) # (B, N, S, C)
#         N = all_cands.size(1)
#         cands_flat = all_cands.reshape(B * N, *all_cands.shape[2:])

#         # Flat mark for candidates
#         seq_x_mark_cand = None
#         if seq_x_mark is not None:
#             seq_x_mark_cand = (seq_x_mark
#                                .unsqueeze(1)
#                                .repeat(1, N, 1, 1)
#                                .reshape(B * N, seq_x_mark.size(1), -1))

#         # Stage 1: encode without grad, save RNG states
#         ctx_state_vecs  = []
#         cand_state_vecs = []
#         ctx_rand_states  = []
#         cand_rand_states = []

#         for sv, rng in tqdm.tqdm(
#             self._iter_context_minibatches(context, seq_x_mark, batch_mask, with_grad=False),
#             total=(B + self.mini_batch_size - 1) // self.mini_batch_size,
#             desc="Encoding context",
#             disable=not self.show_progress_bar,
#         ):
#             ctx_state_vecs.append(sv)
#             ctx_rand_states.append(rng)

#         for sv, rng in tqdm.tqdm(
#             self._iter_candidate_minibatches(cands_flat, seq_x_mark_cand, with_grad=False),
#             total=(B * N + self.mini_batch_size - 1) // self.mini_batch_size,
#             desc="Encoding candidates",
#             disable=not self.show_progress_bar,
#         ):
#             cand_state_vecs.append(sv)
#             cand_rand_states.append(rng)

#         # Stage 2: compute loss on detached reps, cache ∂L/∂reps
#         loss = self._loss_and_cache_grads(
#             ctx_state_vecs, cand_state_vecs, B, N, batch_mask
#         )

#         # Stage 3: surrogate backward — re-encode with grad, chain grads
#         # We need a scalar that has a grad_fn so we can attach a hook.
#         # Create a surrogate zero tied to parameters.
#         surrogate_anchor = sum(p.sum() * 0 for p in self.parameters())

#         def _hook(grad):
#             self._backward_hook(
#                 grad,
#                 context, cands_flat,
#                 seq_x_mark, seq_x_mark_cand,
#                 batch_mask,
#                 ctx_rand_states, cand_rand_states,
#             )

#         surrogate_anchor.register_hook(_hook)
#         # Trigger the hook on the next backward call.
#         # loss is detached; we return (loss + surrogate_anchor) so that
#         # optimizer.step() after loss.backward() fires the hook.
#         return loss + surrogate_anchor
