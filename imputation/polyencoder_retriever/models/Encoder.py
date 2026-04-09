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
from polyencoder_retriever.layers import dot_attention
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
    """
    Captures CPU + GPU RNG state at construction time.
    Each call to __enter__ / __exit__ restores that snapshot — reusable across
    multiple `with rng:` blocks (stage-1 encode and stage-3 re-encode).
    """

    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = _get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        _set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)
        return self

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
        self.batch_size = configs.batch_size
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

    def _encode_context_raw(self, ctx_mb, mark_mb, mask_mb):
        state_vecs_ctx, _ = self.transformer_ctx(ctx_mb, mark_mb, mask=mask_mb)
        return state_vecs_ctx

    def _encode_candidate_raw(self, cand_mb, mark_mb):
        state_vecs_cand, _ = self.transformer_pos(cand_mb, mark_mb)
        return state_vecs_cand

    def _poly_context_vecs(self, state_vecs_ctx, batch_mask=None):
        device = self.poly_code_embeddings.weight.device
        B, T, C, d_model = state_vecs_ctx.shape

        poly_code_ids = (
            torch.arange(self.poly_m, device=device).unsqueeze(0).expand(B, self.poly_m)
        )
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        poly_codes = poly_codes.unsqueeze(2).expand(B, self.poly_m, C, d_model)

        sv = state_vecs_ctx.transpose(1, 2)  # (B, C, T, d_model)
        pc = poly_codes.transpose(1, 2)  # (B, C, self.poly_m, d_model)
        v_mask = batch_mask.transpose(1, 2) if batch_mask is not None else None

        ctx_vecs = dot_attention(q=pc, k=sv, v=sv, v_mask=v_mask, dropout=self.dropout)
        ctx_vecs = ctx_vecs.transpose(1, 2).mean(dim=2)
        ctx_vecs = self.context_fc(self.dropout(ctx_vecs))
        return F.normalize(ctx_vecs, p=2, dim=-1)

    def _poly_candidate_vecs(self, state_vecs, N: int, batch_mask=None):
        device = self.poly_code_embeddings.weight.device
        BN, T, C, d_model = state_vecs.shape
        assert BN % N == 0, f"BN ({BN}) must be divisible by N ({N})"
        B = BN // N

        poly_code_ids = torch.full(
            (BN, 1), self.poly_m + 1, dtype=torch.long, device=device
        )
        poly_codes = (
            self.poly_code_embeddings(poly_code_ids)
            .unsqueeze(2)
            .expand(BN, 1, C, d_model)
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
        """
        context_vecs: (B, self.poly_m, self.vec_dim)
        cand_vecs: (B,N,vec_dim)
        with N: # pos + neg samples (e.g., 16)
        """
        B = context_vecs.size(0)
        N = cand_vecs.size(1)

        q = cand_vecs.reshape(-1, 1, self.vec_dim)
        k = (
            context_vecs.unsqueeze(1)
            .expand(B, N, self.poly_m, self.vec_dim)
            .reshape(-1, self.poly_m, self.vec_dim)
        )

        agg = dot_attention(q, k, k, dropout=self.dropout)
        agg = F.normalize(agg.squeeze(1), p=2, dim=-1)
        scores = torch.sum(agg * q.squeeze(1), dim=-1).reshape(B, N)
        return scores * torch.exp(self.temperature)

    def _build_candidate_pool(
        self,
        pos_candidates: Tensor,
        neg_candidates: Tensor,
        hardness_mode: (
            Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None
        ),
    ) -> tuple[Tensor, Tensor, int]:
        """
        None / "in_batch_negatives" : score (B, B), labels = arange(B)
        "hard_negatives"            : score (B, 1+M), labels = zeros(B)
        "all_negatives"             : score (B, B+M), labels = arange(B)
        """
        B, S, C = pos_candidates.shape
        pos_exp = pos_candidates.unsqueeze(1)

        if hardness_mode is None or hardness_mode == "in_batch_negatives":
            all_cands = pos_candidates.unsqueeze(0).expand(B, B, S, C).contiguous()
            labels = torch.arange(B, dtype=torch.long, device=pos_candidates.device)
            N = B  # #pos + neg samples

        if hardness_mode == "hard_negatives":
            all_cands = torch.cat([pos_exp, neg_candidates], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=pos_candidates.device)
            N = all_cands.size(1)  # #pos + neg samples

        if hardness_mode == "all_negatives":
            in_batch = pos_candidates.unsqueeze(0).expand(B, B, S, C).contiguous()
            all_cands = torch.cat([in_batch, neg_candidates], dim=1)
            labels = torch.arange(B, dtype=torch.long, device=pos_candidates.device)
            N = all_cands.size(1)  # #pos + neg samples

        return all_cands, labels, N
        raise ValueError(f"Unknown hardness_mode: {hardness_mode!r}")

    def _iter_context_minibatches(
        self,
        context: Tensor,
        seq_x_mark,
        batch_mask,
        with_grad: bool,
        random_states: Optional[list[RandContext]] = None,
    ):
        B = context.size(0)
        mb = self.mini_batch_size
        for i, b in enumerate(range(0, B, mb)):
            e = min(b + mb, B)
            ctx_mb = context[b:e]
            mark_mb = seq_x_mark[b:e] if seq_x_mark is not None else None
            mask_mb = batch_mask[b:e] if batch_mask is not None else None

            # Capture RNG state BEFORE the forward pass (stage 1).
            # On stage 3 the same object is reused as a context manager again.
            rng = random_states[i] if random_states is not None else RandContext(ctx_mb)

            grad_cm = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_cm, rng:
                sv = self._encode_context_raw(ctx_mb, mark_mb, mask_mb)

            yield sv, rng

    def _iter_candidate_minibatches(
        self,
        candidates_flat: Tensor,
        seq_x_mark_flat,
        with_grad: bool,
        random_states: Optional[list[RandContext]] = None,
    ):
        BN = candidates_flat.size(0)
        mb = self.mini_batch_size
        for i, b in enumerate(range(0, BN, mb)):
            e = min(b + mb, BN)
            cand_mb = candidates_flat[b:e]
            mark_mb = seq_x_mark_flat[b:e] if seq_x_mark_flat is not None else None

            rng = (
                random_states[i] if random_states is not None else RandContext(cand_mb)
            )

            grad_cm = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_cm, rng:
                sv = self._encode_candidate_raw(cand_mb, mark_mb)

            yield sv, rng

    def _loss_and_cache_grads(
        self,
        ctx_state_vecs_list: list[Tensor],
        cand_state_vecs_list: list[Tensor],
        N: int,
        batch_mask,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ctx_states = torch.cat(ctx_state_vecs_list, dim=0).detach().requires_grad_(True)
        cand_states = (
            torch.cat(cand_state_vecs_list, dim=0).detach().requires_grad_(True)
        )

        B: int = ctx_states.size(0)
        BN: int = cand_states.size(0)
        assert BN == B * N, f"Expected cand BN={B*N}, got {BN}"

        context_vecs = self._poly_context_vecs(ctx_states, batch_mask)
        cand_vecs = self._poly_candidate_vecs(cand_states, N, batch_mask)
        scores = self._compute_scores(context_vecs, cand_vecs)
        labels = labels.to(ctx_states.device)
        loss = F.cross_entropy(scores, labels)

        if self.training:
            loss.backward()
            mb = self.mini_batch_size
            ctx_grads = [ctx_states.grad[b : b + mb].clone() for b in range(0, B, mb)]
            cand_grads = [
                cand_states.grad[b : b + mb].clone() for b in range(0, B * N, mb)
            ]
            self.cache = [ctx_grads, cand_grads, N]

        return loss.detach(), scores

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
        ctx_grads, cand_grads, N = self.cache
        B = context.size(0)
        BN = candidates_flat.size(0)
        assert BN == B * N, f"Expected cand BN={B*N}, got {BN}"
        mb = self.mini_batch_size

        with torch.enable_grad():
            for (b, grad_mb), rng in zip(
                ((b, ctx_grads[i]) for i, b in enumerate(range(0, B, mb))),
                ctx_rand_states,
            ):
                e = min(b + mb, B)
                mark_mb = seq_x_mark_ctx[b:e] if seq_x_mark_ctx is not None else None
                mask_mb = batch_mask[b:e] if batch_mask is not None else None
                with rng:
                    sv = self._encode_context_raw(context[b:e], mark_mb, mask_mb)
                (torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output).backward()

            for (b, grad_mb), rng in zip(
                ((b, cand_grads[i]) for i, b in enumerate(range(0, BN, mb))),
                cand_rand_states,
            ):
                e = min(b + mb, BN)
                mark_mb = (
                    seq_x_mark_cand_flat[b:e]
                    if seq_x_mark_cand_flat is not None
                    else None
                )
                with rng:
                    sv = self._encode_candidate_raw(candidates_flat[b:e], mark_mb)
                (torch.dot(sv.flatten(), grad_mb.flatten()) * grad_output).backward()

        self.cache = None

    @torch.no_grad()
    def encode_context(self, context, seq_x_mark=None, batch_mask=None):
        sv = self._encode_context_raw(context, seq_x_mark, batch_mask)
        return self._poly_context_vecs(sv, batch_mask)

    @torch.no_grad()
    def encode_candidate(self, candidates, seq_x_mark=None):
        return self._encode_candidate_raw(cand_mb=candidates, mark_mb=seq_x_mark)

    # def encode_candidate(self, candidates, seq_x_mark=None, batch_mask=None):
    #     B, N, S, C = candidates.shape
    #     flat = candidates.reshape(B * N, S, C)
    #     if seq_x_mark is not None:
    #         seq_x_mark = seq_x_mark.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, S, -1)
    #     sv = self._encode_candidate_raw(flat, seq_x_mark)
    #     # return self._poly_candidate_vecs(sv, B, N, batch_mask)
    #     return sv

    @torch.no_grad()
    def compute_similarity(self, context_vecs, cand_vecs, N, batch_mask):
        cand_vecs = self._poly_candidate_vecs(cand_vecs, N, batch_mask)
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

        all_cands, labels, N = self._build_candidate_pool(
            pos_candidates, neg_candidates, hardness_mode
        )
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

        loss, scores = self._loss_and_cache_grads(
            ctx_state_vecs, cand_state_vecs, N, batch_mask, labels
        )

        surrogate_anchor = sum(p.sum() * 0 for p in self.parameters())

        if self.training:

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

        return loss + surrogate_anchor, scores


# from .Transformer import Model as TransformerModel
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from polyencoder_retriever.layers import dot_attention

# class TransformerPolyModel(nn.Module):
#     def __init__(self, configs):
#         super(TransformerPolyModel, self).__init__()
#         self.task_name = configs.task_name
#         self.pred_len = configs.pred_len
#         self.vec_dim = 64
#         self.poly_m = configs.poly_m # #context codes
#         self.topm = configs.topm # #positive samples
#         self.n_period = 1 # != RAFT
#         self.channels = configs.enc_in
#         self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.poly_code_embeddings = nn.Embedding(self.poly_m + 2, configs.d_model)

#         self.period_num = 1

#         self.transformer_ctx = TransformerModel(configs=configs)
#         self.transformer_pos = TransformerModel(configs=configs)

#         self.dropout = nn.Dropout(configs.dropout)
#         self.context_fc = nn.Linear(in_features=configs.d_model, out_features=self.vec_dim)
#         self.pos_candidate_fc = nn.Linear(in_features=configs.d_model, out_features=self.vec_dim)


#     def encode_context(self, context, seq_x_mark=None, batch_mask=None):
#         device = self.poly_code_embeddings.weight.device
#         B = context.size(0)
#         state_vecs_ctx, state_vecs_ctx_reshaped = self.transformer_ctx(context, seq_x_mark, mask = batch_mask) # (B, T, C, d_model)

#         T, C, d_model = state_vecs_ctx.shape[1], state_vecs_ctx.shape[2], state_vecs_ctx.shape[3]

#         # poly_codes: [B, poly_m, C, d_model]
#         poly_code_ids = torch.arange(self.poly_m, device=device) + 1
#         poly_code_ids = poly_code_ids.unsqueeze(0).expand(B, self.poly_m)
#         poly_codes = self.poly_code_embeddings(poly_code_ids) # [B, poly_m, d_model]
#         poly_codes = poly_codes.unsqueeze(2).expand(B, self.poly_m, C, d_model)

#         # transpose for 4D attention
#         # state_vecs_ctx: [B, C, T, D]
#         # poly_codes: [B, C, poly_m, D]
#         state_vecs_ctx = state_vecs_ctx.transpose(1, 2)
#         poly_codes = poly_codes.transpose(1, 2)

#         # v_mask: [B, T, C] -> [B, C, T]
#         v_mask = batch_mask.transpose(1, 2) if batch_mask is not None else None

#         context_vecs = dot_attention(q=poly_codes, k=state_vecs_ctx, v=state_vecs_ctx,
#          v_mask=v_mask, dropout=self.dropout)
#         context_vecs = context_vecs.transpose(1, 2)
#         context_vecs = context_vecs.mean(dim=2) # or flatten
#         context_vecs = self.context_fc(self.dropout(context_vecs)) # (B, poly_m, vec_dim)

#         return F.normalize(context_vecs, p=2, dim=-1)

#     def encode_candidate(self, candidates, seq_x_mark=None, batch_mask=None, eval=False):
#         device = self.poly_code_embeddings.weight.device
#         B, N, S, C = candidates.shape
#         candidates_reshaped = candidates.reshape(B * N, S, C)

#         if seq_x_mark is not None:
#             seq_x_mark = seq_x_mark.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, S, -1)

#         state_vecs_pos, _ = self.transformer_pos(candidates_reshaped, seq_x_mark)

#         if eval:
#             return state_vecs_pos

#         T, D = state_vecs_pos.shape[1], state_vecs_pos.shape[3]

#         poly_code_ids = torch.full((B * N, 1), self.poly_m + 1, dtype=torch.long, device=device)
#         poly_codes = self.poly_code_embeddings(poly_code_ids).unsqueeze(2).expand(B * N, 1, C, D)

#         state_vecs_pos = state_vecs_pos.transpose(1, 2) # [B*N, C, T, D]
#         poly_codes = poly_codes.transpose(1, 2) # [B*N, C, 1, D]
#         v_mask = None
#         if batch_mask is not None:
#             # batch_mask: [B, T, C] -> [B, C, T]
#             v_mask = batch_mask.transpose(1, 2)
#             # Expand [B, N, C, T] --> reshape [B*N, C, T]
#             v_mask = v_mask.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, C, T)

#         pos_vec = dot_attention(q=poly_codes, k=state_vecs_pos, v=state_vecs_pos,
#                                  v_mask=v_mask, dropout=self.dropout)

#         pos_vec = pos_vec.transpose(1, 2).squeeze(1).mean(dim=1)
#         pos_vec = self.pos_candidate_fc(self.dropout(pos_vec))

#         return F.normalize(pos_vec, p=2, dim=-1).reshape(B, N, self.vec_dim)


#     def compute_similarity(self, context_vecs, state_vecs_pos):
#         # context_vecs: [B, poly_m, vec_dim]
#         # state_vecs_pos: [B, N, T, C, D]: output of transformer encoder (candidate)

#         B, N, T, C, D = state_vecs_pos.shape
#         device = state_vecs_pos.device

#         states_reshaped = state_vecs_pos.reshape(B * N, T, C, D)
#         # states_reshaped: [B*N, T, C, D]

#         poly_code_ids = torch.full((B * N, 1), self.poly_m + 1, dtype=torch.long, device=device)
#         poly_codes = self.poly_code_embeddings(poly_code_ids).unsqueeze(2).expand(B * N, 1, C, D)

#         states_reshaped = states_reshaped.transpose(1, 2)
#         # states_reshaped: [B*N, C, T, D]
#         poly_codes = poly_codes.transpose(1, 2)
#         # poly_codes: [B*N, C, 1, D]

#         pos_vec = dot_attention(q=poly_codes, k=states_reshaped, v=states_reshaped,
#                                  v_mask=None, dropout=self.dropout)
#         # pos_vec: [B*N, C, 1, D]

#         pos_vec = pos_vec.transpose(1, 2).squeeze(1).mean(dim=1)
#         # pos_vec: [B*N, D]

#         pos_vec = self.pos_candidate_fc(self.dropout(pos_vec))
#         candidate_vecs = F.normalize(pos_vec, p=2, dim=-1).reshape(B, N, self.vec_dim)
#         # candidate_vecs: [B, N, vec_dim]

#         q = candidate_vecs.reshape(-1, 1, self.vec_dim)
#         # q: [B * N, 1, vec_dim]

#         k = context_vecs.unsqueeze(1).expand(B, N, self.poly_m, self.vec_dim).reshape(-1, self.poly_m, self.vec_dim)
#         # k: [B * N, poly_m, vec_dim]

#         aggregated_ctx = dot_attention(q, k, k, dropout=self.dropout)
#         # aggregated_ctx: [B * N, 1, vec_dim]

#         aggregated_ctx = F.normalize(aggregated_ctx.squeeze(1), p=2, dim=-1)
#         # aggregated_ctx: [B * N, vec_dim]

#         scores = torch.sum(aggregated_ctx * q.squeeze(1), dim=-1).reshape(B, N)
#         # scores: [B, N]

#         return scores * torch.exp(self.temperature)


#     def forward(self, context, pos_candidates, neg_candidates, seq_x_mark=None, batch_mask=None):
#         B = context.size(0)
#         device = context.device
#         if batch_mask is not None:
#             batch_mask = batch_mask.to(device)

#         # Encode Context
#         context_vecs = self.encode_context(context, seq_x_mark, batch_mask)

#         # Encode Positive Candidate: [B, 1, S, C]
#         pos_vecs = self.encode_candidate(pos_candidates.unsqueeze(1), seq_x_mark, batch_mask)

#         # Encode Negative Candidates: [B, M, S, C]
#         neg_vecs = self.encode_candidate(neg_candidates, seq_x_mark, batch_mask)

#         all_cand_vecs = torch.cat([pos_vecs, neg_vecs], dim=1)
#         num_cands = all_cand_vecs.size(1)

#         # Poly-Attention
#         q = all_cand_vecs.reshape(-1, 1, self.vec_dim)
#         k = context_vecs.unsqueeze(1).expand(B, num_cands, self.poly_m, self.vec_dim).reshape(-1, self.poly_m, self.vec_dim)

#         aggregated_ctx = dot_attention(q, k, k, dropout=self.dropout)
#         aggregated_ctx = F.normalize(aggregated_ctx.squeeze(1), p=2, dim=-1)

#         scores = torch.sum(aggregated_ctx * q.squeeze(1), dim=-1).reshape(B, num_cands)
#         scores = scores * torch.exp(self.temperature)

#         labels = torch.zeros(B, dtype=torch.long, device=context.device)
#         loss = F.cross_entropy(scores, labels)

#         return loss, scores


#     # def encode_candidate(self, pos_candidate, seq_x_mark=None):
#     #     device = next(self.parameters()).device
#     #     B = pos_candidate.size(0)
#     #     state_vecs_pos = self.transformer_pos(pos_candidate, seq_x_mark)  # [B, seq_len, d_model]

#     #     poly_code_ids = torch.full((B, 1), self.poly_m + 1, dtype=torch.long, device=device)
#     #     poly_codes = self.poly_code_embeddings(poly_code_ids)  # [B, 1, d_model]

#     #     pos_candidate_vec = dot_attention(poly_codes, state_vecs_pos, state_vecs_pos, None, dropout=self.dropout)  # [B, 1, d_model]
#     #     pos_candidate_vec = pos_candidate_vec.squeeze(1)  # [B, d_model]
#     #     pos_candidate_vec = self.pos_candidate_fc(self.dropout(pos_candidate_vec))  # [B, vec_dim]
#     #     return F.normalize(pos_candidate_vec, p=2, dim=-1)

#     # def forward(self, context, pos_candidate, seq_x_mark=None):
#     #     B = context.size(0)

#     #     context_vecs = self.encode_context(context, seq_x_mark)          # [B, poly_m, vec_dim]
#     #     pos_candidate_vec = self.encode_candidate(pos_candidate, seq_x_mark)  # [B, vec_dim]

#     #     context_vecs_expand = context_vecs.unsqueeze(0).expand(B, B, self.poly_m, self.vec_dim)
#     #     context_vecs_expand = context_vecs_expand.reshape(B * B, self.poly_m, self.vec_dim)

#     #     cand_expand = pos_candidate_vec.unsqueeze(1).unsqueeze(0).expand(B, B, 1, self.vec_dim)
#     #     cand_expand = cand_expand.reshape(B * B, 1, self.vec_dim)

#     #     final_context_vec = dot_attention(
#     #         q=cand_expand,
#     #         k=context_vecs_expand,
#     #         v=context_vecs_expand,
#     #         v_mask=None,
#     #         dropout=self.dropout
#     #     )  # [B*B, 1, vec_dim]
#     #     final_context_vec = final_context_vec.squeeze(1).reshape(B, B, self.vec_dim)  # [B, B, vec_dim]
#     #     final_context_vec = F.normalize(final_context_vec, p=2, dim=-1)

#     #     cand_for_dot = pos_candidate_vec.unsqueeze(1).expand(B, B, self.vec_dim)  # [B, B, vec_dim]
#     #     dot_product = torch.sum(final_context_vec * cand_for_dot, dim=-1)  # [B, B]

#     #     mask = torch.eye(B, device=context.device)
#     #     loss = F.log_softmax(dot_product * 20, dim=-1) * mask
#     #     loss = (-loss.sum(dim=1)).mean()

#     #     return loss, dot_product
