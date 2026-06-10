"""evaluate_hgnn — faithful port of the notebook cell-25 evaluation protocol.

Filtered ranking (Bordes et al. 2013): for each positive (head, true_tail),
score the head against ALL tails of that type; set other known-positive tails to
-inf; rank = 1 + count(scores > true_score). Accumulate MRR + Hits@{1,3,10}.

Separately, sampled ROC-AUC / PR-AUC: per positive, take the true tail (label 1)
plus NEG_PER_POS_EVAL=50 random tails not in the known-positive set (label 0),
collect scores, then roc_auc_score / average_precision_score over all pairs.
The 1:50 sampling is what makes DP PR-AUC ~0.58.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

NEG_PER_POS_EVAL = 50


@torch.no_grad()
def evaluate_hgnn(mdl, pos_edge_index, relation_type, all_positive_set, x_dict,
                  num_proteins, num_effects, device, neg_per_pos=NEG_PER_POS_EVAL):
    mdl.eval()
    if relation_type == "dp":
        n_tails = num_proteins
        head_embs = x_dict["drug"]
        tail_embs = x_dict["protein"]
        score_fn = mdl.score_dp
    else:
        n_tails = num_effects
        head_embs = x_dict["drug"]
        tail_embs = x_dict["effect"]
        score_fn = mdl.score_di

    head_to_pos = {}
    for (h, t) in all_positive_set:
        head_to_pos.setdefault(h, set()).add(t)

    n_pos = pos_edge_index.shape[1]
    reciprocal_ranks = []
    hits_at = {1: 0, 3: 0, 10: 0}
    auc_y_true = []
    auc_y_score = []

    EVAL_BATCH = 64
    TAIL_CHUNK = 512

    for start in range(0, n_pos, EVAL_BATCH):
        end = min(start + EVAL_BATCH, n_pos)
        heads = pos_edge_index[0, start:end]
        true_tails = pos_edge_index[1, start:end]
        B = end - start
        h_emb = head_embs[heads]

        all_scores = torch.zeros(B, n_tails, device=device)
        for t0 in range(0, n_tails, TAIL_CHUNK):
            t1 = min(t0 + TAIL_CHUNK, n_tails)
            T = t1 - t0
            t_emb = tail_embs[t0:t1]
            h_exp = h_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            t_exp = t_emb.unsqueeze(0).expand(B, T, -1).reshape(B * T, -1)
            all_scores[:, t0:t1] = score_fn(h_exp, t_exp).reshape(B, T)

        scores_np = all_scores.cpu().numpy()

        for i in range(B):
            h = heads[i].item()
            true_t = true_tails[i].item()
            true_score = scores_np[i, true_t]
            known = head_to_pos.get(h, set())

            filt = scores_np[i].copy()
            for kt in known:
                if kt != true_t and kt < n_tails:
                    filt[kt] = -np.inf
            rank = 1 + int((filt > true_score).sum())
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1

            neg_pool = [t for t in range(n_tails) if t != true_t and t not in known]
            k_s = min(neg_per_pos, len(neg_pool))
            neg_idx = np.random.choice(neg_pool, k_s, replace=False)
            auc_y_true.append(1.0)
            auc_y_score.append(true_score)
            for ns in scores_np[i, neg_idx]:
                auc_y_true.append(0.0)
                auc_y_score.append(float(ns))

    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "hits@1": hits_at[1] / n_pos,
        "hits@3": hits_at[3] / n_pos,
        "hits@10": hits_at[10] / n_pos,
        "roc_auc": float(roc_auc_score(np.array(auc_y_true), np.array(auc_y_score))),
        "pr_auc": float(average_precision_score(np.array(auc_y_true), np.array(auc_y_score))),
    }
