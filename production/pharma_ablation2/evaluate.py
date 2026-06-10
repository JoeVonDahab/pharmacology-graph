"""Evaluation: PR-AUC (1:1 negative sampling) and filtered Hits@K (full-ranking),
matching the metrics reported in the paper's ablation table.

Hits@K protocol (standard filtered KG evaluation): for each held-out positive
(drug d, target t), score d against *every* candidate target, set the scores of
all OTHER known-true targets of d to -inf (filtered setting), then check whether
the true t ranks in the top K.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def _scores_drug_vs_all(model, drug_emb_rows, all_target_emb, relation, chunk=2048):
    """scores[i, j] = model.score(drug_emb_rows[i], all_target_emb[j])."""
    n_d = drug_emb_rows.size(0)
    n_t = all_target_emb.size(0)
    out = torch.empty(n_d, n_t, device=drug_emb_rows.device)
    for i in range(n_d):
        d = drug_emb_rows[i:i + 1].expand(n_t, -1)
        s = torch.empty(n_t, device=drug_emb_rows.device)
        for j0 in range(0, n_t, chunk):
            j1 = min(j0 + chunk, n_t)
            s[j0:j1] = model.score(d[j0:j1], all_target_emb[j0:j1], relation)
        out[i] = s
    return out


@torch.no_grad()
def pr_auc(model, x_dict, pos_edges, target_type, relation, seed=0):
    """1:1 negative sampling PR-AUC (+ ROC-AUC) on held-out positives."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    drug_emb = x_dict["drug"]
    tgt_emb = x_dict[target_type]
    n = pos_edges.size(1)
    pos_src, pos_dst = pos_edges[0], pos_edges[1]
    neg_dst = torch.randint(0, tgt_emb.size(0), (n,), generator=g).to(drug_emb.device)

    src = torch.cat([pos_src, pos_src])
    dst = torch.cat([pos_dst, neg_dst])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    logits = model.score(drug_emb[src], tgt_emb[dst], relation)
    probs = torch.sigmoid(logits).cpu().numpy()
    return {
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


@torch.no_grad()
def hits_at_k(model, x_dict, eval_edges, all_edges, target_type, relation, k=10):
    """Filtered Hits@K and MRR over held-out positives."""
    drug_emb = x_dict["drug"]
    tgt_emb = x_dict[target_type]

    # known true targets per drug (for filtering), from the FULL edge set
    known = {}
    for d, t in zip(all_edges[0].tolist(), all_edges[1].tolist()):
        known.setdefault(d, set()).add(t)

    eval_drugs = eval_edges[0].tolist()
    eval_tgts = eval_edges[1].tolist()
    unique_drugs = sorted(set(eval_drugs))
    rows = _scores_drug_vs_all(model, drug_emb[torch.tensor(unique_drugs, device=drug_emb.device)],
                               tgt_emb, relation)
    drug_to_row = {d: i for i, d in enumerate(unique_drugs)}

    hits, rr = 0, 0.0
    for d, t in zip(eval_drugs, eval_tgts):
        s = rows[drug_to_row[d]].clone()
        for kt in known.get(d, ()):
            if kt != t:
                s[kt] = float("-inf")
        true_score = s[t]
        rank = int((s > true_score).sum().item()) + 1  # 1-based, ties -> best case
        if rank <= k:
            hits += 1
        rr += 1.0 / rank
    n = len(eval_tgts)
    return {f"hits@{k}": hits / n, "mrr": rr / n, "n": n}


@torch.no_grad()
def full_eval(model, graph, splits, cfg, device, split="test"):
    """Compute all reported metrics for the given split ('val' or 'test')."""
    from .model import build_edge_index_dict

    model.eval()
    dp_tr = splits["dp"]["train"].to(device)
    de_tr = splits["de"]["train"].to(device)
    edge_dict = build_edge_index_dict(dp_tr, de_tr)
    x_dict = model.encode(graph.protein_features.to(device), edge_dict)

    dp_eval = splits["dp"][split].to(device)
    de_eval = splits["de"][split].to(device)
    dp_all = graph.dp_edge_index.to(device)
    de_all = graph.de_edge_index.to(device)

    res = {
        "drug_protein": {
            **pr_auc(model, x_dict, dp_eval, "protein", "protein", seed=cfg.seed),
            **hits_at_k(model, x_dict, dp_eval, dp_all, "protein", "protein", k=cfg.hits_k),
        },
        "drug_indication": {
            **pr_auc(model, x_dict, de_eval, "effect", "effect", seed=cfg.seed),
            **hits_at_k(model, x_dict, de_eval, de_all, "effect", "effect", k=cfg.hits_k),
        },
    }
    return res
