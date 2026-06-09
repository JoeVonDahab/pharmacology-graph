"""Train the Ablation-2 model (full-batch) and save checkpoint + metrics.

Unlike the original notebook (which wrapped the forward pass in `torch.no_grad()`,
so only the MLP heads learned), this trains the full model end-to-end: gradients
flow into the drug embeddings, projections and message-passing layers. Optimizer
is AdamW; loss is BCE over 1:1 negative-sampled drug-protein and drug-indication
edges. Early stopping on mean validation PR-AUC.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data import load_graph, split_edges
from .evaluate import full_eval
from .model import PharmacologyNoDrugGraph, build_edge_index_dict


def _sample_neg(pos_edges, num_targets, generator, device):
    n = pos_edges.size(1)
    neg_dst = torch.randint(0, num_targets, (n,), generator=generator).to(device)
    return torch.stack([pos_edges[0], neg_dst])


def _bce(model, x_dict, pos_edges, neg_edges, target_type, relation):
    drug_emb, tgt_emb = x_dict["drug"], x_dict[target_type]
    src = torch.cat([pos_edges[0], neg_edges[0]])
    dst = torch.cat([pos_edges[1], neg_edges[1]])
    labels = torch.cat([
        torch.ones(pos_edges.size(1), device=src.device),
        torch.zeros(neg_edges.size(1), device=src.device),
    ])
    logits = model.score(drug_emb[src], tgt_emb[dst], relation)
    return F.binary_cross_entropy_with_logits(logits, labels)


def train(cfg: Config | None = None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    graph = load_graph(cfg)
    print(f"[train] drugs={graph.num_drugs} proteins={graph.num_proteins} "
          f"effects={graph.num_effects} dp_edges={graph.dp_edge_index.size(1)} "
          f"de_edges={graph.de_edge_index.size(1)}")

    dp_tr, dp_va, dp_te = split_edges(graph.dp_edge_index, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    de_tr, de_va, de_te = split_edges(graph.de_edge_index, cfg.train_ratio, cfg.val_ratio, cfg.seed + 1)
    splits = {
        "dp": {"train": dp_tr, "val": dp_va, "test": dp_te},
        "de": {"train": de_tr, "val": de_va, "test": de_te},
    }

    model = PharmacologyNoDrugGraph(cfg, graph.num_drugs, graph.num_effects).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,} ({n_params/1e6:.2f}M)")

    protein_features = graph.protein_features.to(device)
    dp_tr_d, de_tr_d = dp_tr.to(device), de_tr.to(device)
    train_edge_dict = build_edge_index_dict(dp_tr_d, de_tr_d)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)

    artifacts = Path(cfg.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts / "ablation2_best.pt"

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        x_dict = model.encode(protein_features, train_edge_dict)
        dp_neg = _sample_neg(dp_tr_d, graph.num_proteins, gen, device)
        de_neg = _sample_neg(de_tr_d, graph.num_effects, gen, device)
        loss = (_bce(model, x_dict, dp_tr_d, dp_neg, "protein", "protein")
                + _bce(model, x_dict, de_tr_d, de_neg, "effect", "effect"))
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 1:
            val = full_eval(model, graph, splits, cfg, device, split="val")
            mean_pr = (val["drug_protein"]["pr_auc"] + val["drug_indication"]["pr_auc"]) / 2
            print(f"epoch {epoch:3d} loss={loss.item():.4f} "
                  f"val dp_pr={val['drug_protein']['pr_auc']:.4f} "
                  f"de_pr={val['drug_indication']['pr_auc']:.4f} mean_pr={mean_pr:.4f}")
            if mean_pr > best_val:
                best_val = mean_pr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    print(f"[train] early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "num_drugs": graph.num_drugs, "num_effects": graph.num_effects}, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}")

    test = full_eval(model, graph, splits, cfg, device, split="test")
    metrics_path = artifacts / "metrics.json"
    metrics_path.write_text(json.dumps(test, indent=2))
    print(f"[train] test metrics -> {metrics_path}")
    print(json.dumps(test, indent=2))
    return model, graph, splits, test


if __name__ == "__main__":
    train()
