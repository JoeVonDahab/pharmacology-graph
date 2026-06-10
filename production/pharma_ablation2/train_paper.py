"""Train the FAITHFUL Ablation-2 ("No Drug Graphs") model on the paper data.

Replicates cell 25's training loop for the no_drug_graphs variant only:
  Adam lr=3e-4 wd=1e-5, up to 2000 epochs, full-batch, NEG_RATIO=1,
  margin-ranking loss (dp+di), validate every 10 epochs, early-stop on avg MRR
  with patience=30 checks, save best checkpoint, then test on cumulative graph.

Run:  CUDA_VISIBLE_DEVICES=1 python3 -m pharma_ablation2.train_paper
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from pharma_ablation2.data_paper import (
    REPO_ROOT,
    load_paper_data,
    sample_negatives_dp_dynamic,
    sample_negatives_di_dynamic,
)
from pharma_ablation2.eval_paper import NEG_PER_POS_EVAL, evaluate_hgnn
from pharma_ablation2.model_paper import AblationHeteroGNN

NUM_EPOCHS = 2000
LR = 3e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO = 1
VAL_EVERY = 10
PATIENCE = 30
SEED = 42

ARTIFACTS = REPO_ROOT / "production" / "artifacts"
CKPT = ARTIFACTS / "ablation2_paper_best.pt"
METRICS = ARTIFACTS / "metrics_paper.json"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_paper_data()
    num_drugs = data["num_drugs"]
    num_proteins = data["num_proteins"]
    num_effects = data["num_effects"]
    print(f"drugs={num_drugs} proteins={num_proteins} effects={num_effects}")
    print(f"dp train/val/test = {data['dp_train'].shape[1]}/{data['dp_val'].shape[1]}/{data['dp_test'].shape[1]}")
    print(f"di train/val/test = {data['di_train'].shape[1]}/{data['di_val'].shape[1]}/{data['di_test'].shape[1]}")

    cfg = {
        "shared_dim": 256,
        "protein_feat_dim": data["protein_feat_dim"],
        "effect_feat_dim": data["effect_feat_dim"],
        "num_hetero_layers": 3,
    }

    # static device tensors
    prot_feat = data["protein_features_tensor"].to(device)
    eff_feat = data["effect_features_tensor"].to(device)
    train_ei = {k: v.to(device) for k, v in data["train_edge_index_dict"].items()}
    val_ei = {k: v.to(device) for k, v in data["val_edge_index_dict"].items()}
    test_ei = {k: v.to(device) for k, v in data["test_edge_index_dict"].items()}
    dp_train_pos = data["dp_train"].to(device)
    di_train_pos = data["di_train"].to(device)
    dp_val_dev = data["dp_val"].to(device)
    di_val_dev = data["di_val"].to(device)
    dp_test_dev = data["dp_test"].to(device)
    di_test_dev = data["di_test"].to(device)

    n_dp_train = dp_train_pos.shape[1]
    n_di_train = di_train_pos.shape[1]

    verified_dp_train = data["verified_dp_train"]
    hard_neg_edges = data["hard_neg_edges"]
    med_neg_edges = data["med_neg_edges"]
    existing_dp = data["existing_dp"]
    existing_di = data["existing_di"]
    train_drugs_with_pos = data["train_drugs_with_pos"]
    train_proteins_with_pos = data["train_proteins_with_pos"]
    train_drugs_di = data["train_drugs_di"]
    train_effects_di = data["train_effects_di"]

    def _neg_dp():
        negs = sample_negatives_dp_dynamic(
            n_dp_train * NEG_RATIO, verified_dp_train, existing_dp,
            train_drugs_with_pos, train_proteins_with_pos)
        return torch.tensor(negs, dtype=torch.long).t().to(device)

    def _neg_di():
        negs = sample_negatives_di_dynamic(
            n_di_train * NEG_RATIO, hard_neg_edges, med_neg_edges, existing_di,
            train_drugs_di, train_effects_di)
        return torch.tensor(negs, dtype=torch.long).t().to(device)

    mdl = AblationHeteroGNN(cfg, num_drugs, num_proteins, num_effects,
                            use_drug_graphs=False, use_esm2=True).to(device)
    total_params = sum(p.numel() for p in mdl.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(mdl.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_mrr = 0.0
    patience_counter = 0
    best_state = None
    t_start = time.time()

    def _eval(pos_ei, rel, existing, x_dict):
        return evaluate_hgnn(mdl, pos_ei, rel, existing, x_dict,
                             num_proteins, num_effects, device, NEG_PER_POS_EVAL)

    for epoch in range(1, NUM_EPOCHS + 1):
        mdl.train()
        x_dict = mdl.encode(prot_feat, eff_feat, train_ei)

        dp_neg = _neg_dp()
        di_neg = _neg_di()

        dp_pos = mdl.score_dp(x_dict["drug"][dp_train_pos[0]], x_dict["protein"][dp_train_pos[1]])
        dp_neg_s = mdl.score_dp(x_dict["drug"][dp_neg[0]], x_dict["protein"][dp_neg[1]])
        loss_dp = AblationHeteroGNN.margin_loss(dp_pos, dp_neg_s)

        di_pos = mdl.score_di(x_dict["drug"][di_train_pos[0]], x_dict["effect"][di_train_pos[1]])
        di_neg_s = mdl.score_di(x_dict["drug"][di_neg[0]], x_dict["effect"][di_neg[1]])
        loss_di = AblationHeteroGNN.margin_loss(di_pos, di_neg_s)

        loss = loss_dp + loss_di
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % VAL_EVERY == 0 or epoch == 1:
            mdl.eval()
            with torch.no_grad():
                x_val = mdl.encode(prot_feat, eff_feat, val_ei)
            val_dp = _eval(dp_val_dev, "dp", existing_dp, x_val)
            val_di = _eval(di_val_dev, "di", existing_di, x_val)
            avg_mrr = (val_dp["mrr"] + val_di["mrr"]) / 2
            print(f"Ep {epoch:>4d}  loss={loss.item():.4f} (dp={loss_dp.item():.4f} di={loss_di.item():.4f})  "
                  f"valDP[MRR={val_dp['mrr']:.4f} H@10={val_dp['hits@10']:.4f} PR={val_dp['pr_auc']:.4f}]  "
                  f"valDI[MRR={val_di['mrr']:.4f} H@10={val_di['hits@10']:.4f} PR={val_di['pr_auc']:.4f}]")
            if avg_mrr > best_val_mrr:
                best_val_mrr = avg_mrr
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                print(f"    * new best avg MRR {avg_mrr:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stop at epoch {epoch}")
                    break

    train_time = time.time() - t_start

    # ── test on cumulative graph ─────────────────────────────────────────────
    mdl.load_state_dict(best_state)
    mdl.eval()
    with torch.no_grad():
        x_test = mdl.encode(prot_feat, eff_feat, test_ei)
    test_dp = _eval(dp_test_dev, "dp", existing_dp, x_test)
    test_di = _eval(di_test_dev, "di", existing_di, x_test)

    print("\n==== TEST RESULTS (Ablation 2, paper reconstruction) ====")
    print(f"Drug-Protein : PR-AUC={test_dp['pr_auc']:.4f}  Hits@10={test_dp['hits@10']:.4f}  "
          f"ROC-AUC={test_dp['roc_auc']:.4f}  MRR={test_dp['mrr']:.4f}")
    print(f"Drug-Indic.  : PR-AUC={test_di['pr_auc']:.4f}  Hits@10={test_di['hits@10']:.4f}  "
          f"ROC-AUC={test_di['roc_auc']:.4f}  MRR={test_di['mrr']:.4f}")
    print("Paper target : DP PR-AUC 0.5785 H@10 0.5234 | DI PR-AUC 0.8060 H@10 0.8042")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": cfg,
        "num_drugs": num_drugs,
        "num_proteins": num_proteins,
        "num_effects": num_effects,
        "drug_ids_filtered": data["drug_ids_filtered"],
        "protein_ids_filtered": data["protein_ids_filtered"],
        "effect_ids_filtered": data["effect_ids_filtered"],
        "drug_to_idx": data["drug_to_idx"],
        "protein_to_idx": data["protein_to_idx"],
        "effect_to_idx": data["effect_to_idx"],
        "effect_features_tensor": data["effect_features_tensor"],
        "best_val_mrr": best_val_mrr,
        "train_time_s": train_time,
    }, CKPT)
    print(f"\nSaved checkpoint -> {CKPT}")

    with open(METRICS, "w") as f:
        json.dump({
            "variant": "ablation2_no_drug_graphs_paper",
            "params": total_params,
            "train_time_s": train_time,
            "best_val_mrr": best_val_mrr,
            "test": {"dp": test_dp, "di": test_di},
            "paper_target": {
                "dp_pr_auc": 0.5785, "dp_hits@10": 0.5234,
                "di_pr_auc": 0.8060, "di_hits@10": 0.8042,
            },
        }, f, indent=2)
    print(f"Saved metrics    -> {METRICS}")


if __name__ == "__main__":
    main()
