"""Configuration for the Ablation-2 ("No Drug Graphs") production model.

This reconstructs the model from the paper:
  "Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug
   Repurposing?" (arXiv:2603.01537 / Springer s44163-026-01303-2)

Ablation 2 replaces the GAT molecular drug encoder of the full
PharmacologyHeteroGNN with a *learnable drug embedding table* — i.e. drugs are
represented purely by trainable lookup vectors, no chemical structure. Proteins
use frozen ESM-2 features, indications use fixed random features; everything is
projected into a shared 256-d space and refined by 3 layers of GraphSAGE-style
heterogeneous message passing, then scored by two 2-layer MLP link heads.

Reported (paper Table 3, Ablation 2):
  Drug-Protein    PR-AUC 0.5785   Hits@10 0.5234
  Drug-Indication PR-AUC 0.8060   Hits@10 0.8042
  Params 3.29M    VRAM 353 MB
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Repo root = two levels up from this file (production/pharma_ablation2/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- input data (tracked in the repo) ----
    drug_nodes_csv: str = str(REPO_ROOT / "drug_nodes.csv")
    drug_effects_csv: str = str(REPO_ROOT / "drug_effects.csv")
    drug_interactions_csv: str = str(REPO_ROOT / "drugs_interactions.csv")
    protein_pickle: str = str(REPO_ROOT / "protein_nodes_with_embeddings_v4.pkl")

    # ---- output artifacts ----
    artifacts_dir: str = str(REPO_ROOT / "production" / "artifacts")

    # ---- model dims ----
    shared_dim: int = 256
    protein_feat_dim: int = 2560     # ESM-2 t36 3B mean-pooled
    effect_feat_dim: int = 32        # fixed random features -> projection
    num_hetero_layers: int = 3

    # ---- training ----
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    neg_ratio: float = 1.0
    patience: int = 40
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    device: str = "cuda"  # falls back to cpu automatically in train.py

    # ---- evaluation ----
    hits_k: int = 10

    def to_dict(self) -> dict:
        return asdict(self)
