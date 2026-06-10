"""Data loading and graph construction for the Ablation-2 model.

Pure pandas/numpy/torch — no torch_geometric, no rdkit (drugs are learnable
embeddings, so molecular graphs are not needed).

Index alignment mirrors the original notebook (`3 million paramaters model.ipynb`):
  * drugs    indexed by row order of drug_nodes.csv          (map: drug_internal_id -> idx)
  * proteins indexed by row order of the protein pickle       (map: protein_id -> idx)
  * effects  indexed by drop_duplicates(effect_id,effect_name) (map: effect_id -> idx)

Edges are filtered to endpoints present in the maps and **deduplicated to unique
(src,dst) pairs** — the notebook kept duplicate drug-effect rows, but for link
prediction duplicates can leak identical edges across the train/test split, so we
remove them. This yields 11,493 drug-protein and 5,633 drug-indication edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class GraphData:
    # node metadata (index-aligned lists)
    drug_ids: list           # ChEMBL ids,   len = num_drugs
    drug_names: list
    protein_ids: list        # ChEMBL ids,   len = num_proteins
    protein_names: list
    effect_ids: list         # MeSH ids,     len = num_effects
    effect_names: list

    # features
    protein_features: torch.Tensor   # (num_proteins, 2560) float32
    num_drugs: int
    num_proteins: int
    num_effects: int

    # full edge sets (unique pairs), shape (2, E), row0=drug idx
    dp_edge_index: torch.Tensor
    de_edge_index: torch.Tensor

    # id -> idx maps (handy for prediction / lookups)
    drug_id_to_idx: dict
    protein_id_to_idx: dict
    effect_id_to_idx: dict


def load_graph(cfg) -> GraphData:
    drug_nodes = pd.read_csv(cfg.drug_nodes_csv)
    drug_effects = pd.read_csv(cfg.drug_effects_csv)
    interactions = pd.read_csv(cfg.drug_interactions_csv)
    proteins = pd.read_pickle(cfg.protein_pickle)

    # ---- drugs ----
    drug_internal_to_idx = {int(r): i for i, r in enumerate(drug_nodes["drug_internal_id"])}
    drug_id_to_idx = {str(r): i for i, r in enumerate(drug_nodes["drug_id"])}
    drug_ids = drug_nodes["drug_id"].astype(str).tolist()
    drug_names = drug_nodes["drug_name"].astype(str).tolist()
    num_drugs = len(drug_ids)

    # ---- proteins ----
    protein_id_to_idx = {str(r): i for i, r in enumerate(proteins["protein_id"])}
    protein_ids = proteins["protein_id"].astype(str).tolist()
    protein_names = proteins["protein_name"].astype(str).tolist()
    protein_features = torch.from_numpy(
        np.stack([np.asarray(e, dtype=np.float32) for e in proteins["esm2_embedding"]])
    ).float()
    num_proteins = len(protein_ids)

    # ---- effects / indications ----
    unique_effects = drug_effects[["effect_id", "effect_name"]].drop_duplicates().reset_index(drop=True)
    effect_id_to_idx = {str(r): i for i, r in enumerate(unique_effects["effect_id"])}
    effect_ids = unique_effects["effect_id"].astype(str).tolist()
    effect_names = unique_effects["effect_name"].astype(str).tolist()
    num_effects = len(effect_ids)

    # ---- drug-protein edges (unique pairs) ----
    dp_pairs = set()
    for did, pid in zip(interactions["drug_internal_id"], interactions["protein_id"]):
        di = drug_internal_to_idx.get(int(did))
        pi = protein_id_to_idx.get(str(pid))
        if di is not None and pi is not None:
            dp_pairs.add((di, pi))
    dp_edge_index = torch.tensor(sorted(dp_pairs), dtype=torch.long).t().contiguous()

    # ---- drug-effect edges (unique pairs) ----
    de_pairs = set()
    for did, eid in zip(drug_effects["drug_internal_id"], drug_effects["effect_id"]):
        di = drug_internal_to_idx.get(int(did))
        ei = effect_id_to_idx.get(str(eid))
        if di is not None and ei is not None:
            de_pairs.add((di, ei))
    de_edge_index = torch.tensor(sorted(de_pairs), dtype=torch.long).t().contiguous()

    return GraphData(
        drug_ids=drug_ids, drug_names=drug_names,
        protein_ids=protein_ids, protein_names=protein_names,
        effect_ids=effect_ids, effect_names=effect_names,
        protein_features=protein_features,
        num_drugs=num_drugs, num_proteins=num_proteins, num_effects=num_effects,
        dp_edge_index=dp_edge_index, de_edge_index=de_edge_index,
        drug_id_to_idx=drug_id_to_idx,
        protein_id_to_idx=protein_id_to_idx,
        effect_id_to_idx=effect_id_to_idx,
    )


def split_edges(edge_index: torch.Tensor, train_ratio, val_ratio, seed):
    """Random edge split (seeded). Returns (train, val, test) edge_index tensors."""
    g = torch.Generator().manual_seed(seed)
    num = edge_index.size(1)
    perm = torch.randperm(num, generator=g)
    n_train = int(train_ratio * num)
    n_val = int(val_ratio * num)
    tr = edge_index[:, perm[:n_train]]
    va = edge_index[:, perm[n_train:n_train + n_val]]
    te = edge_index[:, perm[n_train + n_val:]]
    return tr, va, te
