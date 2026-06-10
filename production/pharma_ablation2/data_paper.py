"""Data loading for the FAITHFUL Ablation-2 ("No Drug Graphs") reconstruction.

Pure PyTorch / pandas / numpy — NO torch_geometric, NO rdkit. Replicates the
paper notebook's cells 12-22 exactly:

  * connected-node filtering + reindexing  (cell 14)
  * drug-protein temporal split (pre-split pickles)  (cell 15)
  * drug-indication random 80/10/10 split, seed 42  (cell 16)
  * verified DP negatives pool  (cell 17)
  * hard / medium DI negatives pools  (cell 18)
  * cumulative val/test edge_index_dicts  (cell 19)
  * dynamic negative samplers  (cells 17/18)

The effect feature tensor is fixed random `torch.randn(num_effects, 32)` exactly
like the notebook's cell 11; we seed it so reconstruction is reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAINING_DATA = REPO_ROOT / "training_data"
_FALLBACK_DATA = Path("/tmp/paper_data")

EFFECT_EMBEDDING_DIM = 32  # cell 11: EFFECT_EMBEDDING_DIM = 32


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    if _DEFAULT_TRAINING_DATA.is_dir():
        return _DEFAULT_TRAINING_DATA
    return _FALLBACK_DATA


# ---------------------------------------------------------------------------
# Negative samplers — ported verbatim from cells 17 & 18
# ---------------------------------------------------------------------------
def sample_negatives_dp_dynamic(num_samples, verified_negs, existing_edges,
                                valid_srcs, valid_tgts, ratio=0.5):
    """50% verified + 50% random drug-protein negatives (cell 17)."""
    verified_count = min(len(verified_negs), int(num_samples * ratio))
    negatives = []
    if verified_negs:
        for edge_idx in np.random.choice(
                len(verified_negs), min(verified_count, len(verified_negs)), replace=False):
            negatives.append(verified_negs[edge_idx])

    random_count = num_samples - verified_count
    valid_srcs = list(valid_srcs)
    valid_tgts = list(valid_tgts)
    attempts = 0
    max_attempts = random_count * 20
    neg_set = set(negatives)
    while len(negatives) < num_samples and attempts < max_attempts:
        pair = (np.random.choice(valid_srcs), np.random.choice(valid_tgts))
        if pair not in existing_edges and pair not in neg_set:
            negatives.append(pair)
            neg_set.add(pair)
        attempts += 1
    while len(negatives) < num_samples:
        pair = (np.random.choice(valid_srcs), np.random.choice(valid_tgts))
        if pair not in existing_edges and pair not in neg_set:
            negatives.append(pair)
            neg_set.add(pair)
    return negatives[:num_samples]


def sample_negatives_di_dynamic(num_samples, hard_negs, med_negs, existing_edges,
                                valid_srcs, valid_tgts):
    """33% hard + 33% medium + 33% random drug-indication negatives (cell 18)."""
    hard_count = min(len(hard_negs), num_samples // 3)
    med_count = min(len(med_negs), num_samples // 3)
    negatives = []
    if hard_negs:
        for idx in np.random.choice(
                len(hard_negs), min(hard_count, len(hard_negs)), replace=False):
            negatives.append(hard_negs[idx])
    if med_negs:
        for idx in np.random.choice(
                len(med_negs), min(med_count, len(med_negs)), replace=False):
            negatives.append(med_negs[idx])

    random_count = num_samples - hard_count - med_count
    all_neg_set = set(hard_negs + med_negs)
    valid_srcs = list(valid_srcs)
    valid_tgts = list(valid_tgts)
    neg_set = set(negatives)
    attempts = 0
    max_attempts = random_count * 20
    while len(negatives) < num_samples and attempts < max_attempts:
        pair = (np.random.choice(valid_srcs), np.random.choice(valid_tgts))
        if pair not in existing_edges and pair not in all_neg_set and pair not in neg_set:
            negatives.append(pair)
            neg_set.add(pair)
        attempts += 1
    while len(negatives) < num_samples:
        pair = (np.random.choice(valid_srcs), np.random.choice(valid_tgts))
        if pair not in existing_edges and pair not in neg_set:
            negatives.append(pair)
            neg_set.add(pair)
    return negatives[:num_samples]


def _edges_to_tensor(df, src_col, tgt_col, src_map, tgt_map):
    """cell 15: convert edges to tensor, filtering out missing nodes."""
    src_idx, tgt_idx = [], []
    for _, row in df.iterrows():
        sid = int(row[src_col])
        tid = str(row[tgt_col])
        if sid in src_map and tid in tgt_map:
            src_idx.append(src_map[sid])
            tgt_idx.append(tgt_map[tid])
    if src_idx:
        return torch.LongTensor([src_idx, tgt_idx])
    return torch.zeros((2, 0), dtype=torch.long)


def _existing_set(edge_index):
    if edge_index.shape[1] == 0:
        return set()
    return set(map(tuple, edge_index.t().numpy()))


def build_edge_index_dict(dp_edges, di_edges):
    return {
        ("drug", "binds_to", "protein"): dp_edges,
        ("protein", "rev_binds_to", "drug"): dp_edges.flip(0),
        ("drug", "treats", "effect"): di_edges,
        ("effect", "rev_treats", "drug"): di_edges.flip(0),
    }


def load_paper_data(data_dir=None, effect_seed: int = 42):
    """Load + assemble everything needed to train/eval Ablation 2.

    Returns a dict with node counts, id<->idx mappings, split edge indices,
    edge_index_dicts, feature tensors, negative pools and existing-edge sets.
    """
    d = resolve_data_dir(data_dir)

    drug_nodes = pd.read_pickle(d / "approved_small_molecule_drugs_review.pkl")
    dp_train_df = pd.read_pickle(d / "drug_protein_interactions_train_review.pkl")
    dp_val_df = pd.read_pickle(d / "drug_protein_interactions_validation_review.pkl")
    dp_test_df = pd.read_pickle(d / "drug_protein_interactions_test_review.pkl")
    drug_indications = pd.read_pickle(d / "drug_indications_review.pkl")
    verified_dp_train_df = pd.read_pickle(d / "verified_negatives_time_aware_train.pkl")
    verified_dp_test_df = pd.read_pickle(d / "verified_negatives_time_aware_test.pkl")
    failed_med = pd.read_pickle(d / "failed_indications_medium.pkl")
    failed_hard = pd.read_pickle(d / "failed_indications_hard.pkl")
    protein_nodes = pd.read_pickle(d / "protein_nodes_with_embeddings_v4.pkl")

    # ── cell 14: collect connected nodes across ALL splits, reindex ──────────
    all_dp_src, all_dp_tgt = set(), set()
    all_di_src, all_di_tgt = set(), set()
    for df in (dp_train_df, dp_val_df, dp_test_df):
        all_dp_src.update(int(x) for x in df["drug_internal_id"].values)
        all_dp_tgt.update(str(x) for x in df["protein_id"].values)
    all_di_src.update(int(x) for x in drug_indications["drug_internal_id"].values)
    all_di_tgt.update(str(x) for x in drug_indications["effect_id"].values)

    connected_drugs = all_dp_src | all_di_src
    connected_proteins = all_dp_tgt
    connected_effects = all_di_tgt

    drug_ids_filtered = np.array(sorted(connected_drugs))
    protein_ids_filtered = np.array(sorted(connected_proteins))
    effect_ids_filtered = np.array(sorted(connected_effects))

    drug_to_idx = {did: i for i, did in enumerate(drug_ids_filtered)}
    protein_to_idx = {pid: i for i, pid in enumerate(protein_ids_filtered)}
    effect_to_idx = {eid: i for i, eid in enumerate(effect_ids_filtered)}

    num_drugs = len(drug_to_idx)
    num_proteins = len(protein_to_idx)
    num_effects = len(effect_to_idx)

    # ── cell 15: drug-protein temporal splits ───────────────────────────────
    dp_train = _edges_to_tensor(dp_train_df, "drug_internal_id", "protein_id", drug_to_idx, protein_to_idx)
    dp_val = _edges_to_tensor(dp_val_df, "drug_internal_id", "protein_id", drug_to_idx, protein_to_idx)
    dp_test = _edges_to_tensor(dp_test_df, "drug_internal_id", "protein_id", drug_to_idx, protein_to_idx)

    # ── cell 16: drug-indication random 80/10/10 split, seed 42 ──────────────
    di_edges = _edges_to_tensor(drug_indications, "drug_internal_id", "effect_id", drug_to_idx, effect_to_idx)
    np.random.seed(42)
    n_di = di_edges.shape[1]
    perm = np.random.permutation(n_di)
    tr = int(0.8 * n_di)
    va = int(0.1 * n_di)
    di_train = di_edges[:, perm[:tr]]
    di_val = di_edges[:, perm[tr:tr + va]]
    di_test = di_edges[:, perm[tr + va:]]

    # ── cell 17: verified DP negatives pool ──────────────────────────────────
    chembl_to_internal = dict(zip(drug_nodes["drug_id"], drug_nodes["drug_internal_id"]))
    train_drugs_with_pos = set(int(x) for x in dp_train[0].numpy())
    train_proteins_with_pos = set(int(x) for x in dp_train[1].numpy())

    def _verified_dp(df):
        out = []
        for _, row in df.iterrows():
            dc = str(row["drug_id"])
            pid = str(row["protein_id"])
            if dc in chembl_to_internal and pid in protein_to_idx:
                internal = chembl_to_internal[dc]
                if internal in drug_to_idx:
                    di_ = drug_to_idx[internal]
                    pi_ = protein_to_idx[pid]
                    if di_ in train_drugs_with_pos and pi_ in train_proteins_with_pos:
                        out.append((di_, pi_))
        return list(set(out))

    verified_dp_train = _verified_dp(verified_dp_train_df)
    verified_dp_test = _verified_dp(verified_dp_test_df)

    all_pos_dp = torch.cat([dp_train, dp_val, dp_test], dim=1)
    existing_dp = _existing_set(all_pos_dp)

    # ── cell 18: hard / medium DI negatives pools ────────────────────────────
    effect_name_to_id = dict(zip(drug_indications["effect_name"], drug_indications["effect_id"]))
    train_drugs_di = set(int(x) for x in di_train[0].numpy())
    train_effects_di = set(int(x) for x in di_train[1].numpy())

    def _failed(df):
        out = []
        for _, row in df.iterrows():
            dc = str(row["drug_id"])
            ename = str(row["effect_name"])
            if dc in chembl_to_internal and ename in effect_name_to_id:
                internal = chembl_to_internal[dc]
                eid = effect_name_to_id[ename]
                if internal in drug_to_idx and eid in effect_to_idx:
                    di_ = drug_to_idx[internal]
                    ei_ = effect_to_idx[eid]
                    if di_ in train_drugs_di and ei_ in train_effects_di:
                        out.append((di_, ei_))
        return list(set(out))

    hard_neg_edges = _failed(failed_hard)
    med_neg_edges = _failed(failed_med)

    all_pos_di = torch.cat([di_train, di_val, di_test], dim=1)
    existing_di = _existing_set(all_pos_di)

    # ── cell 19: feature tensors ─────────────────────────────────────────────
    # proteins: ESM-2 embedding per protein_id, zeros if missing
    pid_to_emb = {str(r["protein_id"]): np.asarray(r["esm2_embedding"], dtype=np.float32)
                  for _, r in protein_nodes.iterrows()}
    prot_dim = next(iter(pid_to_emb.values())).shape[0]
    prot_feats = []
    for pid in protein_ids_filtered:
        if pid in pid_to_emb:
            prot_feats.append(torch.from_numpy(pid_to_emb[pid]))
        else:
            prot_feats.append(torch.zeros(prot_dim))
    protein_features_tensor = torch.stack(prot_feats).float()

    # effects: fixed random features (cell 11: torch.randn(1, 32)); seed for repro
    gen = torch.Generator().manual_seed(effect_seed)
    effect_features_tensor = torch.randn(num_effects, EFFECT_EMBEDDING_DIM, generator=gen)

    # ── cell 19: cumulative edge_index_dicts ─────────────────────────────────
    dp_val_cum = torch.cat([dp_train, dp_val], dim=1)
    di_val_cum = torch.cat([di_train, di_val], dim=1)
    dp_test_cum = torch.cat([dp_train, dp_val, dp_test], dim=1)
    di_test_cum = torch.cat([di_train, di_val, di_test], dim=1)

    train_edge_index_dict = build_edge_index_dict(dp_train, di_train)
    val_edge_index_dict = build_edge_index_dict(dp_val_cum, di_val_cum)
    test_edge_index_dict = build_edge_index_dict(dp_test_cum, di_test_cum)

    return {
        "num_drugs": num_drugs,
        "num_proteins": num_proteins,
        "num_effects": num_effects,
        # mappings (idx -> id, for serialization)
        "drug_ids_filtered": [int(x) for x in drug_ids_filtered],
        "protein_ids_filtered": [str(x) for x in protein_ids_filtered],
        "effect_ids_filtered": [str(x) for x in effect_ids_filtered],
        "drug_to_idx": drug_to_idx,
        "protein_to_idx": protein_to_idx,
        "effect_to_idx": effect_to_idx,
        # split edges
        "dp_train": dp_train, "dp_val": dp_val, "dp_test": dp_test,
        "di_train": di_train, "di_val": di_val, "di_test": di_test,
        # edge dicts
        "train_edge_index_dict": train_edge_index_dict,
        "val_edge_index_dict": val_edge_index_dict,
        "test_edge_index_dict": test_edge_index_dict,
        # features
        "protein_features_tensor": protein_features_tensor,
        "effect_features_tensor": effect_features_tensor,
        "protein_feat_dim": prot_dim,
        "effect_feat_dim": EFFECT_EMBEDDING_DIM,
        # negative pools + existing sets
        "verified_dp_train": verified_dp_train,
        "verified_dp_test": verified_dp_test,
        "hard_neg_edges": hard_neg_edges,
        "med_neg_edges": med_neg_edges,
        "existing_dp": existing_dp,
        "existing_di": existing_di,
        "train_drugs_with_pos": train_drugs_with_pos,
        "train_proteins_with_pos": train_proteins_with_pos,
        "train_drugs_di": train_drugs_di,
        "train_effects_di": train_effects_di,
    }


if __name__ == "__main__":
    data = load_paper_data()
    print("num_drugs   ", data["num_drugs"])
    print("num_proteins", data["num_proteins"])
    print("num_effects ", data["num_effects"])
    print("dp train/val/test", data["dp_train"].shape[1], data["dp_val"].shape[1], data["dp_test"].shape[1])
    print("di train/val/test", data["di_train"].shape[1], data["di_val"].shape[1], data["di_test"].shape[1])
    print("verified_dp_train", len(data["verified_dp_train"]))
    print("hard/med neg     ", len(data["hard_neg_edges"]), len(data["med_neg_edges"]))
    print("protein feats    ", tuple(data["protein_features_tensor"].shape))
    print("effect feats     ", tuple(data["effect_features_tensor"].shape))
