"""Export the FULL ranked drug list for a single indication, to CSV.

Usage:  python -m pharma_ablation2.export_indication_ranking <EFFECT_ID> [out.csv]
Ranks every drug for the given indication (no top-K cutoff) by the model's
drug-indication score, flags known vs novel, and writes a CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import torch

from pharma_ablation2.data_paper import load_paper_data, resolve_data_dir
from pharma_ablation2.model_paper import AblationHeteroGNN

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@torch.no_grad()
def export(effect_id: str, out_path: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ARTIFACTS / "ablation2_paper_best.pt", map_location=device, weights_only=False)
    data = load_paper_data()

    if effect_id not in data["effect_to_idx"]:
        raise SystemExit(f"effect_id {effect_id} not found")
    eidx = data["effect_to_idx"][effect_id]

    model = AblationHeteroGNN(ckpt["config"], data["num_drugs"], data["num_proteins"],
                              data["num_effects"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    edict = {k: v.to(device) for k, v in data["test_edge_index_dict"].items()}
    x = model.encode(data["protein_features_tensor"].to(device),
                     data["effect_features_tensor"].to(device), edict)
    drug_emb, eff_emb = x["drug"], x["effect"]

    # score every drug against this one indication
    eff_vec = eff_emb[eidx:eidx + 1].expand(drug_emb.size(0), -1)
    scores = model.score_di(drug_emb, eff_vec)
    lo, hi = float(scores.min()), float(scores.max())
    rng = hi - lo if hi > lo else 1.0

    # known drugs for this indication
    known = {dd for (dd, ee) in data["existing_di"] if ee == eidx}

    # id -> name maps (merged across all sources, like predict_paper)
    d = resolve_data_dir()
    di_df = pd.read_pickle(d / "drug_indications_review.pkl")
    eff_name = dict(zip(di_df["effect_id"].astype(str), di_df["effect_name"].astype(str)))
    dp_dfs = [pd.read_pickle(d / f) for f in [
        "drug_protein_interactions_train_review.pkl",
        "drug_protein_interactions_validation_review.pkl",
        "drug_protein_interactions_test_review.pkl"]]
    internal_to_chembl, internal_to_name = {}, {}
    for df in [pd.read_pickle(d / "approved_small_molecule_drugs_review.pkl"), di_df, *dp_dfs]:
        if {"drug_internal_id", "drug_id", "drug_name"}.issubset(df.columns):
            for iid, cid, nm in zip(df["drug_internal_id"].astype(int),
                                    df["drug_id"].astype(str), df["drug_name"].astype(str)):
                internal_to_chembl.setdefault(iid, cid); internal_to_name.setdefault(iid, nm)
    drug_internal = data["drug_ids_filtered"]

    rows = []
    for di_ in range(drug_emb.size(0)):
        iid = drug_internal[di_]
        rows.append((float(scores[di_]), internal_to_chembl.get(iid, f"INT{iid}"),
                     internal_to_name.get(iid, str(iid)), di_ in known))
    rows.sort(key=lambda r: -r[0])

    out = Path(out_path) if out_path else (
        Path(__file__).resolve().parents[1] / "exports" / f"{effect_id}_{eff_name.get(effect_id, 'indication').replace(' ', '_').replace(',', '')}_all_drugs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "drug_id", "drug_name", "score", "raw_score", "status"])
        for rank, (raw, cid, nm, kn) in enumerate(rows, 1):
            w.writerow([rank, cid, nm, round((raw - lo) / rng, 4), round(raw, 4),
                        "known" if kn else "novel"])
    print(f"indication: {effect_id} {eff_name.get(effect_id)}")
    print(f"ranked {len(rows)} drugs ({sum(1 for r in rows if r[3])} known) -> {out}")
    return out


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
