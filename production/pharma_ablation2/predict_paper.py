"""Generate forward + reverse top-K predictions from the faithful paper model.

Forward : per drug      -> top-K proteins, top-K indications
Reverse : per indication -> top-K drugs   (disease search)
          per protein    -> top-K drugs   (target search)

Each prediction is flagged known (already an edge in the data) or novel. Scores
come from the margin-ranking heads (unbounded), so for display we min-max
normalise within each ranking list to [0,1]; the raw score is kept in the CSVs.

Outputs under production/webapp/data/:
  meta.json
  drugs.json / indications.json / proteins.json     (search indices)
  predictions/<DRUG_CHEMBL>.json                      (forward)
  by_indication/<EFFECT_ID>.json                      (reverse: ranked drugs)
  by_protein/<PROTEIN_CHEMBL>.json                    (reverse: ranked drugs)
  predictions_drug_protein.csv / predictions_drug_indication.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pharma_ablation2.data_paper import load_paper_data, resolve_data_dir
from pharma_ablation2.model_paper import AblationHeteroGNN

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "production" / "artifacts"
OUT = REPO_ROOT / "production" / "webapp" / "data"


@torch.no_grad()
def _score_matrix(model, head_emb, tail_emb, score_fn, chunk=256):
    """Full (n_head, n_tail) score matrix, computed in row chunks."""
    n_h, n_t = head_emb.size(0), tail_emb.size(0)
    out = torch.empty(n_h, n_t, device=head_emb.device)
    for h0 in range(0, n_h, chunk):
        h1 = min(h0 + chunk, n_h)
        B = h1 - h0
        h = head_emb[h0:h1].unsqueeze(1).expand(B, n_t, -1).reshape(B * n_t, -1)
        t = tail_emb.unsqueeze(0).expand(B, n_t, -1).reshape(B * n_t, -1)
        out[h0:h1] = score_fn(h, t).reshape(B, n_t)
    return out


def _norm(vec):
    lo, hi = float(vec.min()), float(vec.max())
    rng = hi - lo if hi > lo else 1.0
    return lo, rng


def _topk_rows(scores, known_by_row, k, id_lookup, name_lookup):
    """For each row, return top-k (col) entries with normalised score + status."""
    results = []
    n = scores.size(0)
    topv, topi = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    for r in range(n):
        lo, rng = _norm(scores[r])
        kn = known_by_row.get(r, set())
        row = []
        for rank, (v, j) in enumerate(zip(topv[r].tolist(), topi[r].tolist()), 1):
            row.append({
                "rank": rank,
                "id": id_lookup(j),
                "name": name_lookup(j),
                "score": round((v - lo) / rng, 4),
                "raw": round(v, 4),
                "status": "known" if j in kn else "novel",
            })
        results.append(row)
    return results


# bulk "download-all" CSVs are capped to stay under GitHub's 100 MB/file limit;
# per-entity JSON and client-side download still give the full top_k.
CSV_CAP = 100


def generate(top_k: int = 500, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ARTIFACTS / "ablation2_paper_best.pt", map_location=device, weights_only=False)
    data = load_paper_data()

    model = AblationHeteroGNN(ckpt["config"], data["num_drugs"], data["num_proteins"],
                              data["num_effects"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # encode with the FULL known graph (train+val+test) for best embeddings
    edict = {k: v.to(device) for k, v in data["test_edge_index_dict"].items()}
    x = model.encode(data["protein_features_tensor"].to(device),
                     data["effect_features_tensor"].to(device), edict)
    drug_emb, prot_emb, eff_emb = x["drug"], x["protein"], x["effect"]

    # ---- id -> human-readable name maps ----
    d = resolve_data_dir()
    di_df = pd.read_pickle(d / "drug_indications_review.pkl")
    dp_files = ["drug_protein_interactions_train_review.pkl",
                "drug_protein_interactions_validation_review.pkl",
                "drug_protein_interactions_test_review.pkl"]
    dp_dfs = [pd.read_pickle(d / f) for f in dp_files]

    # drug internal_id -> ChEMBL id + name, merged across ALL sources
    # (the metadata pickle covers only ~80% of edge drugs; the edge frames carry the rest)
    internal_to_chembl, internal_to_name = {}, {}
    sources = [pd.read_pickle(d / "approved_small_molecule_drugs_review.pkl"), di_df, *dp_dfs]
    for df in sources:
        if {"drug_internal_id", "drug_id", "drug_name"}.issubset(df.columns):
            for iid, cid, nm in zip(df["drug_internal_id"].astype(int),
                                    df["drug_id"].astype(str), df["drug_name"].astype(str)):
                if iid not in internal_to_chembl:
                    internal_to_chembl[iid], internal_to_name[iid] = cid, nm

    prot_name = {}
    for df in dp_dfs:
        prot_name.update(dict(zip(df["protein_id"].astype(str), df["protein_name"].astype(str))))
    eff_name = dict(zip(di_df["effect_id"].astype(str), di_df["effect_name"].astype(str)))

    drug_internal = data["drug_ids_filtered"]        # idx -> internal id
    prot_ids = data["protein_ids_filtered"]          # idx -> chembl
    eff_ids = data["effect_ids_filtered"]            # idx -> mesh id
    drug_chembl = [internal_to_chembl.get(i, f"INT{i}") for i in drug_internal]
    drug_names = [internal_to_name.get(i, drug_chembl[k]) for k, i in enumerate(drug_internal)]

    # ---- known maps (row -> set of col) ----
    kp_by_drug, kd_by_prot = {}, {}
    for (dd, pp) in data["existing_dp"]:
        kp_by_drug.setdefault(dd, set()).add(pp)
        kd_by_prot.setdefault(pp, set()).add(dd)
    ki_by_drug, kd_by_eff = {}, {}
    for (dd, ee) in data["existing_di"]:
        ki_by_drug.setdefault(dd, set()).add(ee)
        kd_by_eff.setdefault(ee, set()).add(dd)

    # ---- score matrices (computed once, reused for forward + reverse) ----
    S_dp = _score_matrix(model, drug_emb, prot_emb, model.score_dp)   # (drugs, proteins)
    S_di = _score_matrix(model, drug_emb, eff_emb, model.score_di)    # (drugs, effects)

    # forward: top-k along rows
    fwd_p = _topk_rows(S_dp, kp_by_drug, top_k, lambda j: prot_ids[j], lambda j: prot_name.get(prot_ids[j], prot_ids[j]))
    fwd_i = _topk_rows(S_di, ki_by_drug, top_k, lambda j: eff_ids[j], lambda j: eff_name.get(eff_ids[j], eff_ids[j]))
    # reverse: top-k along columns -> transpose
    rev_p = _topk_rows(S_dp.t().contiguous(), kd_by_prot, top_k, lambda j: drug_chembl[j], lambda j: drug_names[j])
    rev_i = _topk_rows(S_di.t().contiguous(), kd_by_eff, top_k, lambda j: drug_chembl[j], lambda j: drug_names[j])

    # ---- write files ----
    pred_dir = OUT / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    byi_dir = OUT / "by_indication"; byi_dir.mkdir(parents=True, exist_ok=True)
    byp_dir = OUT / "by_protein"; byp_dir.mkdir(parents=True, exist_ok=True)

    pcsv = open(OUT / "predictions_drug_protein.csv", "w", newline="")
    icsv = open(OUT / "predictions_drug_indication.csv", "w", newline="")
    pw, iw = csv.writer(pcsv), csv.writer(icsv)
    pw.writerow(["drug_id", "drug_name", "rank", "protein_id", "protein_name", "score", "raw_score", "status"])
    iw.writerow(["drug_id", "drug_name", "rank", "indication_id", "indication_name", "score", "raw_score", "status"])

    drugs_index = []
    for di_, cid in enumerate(drug_chembl):
        proteins, indications = fwd_p[di_], fwd_i[di_]
        (pred_dir / f"{cid}.json").write_text(json.dumps({
            "drug": {"id": cid, "name": drug_names[di_]},
            "proteins": proteins, "indications": indications}))
        for r in proteins[:CSV_CAP]:
            pw.writerow([cid, drug_names[di_], r["rank"], r["id"], r["name"], r["score"], r["raw"], r["status"]])
        for r in indications[:CSV_CAP]:
            iw.writerow([cid, drug_names[di_], r["rank"], r["id"], r["name"], r["score"], r["raw"], r["status"]])
        drugs_index.append({"id": cid, "name": drug_names[di_],
                            "kp": len(kp_by_drug.get(di_, ())), "np": sum(1 for r in proteins if r["status"] == "novel"),
                            "ki": len(ki_by_drug.get(di_, ())), "ni": sum(1 for r in indications if r["status"] == "novel")})
    pcsv.close(); icsv.close()

    indications_index = []
    for ei, eid in enumerate(eff_ids):
        drugs = rev_i[ei]
        (byi_dir / f"{eid}.json").write_text(json.dumps({
            "indication": {"id": eid, "name": eff_name.get(eid, eid)}, "drugs": drugs}))
        indications_index.append({"id": eid, "name": eff_name.get(eid, eid),
                                  "kd": len(kd_by_eff.get(ei, ())), "nd": sum(1 for r in drugs if r["status"] == "novel")})

    proteins_index = []
    for pi, pid in enumerate(prot_ids):
        drugs = rev_p[pi]
        (byp_dir / f"{pid}.json").write_text(json.dumps({
            "protein": {"id": pid, "name": prot_name.get(pid, pid)}, "drugs": drugs}))
        proteins_index.append({"id": pid, "name": prot_name.get(pid, pid),
                               "kd": len(kd_by_prot.get(pi, ())), "nd": sum(1 for r in drugs if r["status"] == "novel")})

    (OUT / "drugs.json").write_text(json.dumps(drugs_index))
    (OUT / "indications.json").write_text(json.dumps(indications_index))
    (OUT / "proteins.json").write_text(json.dumps(proteins_index))
    (OUT / "meta.json").write_text(json.dumps({
        "model": "Ablation 2 (No Drug Graphs) — paper reconstruction",
        "params": ckpt.get("config") and sum(p.numel() for p in model.parameters()),
        "num_drugs": data["num_drugs"], "num_proteins": data["num_proteins"],
        "num_indications": data["num_effects"], "top_k": top_k,
        "metrics": json.loads((ARTIFACTS / "metrics_paper.json").read_text()).get("test"),
    }))
    print(f"[predict_paper] wrote {len(drug_chembl)} drugs, {len(eff_ids)} indications, "
          f"{len(prot_ids)} proteins to {OUT}")


if __name__ == "__main__":
    generate()
