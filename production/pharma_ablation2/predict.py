"""Generate top-K drug-protein and drug-indication predictions for every drug.

Loads the trained Ablation-2 checkpoint, runs message passing over the FULL known
graph (all edges) to obtain the best entity embeddings, then for each drug ranks
all proteins and all indications by the link-prediction head. Each prediction is
flagged `known` (already an edge in the data) or novel.

Outputs (under production/app_data/):
  drugs.json                         search index [{id,name,n_known_p,n_known_i}]
  predictions/<DRUG_ID>.json         per-drug top-K proteins + indications
  predictions_drug_protein.csv       all drugs, long format (downloadable)
  predictions_drug_indication.csv    all drugs, long format (downloadable)
And the raw embeddings under production/artifacts/entity_embeddings.npz.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import load_graph
from .model import PharmacologyNoDrugGraph, build_edge_index_dict


@torch.no_grad()
def _rank_all(model, drug_emb, target_emb, relation, top_k, known_sets, device, batch=256):
    """For each drug return list of (target_idx, score, known) for top_k targets."""
    n_d, n_t = drug_emb.size(0), target_emb.size(0)
    results = []
    for d0 in range(0, n_d, batch):
        d1 = min(d0 + batch, n_d)
        for di in range(d0, d1):
            d = drug_emb[di:di + 1].expand(n_t, -1)
            logits = model.score(d, target_emb, relation)
            scores = torch.sigmoid(logits)
            topv, topi = torch.topk(scores, k=min(top_k, n_t))
            kn = known_sets.get(di, set())
            results.append([(int(j), float(v), int(j in kn))
                            for v, j in zip(topv.tolist(), topi.tolist())])
    return results


def _known_map(edge_index):
    m = {}
    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        m.setdefault(s, set()).add(t)
    return m


@torch.no_grad()
def generate(cfg: Config | None = None, top_k: int = 50):
    cfg = cfg or Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    artifacts = Path(cfg.artifacts_dir)
    ckpt = torch.load(artifacts / "ablation2_best.pt", map_location=device)

    graph = load_graph(cfg)
    model = PharmacologyNoDrugGraph(cfg, ckpt["num_drugs"], ckpt["num_effects"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # encode with the FULL known graph for best representations
    dp_all = graph.dp_edge_index.to(device)
    de_all = graph.de_edge_index.to(device)
    edge_dict = build_edge_index_dict(dp_all, de_all)
    x = model.encode(graph.protein_features.to(device), edge_dict)
    drug_emb, prot_emb, eff_emb = x["drug"], x["protein"], x["effect"]

    # save raw embeddings
    np.savez_compressed(
        artifacts / "entity_embeddings.npz",
        drug=drug_emb.cpu().numpy(), protein=prot_emb.cpu().numpy(), effect=eff_emb.cpu().numpy(),
        drug_ids=np.array(graph.drug_ids), protein_ids=np.array(graph.protein_ids),
        effect_ids=np.array(graph.effect_ids),
    )

    known_p = _known_map(graph.dp_edge_index)
    known_i = _known_map(graph.de_edge_index)
    prot_top = _rank_all(model, drug_emb, prot_emb, "protein", top_k, known_p, device)
    ind_top = _rank_all(model, drug_emb, eff_emb, "effect", top_k, known_i, device)

    out = Path(cfg.artifacts_dir).parent / "webapp" / "data"
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    drugs_index = []
    p_csv = open(out / "predictions_drug_protein.csv", "w", newline="")
    i_csv = open(out / "predictions_drug_indication.csv", "w", newline="")
    pw = csv.writer(p_csv); iw = csv.writer(i_csv)
    pw.writerow(["drug_id", "drug_name", "rank", "protein_id", "protein_name", "score", "status"])
    iw.writerow(["drug_id", "drug_name", "rank", "indication_id", "indication_name", "score", "status"])

    for di in range(graph.num_drugs):
        d_id, d_name = graph.drug_ids[di], graph.drug_names[di]
        proteins, indications = [], []
        for rank, (j, score, kn) in enumerate(prot_top[di], 1):
            status = "known" if kn else "novel"
            proteins.append({"rank": rank, "id": graph.protein_ids[j],
                             "name": graph.protein_names[j], "score": round(score, 4), "status": status})
            pw.writerow([d_id, d_name, rank, graph.protein_ids[j], graph.protein_names[j],
                         f"{score:.4f}", status])
        for rank, (j, score, kn) in enumerate(ind_top[di], 1):
            status = "known" if kn else "novel"
            indications.append({"rank": rank, "id": graph.effect_ids[j],
                                "name": graph.effect_names[j], "score": round(score, 4), "status": status})
            iw.writerow([d_id, d_name, rank, graph.effect_ids[j], graph.effect_names[j],
                         f"{score:.4f}", status])

        (pred_dir / f"{d_id}.json").write_text(json.dumps({
            "drug": {"id": d_id, "name": d_name},
            "proteins": proteins, "indications": indications,
        }))
        drugs_index.append({
            "id": d_id, "name": d_name,
            "n_known_p": len(known_p.get(di, ())), "n_known_i": len(known_i.get(di, ())),
            "n_novel_p": sum(1 for p in proteins if p["status"] == "novel"),
            "n_novel_i": sum(1 for p in indications if p["status"] == "novel"),
        })

    p_csv.close(); i_csv.close()
    (out / "drugs.json").write_text(json.dumps(drugs_index))
    (out / "meta.json").write_text(json.dumps({
        "num_drugs": graph.num_drugs, "num_proteins": graph.num_proteins,
        "num_indications": graph.num_effects, "top_k": top_k,
        "model": "Ablation-2 (No Drug Graphs)", "params": sum(p.numel() for p in model.parameters()),
    }))
    print(f"[predict] wrote {graph.num_drugs} per-drug files + CSVs to {out}")
    return out


if __name__ == "__main__":
    generate()
