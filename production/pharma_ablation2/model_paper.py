"""AblationHeteroGNN (Ablation 2: No Drug Graphs) — pure PyTorch port of cell 24.

Faithful to the notebook, with torch_geometric's `SAGEConv((d,d),d,aggr='mean')`
replaced by `SAGEConvMean` (verified equivalent: out = lin_l(mean_nbr) + lin_r(dst),
lin_l has bias, lin_r no bias). The key detail vs the older model.py: the
HeteroSAGELayer keeps **one LayerNorm per destination node type** (drug / protein
/ effect), exactly as `nn.ModuleDict({t: LayerNorm for t in dst_types})`.

For Ablation 2: use_drug_graphs=False (drug = nn.Embedding, xavier_uniform),
use_esm2=True (protein = Linear+LN+ReLU+Dropout over ESM-2 features).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pharma_ablation2.model import SAGEConvMean  # reuse verified PyG-equivalent conv

EDGE_TYPES = [
    ("drug", "binds_to", "protein"),
    ("protein", "rev_binds_to", "drug"),
    ("drug", "treats", "effect"),
    ("effect", "rev_treats", "drug"),
]


def _rel_key(et):
    s, r, d = et
    return f"{s}__{r}__{d}"


class HeteroSAGELayer(nn.Module):
    """Per-relation SAGEConv-mean, per-destination-type LayerNorm over residual."""

    def __init__(self, hidden_dim, edge_types=EDGE_TYPES):
        super().__init__()
        self.edge_types = edge_types
        self.convs = nn.ModuleDict()
        dst_types = set()
        for src, rel, dst in edge_types:
            self.convs[_rel_key((src, rel, dst))] = SAGEConvMean(hidden_dim, hidden_dim)
            dst_types.add(dst)
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in sorted(dst_types)})

    def forward(self, x_dict, edge_index_dict):
        contributions = {nt: [] for nt in x_dict}
        for et in self.edge_types:
            src, _, dst = et
            ei = edge_index_dict.get(et)
            if ei is None or ei.shape[1] == 0:
                continue
            out = self.convs[_rel_key(et)](x_dict[src], x_dict[dst], ei)
            contributions[dst].append(out)
        out_dict = {}
        for nt, feats in x_dict.items():
            if contributions[nt]:
                agg = torch.stack(contributions[nt]).mean(dim=0)
                out_dict[nt] = self.norms[nt](agg + feats)
            else:
                out_dict[nt] = feats
        return out_dict


class LinkHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim, 1)
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=-1)).squeeze(-1)


class AblationHeteroGNN(nn.Module):
    """Ablation 2 (use_drug_graphs=False, use_esm2=True). Pure PyTorch."""

    def __init__(self, cfg, num_drugs, num_proteins, num_effects,
                 use_drug_graphs=False, use_esm2=True):
        super().__init__()
        assert not use_drug_graphs, "Ablation 2 uses drug entity embeddings only"
        self.use_drug_graphs = use_drug_graphs
        self.use_esm2 = use_esm2
        sd = cfg["shared_dim"]
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.num_effects = num_effects

        # drugs: one trainable vector per drug (xavier_uniform), graphs ignored
        self.drug_entity_emb = nn.Embedding(num_drugs, sd)
        nn.init.xavier_uniform_(self.drug_entity_emb.weight)

        if use_esm2:
            self.protein_proj = nn.Sequential(
                nn.Linear(cfg["protein_feat_dim"], sd), nn.LayerNorm(sd), nn.ReLU(), nn.Dropout(0.1)
            )
        else:
            self.protein_entity_emb = nn.Embedding(num_proteins, sd)
            nn.init.xavier_uniform_(self.protein_entity_emb.weight)

        self.effect_proj = nn.Sequential(
            nn.Linear(cfg["effect_feat_dim"], sd), nn.LayerNorm(sd), nn.ReLU(), nn.Dropout(0.1)
        )

        self.hetero_layers = nn.ModuleList(
            [HeteroSAGELayer(sd, EDGE_TYPES) for _ in range(cfg["num_hetero_layers"])]
        )

        self.dp_head = LinkHead(sd)
        self.di_head = LinkHead(sd)

    def encode(self, protein_features, effect_features, edge_index_dict):
        dev = effect_features.device
        drug_emb = self.drug_entity_emb(torch.arange(self.num_drugs, device=dev))
        if self.use_esm2:
            protein_emb = self.protein_proj(protein_features)
        else:
            protein_emb = self.protein_entity_emb(torch.arange(self.num_proteins, device=dev))
        effect_emb = self.effect_proj(effect_features)

        x_dict = {"drug": drug_emb, "protein": protein_emb, "effect": effect_emb}
        for layer in self.hetero_layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict

    def score_dp(self, drug_emb, protein_emb):
        return self.dp_head(drug_emb, protein_emb)

    def score_di(self, drug_emb, effect_emb):
        return self.di_head(drug_emb, effect_emb)

    @staticmethod
    def margin_loss(pos_score, neg_score, margin=1.0):
        return F.relu(margin - pos_score + neg_score).mean()
