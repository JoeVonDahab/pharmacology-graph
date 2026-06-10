"""Ablation-2 ("No Drug Graphs") heterogeneous GNN — pure PyTorch.

Faithful reimplementation of the notebook's `PharmacologyHeteroGNN`, with the GAT
`DrugMolecularEncoder` replaced by a learnable `nn.Embedding` over drugs. The
heterogeneous message passing replicates torch_geometric's `SAGEConv` with mean
aggregation, so no torch_geometric dependency is required:

    SAGEConv((src,dst)->out):  out = lin_l( mean_neighbors(x_src) ) + lin_r(x_dst)

`HeteroGraphConvLayer` runs one such conv per relation, mean-combines the messages
arriving at each node type, and applies LayerNorm over a residual:

    out[t] = LayerNorm( mean_r(conv_r) + x[t] )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# canonical relations (drug-protein "binds_to", drug-effect "treats", + reverses)
EDGE_TYPES = [
    ("drug", "binds_to", "protein"),
    ("protein", "rev_binds_to", "drug"),
    ("drug", "treats", "effect"),
    ("effect", "rev_treats", "drug"),
]


def _rel_key(edge_type):
    s, r, d = edge_type
    return f"{s}__{r}__{d}"


class SAGEConvMean(nn.Module):
    """Bipartite GraphSAGE layer with mean aggregation (matches PyG SAGEConv)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)   # aggregated neighbors
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)  # root / self

    def forward(self, x_src, x_dst, edge_index):
        # edge_index: (2, E) with row0 indexing x_src, row1 indexing x_dst
        num_dst = x_dst.size(0)
        out_dim_in = x_src.size(1)
        if edge_index.numel() == 0:
            agg = x_src.new_zeros(num_dst, out_dim_in)
        else:
            src, dst = edge_index[0], edge_index[1]
            messages = x_src.index_select(0, src)            # (E, in)
            agg = x_src.new_zeros(num_dst, out_dim_in)
            agg.index_add_(0, dst, messages)
            deg = x_src.new_zeros(num_dst)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x_src.dtype))
            agg = agg / deg.clamp(min=1).unsqueeze(1)
        return self.lin_l(agg) + self.lin_r(x_dst)


class HeteroGraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_types=EDGE_TYPES):
        super().__init__()
        self.edge_types = edge_types
        self.convs = nn.ModuleDict(
            {_rel_key(et): SAGEConvMean(hidden_dim, hidden_dim) for et in edge_types}
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        gathered = {nt: [] for nt in x_dict}
        for et in self.edge_types:
            s, _, d = et
            ei = edge_index_dict.get(et)
            if ei is None:
                continue
            out = self.convs[_rel_key(et)](x_dict[s], x_dict[d], ei)
            gathered[d].append(out)
        out_dict = {}
        for nt, msgs in gathered.items():
            if msgs:
                agg = torch.stack(msgs, dim=0).mean(dim=0)
                out_dict[nt] = self.norm(agg + x_dict[nt])
            else:
                out_dict[nt] = x_dict[nt]
        return out_dict


class LinkPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, 1),
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1)).squeeze(-1)


class PharmacologyNoDrugGraph(nn.Module):
    """Ablation-2 model: learnable drug embeddings + ESM-2 proteins + random-init
    indication features, refined by heterogeneous GraphSAGE message passing."""

    def __init__(self, cfg, num_drugs: int, num_effects: int):
        super().__init__()
        dim = cfg.shared_dim
        self.num_drugs = num_drugs
        self.num_effects = num_effects

        # drugs: learnable embedding table (replaces the GAT molecular encoder)
        self.drug_embedding = nn.Embedding(num_drugs, dim)
        nn.init.xavier_uniform_(self.drug_embedding.weight)

        # proteins: project frozen ESM-2 features
        self.protein_proj = nn.Sequential(
            nn.Linear(cfg.protein_feat_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(0.1)
        )
        # indications: fixed random features -> projection (matches notebook)
        self.register_buffer(
            "effect_features",
            torch.randn(num_effects, cfg.effect_feat_dim,
                        generator=torch.Generator().manual_seed(cfg.seed)),
        )
        self.effect_proj = nn.Sequential(
            nn.Linear(cfg.effect_feat_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(0.1)
        )

        self.layers = nn.ModuleList(
            [HeteroGraphConvLayer(dim) for _ in range(cfg.num_hetero_layers)]
        )

        self.dp_predictor = LinkPredictor(dim)
        self.de_predictor = LinkPredictor(dim)

    def encode(self, protein_features, edge_index_dict):
        """Run message passing, return refined {drug, protein, effect} embeddings."""
        x_dict = {
            "drug": self.drug_embedding.weight,
            "protein": self.protein_proj(protein_features),
            "effect": self.effect_proj(self.effect_features),
        }
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict

    def score(self, drug_emb, target_emb, relation: str):
        head = self.dp_predictor if relation == "protein" else self.de_predictor
        return head(drug_emb, target_emb)


def build_edge_index_dict(dp_edges, de_edges):
    """Build the bidirectional edge_index_dict from forward dp/de edges."""
    return {
        ("drug", "binds_to", "protein"): dp_edges,
        ("protein", "rev_binds_to", "drug"): dp_edges.flip(0),
        ("drug", "treats", "effect"): de_edges,
        ("effect", "rev_treats", "drug"): de_edges.flip(0),
    }
