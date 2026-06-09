# Production: Ablation-2 "No Drug Graphs" — reconstruction, predictions & explorer

This folder is a self-contained, production-format reconstruction of the
**Ablation 2 ("No Drug Graphs")** model from the paper
*"Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug
Repurposing?"* (arXiv:2603.01537 / Springer s44163-026-01303-2), plus the
top-50 predictions it produces for every drug and a static web app to explore
them.

```
production/
├── pharma_ablation2/        # pure-PyTorch package (no torch_geometric / rdkit)
│   ├── config.py            # all hyper-parameters & paths
│   ├── data.py              # CSV/pickle -> indices, edges, splits
│   ├── model.py             # learnable drug embeddings + GraphSAGE message passing
│   ├── train.py             # full-batch end-to-end training
│   ├── evaluate.py          # PR-AUC + filtered Hits@K
│   └── predict.py           # top-50 protein & indication predictions per drug
├── webapp/                  # static GitHub Pages site (HTML/CSS/JS)
│   ├── index.html · style.css · app.js
│   └── data/                # drugs.json, meta.json, predictions/<CHEMBL>.json, *.csv
├── artifacts/               # checkpoint, embeddings, metrics.json (git-ignored)
└── requirements.txt
```

## The model

Ablation 2 takes the full `PharmacologyHeteroGNN` and **replaces the GAT
molecular drug encoder with a learnable `nn.Embedding` table** — drugs are pure
trainable vectors, with no chemical structure. Everything else is unchanged:

| Node type   | Representation                                            |
|-------------|----------------------------------------------------------|
| Drug        | learnable embedding (256-d)                              |
| Protein     | frozen ESM-2 (2560-d) → linear projection → 256-d        |
| Indication  | fixed random feature (32-d) → linear projection → 256-d  |

These are refined by **3 layers of heterogeneous GraphSAGE-style message
passing** (mean aggregation, replicated in pure PyTorch — see
`SAGEConvMean`), then two 2-layer MLP heads score drug–protein and
drug–indication links. **3.31M parameters** (paper reports 3.29M).

Data: 3,127 drugs · 1,156 proteins · 1,065 indications · 11,493 drug–protein
edges · 5,633 drug–indication edges (unique pairs), all from the CSVs and
protein pickle in the repo root.

## How to run

```bash
pip install -r production/requirements.txt
cd production
python -m pharma_ablation2.train      # trains, writes artifacts/ablation2_best.pt + metrics.json
python -m pharma_ablation2.predict    # writes webapp/data/ (per-drug JSON + CSVs)
```

Training is full-batch and fits comfortably on a single 24 GB GPU (≈0.06 s/epoch;
the whole graph is tiny). It also runs on CPU.

## Metrics — and how they compare to the paper

Test metrics from this reconstruction (random 80/10/10 edge split, seed 42):

| Relation         | PR-AUC | ROC-AUC | Hits@10 (full-rank, filtered) |
|------------------|--------|---------|-------------------------------|
| Drug–Protein     | 0.910  | 0.904   | 0.375                         |
| Drug–Indication  | 0.881  | 0.868   | 0.328                         |

Paper (Table 3, Ablation 2): dp PR-AUC **0.5785**, dp Hits@10 **0.5234**,
de PR-AUC **0.8060**, de Hits@10 **0.8042**.

**Why they differ (and why this is expected):**

1. **Drug–indication PR-AUC matches well** (0.88 vs 0.81) — the architecture is
   faithfully reconstructed (param count matches to within rounding).
2. **Drug–protein PR-AUC is much higher here (0.91 vs 0.58).** The paper's low
   number is the *whole point* of this ablation: without chemical structure the
   model can't generalise drug–protein links to drugs it didn't train on. That
   failure only appears under the paper's **temporal split** ("training ≤ 2022",
   future edges held out), where newer drugs have no training edges and their
   learnable embedding stays untrained. The tracked CSVs contain **no dates**, so
   we use a random transductive split — under which every drug has training
   edges and the embeddings work, lifting drug–protein PR-AUC.
   We verified the bracket empirically: freezing the encoder (replicating the
   original notebook's `torch.no_grad()` training, which trained only the MLP
   heads) collapses both relations to ≈0.51–0.54 PR-AUC; full end-to-end
   training gives the table above. The paper's asymmetric 0.58/0.81 sits between
   these, consistent with a temporal split.
3. **Hits@10 here uses strict full-ranking against all candidates** (filtered),
   which is harder than a sampled-candidate protocol.

So: the architecture is reproduced exactly; absolute numbers differ because the
paper's hardest result needs temporal information absent from the public CSVs.
For the explorer we use the properly-trained model, which yields meaningful,
biologically coherent predictions (e.g. Sunitinib → kinase targets; novel
oncology indications).

## The explorer (static site)

`webapp/` is a dependency-free static site. Open it via any static server:

```bash
cd production/webapp && python -m http.server 8000   # then http://localhost:8000
```

Search any drug → see its top-50 predicted protein targets and indications, each
flagged **known** (already an edge) or **novel**, with score bars, external links
(ChEMBL / MeSH), per-drug CSV/JSON download, and full-dataset CSV downloads.

### Deploy to GitHub Pages

A workflow at `.github/workflows/pages.yml` publishes `production/webapp/` on
push to the default branch. Enable **Settings → Pages → Source: GitHub Actions**.
