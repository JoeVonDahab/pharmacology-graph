# Production: Ablation-2 "No Drug Graphs" — faithful reconstruction, predictions & explorer

Self-contained, production-format reconstruction of the **Ablation 2 ("No Drug
Graphs")** model from *"Pharmacology Knowledge Graphs: Do We Need Chemical
Structure for Drug Repurposing?"* (arXiv:2603.01537 / Springer s44163-026-01303-2),
the top-50 predictions it produces in **both directions**, and a static web app to
explore them. Pure PyTorch — no torch_geometric, no rdkit.

```
production/
├── pharma_ablation2/
│   ├── model.py        SAGEConvMean (PyG-equivalent) + a simple variant
│   ├── data_paper.py   loads the paper's real splits + curated negatives (cells 12-22)
│   ├── model_paper.py  AblationHeteroGNN — faithful port of the paper's cell 24
│   ├── eval_paper.py   filtered ranking (MRR/Hits@k) + sampled 1:50 ROC/PR-AUC (cell 25)
│   ├── train_paper.py  margin-ranking training, Adam 3e-4, early-stop on avg MRR
│   ├── predict_paper.py  forward + REVERSE top-50 predictions → webapp/data
│   └── (config.py, data.py, train.py, evaluate.py, predict.py — earlier random-split baseline)
├── webapp/             static GitHub Pages explorer (drug · disease · protein search)
│   ├── index.html · style.css · app.js
│   └── data/           drugs/indications/proteins indices + per-entity JSON + CSVs
└── artifacts/          checkpoints, metrics (git-ignored)
```

## The model (Ablation 2)

The full `PharmacologyHeteroGNN` with the GAT molecular drug encoder **replaced by
a learnable `nn.Embedding` per drug** — drugs carry no chemical structure.

| Node type   | Representation                                            |
|-------------|----------------------------------------------------------|
| Drug        | learnable embedding (256-d), xavier-uniform              |
| Protein     | frozen ESM-2 (2560-d) → Linear+LayerNorm+ReLU → 256-d    |
| Indication  | fixed random feature (32-d) → projection → 256-d         |

3 layers of heterogeneous GraphSAGE message passing (mean aggregation, per-dst-type
LayerNorm over a residual), then two 2-layer MLP heads scoring drug–protein and
drug–indication links. **Margin-ranking loss**, Adam (lr 3e-4). **3,294,978 params**
(paper: 3.29M).

## Data (the paper's real training data)

From `training_data/` (the `*_review.pkl` set added in the paper commit):
- **Temporal drug–protein split** by `first_published_year`: train 10,409 / val 1,901 / test 1,901.
- Drug–indication positives (`drug_indications_review.pkl`), random 80/10/10 split (seed 42).
- **Curated negatives** (this is what makes the metrics realistic):
  `verified_negatives_time_aware_*` for drug–protein, and `failed_indications_{hard,medium}`
  for drug–indication — not easy random negatives.
- ESM-2 features from `protein_nodes_with_embeddings_v4.pkl` (proteins without an
  embedding are zero-filled, mirroring the notebook).

`data_paper.py` reads `training_data/` at the repo root (falls back to `/tmp/paper_data`).

## How to run

```bash
pip install -r production/requirements.txt
cd production
python -m pharma_ablation2.train_paper     # → artifacts/ablation2_paper_best.pt + metrics_paper.json
python -m pharma_ablation2.predict_paper   # → webapp/data/ (forward + reverse, per-entity JSON + CSVs)
```

Full-batch, fits a single 24 GB GPU; ~1000 s for 2000 epochs (also runs on CPU).

## Metrics — reproduced within ±0.025 of the paper

Test set, exact eval protocol from the paper (filtered full ranking for MRR/Hits@k;
ROC-AUC/PR-AUC sampled at 1 positive : 50 negatives):

| Metric                  | This reconstruction | Paper  | Δ      |
|-------------------------|--------------------|--------|--------|
| Drug–Protein PR-AUC     | 0.601              | 0.5785 | +0.022 |
| Drug–Protein Hits@10    | 0.546              | 0.5234 | +0.023 |
| Drug–Indication PR-AUC  | 0.786              | 0.8060 | −0.021 |
| Drug–Indication Hits@10 | 0.807              | 0.8042 | +0.003 |

The low drug–protein PR-AUC is the ablation's point: without chemical structure the
model can't rank true targets above *hard verified negatives* under a temporal split.
(An earlier `train.py`/`predict.py` baseline used a random split + 1:1-sampled PR-AUC
and is kept for reference; it scores higher precisely because that protocol is easier.)

## The explorer (static site)

`webapp/` is dependency-free. Serve it and open in a browser:

```bash
cd production/webapp && python -m http.server 8000   # http://localhost:8000
```

**Bidirectional search:**
- Search a **drug** → its top-50 predicted protein targets and indications.
- Search a **disease / indication** → the drugs most likely linked to it.
- Search a **protein** → the drugs most likely to bind it.

Every prediction is flagged **known** (already an edge) or **novel**, with a relative
score bar, external links (ChEMBL / MeSH), per-query CSV/JSON download, and
full-dataset CSV downloads.

### Deploy to GitHub Pages

`.github/workflows/pages.yml` publishes `production/webapp/` on push to the default
branch. Enable **Settings → Pages → Source: GitHub Actions**. Live demo:
https://joevondahab.github.io/pharmacology-graph/
