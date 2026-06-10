# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CS224W (Stanford) final project: heterogeneous graph link prediction over a pharmacology knowledge graph built from ChEMBL data. The goal is to predict two relation types — drug→protein binding and drug→clinical-effect — from a graph of 3,127 drugs, 1,156 proteins, and 1,065 effects.

**All work lives in Jupyter notebooks.** There is no `app.py`, package `src/`, or test suite in this repository despite what the README's "Repository Structure" section claims. Treat that README section as aspirational/HF-Spaces-oriented, not a description of the tracked files. The actual tracked content is: four notebooks, three input CSVs, one protein-embedding pickle, and two result CSVs.

## Three models (each in its own notebook)

| Notebook | Model | Approach |
|----------|-------|----------|
| `3 million paramaters model.ipynb` | V1 (~3.4M params) | GraphSAGE/GAT baseline, manual SGD |
| `800 million parmaters model.ipynb` | V2 (~888M params) | NNConv drug encoder + attention HeteroConv + InfoNCE contrastive loss, AdamW |
| `pure_knowledge_embeddings.ipynb` | TransR (~246M params) | Pure KG embedding lookup with relation-specific projection matrices; best AUC (~0.896) |
| `old_model_transE_and_data.ipynb` | TransE (archived) | Earlier baseline; also contains the ESM-2 embedding-generation code |

V1 and V2 share the same data-loading and graph-construction pipeline (identical markdown section headers); they differ only in the model definition and training cells. The TransR notebook is independent and uses pickle files **not tracked here** (download ~2.8GB package from the Google Drive link in README.md, extract to a `transR/` directory).

## Running

There is no build/lint/test. Work is run cell-by-cell in the notebooks. Dependencies are split:

- `pyproject.toml` / `uv.lock` — the uv-managed env (dspy, gradio, plotly). Sync with `uv sync`.
- `requirements.txt` — pinned versions for Hugging Face Spaces deployment (gradio, pandas, numpy<2, plotly, networkx, scikit-learn). **Neither file lists the heavy training deps** — notebooks additionally require `torch`, `torch_geometric`, `rdkit`, `tqdm`, and (for ESM-2 generation) `transformers`/`fair-esm`. These are expected to be installed in the GPU runtime separately.

Training needs a GPU (V2 is ~96GB, TransR ~96GB; V1 fits in 24GB per the README). Trained weights (`best_model_improved.pt`, `best_model_clean.pt`, etc.) are git-ignored and not present.

## Data pipeline (V1/V2 notebooks)

The graph is assembled in-notebook from CSVs into a PyG `HeteroData` object. Loading order matters:

1. `drug_nodes.csv` — `drug_internal_id, drug_id (ChEMBL), drug_name, smile`. SMILES → RDKit → per-molecule PyG `Data` graph via `smiles_to_graph()`. Atom features are 57-dim one-hot (`encode_atom`), bond features 4-dim (`encode_bond`). Each drug is a sub-graph that the drug encoder pools to a 256-dim vector.
2. `protein_nodes_with_embeddings_v4.pkl` — DataFrame with `esm2_embedding` (2,560-dim, mean-pooled `facebook/esm2_t36_3B_UR50D`) plus `amino_acid_sequence`, `uniprot_id`. Proteins are pre-encoded; the ESM-2 generation step lives in `old_model_transE_and_data.ipynb` and is slow (~1hr).
3. `drug_effects.csv` — drug→effect edges (`effect_id`, `efo_id`, `indication_phase`). Effect features are learnable 32-dim embeddings.
4. `drugs_interactions.csv` — drug→protein edges with `pchembl_max/avg`, `best_value`, `confidence`.

Edges use string-keyed relation tuples: `('drug','binds_to','protein')` and `('drug','causes','effect')`. The `*_internal_id` columns are the integer node indices that tie the CSVs together — join on these, not on the ChEMBL string IDs.

## Training conventions

- Edge-level transductive split via `split_edges(train=0.8/val=0.1/test=0.1, seed=42)`. All nodes are visible during training; only test *edges* are held out of message passing.
- 1:1 negative sampling (same drug, random target), BCE loss summed across both relation types. V2 adds an InfoNCE contrastive term.
- Drugs are encoded fresh each epoch from their molecular graphs (`encode_drugs`), so the per-epoch cost is dominated by running the drug encoder over all molecule sub-graphs — not a simple embedding lookup. Drug graphs and feature tensors are moved to `device` once up front.

## Conventions to preserve

- Filenames contain spaces and typos (`paramaters`, `parmaters`); match them exactly when referencing or globbing.
- `.gitignore` ignores everything by default and re-allows only `*.csv`, `*.ipynb`, `*.py`, and directory traversal — model weights (`*.pt`), embeddings (`*.npy`), images, and DBs are intentionally untracked. New large artifacts won't be committed unless you adjust `.gitignore`, and that is usually the intended behavior (large files go to LFS per `.gitattributes` or to the Google Drive package).
