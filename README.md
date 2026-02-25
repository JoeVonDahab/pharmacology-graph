# Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug Repurposing?

**Authors:** Youssef Abo-Dahab¹, Ruby Hernandez², Ismael Caleb Arechiga Duran²  
**Affiliations:** ¹University of California, San Francisco · ²Stanford University  
**Course:** CS224W: Machine Learning with Graphs, Stanford University (Instructor: Prof. Jure Leskovec)  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)

---

## Abstract

We constructed a pharmacology knowledge graph from ChEMBL 36 comprising **5,348 entities** (3,127 drugs, 1,156 proteins, 1,065 indications) and **20,015 edges** across 4 relation types. We enforced a strict **temporal split** (training: ≤2022; testing: 2023–2025) with biologically verified hard negatives mined from failed assays and clinical trials.

**Key finding:** Removing the GAT-based drug structure encoder entirely from the GNN and retaining only topological embeddings combined with ESM-2 protein features **improved** drug–protein PR-AUC from 0.5631 → 0.5785, while simultaneously reducing VRAM from 5.30 GB to **353 MB**. Explicit chemical structure is not only redundant — it is detrimental.

---

## Results Summary

### Benchmark: KGEs vs. GNNs

| Model | Params | VRAM | DP PR-AUC | DP Hits@10 | DI PR-AUC | DI Hits@10 |
|-------|--------|------|-----------|------------|-----------|------------|
| TransE | 0.78M | 1.0 GB | 0.1871 | 0.1594 | 0.2954 | 0.4268 |
| TransR | 0.81M | 1.0 GB | 0.2017 | 0.1620 | 0.2402 | 0.3028 |
| RotatE | 0.78M | 1.5 GB | 0.1855 | 0.1783 | 0.0720 | 0.2718 |
| ComplEx | 0.78M | 0.9 GB | 0.2262 | 0.2188 | 0.2989 | 0.4028 |
| DistMult | 0.78M | 1.0 GB | 0.2314 | 0.2241 | **0.3246** | 0.3803 |
| Standard GNN | 3.44M | 5.3 GB | 0.5631 | 0.5134 | 0.8175 | 0.8620 |
| Blackwell GNN | 9.73M | 36.8 GB | **0.5910** | **0.5250** | **0.8658** | **0.9014** |

> DP = Drug–Protein, DI = Drug–Indication. Metrics: PR-AUC and Hits@10 on temporally held-out test edges.

### Feature Ablation Study

| Variant | Params | VRAM | DP PR-AUC | DP Hits@10 | DI PR-AUC | DI Hits@10 |
|---------|--------|------|-----------|------------|-----------|------------|
| Standard GNN (GAT + ESM-2) | 3.44M | 5.30 GB | 0.5631 | 0.5134 | **0.8175** | **0.8620** |
| Ablation 1: No ESM-2 | 3.29M | 5.26 GB | 0.5186 | 0.4419 | 0.7937 | 0.8310 |
| **Ablation 2: No Drug Graphs** ⭐ | **3.29M** | **353 MB** | **0.5785** | **0.5234** | 0.8060 | 0.8042 |
| Ablation 3: Both Ablated | 3.14M | 345 MB | 0.4658 | 0.4340 | 0.7952 | 0.8183 |
| Morgan FP MLP + ESM-2 | 3.69M | ~350 MB | 0.5286 | 0.4813 | 0.7937 | 0.8254 |

### Scaling Laws

| Experiment | Params | DP PR-AUC | DI PR-AUC |
|-----------|--------|-----------|-----------|
| **Data Scaling** (fixed 3.44M params) | | | |
| 25% Data | 3.44M | 0.4320 | 0.5749 |
| 50% Data | 3.44M | 0.5140 | 0.6882 |
| 75% Data | 3.44M | 0.5381 | 0.7919 |
| 100% Data | 3.44M | 0.5631 | 0.8175 |
| **Parameter Scaling** (fixed 100% data) | | | |
| sd=64 | 1.12M | 0.4480 | 0.5698 |
| sd=128 | 1.66M | 0.5216 | 0.6868 |
| sd=192 | 2.44M | **0.5650** | 0.7679 |
| sd=256 (Standard GNN) | 3.44M | 0.5631 | 0.8175 |
| sd=512 | 9.75M | 0.5349 | **0.8335** |

> A 1.66M parameter model on 100% data (PR-AUC 0.5216) beats a 3.44M model on 50% data (PR-AUC 0.5140). **Data volume is the tighter bottleneck.**

---

## Repository Structure

```
pharmacology-graph/
│
├── README.md
│
├── data_preparation_new.ipynb              # Data extraction from ChEMBL 36, graph construction,
│                                           # temporal split, and hard negative mining
│
├── protien_embedings_prepare.ipynb         # ESM-2 (3B) protein embedding computation
│                                           # (requires GPU; ~1 hr for 1,156 proteins)
│
├── Standard GNN Model.ipynb               # Standard GNN: GAT drug encoder + ESM-2 proteins
│                                           # + GraphSAGE message passing (3.44M params, 5.30 GB VRAM)
│
├── Standard Model with fingerprints        # Morgan Fingerprint baseline: replaces GAT with
│   instead of molecular subgraphs.ipynb   # 2048-dim MFP + MLP (3.69M params, ~350 MB VRAM)
│
├── Ablation Study, Nodes Featrures,        # Feature ablation: systematically removes ESM-2
│   esm2 and GAT.ipynb                     # and/or GAT encoder to isolate contribution of
│                                           # each feature modality (produces the 353 MB model)
│
├── Ablation Study, Scaling Dimenisons      # Scaling laws: varies data volume (25/50/75/100%)
│   and Data, Main.ipynb                   # and hidden dimension (sd=64/128/192/256/512)
│
├── TransR Model.ipynb                     # KGE: TransR (relation-specific projection matrices)
├── RotateE Model.ipynb                    # KGE: RotatE (complex rotation in embedding space)
├── ComplEX model.ipynb                    # KGE: ComplEx (complex-valued embeddings)
├── DistMult Model.ipynb                   # KGE: DistMult (bilinear diagonal scoring)
│                                           # (TransE is included in data_preparation_new.ipynb)
│
├── plots.ipynb                            # Generates all paper figures (benchmark comparison,
│                                           # feature ablation, scaling laws, efficiency tradeoff)
│
└── training_data/                         # Preprocessed data files (output of data_preparation)
    ├── approved_small_molecule_drugs_review.pkl
    ├── drug_indications_review.pkl
    ├── drug_protein_interactions_review.pkl
    ├── drug_protein_interactions_train_review.pkl
    ├── drug_protein_interactions_validation_review.pkl
    ├── drug_protein_interactions_test_review.pkl
    ├── failed_indications_hard.pkl         # Phase 3 clinical trial failures (hard negatives)
    ├── failed_indications_medium.pkl       # Phase 2 clinical trial failures (medium negatives)
    ├── failed_indications_review.pkl
    ├── verified_negatives_time_aware_train.pkl
    ├── verified_negatives_time_aware_test.pkl
    ├── verified_negatives_v2_review.pkl
    ├── protein_nodes_with_embeddings_v4.pkl        # ESM-2 embeddings (1,156 proteins)
    └── protein_nodes_with_embeddings_extended.pkl
```

---

## Methodology

### Data

All data sourced from **ChEMBL 36** (SQLite release).

- **Drugs:** 3,127 approved small molecules with valid SMILES
- **Proteins:** 1,156 human single-protein targets (pChEMBL ≥ 5.5, assay confidence ≥ 8)
- **Indications:** 1,065 approved indications with MeSH identifiers
- **Edges:** 11,703 drug–protein + 8,312 drug–indication

### Temporal Split

- **Train:** edges with timestamp ≤ 2022
- **Test:** edges with timestamp > 2022 (only for entities already seen in training)
- Cold-start filtering relocated 1,199 edges back to training; final test set: 1,901 validation + 1,901 test edges

### Hard Negatives

- **Drug–protein:** experimentally confirmed inactives (pChEMBL < 4.5 or ≥ 10,000 nM)
- **Drug–indication:** Phase 3 failures (hard) and Phase 2 failures (medium) from ChEMBL
- 78,486 training negatives / 65,036 test negatives
- Dynamic 50/50 mix (verified + random) for drug–protein; 33/33/33 mix (hard/medium/random) for drug–indication

### Models

**Knowledge Graph Embeddings** — identity features only, no message passing:
- TransE, TransR, RotatE, ComplEx, DistMult (0.78–0.81M params, ≤1.5 GB VRAM)

**Standard GNN** — multi-modal features + GraphSAGE:
- Drug: GAT encoder (57-dim atom features → 256-dim molecule embedding)
- Protein: ESM-2 (`facebook/esm2_t36_3B_UR50D`, 2560-dim) linearly projected to 256-dim
- Indication: learnable 32-dim embeddings projected to 256-dim
- 3-layer heterogeneous GraphSAGE backbone, MLP link predictors
- 3.44M params, 5.30 GB VRAM, ~45 min training (RTX 3090)

**Blackwell GNN** — expressive upper bound:
- NNConv drug encoder + 3-layer AttentionHeteroConv + InfoNCE contrastive loss
- 9.73M params, 36.8 GB VRAM, ~60 min training (RTX PRO 6000)

**Efficient Topological Model** (Ablation 2) — main contribution:
- Drug structure encoder removed; drugs represented as learnable embeddings
- Protein: ESM-2 + linear projection (unchanged)
- 3.29M params, **353 MB VRAM**, ~42 min training (RTX 3090)
- Achieves **~95% of Blackwell ceiling** at <1% of its memory footprint

---

## Setup

### Requirements

```bash
pip install torch torch-geometric rdkit esm
```

ChEMBL 36 SQLite database (~4 GB) required for `data_preparation_new.ipynb`:

```
training_data/chembl_36/chembl_36_sqlite/chembl_36.db
```

Download from: https://www.ebi.ac.uk/chembl/

### Recommended Run Order

1. `data_preparation_new.ipynb` — build graph, apply temporal split, mine hard negatives
2. `protien_embedings_prepare.ipynb` — compute ESM-2 embeddings (GPU required)
3. Any model notebook (KGE or GNN)
4. `plots.ipynb` — generate paper figures

Pre-computed `training_data/` files are included so steps 1–2 can be skipped.

---

## External Validation

Top novel predictions from the Blackwell GNN (post-2022, absent from training graph):

| Rank | Drug | Predicted Indication | Validation |
|------|------|----------------------|------------|
| 4 | Diltiazem Hydrochloride | Stroke | NORDIL Trial |
| 5 | Dabigatran Etexilate Mesylate | Pulmonary Embolism | FDA Approved (2014) |
| 7 | Cortisone Acetate | Serum Sickness | FDA Approved |
| 8 | Captopril | Stroke | PROGRESS/HOPE Trials |
| 9 | Methotrexate Sodium | Immune System Diseases | Standard of Care |
| 14 | Pantoprazole Sodium | Stomach Ulcer | Standard of Care |
| 22 | Etrasimod | Atopic Dermatitis | Phase 2 ADVISE Trial |

**6 of the top 14 novel predictions (42.9%) confirmed as true therapeutic indications.**

---

## Key Takeaways

1. **Drop the drug structure encoder.** For macro-scale repurposing of approved drugs, explicit 2D/3D chemical structure (GAT or Morgan fingerprints) hurts drug–protein PR-AUC and wastes 93% of VRAM. Pure topological embeddings + ESM-2 protein features outperform all structural approaches.

2. **Data beats parameters.** A smaller model (1.66M params) on full data outperforms a larger model (3.44M params) on half the data. Scaling beyond 2.44M parameters yields diminishing returns; adding data never plateaus.

3. **Budget hardware is sufficient.** The 353 MB model runs on any GPU with >400 MB VRAM and achieves ~95% of a 36.8 GB state-of-the-art baseline.

4. **GNNs dominate KGEs** when ESM-2 protein embeddings are available (+150% PR-AUC). KGEs remain competitive for memory-constrained settings without pre-computed language model features.

---

## Citation

```bibtex
@article{abodahab2025pharmacology,
  author    = {Abo-Dahab, Youssef and Hernandez, Ruby and Arechiga Duran, Ismael Caleb},
  title     = {Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug Repurposing?},
  year      = {2025},
  institution = {University of California, San Francisco and Stanford University},
  url       = {https://github.com/JoeVonDahab/pharmacology-graph}
}
```

---

## Contact

**Youssef Abo-Dahab** — youssef.abo-dahab@ucsf.edu  
**Ruby Hernandez** — rubyh@stanford.edu  
**Ismael Caleb Arechiga Duran** — iaredur@stanford.edu
