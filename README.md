# Pharmacology Knowledge Graph: Drug-Target-Effect Prediction

- **CS224W**: Machine Learning with Graphs - Final Project, Fall 2025.  
- **Students:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran.
- **Teaching Staff**: Jure Leskovec, Charilaos Kanatsoulis, Ayush Agrawal.
- **Affiliation**: Stanford University, Computer Science Department. 

> **📦 TransR Model Data Package**  
> To run the TransR knowledge embedding model (`pure_knowledge_embeddings.ipynb`), download the required data package (~2.8GB) from:  
> **[📥 Download TransR Full Data](https://drive.google.com/file/d/1cDcCLBwwPVpEtgfracVmW1UkLqxaqC2M/view?usp=sharing)**  
>  
> The ZIP contains all pickle data files and trained model weights:  
> - `approved_small_molecule_drugs.pkl` - Drug metadata
> - `drug_clinical_effects.pkl` - Drug-effect relationships
> - `drug_mechanism_of_action.pkl` - Mechanism of action data
> - `drug_protein_interactions.pkl` - Drug-protein binding data
> - `drug_therapeutic_classes_atc.pkl` - ATC therapeutic classifications
> - `drug_warnings_adverse_effects.pkl` - Safety/warning information
> - `transr_full_model.pt` - Full trained TransR model
> - `transr_knowledge_embeddings_best.pt` - Best checkpoint
> - `transr_entity_embeddings.npz` - Extracted entity embeddings
> - `transr_relation_data.npz` - Relation embeddings and projection matrices
> - Visualization outputs (`.png` files)
>  
> Extract contents to the `transR/` directory before running the notebook.

---

  Heterogeneous graph neural network for predicting drug-target interactions and therapeutic effects. Built from ChEMBL data with 3,127 drugs, 1,156 proteins, and 1,065 effects.

**📊 Three Models:**
- **Model V1** (3.4M params): GraphSAGE baseline → **69.2% AUC**
- **Model V2** (888M params): NNConv + Attention + Contrastive → **88.2% AUC** 
- **TransR** (Knowledge Embeddings): Pure embedding lookup with relation-specific projections  **89.6% AUC** 🏆

**📓 Notebooks:** [Model V1](code%20copy.ipynb) | [Model V2](800%20million%20parmaters%20model.ipynb) | [TransR](pure_knowledge_embeddings.ipynb)

## 📊 Model Comparison

| Feature | Model V1 (GraphSAGE) | Model V2 (Attention) | TransR (Embeddings) |
|---------|---------------------|---------------------|---------------------|
| **Parameters** | 3.4M | 888M | 246M |
| **Architecture** | GAT + GraphSAGE | NNConv + Attention + Contrastive | Embedding Lookup + Relation Projections |
| **Test AUC** | 69.2% | 88.2% | **89.6%** 🏆 |
| **Precision** | 60.7% | **86.8%** 🏆 | 78.9% |
| **Training Time** | **20 min** 🏆  | 130 min | 5hr |
| **GPU Memory USED** | ~24GB | ~96GB |  ~12GB 🏆 |

### TransR Model Details
- **Approach:** Pure knowledge graph embeddings without complex feature encoders
- **Scoring:** $\text{Score}(h,r,t) = -\|\mathbf{M}_r \mathbf{h} + \mathbf{r} - \mathbf{M}_r \mathbf{t}\|_2$
- **Entity Types:** Drugs (1.9M), Proteins (4,040), Effects (2,178), Targets (1,518), Warnings (462), Therapeutics (666)
- **Total Entities:** 1,924,278
- **Relation Types:** drug_binds_protein, drug_causes_effect, drug_acts_on_target, drug_has_warning, drug_has_therapeutic
- **Dimensions:** Entity=128, Relation=64
- **Total Parameters:** 246,348,864 (~985 MB)
- **Loss:** Margin-based ranking with negative sampling (margin=1.0)

**Classification Metrics:**
| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.8961 |
| **Accuracy** | 0.8157 |
| **Precision** | 0.7889 |
| **Recall** | 0.8620 |
| **F1 Score** | 0.8239 |

---

## 🎯 Overview

**Data:** ChEMBL database
- **Nodes:** 3,127 drugs | 1,156 proteins | 1,065 effects
- **Edges:** 11,493 drug-protein | 6,496 drug-effect

**Features:**
- **Drugs:** Molecular graphs from SMILES (GAT/NNConv encoding)
- **Proteins:** ESM-2 embeddings (2,560-dim from 3B parameter model)
- **Effects:** Learnable embeddings (32-dim)

**Training:** 80/10/10 train/val/test split, BCE loss with negative sampling

---

## 🏗️ Architecture

### Model V1: GraphSAGE Baseline (3.4M params)
```
Drug SMILES → GAT (3 layers, 4 heads) → 256-dim
Protein Seq → ESM-2 projection → 256-dim
Effect → Learnable embedding → 256-dim
    ↓
GraphSAGE (3 layers, mean aggregation)
    ↓
MLP Link Predictors → Drug-Protein & Drug-Effect scores
```
- Optimizer: SGD (lr=0.001)
- Training: 20 Minutes

### Model V2: Attention-Enhanced (888M params)
```
Drug SMILES → NNConv (edge-conditioned, 3 layers) → 256-dim
Protein Seq → ESM-2 projection → 256-dim
Effect → Learnable embedding → 256-dim
    ↓
Attention-based Hetero Conv (3 layers, 4 heads)
+ Edge type embeddings
+ Contrastive loss (InfoNCE)
    ↓
MLP Link Predictors → Drug-Protein & Drug-Effect scores
```
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Training: 130 Minutes

---

## 📊 Results

### Model V1 (GraphSAGE Baseline)
| Metric | Drug-Protein | Drug-Effect | Average |
|--------|--------------|-------------|---------|
| **AUC** | 0.684 | 0.700 | **0.692** |
| **Precision** | 0.593 | 0.621 | 0.607 |
| **Recall** | 0.802 | 0.724 | 0.763 |
| **F1** | 0.682 | 0.668 | 0.675 |

### Model V2 (Attention-Enhanced)
| Metric | Drug-Protein | Drug-Effect | Average |
|--------|--------------|-------------|---------|
| **AUC** | **0.930** | **0.834** | **0.882** |
| **Precision** | 0.875 | 0.862 | **0.868** |
| **Recall** | 0.869 | 0.613 | 0.741 |
| **F1** | 0.872 | 0.716 | 0.794 |

**Improvement:** +27.4% AUC, +43.1% Precision, 300x Paramaters

---

**Data:** ChEMBL 36 database (place in `chembl_36/chembl_36_sqlite/chembl_36.db`)
---

## � References

## 📚 References

- **GAT:** Veličković et al., "Graph Attention Networks" (ICLR 2018)
- **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- **ESM-2:** Lin et al., "Evolutionary-scale prediction of atomic-level protein structure" (Science 2023)
- **ChEMBL:** Gaulton et al., "The ChEMBL database in 2017" (Nucleic Acids Research 2017)

---

## 🔬 Technical Details

### Node Feature Engineering

#### **1. Drug Features (Molecular Graphs)**

**SMILES → RDKit → PyTorch Geometric `Data` objects**

**Atom Features (57-dim one-hot):**
```python
- Atom symbol (20): C, N, O, S, P, F, Cl, Br, I, etc.
- Atom degree (6): 0, 1, 2, 3, 4, 5+
- Formal charge (5): -2, -1, 0, +1, +2
- Is aromatic (2): True, False
- Hybridization (8): SP, SP2, SP3, SP3D, SP3D2, etc.
- Num hydrogens (16): 0, 1, 2, 3, 4+
```

**Bond Features (4-dim one-hot):**
```python
- Bond type: SINGLE, DOUBLE, TRIPLE, AROMATIC
```

**Graph Structure:**
- Nodes: Atoms in the molecule
- Edges: Chemical bonds (undirected)
- Variable size: 10-100 atoms per drug

#### **2. Protein Features (ESM-2 Embeddings)**

**Model:** `facebook/esm2_t36_3B_UR50D`
- **Parameters:** 3 billion
- **Architecture:** Transformer (36 layers, 2560 hidden dim)
- **Pre-training:** 250M protein sequences from UniRef50
- **Embedding:** Mean-pooled over sequence length → 2560-dim vector

**Processing:**
```python
batch_size = 4 proteins
total_proteins = 1,156
# compute_time = ~1 hour (RTX GeForce 3090)
```

#### **3. Effect Features (Learnable)**

**Initialization:** Random normal distribution
- **Dimension:** 32
- **Trainable:** Yes (updated via backpropagation)
- **Purpose:** Learn task-specific representations for clinical effects

### Training Procedure

#### **Loss Function**

**Binary Cross-Entropy with Negative Sampling:**

```python
For each batch:
  1. Sample positive edges: (drug, protein/effect) from training set
  2. Sample negative edges: (drug, random_protein/effect)
  3. Compute predictions: σ(MLP(concat(drug_emb, target_emb)))
  4. Loss = BCE(pos_pred, 1) + BCE(neg_pred, 0)
  5. Total = loss_drug_protein + loss_drug_effect
```

**Negative Sampling Ratio:** 1:1 (equal positives and negatives)

#### **Optimization**

```python
optimizer = Manual SGD
learning_rate = 0.001
batch_size = 256 drugs
epochs = 300 (with early stopping)
early_stopping_patience = 80 epochs
```

**Gradient Update:**
```python
# Manual SGD step
for param in model.parameters():
    param.data -= learning_rate * param.grad.data
```

### Design Decisions (CS224W Concepts)

#### **1. Why Heterogeneous GNN?**
- **Multiple node types:** Drugs ≠ Proteins ≠ Effects (different modalities)
- **Multiple edge types:** binds_to vs treats (different semantics)
- **Type-specific encoders:** GAT for molecular graphs, MLP for embeddings

#### **2. Why GAT for Drugs?**
- **Attention mechanism:** Learn the importance of atoms/bonds
- **Permutation invariant:** Order of atoms doesn't matter
- **Variable graphs:** Handle molecules of different sizes

#### **3. Why GraphSAGE for Message Passing?**
- **Scalable aggregation:** Mean pooling over neighbors
- **Inductive capability:** Can generalize to new nodes
- **Flexible:** Works with different node feature dimensions

#### **4. Why Bidirectional Edges?**
- **Information flow:** Drugs ↔ Proteins ↔ Effects
- **Richer embeddings:** Nodes learn from multiple hops
- **Nodes that are connected are different types --> no need for directionality**
- **Symmetry:** Both endpoints benefit from the relationship

---

## 🧪 Methodology & Validation

### Problem Formulation (CS224W Framework)

**Task:** Link prediction in a heterogeneous knowledge graph

**Input:**
- Graph: G = (V, E) where V = V_drug ∪ V_protein ∪ V_effect
- Node features: X_drug (molecular graphs), X_protein (ESM-2), X_effect (learned)
- Training edges: E_train ⊂ E
- Edge types: R = {binds_to, treats}

**Output:**
- Score function: f(drug, protein) → [0,1] (binding probability)
- Score function: f(drug, effect) → [0,1] (treatment probability)

**Objective:** Maximize AUC-ROC on held-out test edges E_test

### Evaluation Protocol

#### **1. Edge-Level Splitting (Transductive)**

```python
# For each edge type (drug-protein, drug-effect):
E_train = 80% of edges   # Message passing + supervision
E_val = 10% of edges     # Hyperparameter tuning
E_test = 10% of edges    # Final evaluation

# All nodes visible during training (transductive)
# Test edges hidden from message passing
```

#### **2. Negative Sampling Strategy**

```python
# For each positive edge (drug_i, target_j):
negative_sample = (drug_i, random_target_k)

# Ensures:
# - Same drug, different target (hard negatives)
# - Balanced classes (1:1 ratio)
# - Realistic evaluation
```

---

## 📝 Citation

```bibtex
@misc{abodahab2025pharmacology,
  author = {Abo-Dahab, Youssef},
  title = {Heterogeneous Graph Neural Networks for Pharmacology Link Prediction},
  year = {2024},
  institutions = {Stanford University, UCSF},
  course = {CS224W: Machine Learning with Graphs},
  url = {https://github.com/JoeVonDahab/pharmacology-graph}
}
```

## 📁 Repository Structure

```
pharmacology-graph/
├── README.md                                    # This file
├── SETUP.md                                     # Detailed setup instructions
├── requirements.txt                             # Python dependencies (full training)
├── requirements_app.txt                         # Python dependencies (app only)
├── start_app.sh                                 # Launch interactive demo
├── app.py                                       # Gradio web interface
│
├── code copy.ipynb                              # Model V1 (GraphSAGE, 3.4M params)
├── 800 million parmaters model.ipynb           # Model V2 (Attention, 888M params)
├── 3 million paramaters model.ipynb            # Early experiments
├── old_model_transE_and_data.ipynb            # TransE baseline (archived)
│
├── transR/                                      # TransR Knowledge Embeddings
│   ├── pure_knowledge_embeddings.ipynb         # TransR model notebook
│   └── TransR Full Data/                       # ⚠️ Download from Google Drive (see note above)
│       ├── approved_small_molecule_drugs.pkl   # Drug metadata
│       ├── drug_clinical_effects.pkl           # Drug-effect relationships
│       ├── drug_mechanism_of_action.pkl        # MOA data
│       ├── drug_protein_interactions.pkl       # Drug-protein bindings
│       ├── drug_therapeutic_classes_atc.pkl    # ATC classifications
│       ├── drug_warnings_adverse_effects.pkl   # Safety data
│       ├── transr_full_model.pt                # Trained model weights
│       ├── transr_knowledge_embeddings_best.pt # Best checkpoint
│       ├── transr_entity_embeddings.npz        # Entity embeddings
│       ├── transr_relation_data.npz            # Relation data
│       └── *.png                               # Visualization outputs
│
├── best_model_clean.pt                          # Trained Model V1 weights
├── best_model_improved.pt                       # Trained Model V2 weights (if generated)
├── pharmacology_graph_model.pt                  # Legacy model weights
│
├── drug_nodes.csv                               # Drug metadata (3,127 drugs)
├── drug_effects.csv                             # Drug-indication mappings
├── drugs_interactions.csv                       # Drug-protein interactions
├── protein_nodes_with_embeddings_v4.pkl        # Protein features + ESM-2 embeddings
│
├── chembl_36/                                   # ChEMBL database (optional)
│   └── chembl_36_sqlite/
│       └── chembl_36.db                         # SQLite database (~4GB)
│
├── Gemini_approach/                             # Alternative approaches
│   ├── experiments/
│   │   ├── chemberta_embeddings.npy            # ChemBERTa drug embeddings
│   │   └── exp1.ipynb                          # ChemBERTa experiments
│   └── src/
│
└── old_model_results_tramse/                    # TransE baseline results
    ├── top_50_predicted_drug_effects.csv
    └── top_50_predicted_drug_protein.csv
```

### Key Files

**Notebooks (Model Training):**
- `code copy.ipynb` - **Model V1**: GraphSAGE baseline (recommended for learning)
- `800 million parmaters model.ipynb` - **Model V2**: Attention-enhanced (state-of-the-art)
- `transR/pure_knowledge_embeddings.ipynb` - **TransR**: Pure knowledge embeddings (lightweight)

**Model Weights:**
- `best_model_clean.pt` - Trained Model V1 (69.2% AUC)
- `best_model_improved.pt` - Trained Model V2 (88.2% AUC)
- `transR/transr_full_model.pt` - Trained TransR model ([download from Google Drive](https://drive.google.com/file/d/1cDcCLBwwPVpEtgfracVmW1UkLqxaqC2M/view?usp=sharing))

**Data Files:**
- `drug_nodes.csv` - Drug metadata from ChEMBL
- `drug_effects.csv` - Drug-indication relationships
- `drugs_interactions.csv` - Known drug-target interactions
- `protein_nodes_with_embeddings_v4.pkl` - Protein sequences + ESM-2 embeddings (2.8GB)
- `transR/*.pkl` - TransR pickle data files ([download from Google Drive](https://drive.google.com/file/d/1cDcCLBwwPVpEtgfracVmW1UkLqxaqC2M/view?usp=sharing))

---

## 📧 Contact

**Authors:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  
**Email:** abodahab@stanford.edu, iaredur@stanford.edu, rubyh@stanford.edu  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)
