# Pharmacology Knowledge Graph: Drug-Target-Effect Prediction

**CS224W: Machine Learning with Graphs - Final Project**  
**Students:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  
**Fall 2025**

Heterogeneous graph neural network for predicting drug-target interactions and therapeutic effects. Built from ChEMBL data with 3,127 drugs, 1,156 proteins, and 1,065 effects.

**📊 Two Models:**
- **Model V1** (3.4M params): GraphSAGE baseline → **69.2% AUC**
- **Model V2** (888M params): NNConv + Attention + Contrastive → **88.2% AUC** 🏆

**� Notebooks:** [Model V1](code%20copy.ipynb) | [Model V2](800%20million%20parmaters%20model.ipynb)

## 📊 Model Comparison

| Feature | Model V1 (GraphSAGE) | Model V2 (Attention) |
|---------|---------------------|---------------------|
| **Parameters** | 3.4M | 888M |
| **Architecture** | GAT + GraphSAGE | NNConv + Attention + Contrastive |
| **Test AUC** | 69.2% | **88.2%** 🏆 |
| **Precision** | 60.7% | **86.8%** 🏆 |
| **Training Time** | 20 min 🏆 | 130 min |
| **GPU Memory** | 24GB 🏆 | 96GB |

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
- Training: 45 min (NVIDIA GPU, 16GB+)

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

## 🚀 Usage

**Choose Model V2** (88.2% AUC) for production - requires 16GB+ GPU  
**Choose Model V1** (69.2% AUC) for baselines/education - works on 8GB GPU

**Notebooks:**
- Model V1: [`code copy.ipynb`](code%20copy.ipynb)
- Model V2: [`800 million parmaters model.ipynb`](800%20million%20parmaters%20model.ipynb)


**Data:** ChEMBL 36 database (place in `chembl_36/chembl_36_sqlite/chembl_36.db`)

---

## � References

## 📚 References

- **GAT:** Veličković et al., "Graph Attention Networks" (ICLR 2018)
- **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- **ESM-2:** Lin et al., "Evolutionary-scale prediction of atomic-level protein structure" (Science 2023)
- **ChEMBL:** Gaulton et al., "The ChEMBL database in 2017" (Nucleic Acids Research 2017)

---

## 📧 Contact

**Authors:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  
**Email:** abodahab@stanford.edu  
**Course:** CS224W: Machine Learning with Graphs (Fall 2025)  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)


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
compute_time = ~1 hour (RTX GeForce 3090)
```

#### **3. Effect Features (Learnable)**

**Initialization:** Random normal distribution
- **Dimension:** 32
- **Trainable:** Yes (updated via backpropagation)
- **Purpose:** Learn task-specific representations for clinical effects

### Training Procedure

#### **Data Splits (No Leakage)**

```python
# Edge-level splitting (transductive setting)
train_edges = 80%  # Used for GNN message passing
val_edges = 10%    # Held out for hyperparameter tuning
test_edges = 10%   # Final evaluation only

# Critical: Message passing uses ONLY training edges
# Validation/test edges are predicted but not used for aggregation
```

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
epochs = 100 (with early stopping)
early_stopping_patience = 10 epochs
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
- **Attention mechanism:** Learn importance of atoms/bonds
- **Permutation invariant:** Order of atoms doesn't matter
- **Variable graphs:** Handle molecules of different sizes

#### **3. Why GraphSAGE for Message Passing?**
- **Scalable aggregation:** Mean pooling over neighbors
- **Inductive capability:** Can generalize to new nodes
- **Flexible:** Works with different node feature dimensions

#### **4. Why Bidirectional Edges?**
- **Information flow:** Drugs ↔ Proteins ↔ Effects
- **Richer embeddings:** Nodes learn from multiple hops
- **Symmetry:** Both endpoints benefit from relationship

---

## 🧪 Methodology & Validation

### Problem Formulation (CS224W Framework)

**Task:** Link prediction in heterogeneous knowledge graph

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

#### **3. Metrics**

**Primary Metrics:**
- **AUC-ROC:** Area under receiver operating characteristic curve
- **Average Precision:** Area under precision-recall curve

**Secondary Metrics:**
- **Precision@0.5:** Fraction of predictions >0.5 that are correct
- **Recall@0.5:** Fraction of true edges with score >0.5
- **F1-Score:** Harmonic mean of precision and recall

### Cross-Validation Strategy

**Validation Set Usage:**
- Monitor training progress (every epoch)
- Early stopping (patience = 10 epochs)
- Hyperparameter selection (learning rate, hidden dims, num layers)

**Test Set Usage:**
- Single evaluation after training completes
- Load best model (highest validation AUC)
- Report final performance

### Reproducibility

**Random Seeds:**
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```
## 📈 Use Cases

### 1. **Drug Repurposing**

Find new therapeutic uses for approved drugs:

```python
# Example: Query novel indications for Aspirin
drug_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
predictions = predict_new_drug_embedding(drug_smiles, top_k=10)

# Output: Predicted new effects beyond pain/inflammation
# - Cardiovascular prevention (known)
# - Colorectal cancer prevention (emerging evidence)
```

### 2. **Target Identification**

Predict protein targets for experimental compounds:

```python
# Novel kinase inhibitor candidate
novel_smiles = "Cc1ccc(Nc2nccc(...)...)cc1"
targets = predict_protein_targets(novel_smiles, top_k=20)

# Helps prioritize biochemical assays
```

### 3. **Off-target Prediction**

Identify safety liabilities early:

```python
# Check for unintended receptor binding
all_targets = predict_all_targets(drug_smiles, threshold=0.45)
safety_flags = [t for t in all_targets if t in ['hERG', 'CYP3A4', 'Opioid']]
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

## � Repository Structure

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

**Model Weights:**
- `best_model_clean.pt` - Trained Model V1 (69.2% AUC)
- `best_model_improved.pt` - Trained Model V2 (88.2% AUC)

**Data Files:**
- `drug_nodes.csv` - Drug metadata from ChEMBL
- `drug_effects.csv` - Drug-indication relationships
- `drugs_interactions.csv` - Known drug-target interactions
- `protein_nodes_with_embeddings_v4.pkl` - Protein sequences + ESM-2 embeddings (2.8GB)

---

## 📧 Contact

**Authors:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  
**Email:** abodahab@stanford.edu, iaredur@stanford.edu, rubyh@stanford.edu  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)
