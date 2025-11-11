---
title: Pharmacology Knowledge Graph
emoji: 💊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---
- https://www.mediafire.com/file/i33yodulcc44q9g/protein_nodes_with_embeddings_v4.pkl/file

# Pharmacology Knowledge Graph: Drug-Target-Effect Prediction

**CS224W: Machine Learning with Graphs - Final Project**  
**Student:** Youssef Abo-Dahab (abodahab@stanford.edu)  
**Fall 2024**

A heterogeneous graph neural network that predicts drug-target interactions and therapeutic effects by learning joint representations of molecular structures, protein sequences, and clinical outcomes. The model uses Graph Attention Networks (GAT) for molecular encoding and heterogeneous message passing for multi-relational link prediction.

**🚀 [Try the Interactive Demo](https://huggingface.co/spaces/JoeVonDahab/pharmacology-graph)** | **📖 [Full Setup Guide](SETUP.md)** | **🎓 [Research Notebook](code.ipynb)**

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JoeVonDahab/pharmacology-graph.git
cd pharmacology-graph

# 2. Install dependencies
pip install -r requirements_app.txt

# 3. Run the interactive app
./start_app.sh
# Open http://localhost:7860 in your browser
```

**Requirements**: Python 3.9+, 8GB RAM, 10MB disk space (app only)

For complete setup instructions and troubleshooting, see **[SETUP.md](SETUP.md)**.

---

## 🎯 Project Overview

This CS224W project builds a heterogeneous graph neural network for pharmacological link prediction:

1. **Graph Construction:** Build multi-relational knowledge graph from ChEMBL database
   - **Nodes:** Drugs (3,127), Proteins (1,156), Effects (1,065)
   - **Edges:** Drug-Protein bindings (11,493), Drug-Effect relationships (6,496)

2. **Node Feature Extraction:**
   - **Drugs:** Molecular graphs from SMILES (57-dim atom features, 4-dim bond features)
   - **Proteins:** ESM-2 language model embeddings (2,560-dim)
   - **Effects:** Learnable embeddings (32-dim)

3. **Model Architecture:** Heterogeneous GNN with specialized encoders
   - **Drug Encoder:** 3-layer Graph Attention Network (GAT) for molecular graphs
   - **Projection Layers:** Map all node types to shared 256-dim space
   - **Message Passing:** 3 layers of heterogeneous graph convolution (GraphSAGE)
   - **Link Prediction:** MLP heads for drug-protein and drug-effect predictions

4. **Training:** Link prediction with proper train/val/test splits (80/10/10)
   - Binary cross-entropy loss with negative sampling
   - Early stopping based on validation AUC

### Key Innovation: Multi-Modal Heterogeneous GNN

Unlike traditional methods that treat drugs as fixed fingerprints, our model:
- **Learns molecular representations** via attention over atoms/bonds
- **Integrates protein biology** through pre-trained ESM-2 embeddings
- **Captures pharmacological relationships** via heterogeneous message passing
- **Predicts both targets and effects** in a unified framework

---

## 🏗️ Model Architecture

### High-Level Pipeline

```
ChEMBL Database (3,127 drugs, 1,156 proteins, 1,065 effects)
    ↓
[1. Node Feature Extraction]
    ├── Drugs: SMILES → RDKit → Molecular Graphs (atoms + bonds)
    ├── Proteins: Sequences → ESM-2 (3B params) → 2560-dim embeddings
    └── Effects: Random init → 32-dim learnable embeddings
    ↓
[2. Heterogeneous Graph Construction]
    ├── Nodes: {drug, protein, effect}
    └── Edges: {(drug, binds_to, protein), (drug, treats, effect)}
    ↓
[3. PharmacologyHeteroGNN Model]
    ├── Drug Molecular Encoder (GAT): 57-dim → 256-dim
    ├── Protein Projection: 2560-dim → 256-dim
    ├── Effect Projection: 32-dim → 256-dim
    ├── 3× Heterogeneous Graph Conv Layers (GraphSAGE)
    └── Link Prediction Heads (MLPs)
    ↓
[4. Training & Evaluation]
    ├── Train/Val/Test Split: 80%/10%/10%
    ├── Loss: Binary Cross-Entropy + Negative Sampling
    └── Metrics: AUC-ROC, Precision, Recall, F1
```

### Detailed Architecture

#### **Layer 1: Drug Molecular Encoder (GAT)**
```
Input: Molecular graphs (variable size)
  ├── Atoms: (num_atoms, 57) one-hot features
  │   └── [symbol, degree, charge, aromatic, hybridization, hydrogens]
  └── Bonds: (num_bonds, 4) one-hot features
      └── [SINGLE, DOUBLE, TRIPLE, AROMATIC]

Processing:
  1. Linear(57 → 128) for atoms
  2. Linear(4 → 128) for bonds
  3. GAT Layer 1: (128) → (128 × 4 heads) = 512
  4. GAT Layer 2: (512) → (128 × 4 heads) = 512
  5. GAT Layer 3: (512) → (128 × 4 heads) = 512
  6. Global Mean Pool: (num_atoms, 512) → (1, 512)
  7. Linear(512 → 256)

Output: (num_drugs, 256)
```

#### **Layer 2: Protein & Effect Projections**
```
Protein:
  Input: (1156, 2560) ESM-2 embeddings
    → Linear(2560 → 256)
    → LayerNorm + ReLU + Dropout(0.1)
  Output: (1156, 256)

Effect:
  Input: (1065, 32) learnable embeddings
    → Linear(32 → 256)
    → LayerNorm + ReLU + Dropout(0.1)
  Output: (1065, 256)
```

#### **Layer 3-5: Heterogeneous Graph Convolution (3 layers)**
```
Edge Types (bidirectional):
  - (drug, binds_to, protein) ↔ (protein, binds_to_by, drug)
  - (drug, treats, effect) ↔ (effect, treated_by, drug)

For each layer:
  For each edge type (src → dst):
    1. GraphSAGE: Message(src) → dst
    2. Aggregate: Mean over incoming messages
    3. Residual: output = LayerNorm(aggregated + input)
  Apply ReLU to all node types

Output: Refined embeddings in 256-dim shared space
```

#### **Layer 6: Link Prediction Heads**
```
Drug-Protein Predictor:
  Input: Concat(drug_emb, protein_emb) = 512-dim
    → Linear(512 → 256) → ReLU → Dropout(0.2)
    → Linear(256 → 1) → Sigmoid
  Output: P(drug binds to protein)

Drug-Effect Predictor:
  Input: Concat(drug_emb, effect_emb) = 512-dim
    → Linear(512 → 256) → ReLU → Dropout(0.2)
    → Linear(256 → 1) → Sigmoid
  Output: P(drug treats effect)
```

### Model Statistics

| Component | Parameters |
|-----------|------------|
| Drug GAT Encoder | ~1.5M |
| Protein Projection | ~655K |
| Effect Projection | ~8K |
| Hetero Graph Conv (3×) | ~590K |
| Link Prediction Heads (2×) | ~264K |
| **Total** | **~3.0M** |

**Key Hyperparameters:**
- Shared embedding dimension: 256
- GAT heads: 4
- GAT layers: 3
- Hetero conv layers: 3
- Learning rate: 0.001 (manual SGD)
- Batch size: 256 drugs
- Negative sampling ratio: 1:1

---

## 📊 Experimental Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Nodes** | |
| Total drugs | 3,127 |
| Total proteins | 1,156 |
| Total effects | 1,065 |
| **Edges** | |
| Drug-Protein interactions | 11,493 |
| Drug-Effect relationships | 6,496 |
| **Splits** | |
| Training edges (drug-protein) | 9,194 (80%) |
| Validation edges (drug-protein) | 1,149 (10%) |
| Test edges (drug-protein) | 1,150 (10%) |
| Training edges (drug-effect) | 5,196 (80%) |
| Validation edges (drug-effect) | 649 (10%) |
| Test edges (drug-effect) | 651 (10%) |

### Model Performance (Without Data Leakage)

**Test Set Results (Final Model):**

| Metric | Drug-Protein Links | Drug-Effect Links | Average |
|--------|-------------------|-------------------|---------|
| **AUC-ROC** | 0.6225 | 0.6464 | **0.6344** |
| **Precision** | 0.5796 | 0.5919 | 0.5858 |
| **Recall** | 0.6809 | 0.7220 | 0.7015 |
| **F1-Score** | 0.6261 | 0.6505 | 0.6383 |
| **Avg Precision** | 0.5926 | 0.6282 | 0.6104 |

### Training Progress

- **Total Epochs:** 83 (early stopping triggered)
- **Training Time:** ~7.5 minutes (NVIDIA GPU)
- **Best Validation AUC:** 0.6369 (epoch 73)
- **Final Test AUC:** 0.6344

**Learning Curve:**
- Initial validation AUC: 0.5165 (epoch 1, near random)
- Final validation AUC: 0.6369 (epoch 73)
- Improvement: **+23.3%** over random baseline

### Ablation Studies

| Configuration | Test AUC-ROC | Δ from Full Model |
|--------------|--------------|-------------------|
| **Full Model** | **0.6344** | baseline |
| Without GAT (random drug init) | 0.5421 | -14.5% |
| Without ESM-2 (random protein init) | 0.5789 | -8.7% |
| Without hetero message passing (1 layer) | 0.5912 | -6.8% |
| Without negative sampling | 0.5234 | -17.5% |

### Comparison to Baselines

| Method | Test AUC-ROC | Description |
|--------|-------------|-------------|
| **PharmacologyHeteroGNN (Ours)** | **0.6344** | GAT + ESM-2 + Hetero GNN |
| Morgan Fingerprint + MLP | 0.5687 | Traditional cheminformatics |
| ESM-2 Embeddings + Dot Product | 0.5423 | Protein-only similarity |
| Random Predictor | 0.5012 | Uniform random scores |
| TransE (prior work) | 0.5821 | Shallow graph embeddings |

**Key Findings:**
- Heterogeneous GNN outperforms shallow methods by **+8.2%**
- Molecular graph encoder (GAT) crucial for drug representation
- Multi-relational message passing captures complex pharmacology

---

## 🚀 Quick Start

### Requirements

```bash
# Install dependencies with UV
uv pip install -r requirements.txt

# Core dependencies:
- pandas
- numpy
- torch
- transformers (ESM-2)
- rdkit
- networkx
- scikit-learn
- tqdm
- matplotlib
```

### Data

Download ChEMBL 36 SQLite database:
```bash
# Place in: chembl_36/chembl_36_sqlite/chembl_36.db
# Size: ~4.2 GB
```

### Run the Pipeline

```bash
# Open the Jupyter notebook
jupyter notebook code.ipynb

# Or run as Python script (convert cells first)
jupyter nbconvert --to script code.ipynb
python code.py
```

**Pipeline stages:**

1. **Data extraction** (cells 1-14): Query ChEMBL for drugs, targets, effects
2. **Protein embeddings** (cells 15-18): Generate ESM-2 embeddings (GPU recommended, ~30 min)
3. **Graph construction** (cells 19-35): Build NetworkX graph with nodes/edges
4. **TransE training** (cell 36): Train embedding model (100 epochs, ~15 min on GPU)
5. **Prediction** (cells 37-42): Generate novel drug-target and drug-effect predictions
6. **Visualization** (cells 43-45): t-SNE plots, neighbor analysis

---

## 📁 Project Structure

```
pharmacology-graph/
├── code.ipynb                          # Main analysis notebook
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies (UV)
│
├── chembl_36/
│   └── chembl_36_sqlite/
│       └── chembl_36.db               # ChEMBL database (not in git)
│
├── protein_nodes_with_embeddings.pkl  # ESM-2 protein embeddings (not in git)
├── drug_nodes.pkl                     # Drug metadata (not in git)
├── drug_effects.pkl                   # Drug-indication mappings (not in git)
├── drug_protein_interactions.pkl      # Known drug-target edges (not in git)
│
├── graph_embeddings.npy               # Trained TransE embeddings (not in git)
├── node_to_idx.npy                    # Node index mapping (not in git)
│
├── top_50_predicted_drug_protein.csv  # Novel target predictions
├── top_50_predicted_drug_effects.csv  # Novel indication predictions
└── drug_neighbors_visualization.png   # t-SNE cluster plot (not in git)
```

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
compute_time = ~2 hours (NVIDIA A100)
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

### Computational Requirements

| Resource | Training | Inference |
|----------|----------|-----------|
| **GPU Memory** | 8 GB (NVIDIA RTX 3070) | 4 GB |
| **Training Time** | ~8 min (83 epochs) | - |
| **Prediction Time** | - | ~0.5 sec (1000 predictions) |
| **Disk Space** | 500 MB (model + data) | 100 MB |

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

**Prevents Data Leakage:**
- GNN aggregation uses ONLY E_train
- Validation/test edges never influence node embeddings
- Proper evaluation of generalization to unseen relationships

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

**Fixed Splits:**
- Same train/val/test split across all experiments
- Deterministic negative sampling (seeded)

**Hardware:**
- NVIDIA RTX 3070 (8GB VRAM)
- CUDA 11.8
- PyTorch 2.0+

---

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

## 🎓 Key Contributions (CS224W Concepts)

### 1. **Heterogeneous Graph Neural Networks**
- **Multi-modal nodes:** Integrated molecular graphs, protein sequences, and clinical labels
- **Type-specific encoders:** GAT for graphs, MLPs for embeddings
- **Heterogeneous message passing:** GraphSAGE with type-specific aggregation
- **Demonstrates:** Power of GNNs beyond homogeneous graphs

### 2. **Graph Attention for Molecular Encoding**
- **Attention mechanism:** Learn importance of atoms and bonds
- **Inductive bias:** Chemistry-aware (aromatic rings, functional groups)
- **Permutation invariance:** Order-independent molecular representation
- **Demonstrates:** GAT effectiveness on irregular graph structures

### 3. **Link Prediction with Proper Evaluation**
- **No data leakage:** Test edges hidden from message passing
- **Transductive setting:** All nodes visible, predict missing edges
- **Negative sampling:** Balanced evaluation with hard negatives
- **Demonstrates:** Rigorous evaluation methodology

### 4. **Multi-Relational Knowledge Graphs**
- **Multiple edge types:** binds_to vs treats (different semantics)
- **Bidirectional edges:** Information flows both ways
- **Joint prediction:** Unified model for targets + effects
- **Demonstrates:** Scalability to complex real-world graphs

### 5. **Practical Drug Discovery Application**
- **Real-world data:** ChEMBL database (clinical + experimental)
- **Interpretable predictions:** Can explain via attention weights
- **Computational efficiency:** ~8 min training, real-time inference
- **Demonstrates:** ML with graphs for scientific discovery

---

## 📝 Citation

```bibtex
@misc{abodahab2024pharmacology,
  author = {Abo-Dahab, Youssef},
  title = {Heterogeneous Graph Neural Networks for Pharmacology Link Prediction},
  year = {2024},
  institution = {Stanford University},
  course = {CS224W: Machine Learning with Graphs},
  url = {https://github.com/JoeVonDahab/pharmacology-graph}
}
```

## 📚 Related Work & References

### Graph Neural Networks
- **GAT:** Veličković et al., "Graph Attention Networks" (ICLR 2018)
- **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- **Heterogeneous GNNs:** Wang et al., "Heterogeneous Graph Attention Network" (WWW 2019)

### Drug Discovery with ML
- **MoleculeNet:** Wu et al., "MoleculeNet: A Benchmark for Molecular ML" (Chemical Science 2018)
- **ChemBERTa:** Chithrananda et al., "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction" (2020)
- **ESM-2:** Lin et al., "Evolutionary-scale prediction of atomic-level protein structure" (Science 2023)

### Knowledge Graphs
- **TransE:** Bordes et al., "Translating Embeddings for Modeling Multi-relational Data" (NeurIPS 2013)
- **Link Prediction:** Zhang & Chen, "Link Prediction Based on Graph Neural Networks" (NeurIPS 2018)

### Databases
- **ChEMBL:** Gaulton et al., "The ChEMBL database in 2017" (Nucleic Acids Research 2017)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add attention-based graph neural network (GAT/RGCN)
- [ ] Incorporate 3D protein structures (AlphaFold2)
- [ ] Multi-task learning (toxicity + efficacy)
- [ ] Temporal dynamics (drug resistance evolution)
- [ ] Web interface (Streamlit/Gradio demo)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **ChEMBL** for curated pharmacological data
- **Meta AI** for ESM-2 protein language models
- **RDKit** for cheminformatics tools
- **PyTorch** ecosystem for deep learning infrastructure

---

## 📧 Contact

**Author:** Youssef Abo-Dahab  
**Email:** abodahab@stanford.edu  
**Course:** CS224W: Machine Learning with Graphs (Fall 2024)  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)

For questions or collaboration: [create an issue](https://github.com/JoeVonDahab/pharmacology-graph/issues)

---

## 🙏 Acknowledgments

- **CS224W Teaching Team** for course materials and guidance on graph neural networks
- **ChEMBL** for curated pharmacological data
- **Meta AI** for ESM-2 protein language models  
- **RDKit** for cheminformatics tools
- **PyTorch Geometric** for GNN implementations

---

*CS224W Final Project - Fall 2024*  
*Last updated: November 2024*
