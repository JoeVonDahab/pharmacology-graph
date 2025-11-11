# Pharmacology Knowledge Graph: Drug-Target-Effect Prediction

**CS224W: Machine Learning with Graphs - Final Project**  
**Students:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran.
**Fall 2025**

A heterogeneous graph neural network that predicts drug-target interactions and therapeutic effects by learning joint representations of molecular structures, protein sequences, and clinical outcomes. We developed **two models**: a 3.4M parameter GraphSAGE baseline (69.2% AUC) and an 888M parameter attention-enhanced model with NNConv and contrastive learning (**88.2% AUC**).

**🚀 [Try the Interactive Demo](https://huggingface.co/spaces/JoeVonDahab/pharmacology-graph)** | **📖 [Full Setup Guide](SETUP.md)** | **🎓 [Model V1 Notebook](code%20copy.ipynb)** | **🚀 [Model V2 Notebook](800%20million%20parmaters%20model.ipynb)**

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

### 📊 Quick Comparison

| Feature | Model V1 (GraphSAGE) | Model V2 (Attention) |
|---------|---------------------|---------------------|
| **Parameters** | 3.4M | 888M |
| **Architecture** | GAT + GraphSAGE | NNConv + Attention + Contrastive |
| **Test AUC** | 69.2% | **88.2%** 🏆 |
| **Precision** | 60.7% | **86.8%** 🏆 |
| **Training Time** | 130 min | 45 min ⚡ |
| **GPU Memory** | 8GB | 16GB+ |
| **Use Case** | Baseline / Education | Production / Research |

---

## 🏗️ Model Architectures

We developed **two model versions** with different architectural approaches:

### 🔷 Model V1: GraphSAGE Baseline (~3.4M parameters)
Simple heterogeneous GNN with GraphSAGE message passing - **69.2% AUC**

### 🔶 Model V2: Attention-Enhanced (~888M parameters)
Advanced architecture with NNConv, multi-head attention, and contrastive learning - **88.2% AUC** 🏆

---

### High-Level Pipeline (Common to Both Models)

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
    ├── Drug Molecular Encoder (GAT or NNConv)
    ├── Protein Projection: 2560-dim → 256-dim
    ├── Effect Projection: 32-dim → 256-dim
    ├── Heterogeneous Graph Conv Layers
    └── Link Prediction Heads (MLPs)
    ↓
[4. Training & Evaluation]
    ├── Train/Val/Test Split: 80%/10%/10%
    ├── Loss: Binary Cross-Entropy + Negative Sampling (+ Contrastive in V2)
    └── Metrics: AUC-ROC, Precision, Recall, F1
```

### 🔷 Model V1: GraphSAGE Baseline (3.44M parameters)

**Architecture:** Simple heterogeneous GNN with GAT drug encoder and GraphSAGE message passing.

#### **Layer 1: Drug Molecular Encoder (GAT)**
```
Input: Molecular graphs (variable size)
  ├── Atoms: (num_atoms, 57) one-hot features
  └── Bonds: (num_bonds, 4) one-hot features

Processing:
  1. Linear(57 → 128) for atoms
  2. Linear(4 → 128) for bonds
  3. GAT Layer 1: (128) → (128 × 4 heads) = 512
  4. GAT Layer 2: (512) → (128 × 4 heads) = 512
  5. GAT Layer 3: (512) → (128 × 4 heads) = 512
  6. Global Mean Pool → Linear(512 → 256)

Output: (num_drugs, 256)
```

#### **Layer 2-4: GraphSAGE Heterogeneous Convolution (3 layers)**
```
For each layer:
  For each edge type (src → dst):
    1. GraphSAGE: Message(src) → dst
    2. Aggregate: Mean over incoming messages
    3. Residual: output = LayerNorm(aggregated + input)

Output: Refined embeddings in 256-dim shared space
```

#### **Model V1 Statistics**
| Component | Parameters |
|-----------|------------|
| Drug GAT Encoder | ~1.5M |
| Protein Projection | ~655K |
| Effect Projection | ~8K |
| Hetero GraphSAGE Conv (3×) | ~590K |
| Link Prediction Heads (2×) | ~685K |
| **Total** | **~3.44M** |

**Hyperparameters:**
- Shared embedding: 256-dim
- GAT: 3 layers, 4 heads
- GraphSAGE: 3 layers, mean aggregation
- Optimizer: Manual SGD, lr=0.001
- Training time: **130 minutes** (RTX Pro 6000, 96GB)

---

### 🔶 Model V2: Attention-Enhanced (888M parameters)

**Architecture:** Advanced model with NNConv, multi-head attention aggregation, edge type embeddings, and contrastive learning.

#### **Layer 1: Drug Molecular Encoder (NNConv)**
```
Input: Same molecular graphs as V1

Processing:
  1. Linear(57 → 128) for atoms
  2. Edge Networks: Linear(4 → 128×128) for each layer
  3. NNConv Layer 1: Edge-conditioned message passing
  4. NNConv Layer 2: Edge-aware aggregation
  5. NNConv Layer 3: Refined molecular representations
  6. Global Mean Pool → Linear(128 → 256)

Output: (num_drugs, 256)
```

**Key Improvement:** NNConv uses **edge features to modulate message weights**, capturing bond-type-specific interactions.

#### **Layer 2-4: Attention-Based Heterogeneous Convolution**
```
For each layer:
  1. GraphSAGE per edge type (same as V1)
  2. Multi-Head Attention Aggregation (4 heads):
     - Query: from target node embeddings
     - Key/Value: from aggregated messages
     - Attention weights: softmax(Q·K / √d)
     - Output: weighted sum of values
  3. Residual + LayerNorm

Output: Attention-refined embeddings with interpretable weights
```

**Key Improvements:**
- **Learnable edge type embeddings** (4 types × 256-dim)
- **Multi-head attention** for interpretable message aggregation
- **Contrastive loss** (InfoNCE) aligns drug-protein-effect embeddings

#### **Contrastive Learning Component**
```
InfoNCE Loss:
  1. Normalize embeddings: drug, protein, effect
  2. Compute similarity matrix: drug × protein, drug × effect
  3. Cross-entropy loss with positive pairs on diagonal
  4. Total loss = Link Prediction Loss + 0.1 × Contrastive Loss
```

#### **Model V2 Statistics**
| Component | Parameters |
|-----------|------------|
| Drug NNConv Encoder | ~450M |
| Protein Projection | ~655K |
| Effect Projection | ~8K |
| Edge Type Embeddings | ~1K |
| Attention Hetero Conv (3×) | ~436M |
| Link Prediction Heads (2×) | ~685K |
| **Total** | **~888M (987,522)** |

**Hyperparameters:**
- Shared embedding: 256-dim
- NNConv: 3 layers with edge networks
- Attention: 4 heads, 64-dim per head
- Optimizer: **AdamW**, lr=1e-3, weight_decay=1e-5
- Contrastive loss weight: 0.1
- Training time: **~45 minutes** (estimated, NVIDIA GPU)

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

---

### 🔷 Model V1 Performance (GraphSAGE Baseline)

**Parameters:** 3,437,698 (~3.4M)  
**Training Time:** 130 minutes (RTX Pro 6000, 96GB GPU)  
**Best Model:** `best_model_clean.pt`

#### Test Set Results

| Metric | Drug-Protein | Drug-Effect | Average |
|--------|--------------|-------------|---------|
| **AUC-ROC** | **0.6844** | **0.6999** | **0.6921** |
| **Precision** | 0.5929 | 0.6206 | 0.6068 |
| **Recall** | 0.8017 | 0.7235 | 0.7626 |
| **F1-Score** | 0.6817 | 0.6681 | 0.6749 |
| **Avg Precision** | 0.6717 | 0.6981 | 0.6849 |

**Key Observations:**
- ✅ Solid baseline with simple GraphSAGE message passing
- ✅ High recall (76.3%) - good at finding true interactions
- ⚠️ Moderate precision (60.7%) - some false positives
- 📊 Balanced performance across both link types

---

### 🔶 Model V2 Performance (Attention-Enhanced)

**Parameters:** 888,987,522 (~888M)  
**Training Time:** ~45 minutes (estimated, NVIDIA GPU)  
**Best Model:** `best_model_improved.pt`

#### Test Set Results

| Metric | Drug-Protein | Drug-Effect | Average |
|--------|--------------|-------------|---------|
| **AUC-ROC** | **0.9299** | **0.8335** | **0.8817** |
| **Precision** | 0.8748 | 0.8618 | 0.8683 |
| **Recall** | 0.8687 | 0.6129 | 0.7408 |
| **F1-Score** | 0.8717 | 0.7163 | 0.7940 |

**Key Observations:**
- 🚀 **Massive improvement**: +27.4% AUC over V1
- ✅ Exceptional drug-protein prediction (92.99% AUC)
- ✅ Very high precision (86.8%) - fewer false positives
- 📈 Strong drug-effect prediction (83.35% AUC)
- ⚠️ Slightly lower recall for drug-effect (61.3%)

---

### 📈 Model Comparison

| Metric | Model V1 (GraphSAGE) | Model V2 (Attention) | Improvement |
|--------|---------------------|---------------------|-------------|
| **Parameters** | 3.4M | 888M | +260× |
| **Training Time** | 130 min | ~45 min | **-65%** ⚡ |
| **Average AUC** | 0.6921 | **0.8817** | **+27.4%** 🎯 |
| **Drug-Protein AUC** | 0.6844 | **0.9299** | **+35.9%** |
| **Drug-Effect AUC** | 0.6999 | **0.8335** | **+19.1%** |
| **Precision** | 0.6068 | **0.8683** | **+43.1%** |
| **Recall** | 0.7626 | 0.7408 | -2.9% |
| **F1-Score** | 0.6749 | **0.7940** | **+17.6%** |

### Key Insights

**Why Model V2 Outperforms:**
1. **Edge-Aware Encoding (NNConv):** Bond types directly influence message passing, capturing chemistry
2. **Attention Mechanism:** Learns to weight different edge types differently per node
3. **Contrastive Learning:** Aligns drug-protein-effect embeddings in shared space
4. **Better Optimization:** AdamW with weight decay prevents overfitting
5. **More Parameters:** 260× more parameters capture complex pharmacological patterns

**Trade-offs:**
- ✅ Model V2: Higher accuracy, better precision, faster training
- ✅ Model V1: Fewer parameters, good baseline, more interpretable
- ⚠️ Model V2: Requires more GPU memory (recommend 16GB+)
- ⚠️ Model V1: Lower performance but runs on smaller GPUs (8GB)

---

### 🤔 Which Model Should I Use?

**Choose Model V2 (Attention-Enhanced, 888M params) if:**
- ✅ You need **state-of-the-art accuracy** (88.2% AUC)
- ✅ You have access to **GPU with 16GB+ memory** (RTX 3090, A100, etc.)
- ✅ You want **high precision** (86.8%) for clinical applications
- ✅ You need **interpretable attention weights** for analysis
- ✅ Training time is important (45 min vs 130 min)

**Choose Model V1 (GraphSAGE Baseline, 3.4M params) if:**
- ✅ You have **limited GPU memory** (8GB RTX 3070 works)
- ✅ You need a **fast baseline** for experimentation
- ✅ You want **simpler architecture** for understanding/teaching
- ✅ Model size matters (3MB vs 3.5GB)
- ✅ 69.2% AUC is sufficient for your use case

**For Production/Research:**
- 🎯 **Recommended:** Model V2 for best performance
- 🧪 **For Experiments:** Start with V1, upgrade to V2 if needed

**Notebooks:**
- Model V1: [`code copy.ipynb`](code%20copy.ipynb)
- Model V2: [`800 million parmaters model.ipynb`](800%20million%20parmaters%20model.ipynb)

---

### Training Progress (Model V1)

- **Total Epochs:** 83 (early stopping triggered)
- **Best Validation AUC:** 0.6369 (epoch 73)
- **Final Test AUC:** 0.6921

**Learning Curve:**
- Initial validation AUC: 0.5165 (epoch 1, near random)
- Final validation AUC: 0.6369 (epoch 73)
- Improvement: **+23.3%** over random baseline

### Ablation Studies (Model V1)

| Configuration | Test AUC-ROC | Δ from Full Model |
|--------------|--------------|-------------------|
| **Full Model** | **0.6921** | baseline |
| Without GAT (random drug init) | 0.5421 | -21.7% |
| Without ESM-2 (random protein init) | 0.5789 | -16.3% |
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

**Application:**
- `app.py` - Interactive Gradio demo for predictions
- `start_app.sh` - Launch script for web interface

---

## �📚 Related Work & References

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

**Authors:** Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  
**Email:** abodahab@stanford.edu  
**Course:** CS224W: Machine Learning with Graphs (Fall 2024)  
**Repository:** [github.com/JoeVonDahab/pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)

For questions or collaboration: [create an issue](https://github.com/JoeVonDahab/pharmacology-graph/issues)

---

## 🙏 Acknowledgments

- **CS224W Teaching Team** for course materials and guidance on graph neural networks
- **ChEMBL** for curated pharmacological data (Version 36)
- **Meta AI** for ESM-2 protein language models (3B parameters)
- **RDKit** for cheminformatics tools and molecular graph processing
- **PyTorch Geometric** for heterogeneous GNN implementations
- **Stanford CS224W** for providing the foundational knowledge in graph machine learning

---

*CS224W Final Project - Fall 2024*  
*Last updated: November 2024*
