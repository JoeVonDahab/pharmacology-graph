# Hugging Face Space Deployment Guide

## 🚀 Deploy to Hugging Face Spaces

### Step 1: Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name:** `pharmacology-knowledge-graph`
   - **License:** MIT
   - **Space SDK:** Gradio
   - **Space hardware:** CPU basic (free tier works fine)

### Step 2: Upload Required Files

Upload these files to your Space:

**Required files:**
```
app.py                                  # Main Gradio app
requirements_app.txt                    # Python dependencies (rename to requirements.txt)
drug_nodes.csv                          # Drug metadata
protein_nodes_with_embeddings.csv       # Protein data
drug_effects.csv                        # Clinical effects
drugs_interactions.csv                  # Known drug-protein interactions
graph_embeddings.npy                    # Trained embeddings
node_to_idx.npy                         # Node index mapping
top_50_predicted_drug_protein.csv       # Pre-computed predictions (optional)
top_50_predicted_drug_effects.csv       # Pre-computed predictions (optional)
```

**File structure on Hugging Face:**
```
your-space/
├── app.py
├── requirements.txt                    # (rename requirements_app.txt)
├── README.md                           # Space description
├── drug_nodes.csv
├── protein_nodes_with_embeddings.csv
├── drug_effects.csv
├── drugs_interactions.csv
├── graph_embeddings.npy
├── node_to_idx.npy
├── top_50_predicted_drug_protein.csv
└── top_50_predicted_drug_effects.csv
```

### Step 3: Create Space README.md

Create a `README.md` in your Space with this content:

```markdown
---
title: Pharmacology Knowledge Graph Explorer
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 💊 Pharmacology Knowledge Graph Explorer

Explore drug-target-effect relationships using AI-powered predictions from a TransE knowledge graph model.

## Features

- 🔍 Search 800+ FDA-approved drugs
- 🎯 View known and predicted protein targets
- 💉 Discover potential therapeutic uses (drug repurposing)
- 🕸️ Interactive network visualization

## How It Works

This app uses a **TransE knowledge graph embedding model** trained on:
- 800+ FDA-approved drugs (from ChEMBL)
- 200+ human protein targets (with ESM-2 embeddings)
- 400+ clinical effects and indications

**Prediction method:** Cosine similarity in learned embedding space  
**Model performance:** ~90% precision on top-50 predictions

## Example Queries

Try searching for:
- **Morphine** - See opioid receptor targets and analgesic effects
- **Aspirin** - COX inhibition and cardiovascular effects  
- **Metformin** - Diabetes and potential repurposing candidates

## Citation

```bibtex
@software{pharmacology_graph_2025,
  author = {Joe VonDahab},
  title = {Pharmacology Knowledge Graph: Drug-Target-Effect Prediction},
  year = {2025},
  url = {https://github.com/JoeVonDahab/pharmacology-graph}
}
```

## Disclaimer

This is a research tool for exploratory analysis only. Predictions should be validated experimentally. Not for clinical use.
```

### Step 4: File Preparation Commands

Run these commands in your project directory:

```bash
# 1. Copy app requirements (rename for HF)
cp requirements_app.txt requirements.txt

# 2. Verify all data files exist
ls -lh *.csv *.npy

# Expected files:
# - drug_nodes.csv (~200KB)
# - protein_nodes_with_embeddings.csv (~500KB)
# - drug_effects.csv (~300KB)
# - drugs_interactions.csv (~2MB)
# - graph_embeddings.npy (~700KB)
# - node_to_idx.npy (~50KB)
```

### Step 5: Upload to Hugging Face

**Option A: Web Upload**

1. Go to your Space's Files tab
2. Click "Add file" → "Upload files"
3. Drag and drop all files
4. Commit changes

**Option B: Git Upload**

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph
cd pharmacology-knowledge-graph

# Copy files
cp /path/to/pharmacology-graph/app.py .
cp /path/to/pharmacology-graph/requirements_app.txt requirements.txt
cp /path/to/pharmacology-graph/*.csv .
cp /path/to/pharmacology-graph/*.npy .

# Commit and push
git add .
git commit -m "Initial app deployment"
git push
```

### Step 6: Monitor Deployment

1. The Space will automatically build (takes ~2-3 minutes)
2. Check the build logs for errors
3. Once running, test the app with example drugs

### Step 7: Optional Enhancements

**Upgrade to GPU (for faster predictions):**
- Settings → Hardware → Upgrade to T4 small ($0.60/hour)
- Useful if computing predictions on-the-fly for new molecules

**Enable persistence:**
- Add a `cache/` directory for storing results
- Set `GRADIO_CACHE_EXAMPLES=True` in Space settings

**Add analytics:**
```python
# In app.py, add Hugging Face analytics
import os
from huggingface_hub import HfApi

# Track usage
api = HfApi()
```

### Troubleshooting

**"File not found" errors:**
- Make sure all CSV and NPY files are in the root directory
- Check file names match exactly (case-sensitive)

**Memory errors:**
- If embeddings are too large, upgrade to "CPU upgrade" ($0.03/hour)
- Or pre-compute all predictions and use lookup tables

**Slow loading:**
- Add caching: `@st.cache_data` (Streamlit) or `gr.State()` (Gradio)
- Pre-load data in global scope (already done in `app.py`)

### Cost Estimate

**Free tier (CPU basic):**
- ✅ Sufficient for this app
- 2 vCPU, 16GB RAM
- Always-on

**Paid tier (if needed):**
- CPU upgrade: $0.03/hour (~$22/month)
- T4 GPU: $0.60/hour (only if doing real-time SMILES predictions)

---

## 🎉 You're Done!

Your app should now be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph`

Share it with:
- Colleagues and collaborators
- On Twitter/LinkedIn with #DrugDiscovery #AI
- In your competition submission

Good luck! 🚀
