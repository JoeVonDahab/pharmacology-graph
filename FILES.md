# File Manifest - Pharmacology Graph

Quick reference for what each file does.

---

## 🚀 **ESSENTIAL FILES** (Keep These)

### Application Files
- **`app.py`** - Main Gradio web application (604 lines)
- **`requirements_app.txt`** - Python dependencies for the app
- **`start_app.sh`** - Launch script (recommended)
- **`test_app.py`** - Diagnostic test for data loading

### Data Files (Required)
- **`drug_nodes.csv`** - 3,127 approved drugs with SMILES structures
- **`protein_nodes_with_embeddings.csv`** - 1,156 protein targets
- **`drug_effects.csv`** - 8,312 clinical effects/indications
- **`drugs_interactions.csv`** - 11,703 known drug-protein interactions
- **`graph_embeddings.npy`** - Trained 128-dim TransE embeddings (5,201 nodes)
- **`node_to_idx.npy`** - Node ID to embedding index mapping
- **`top_50_predicted_drug_protein.csv`** - Top 50 predicted drug-target pairs
- **`top_50_predicted_drug_effects.csv`** - Top 50 predicted drug-effect pairs

### Documentation
- **`README.md`** - Project overview, results, and citations
- **`SETUP.md`** - Complete setup guide (START HERE!)
- **`DEPLOYMENT.md`** - Guide for deploying to Hugging Face Spaces
- **`APP_GUIDE.md`** - User guide for the web application

---

## 🔬 **DEVELOPMENT FILES** (Optional)

### Research Notebook
- **`code.ipynb`** - Main Jupyter notebook with full pipeline:
  - Data extraction from ChEMBL
  - ESM-2 protein embedding generation (~30 min GPU)
  - TransE graph model training (100 epochs)
  - Link prediction and validation
  
**Note**: Only needed if you want to retrain the model or explore the data processing pipeline.

### Full Dependencies
- **`requirements.txt`** - Complete dependencies (includes transformers, torch, rdkit)
  - **Use this if**: You want to run `code.ipynb`
  - **Skip if**: You just want to run the app (use `requirements_app.txt` instead)

### ChEMBL Database
- **`chembl_36/chembl_36_sqlite/chembl_36.db`** - Full ChEMBL 36 database (4.2GB)
  - **Needed for**: Running `code.ipynb` to extract fresh data
  - **Skip if**: You're using the pre-extracted CSV files

---

## 🗑️ **REMOVABLE FILES** (Can Delete)

### Duplicates/Old Versions
- `run_app.sh` - Old launch script (use `start_app.sh` instead)
- `APP_SUMMARY.md` - Duplicate of `APP_GUIDE.md`
- `CHECKLIST.md` - Deployment checklist (one-time use)
- `BUGFIX_SUMMARY.md` - Development notes about version fixes

### Testing Scripts
- `test_search.py` - Manual testing guide (not automated)

### Intermediate Files (Can Regenerate)
- `main.py` - Old/experimental code
- `.python-version` - Python version pin (use pyproject.toml instead)
- `*.png` - Visualization outputs (can regenerate from notebook)
- `*.pkl` - Pickle serializations (can regenerate from CSV)
- `drug_mechanism_filtered.csv` - Intermediate processing file
- `drug_warnings.csv` - Intermediate processing file
- `edges_drug_protein.csv` - Intermediate processing file
- `proteins_for_embedding.csv` - Intermediate processing file
- `top_50_predicted_drug_targets.csv` - Duplicate of `top_50_predicted_drug_protein.csv`

**Run cleanup script**: `./cleanup.sh` to remove these automatically.

---

## 📏 **FILE SIZES**

### App Files (~10 MB total)
```
drug_nodes.csv                          700 KB
protein_nodes_with_embeddings.csv       400 KB
drug_effects.csv                        800 KB
drugs_interactions.csv                  1.5 MB
graph_embeddings.npy                    5 MB
node_to_idx.npy                         50 KB
top_50_predicted_*.csv                  10 KB each
app.py                                  50 KB
```

### Development Files (4.2 GB)
```
chembl_36/chembl_36_sqlite/chembl_36.db  4.2 GB
code.ipynb                               500 KB
```

---

## 🎯 **WHAT YOU NEED FOR...**

### Running the App Only
```
✓ app.py
✓ requirements_app.txt
✓ start_app.sh (or test_app.py)
✓ All CSV files
✓ graph_embeddings.npy
✓ node_to_idx.npy
✓ README.md, SETUP.md (for reference)
```
**Total**: ~10 MB

### Retraining the Model
```
✓ Everything from "Running the App"
✓ code.ipynb
✓ requirements.txt
✓ chembl_36/ directory (ChEMBL database)
```
**Total**: ~4.2 GB

### Deploying to Hugging Face Spaces
```
✓ app.py
✓ All CSV files
✓ graph_embeddings.npy
✓ node_to_idx.npy
✓ requirements_app.txt → rename to requirements.txt
✓ SPACE_README.md → use as README.md
```
**Total**: ~10 MB (HF Spaces free tier supports up to 50GB)

---

## 🧹 **CLEANUP COMMANDS**

### Remove all unnecessary files
```bash
./cleanup.sh
```

### Manual cleanup
```bash
# Remove duplicate documentation
rm -f BUGFIX_SUMMARY.md APP_SUMMARY.md CHECKLIST.md

# Remove old scripts
rm -f run_app.sh test_search.py main.py

# Remove generated outputs
rm -f *.png *.pkl

# Remove intermediate CSV files
rm -f drug_mechanism_filtered.csv drug_warnings.csv edges_drug_protein.csv proteins_for_embedding.csv
```

### Remove development files (if you don't need to retrain)
```bash
# Remove notebook and database (WARNING: Large files!)
rm -rf code.ipynb chembl_36/ requirements.txt
```

---

## 📦 **WHAT'S IN GIT**

The `.gitignore` is configured to:

**✅ Track:**
- All Python files (`*.py`)
- All CSV files (`*.csv`)
- All Jupyter notebooks (`*.ipynb`)
- Documentation (`*.md`)
- Config files (`requirements*.txt`, `pyproject.toml`)

**❌ Ignore:**
- Large binary files (`*.npy`, `*.pkl`, `*.db`)
- Images (`*.png`, `*.jpg`)
- Python cache (`__pycache__/`, `*.pyc`)
- Model checkpoints (`*.pth`, `*.pt`)
- Compressed files (`*.zip`, `*.tar.gz`)

**Why NPY files aren't tracked**: They're 5MB+ and can be regenerated from the notebook. Download them separately or run the notebook to create them.

---

## 🔄 **REGENERATING FILES**

If you're missing `.npy` files or want to retrain:

```bash
# Install full dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook code.ipynb

# Run all cells (takes ~40 minutes on GPU)
# This will regenerate:
# - All CSV files
# - graph_embeddings.npy
# - node_to_idx.npy
# - Prediction CSVs
```

---

**Last Updated**: October 10, 2025  
**Total Project Size**: 4.2 GB (with database) or 10 MB (app only)
