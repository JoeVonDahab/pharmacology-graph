# Setup Guide - Pharmacology Knowledge Graph

Complete guide for setting up and running the Pharmacology Graph Explorer on your local machine.

---

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **RAM**: At least 8GB (16GB recommended for large datasets)
- **Disk Space**: ~5GB for ChEMBL database and model files

---

## 🚀 Quick Start (3 Steps)

### 1. Clone the Repository

```bash
git clone https://github.com/JoeVonDahab/pharmacology-graph.git
cd pharmacology-graph
```

### 2. Install Dependencies

**Option A: Using `uv` (Recommended - Fast)**
```bash
# Install uv if you don't have it
pip install uv

# Install project dependencies
uv pip install -r requirements_app.txt
```

**Option B: Using `pip`**
```bash
pip install -r requirements_app.txt
```

**Critical Version Note**: Make sure you have:
- `gradio>=4.44.0` (not 4.36.x)
- `numpy<2.0` (NumPy 2.x breaks compatibility)

### 3. Run the App

```bash
./start_app.sh
```

Or manually:
```bash
python app.py
```

Then open your browser to: **http://localhost:7860**

---

## 📦 What's Included

### Essential Files (Required to Run)

| File | Purpose | Size |
|------|---------|------|
| `app.py` | Main Gradio web application | 604 lines |
| `requirements_app.txt` | Python dependencies | - |
| `start_app.sh` | Convenience launcher script | - |

### Data Files (Required to Run)

All these CSV files are **required** for the app to work:

| File | Description | Rows |
|------|-------------|------|
| `drug_nodes.csv` | Drug metadata (name, SMILES, ChEMBL ID) | 3,127 |
| `protein_nodes_with_embeddings.csv` | Protein targets with ESM-2 features | 1,156 |
| `drug_effects.csv` | Clinical effects and indications | 8,312 |
| `drugs_interactions.csv` | Known drug-protein interactions | 11,703 |
| `graph_embeddings.npy` | Trained TransE embeddings (128-dim) | 5,201 nodes |
| `node_to_idx.npy` | Node ID mappings | - |
| `top_50_predicted_drug_protein.csv` | Top predicted drug-target pairs | 50 |
| `top_50_predicted_drug_effects.csv` | Top predicted drug-effect pairs | 50 |

**Note**: The `.npy` files are NOT tracked in git (they're large). You need to generate them by running the notebook.

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and results |
| `DEPLOYMENT.md` | Guide for deploying to Hugging Face Spaces |
| `APP_GUIDE.md` | User guide for the web application |
| `BUGFIX_SUMMARY.md` | Recent bug fixes (version compatibility) |

### Development Files

| File | Purpose |
|------|---------|
| `code.ipynb` | Main research notebook (data extraction, training, prediction) |
| `requirements.txt` | Full dependencies for running the notebook |
| `test_app.py` | Diagnostic test for app data loading |
| `test_search.py` | Manual testing guide for search functionality |

---

## 🔄 Full Setup (If You Want to Retrain the Model)

If you want to regenerate the embeddings and predictions from scratch:

### 1. Download ChEMBL Database

```bash
# The ChEMBL 36 database is large (~4.2GB)
# It's already in chembl_36/chembl_36_sqlite/chembl_36.db
# If missing, download from: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
```

### 2. Install Full Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `transformers` (for ESM-2 protein embeddings)
- `torch` (for TransE training)
- `rdkit` (for molecular fingerprints)
- `sqlite3` (for ChEMBL database queries)

### 3. Run the Notebook

Open `code.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook code.ipynb
```

This will:
1. Extract data from ChEMBL (drugs, proteins, effects)
2. Generate ESM-2 embeddings for proteins (~30 min on GPU)
3. Train TransE graph model (100 epochs, ~10 min)
4. Generate predictions using cosine similarity
5. Export all CSV and NPY files

---

## 🧪 Testing the App

### Quick Test

```bash
python test_app.py
```

This checks:
- ✓ All data files are present
- ✓ Libraries are correctly installed
- ✓ Search function works
- ✓ Data structure is valid

### Manual Testing

1. Start the app: `./start_app.sh`
2. Open: http://localhost:7860
3. Search for: "Aspirin", "Morphine", or "Imatinib"
4. Verify:
   - Dropdown shows matching results
   - Selecting a drug displays data tables
   - Network visualization renders correctly
   - No errors in console

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Kill existing app instance
pkill -f "python.*app.py"

# Or use a different port
GRADIO_SERVER_PORT=7861 python app.py
```

### Import Errors

```bash
# Check gradio version (must be >=4.44.0)
pip show gradio

# Check numpy version (must be <2.0)
pip show numpy

# Reinstall with correct versions
pip install --force-reinstall "gradio>=4.44.0" "numpy<2.0"
```

### Missing Data Files

If you see errors about missing CSV files:
1. Make sure you cloned the full repository
2. Run `code.ipynb` to regenerate data files
3. Check `.gitignore` - some large files aren't tracked

### Search Not Showing Results

Make sure you:
1. Updated to Gradio 4.44+ (older versions have bugs)
2. Refreshed your browser after restarting the app
3. Check console for debug output (should show "Search 'xxx' found N results")

---

## 📊 Understanding the Output

### When You Search for a Drug

**Example: Searching "Aspirin"**

You'll see:
1. **Basic Info**: ChEMBL ID, SMILES structure
2. **Known Targets**: Proteins with measured pChEMBL values
3. **Predicted Targets**: Novel protein interactions (cosine similarity >0.7)
4. **Clinical Effects**: Known indications and phases
5. **Predicted Effects**: Potential repurposing opportunities
6. **Network Graph**: Interactive visualization of relationships

### Interpreting Predictions

- **Similarity Score**: 0.0-1.0 (higher = more confident)
  - >0.8: Very high confidence
  - 0.7-0.8: High confidence
  - 0.6-0.7: Moderate confidence
  - <0.6: Low confidence (not shown)

- **Confidence**: "Very High", "High", "Medium" based on similarity threshold

---

## 🚢 Deploying to Hugging Face Spaces

See `DEPLOYMENT.md` for complete step-by-step guide.

**Quick version:**

1. Create a Space at https://huggingface.co/spaces
2. Upload these files:
   - `app.py`
   - All CSV files
   - `graph_embeddings.npy` and `node_to_idx.npy`
   - Rename `requirements_app.txt` → `requirements.txt`
   - Use `SPACE_README.md` content for README.md
3. Wait 2-3 minutes for build
4. Your app will be live!

---

## 🗑️ Unnecessary Files (Can Be Deleted)

**For end users who just want to run the app:**

You can safely delete:
- `BUGFIX_SUMMARY.md` (development notes)
- `APP_SUMMARY.md` (duplicate of APP_GUIDE.md)
- `CHECKLIST.md` (deployment checklist)
- `run_app.sh` (use `start_app.sh` instead)
- `test_search.py` (manual testing guide, not needed if app works)
- `requirements.txt` (only needed for notebook, use `requirements_app.txt`)

**If you're NOT retraining the model:**
- `code.ipynb` (the notebook)
- `chembl_36/` directory (large database)
- `full database.xml` (if present)

**Keep these minimal files:**
```
pharmacology-graph/
├── app.py
├── requirements_app.txt
├── start_app.sh
├── test_app.py
├── README.md
├── DEPLOYMENT.md
├── APP_GUIDE.md
├── *.csv (all CSV files)
├── graph_embeddings.npy
└── node_to_idx.npy
```

---

## 📝 File Size Reference

```
drug_nodes.csv                          ~700 KB
protein_nodes_with_embeddings.csv       ~400 KB
drug_effects.csv                        ~800 KB
drugs_interactions.csv                  ~1.5 MB
graph_embeddings.npy                    ~5 MB
node_to_idx.npy                         ~50 KB
top_50_predicted_*.csv                  ~10 KB each
```

**Total app files**: ~10 MB  
**With ChEMBL database**: ~4.2 GB  

---

## 🤝 Contributing

If you want to improve the project:

1. Fork the repository
2. Run the full notebook to understand the pipeline
3. Make changes (better predictions, UI improvements, etc.)
4. Test with `python test_app.py`
5. Submit a pull request

---

## 📚 Additional Resources

- **ChEMBL Database**: https://www.ebi.ac.uk/chembl/
- **ESM-2 Model**: https://github.com/facebookresearch/esm
- **TransE Paper**: "Translating Embeddings for Modeling Multi-relational Data" (Bordes et al., 2013)
- **Gradio Docs**: https://gradio.app/docs/

---

## ⚖️ License

This project uses publicly available ChEMBL data (CC BY-SA 3.0).  
Model and code are provided as-is for research purposes only.

**Not for clinical use.**

---

## 📧 Support

Issues? Questions?
- Open an issue: https://github.com/JoeVonDahab/pharmacology-graph/issues
- Check existing documentation: README.md, APP_GUIDE.md, DEPLOYMENT.md

---

**Last Updated**: October 10, 2025  
**Version**: 1.0.0  
**Python**: 3.9+  
**Gradio**: 4.44.1
