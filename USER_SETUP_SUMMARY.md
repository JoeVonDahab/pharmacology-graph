# 📋 Summary: Setup for New Users

## What Someone Needs to Do to Run Your App

### **Method 1: Just Run the App (Recommended for Most Users)**

```bash
# 1. Clone repository
git clone https://github.com/JoeVonDahab/pharmacology-graph.git
cd pharmacology-graph

# 2. Install dependencies
pip install -r requirements_app.txt

# 3. Download the .npy files (NOT in git)
# They need to either:
#   a) Run the notebook to generate them, OR
#   b) Download from a release/separate link you provide

# 4. Run the app
./start_app.sh
# Open browser to http://localhost:7860
```

**Requirements:**
- Python 3.9+
- 8GB RAM
- ~10MB disk space

**What they get:**
- ✅ Interactive web app
- ✅ Search 3,127 drugs
- ✅ View predictions
- ✅ Network visualization

---

### **Method 2: Full Research Pipeline (Advanced Users)**

```bash
# 1-2. Same as above

# 3. Download ChEMBL database (4.2GB)
# Already included in repo if you pushed it, otherwise:
# wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_sqlite.tar.gz

# 4. Install full dependencies
pip install -r requirements.txt

# 5. Run the notebook
jupyter notebook code.ipynb
# Run all cells (~40 minutes)

# 6. Run the app
./start_app.sh
```

**Requirements:**
- Python 3.9+
- 16GB RAM (for ESM-2)
- GPU recommended (or 30+ min on CPU)
- ~5GB disk space

**What they get:**
- ✅ Everything from Method 1
- ✅ Ability to retrain model
- ✅ Explore data processing
- ✅ Modify predictions

---

## 🗑️ Files You Can Remove Before Sharing

Run this to clean up unnecessary files:

```bash
./cleanup.sh
```

Or manually remove:

### Definitely Remove (Duplicates/Debugging)
```bash
rm -f BUGFIX_SUMMARY.md      # Development notes
rm -f APP_SUMMARY.md          # Duplicate of APP_GUIDE.md
rm -f CHECKLIST.md            # One-time deployment checklist
rm -f run_app.sh              # Old script (use start_app.sh)
rm -f test_search.py          # Manual testing guide
rm -f main.py                 # Old experimental code
rm -f .python-version         # Unnecessary version pin
```

### Optional Remove (Can Regenerate)
```bash
rm -f *.png                   # Visualization outputs
rm -f *.pkl                   # Pickle files
rm -f drug_mechanism_filtered.csv
rm -f drug_warnings.csv
rm -f edges_drug_protein.csv
rm -f proteins_for_embedding.csv
rm -f top_50_predicted_drug_targets.csv  # Duplicate
```

### Consider Removing (If Users Don't Need to Retrain)
```bash
rm -f code.ipynb              # Research notebook (750KB)
rm -f requirements.txt        # Full deps (use requirements_app.txt only)
rm -rf chembl_36/             # Database (4.2GB!)
```

---

## 📦 Minimal File Set for App-Only Distribution

After cleanup, you'd have:

```
pharmacology-graph/
├── app.py                                    # Main app
├── requirements_app.txt                      # Dependencies
├── start_app.sh                              # Launcher
├── test_app.py                               # Diagnostic
├── README.md                                 # Overview
├── SETUP.md                                  # Setup guide ⭐
├── FILES.md                                  # This file list
├── DEPLOYMENT.md                             # HF Spaces guide
├── APP_GUIDE.md                              # User guide
├── SPACE_README.md                           # For HF deployment
├── drug_nodes.csv                            # Data
├── protein_nodes_with_embeddings.csv         # Data
├── drug_effects.csv                          # Data
├── drugs_interactions.csv                    # Data
├── top_50_predicted_drug_protein.csv         # Predictions
├── top_50_predicted_drug_effects.csv         # Predictions
├── graph_embeddings.npy                      # Model ⚠️ NOT IN GIT
├── node_to_idx.npy                           # Mapping ⚠️ NOT IN GIT
└── .gitignore                                # Git config
```

**Total size**: ~10 MB (without .npy files)

---

## ⚠️ IMPORTANT: The .npy Files Problem

### The Issue
The `.npy` files (`graph_embeddings.npy` and `node_to_idx.npy`) are:
- **Essential** for the app to run
- **NOT tracked in git** (they're in `.gitignore`)
- **~5MB total** (not huge, but binary)

### Solutions

**Option 1: Add them to git** (Simplest)
```bash
# Edit .gitignore to allow these specific files
echo '!graph_embeddings.npy' >> .gitignore
echo '!node_to_idx.npy' >> .gitignore

# Add and push
git add graph_embeddings.npy node_to_idx.npy
git commit -m "Add model embeddings"
git push
```

**Option 2: GitHub Release** (Cleaner)
1. Go to GitHub → Releases → Create new release
2. Upload `graph_embeddings.npy` and `node_to_idx.npy`
3. Update SETUP.md with download instructions:
```bash
# Download embeddings
wget https://github.com/YOUR_USERNAME/pharmacology-graph/releases/download/v1.0/graph_embeddings.npy
wget https://github.com/YOUR_USERNAME/pharmacology-graph/releases/download/v1.0/node_to_idx.npy
```

**Option 3: Git LFS** (Professional)
```bash
# Install Git LFS
git lfs install

# Track .npy files
git lfs track "*.npy"
git add .gitattributes

# Add and push
git add graph_embeddings.npy node_to_idx.npy
git commit -m "Add model embeddings via LFS"
git push
```

**Option 4: External Storage** (If files are huge)
- Upload to Hugging Face Hub, Google Drive, Dropbox
- Provide download link in SETUP.md

### Recommendation
For 5MB files, **Option 1** (just add to git) is fine. GitHub allows files up to 100MB.

---

## 📝 Updated SETUP.md Instructions

I've created **`SETUP.md`** with complete instructions including:

✅ **Quick Start** (3 commands)  
✅ **Full Setup** (retrain from scratch)  
✅ **Troubleshooting** (version issues, port conflicts)  
✅ **Testing** (verify it works)  
✅ **Deployment** (Hugging Face Spaces)  
✅ **File cleanup** (remove unnecessary files)

**For new users, tell them:**
> "Read **SETUP.md** for complete installation and usage instructions."

---

## 🎯 Recommended Git Workflow

### Before sharing on GitHub:

1. **Clean up files**:
   ```bash
   ./cleanup.sh
   ```

2. **Add .npy files to git** (if you choose Option 1):
   ```bash
   echo '!graph_embeddings.npy' >> .gitignore
   echo '!node_to_idx.npy' >> .gitignore
   git add graph_embeddings.npy node_to_idx.npy .gitignore
   git commit -m "Add model embeddings for app"
   ```

3. **Update README**:
   - Already updated with Quick Start section
   - Points to SETUP.md for details

4. **Push everything**:
   ```bash
   git add .
   git commit -m "Add setup documentation and cleanup scripts"
   git push
   ```

### User experience:
```bash
git clone https://github.com/YOUR_USERNAME/pharmacology-graph
cd pharmacology-graph
pip install -r requirements_app.txt
./start_app.sh
```

**That's it!** 3 commands and they're running.

---

## 📊 File Organization Summary

| Category | Files | Purpose | Size | In Git? |
|----------|-------|---------|------|---------|
| **App Core** | app.py, start_app.sh | Run the interface | 50KB | ✅ Yes |
| **Data** | 6 CSV files | Drug/protein/effect data | 3.5MB | ✅ Yes |
| **Model** | 2 NPY files | Trained embeddings | 5MB | ⚠️ **No** (need to add) |
| **Predictions** | 2 CSV files | Top 50 predictions | 20KB | ✅ Yes |
| **Docs** | 7 MD files | Setup, deployment, usage | 50KB | ✅ Yes |
| **Research** | code.ipynb | Full pipeline | 500KB | ✅ Yes |
| **Database** | chembl_36/ | ChEMBL SQLite | 4.2GB | ❌ No (too large) |
| **Cleanup** | 10+ files | Duplicates, old versions | 2MB | ✅ Yes (but can remove) |

**Action Items:**
1. ✅ Created SETUP.md (complete guide)
2. ✅ Created FILES.md (file manifest)
3. ✅ Created cleanup.sh (remove junk)
4. ✅ Updated README.md (quick start)
5. ⚠️ **TODO**: Decide how to distribute .npy files (Option 1-4 above)

---

## 🎓 Bottom Line

**For someone to download and run your app, they need:**

### Minimum (App Only):
1. Clone repo
2. `pip install -r requirements_app.txt`
3. Download/generate `.npy` files
4. `./start_app.sh`

### Full (Research):
1. Clone repo
2. `pip install -r requirements.txt`
3. Download ChEMBL (or use included)
4. Run notebook → generates everything
5. `./start_app.sh`

**Unnecessary files removed**: ~15 files, saving 2MB and reducing clutter

**New documentation**: 
- ✅ SETUP.md (start here!)
- ✅ FILES.md (what each file does)
- ✅ cleanup.sh (automated cleanup)

Users should read **SETUP.md** first. It has everything they need.
