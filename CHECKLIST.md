# 📋 Deployment Checklist

Use this checklist to ensure smooth deployment to Hugging Face Spaces.

## ✅ Pre-Deployment (Local Testing)

- [ ] **Install dependencies**
  ```bash
  uv pip install gradio plotly networkx
  ```

- [ ] **Verify data files exist**
  ```bash
  ls drug_nodes.csv
  ls protein_nodes_with_embeddings.csv
  ls drug_effects.csv
  ls drugs_interactions.csv
  ls graph_embeddings.npy
  ls node_to_idx.npy
  ```

- [ ] **Test app locally**
  ```bash
  ./run_app.sh
  # OR
  python app.py
  ```

- [ ] **Open browser** to `http://localhost:7860`

- [ ] **Test functionality:**
  - [ ] Search works (try "Morphine")
  - [ ] Dropdown populates
  - [ ] Drug info displays
  - [ ] Known targets table shows
  - [ ] Known effects table shows
  - [ ] Predicted targets table shows
  - [ ] Predicted effects table shows
  - [ ] Network visualization renders
  - [ ] Can zoom/pan network
  - [ ] Hover shows node details

- [ ] **Try multiple drugs:**
  - [ ] Aspirin
  - [ ] Metformin
  - [ ] Talazoparib
  - [ ] Ibuprofen
  - [ ] Rivaroxaban

- [ ] **Test edge cases:**
  - [ ] Drug with no known targets
  - [ ] Drug with many interactions
  - [ ] Toggle known/predicted checkboxes
  - [ ] Adjust max nodes slider

- [ ] **Take screenshots** for presentation

---

## 🌐 Hugging Face Account Setup

- [ ] **Create account** at [huggingface.co](https://huggingface.co)
- [ ] **Verify email**
- [ ] **Optional:** Add profile picture and bio
- [ ] **Optional:** Connect GitHub account

---

## 🚀 Hugging Face Spaces Creation

- [ ] **Go to** [huggingface.co/spaces](https://huggingface.co/spaces)
- [ ] **Click** "Create new Space"
- [ ] **Fill in details:**
  - Space name: `pharmacology-knowledge-graph`
  - License: `mit`
  - Select Space SDK: `Gradio`
  - Hardware: `CPU basic (free)`
  - Visibility: `Public`

- [ ] **Click** "Create Space"

---

## 📤 File Upload

### Option A: Web Upload (Easiest)

- [ ] **Go to Files tab** in your Space
- [ ] **Upload each file:**

  **Code files:**
  - [ ] `app.py` (upload as-is)
  - [ ] `requirements_app.txt` → **RENAME to `requirements.txt` before upload**

  **Data files (CSV):**
  - [ ] `drug_nodes.csv`
  - [ ] `protein_nodes_with_embeddings.csv`
  - [ ] `drug_effects.csv`
  - [ ] `drugs_interactions.csv`

  **Model files (NPY):**
  - [ ] `graph_embeddings.npy`
  - [ ] `node_to_idx.npy`

  **Optional (pre-computed predictions):**
  - [ ] `top_50_predicted_drug_protein.csv`
  - [ ] `top_50_predicted_drug_effects.csv`

- [ ] **Add Space README:**
  - [ ] Click "Edit README"
  - [ ] Copy content from `SPACE_README.md`
  - [ ] Paste and save

### Option B: Git Upload (Advanced)

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph
cd pharmacology-knowledge-graph

# Copy files
cp /path/to/app.py .
cp /path/to/requirements_app.txt requirements.txt
cp /path/to/*.csv .
cp /path/to/*.npy .

# Add and commit
git add .
git commit -m "Initial deployment"
git push
```

- [ ] **Clone Space repository**
- [ ] **Copy all required files**
- [ ] **Rename requirements file**
- [ ] **Commit and push**

---

## ⏳ Build Monitoring

- [ ] **Wait for build** (usually 2-3 minutes)
- [ ] **Monitor build logs** for errors
- [ ] **Check for these messages:**
  - `Successfully installed gradio-4.44.0`
  - `Successfully installed plotly-6.3.1`
  - `Running on local URL: http://0.0.0.0:7860`
  - `Running on public URL: https://xxx.gradio.live`

### Common Build Errors

**"File not found"**
- [ ] Verify all CSV/NPY files uploaded
- [ ] Check file names match exactly (case-sensitive)

**"Module not found"**
- [ ] Check `requirements.txt` uploaded (not `requirements_app.txt`)
- [ ] Verify gradio version specified

**"Out of memory"**
- [ ] Try upgrading to CPU Upgrade ($0.03/hr)
- [ ] Or reduce dataset size temporarily

---

## ✅ Post-Deployment Testing

- [ ] **App is running** (green "Running" badge)
- [ ] **Open the app** (click the embedded iframe or public link)
- [ ] **Test all features again:**
  - [ ] Search works
  - [ ] Predictions load
  - [ ] Network visualizes
  - [ ] No errors in console

- [ ] **Test from different device** (phone, tablet)
- [ ] **Test in incognito/private mode**

---

## 📢 Sharing & Promotion

- [ ] **Copy public URL:**
  `https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph`

- [ ] **Add to competition submission**

- [ ] **Share on social media:**
  - [ ] Twitter/X
  - [ ] LinkedIn
  - [ ] Reddit (r/MachineLearning, r/bioinformatics)

- [ ] **Add to GitHub README:**
  ```markdown
  ## 🌐 Live Demo
  
  Try the interactive app: [Pharmacology Knowledge Graph on HF Spaces](https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph)
  ```

- [ ] **Update your portfolio/CV**

---

## 🎬 Demo Preparation

- [ ] **Prepare 2-minute walkthrough**
- [ ] **Have 3-5 example drugs ready:**
  1. Morphine (opioid receptors)
  2. Talazoparib (PARP predictions - novel!)
  3. Rivaroxaban (anticoagulant repurposing)
  4. Metformin (diabetes + aging)
  5. Your choice

- [ ] **Key talking points:**
  - [ ] 800 drugs, 200 proteins, 400 effects
  - [ ] TransE graph embeddings
  - [ ] 90% prediction accuracy
  - [ ] Drug repurposing applications
  - [ ] Interactive visualization

- [ ] **Practice demo** 3x

---

## 🔧 Optional Enhancements

- [ ] **Add Google Analytics** (track usage)
- [ ] **Enable Space caching** (faster loads)
- [ ] **Add "Featured" badge** (if eligible)
- [ ] **Create demo video** (Loom/OBS)
- [ ] **Write blog post** about the project

---

## 📊 Monitoring & Maintenance

- [ ] **Check Space analytics** weekly
- [ ] **Monitor for errors** in community tab
- [ ] **Respond to comments/questions**
- [ ] **Update data** monthly (optional)
- [ ] **Add new features** based on feedback

---

## 🎓 Competition Submission

- [ ] **Include HF Spaces link** in submission
- [ ] **Add screenshots** of the app
- [ ] **Mention interactivity** as key feature
- [ ] **Highlight validation** (90% accuracy)
- [ ] **Explain impact** (drug repurposing, cost savings)

---

## ✨ Success Criteria

You'll know it's working when:

✅ Public URL loads without errors  
✅ Can search and select drugs  
✅ Predictions populate tables  
✅ Network graph renders beautifully  
✅ Friends/colleagues can use it without help  
✅ Judges say "Wow, this is impressive!"  

---

## 🆘 Troubleshooting Resources

**If something goes wrong:**

1. **Check build logs** in HF Spaces
2. **Test locally first** to isolate issues
3. **Review error messages** carefully
4. **Common fixes:**
   - Clear cache and rebuild
   - Check file paths (must be in root directory)
   - Verify requirements.txt has correct package versions
   - Try CPU upgrade if memory issues

5. **Get help:**
   - HF Spaces documentation: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
   - Gradio docs: [gradio.app/docs](https://gradio.app/docs)
   - Community forum: [discuss.huggingface.co](https://discuss.huggingface.co)

---

## 🎉 You're Ready!

Once you've checked off all items above, you'll have:

✅ Fully functional local app  
✅ Deployed public app on HF Spaces  
✅ Professional demo ready  
✅ Shareable link for promotion  
✅ Competition-winning project  

**Now go deploy it and make an impact! 🚀**

---

**Last updated:** October 2025  
**Estimated time:** 30-60 minutes total  
**Difficulty:** ⭐⭐☆☆☆ (Easy-Medium)
