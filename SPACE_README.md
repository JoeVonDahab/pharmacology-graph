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

**Explore drug-target-effect relationships using AI-powered predictions**

## 🎯 What is this?

An interactive web app that lets you explore:
- **Known interactions** between drugs and protein targets
- **Predicted novel targets** for drug repurposing
- **Clinical effects** and therapeutic indications
- **Interactive network visualizations** of drug mechanisms

## 🧬 How It Works

This app uses a **TransE knowledge graph embedding model** trained on:
- **800+ FDA-approved drugs** from ChEMBL database
- **200+ human protein targets** with ESM-2 sequence embeddings
- **400+ clinical effects** and disease indications

The model learns a unified embedding space where:
- Drugs with similar mechanisms cluster together
- Predictions are based on cosine similarity in latent space
- Novel drug-target pairs are ranked by confidence

## 🔍 Example Queries

Try searching for these drugs to see interesting results:

| Drug | What you'll see |
|------|----------------|
| **Morphine** | Opioid receptor targets, analgesic effects, similar opioids (Fentanyl, Oxycodone) |
| **Aspirin** | COX enzyme inhibition, anti-inflammatory effects, cardiovascular prevention |
| **Metformin** | Diabetes targets, metabolic effects, potential repurposing for cancer/aging |
| **Ibuprofen** | COX-1/COX-2 targets, NSAID effects, similar drugs (Naproxen, Diclofenac) |
| **Talazoparib** | PARP family targets, cancer indications, predicted PARP homologs |

## 📊 Model Performance

**Validation metrics:**
- **Precision@50:** ~90% pharmacologically coherent predictions
- **Mean similarity (top predictions):** 0.48-0.60
- **Baseline (random):** 0.23

**Key achievements:**
- ✅ Correctly clusters drugs by therapeutic class (e.g., opioids, antivirals, statins)
- ✅ Recovers known drug-target relationships with high accuracy
- ✅ Predicts plausible novel targets for experimental validation

## 🕸️ Network Visualization Features

The interactive graph shows:
- 🔴 **Central drug node** (red)
- 🔵 **Known protein targets** (blue, solid edges)
- 🟢 **Predicted novel targets** (green, dashed edges)
- 🟡 **Known clinical effects** (yellow, solid edges)
- 🟨 **Predicted effects** (light yellow, dashed edges)

**Controls:**
- Toggle known/predicted interactions
- Adjust max nodes displayed
- Hover over nodes for details

## 🚀 Use Cases

1. **Drug Repurposing:** Find new therapeutic uses for existing drugs
2. **Target Discovery:** Identify potential protein targets for lead compounds
3. **Safety Assessment:** Predict off-target effects and side effects
4. **Mechanism Exploration:** Understand drug action through network context

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{pharmacology_graph_2025,
  author = {Joe VonDahab},
  title = {Pharmacology Knowledge Graph: Drug-Target-Effect Prediction},
  year = {2025},
  url = {https://github.com/JoeVonDahab/pharmacology-graph},
  note = {Hugging Face Space: https://huggingface.co/spaces/YOUR_USERNAME/pharmacology-knowledge-graph}
}
```

## 🔗 Links

- **GitHub Repository:** [pharmacology-graph](https://github.com/JoeVonDahab/pharmacology-graph)
- **Paper/Blog:** *(Coming soon)*
- **Dataset:** [ChEMBL 36](https://www.ebi.ac.uk/chembl/)
- **Protein Model:** [ESM-2 (Meta AI)](https://github.com/facebookresearch/esm)

## ⚠️ Disclaimer

**This is a research tool for exploratory analysis only.**

- Predictions are computational and require experimental validation
- Not intended for clinical decision-making or medical diagnosis
- Drug repurposing candidates need rigorous testing before clinical use
- Always consult scientific literature and regulatory databases

## 🛠️ Technical Details

**Model Architecture:**
- **Graph embedding:** TransE (Translation-based)
- **Embedding dimension:** 128
- **Protein featurization:** ESM-2 (2560-dim) → mean pooling
- **Drug featurization:** Morgan fingerprints (2048-bit)

**Training:**
- Epochs: 100
- Learning rate: 0.01
- Negative sampling: 1:1 ratio
- Margin loss: 1.0

**Data sources:**
- ChEMBL 36 (drugs, targets, indications)
- UniProt (protein sequences)
- MeSH (clinical effects taxonomy)

## 💡 Future Enhancements

Planned features:
- [ ] SMILES input for novel molecules
- [ ] 3D protein structure integration (AlphaFold2)
- [ ] Multi-task learning (toxicity + efficacy)
- [ ] Batch analysis and export
- [ ] API endpoint for programmatic access

## 🤝 Contributing

Found a bug or have a feature request? 
- Open an issue on [GitHub](https://github.com/JoeVonDahab/pharmacology-graph/issues)
- Submit a pull request
- Contact: [create an issue](https://github.com/JoeVonDahab/pharmacology-graph/issues)

## 📜 License

MIT License - See [LICENSE](https://github.com/JoeVonDahab/pharmacology-graph/blob/main/LICENSE) for details

---

**Built with:** 🤗 Gradio • PyTorch • NetworkX • Plotly • ESM-2

**Powered by:** ChEMBL • Hugging Face Spaces

*Last updated: October 2025*
