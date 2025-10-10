#!/bin/bash

# Test Gradio app locally before deploying to Hugging Face

echo "🚀 Setting up local Gradio app test..."

# Install requirements
echo "📦 Installing dependencies..."
pip install gradio plotly networkx

# Check if data files exist
echo "✅ Checking data files..."
required_files=(
    "drug_nodes.csv"
    "protein_nodes_with_embeddings.csv"
    "drug_effects.csv"
    "drugs_interactions.csv"
    "graph_embeddings.npy"
    "node_to_idx.npy"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    printf '   - %s\n' "${missing_files[@]}"
    echo ""
    echo "Please run the Jupyter notebook to generate these files first."
    exit 1
fi

echo "✅ All required files found!"
echo ""
echo "🎯 Starting Gradio app..."
echo "   App will be available at: http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo ""

# Run the app
python app.py
