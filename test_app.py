#!/usr/bin/env python3
"""
Quick test script to verify app.py data loading and functions work
Run this before launching the full Gradio app to catch errors early
"""

import sys

print("="*80)
print("TESTING APP.PY DATA LOADING AND FUNCTIONS")
print("="*80)

# Test 1: Import libraries
print("\n1. Testing library imports...")
try:
    import pandas as pd
    import numpy as np
    import networkx as nx
    print("   ✓ Core libraries imported")
except Exception as e:
    print(f"   ✗ Error importing libraries: {e}")
    sys.exit(1)

try:
    import gradio as gr
    print("   ✓ Gradio imported")
except Exception as e:
    print(f"   ✗ Error importing Gradio: {e}")
    print("   → Install with: uv pip install gradio")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    print("   ✓ Plotly imported")
except Exception as e:
    print(f"   ✗ Error importing Plotly: {e}")
    print("   → Install with: uv pip install plotly")
    sys.exit(1)

# Test 2: Load data files
print("\n2. Testing data file loading...")

required_files = {
    'drug_nodes.csv': None,
    'protein_nodes_with_embeddings.csv': None,
    'drug_effects.csv': None,
    'drugs_interactions.csv': None,
    'graph_embeddings.npy': None,
    'node_to_idx.npy': None
}

errors = []

try:
    drug_nodes = pd.read_csv('drug_nodes.csv')
    required_files['drug_nodes.csv'] = f"✓ {len(drug_nodes)} drugs"
    print(f"   ✓ drug_nodes.csv: {len(drug_nodes)} rows")
except Exception as e:
    errors.append(f"drug_nodes.csv: {e}")
    print(f"   ✗ drug_nodes.csv: {e}")

try:
    protein_nodes = pd.read_csv('protein_nodes_with_embeddings.csv')
    required_files['protein_nodes_with_embeddings.csv'] = f"✓ {len(protein_nodes)} proteins"
    print(f"   ✓ protein_nodes_with_embeddings.csv: {len(protein_nodes)} rows")
except Exception as e:
    errors.append(f"protein_nodes_with_embeddings.csv: {e}")
    print(f"   ✗ protein_nodes_with_embeddings.csv: {e}")

try:
    drug_effects = pd.read_csv('drug_effects.csv')
    required_files['drug_effects.csv'] = f"✓ {len(drug_effects)} effects"
    print(f"   ✓ drug_effects.csv: {len(drug_effects)} rows")
except Exception as e:
    errors.append(f"drug_effects.csv: {e}")
    print(f"   ✗ drug_effects.csv: {e}")

try:
    drugs_interactions = pd.read_csv('drugs_interactions.csv')
    required_files['drugs_interactions.csv'] = f"✓ {len(drugs_interactions)} interactions"
    print(f"   ✓ drugs_interactions.csv: {len(drugs_interactions)} rows")
except Exception as e:
    errors.append(f"drugs_interactions.csv: {e}")
    print(f"   ✗ drugs_interactions.csv: {e}")

try:
    embeddings = np.load('graph_embeddings.npy')
    required_files['graph_embeddings.npy'] = f"✓ {embeddings.shape}"
    print(f"   ✓ graph_embeddings.npy: shape {embeddings.shape}")
except Exception as e:
    errors.append(f"graph_embeddings.npy: {e}")
    print(f"   ✗ graph_embeddings.npy: {e}")

try:
    node_to_idx = np.load('node_to_idx.npy', allow_pickle=True).item()
    required_files['node_to_idx.npy'] = f"✓ {len(node_to_idx)} nodes"
    print(f"   ✓ node_to_idx.npy: {len(node_to_idx)} nodes")
except Exception as e:
    errors.append(f"node_to_idx.npy: {e}")
    print(f"   ✗ node_to_idx.npy: {e}")

# Test 3: Check data structure
if not errors:
    print("\n3. Testing data structure...")
    
    # Check drug_nodes columns
    required_cols = ['drug_internal_id', 'drug_id', 'drug_name', 'smile']
    missing_cols = [col for col in required_cols if col not in drug_nodes.columns]
    if missing_cols:
        print(f"   ✗ drug_nodes missing columns: {missing_cols}")
        errors.append(f"Missing columns in drug_nodes: {missing_cols}")
    else:
        print(f"   ✓ drug_nodes has all required columns")
    
    # Check protein_nodes columns
    required_cols = ['protein_internal_id', 'protein_id', 'protein_name']
    missing_cols = [col for col in required_cols if col not in protein_nodes.columns]
    if missing_cols:
        print(f"   ✗ protein_nodes missing columns: {missing_cols}")
        errors.append(f"Missing columns in protein_nodes: {missing_cols}")
    else:
        print(f"   ✓ protein_nodes has all required columns")
    
    # Test search function
    print("\n4. Testing search function...")
    try:
        # Search for common drug
        test_queries = ['Morphine', 'Aspirin', 'CHEMBL25']
        for query in test_queries:
            matches = drug_nodes[
                drug_nodes['drug_name'].str.lower().str.contains(query.lower(), na=False) |
                drug_nodes['drug_id'].str.lower().str.contains(query.lower(), na=False)
            ]
            if len(matches) > 0:
                print(f"   ✓ Search '{query}': found {len(matches)} results")
            else:
                print(f"   ⚠ Search '{query}': no results (might be OK)")
    except Exception as e:
        print(f"   ✗ Search function error: {e}")
        errors.append(f"Search error: {e}")

# Summary
print("\n" + "="*80)
if errors:
    print("❌ TESTING FAILED")
    print("\nErrors found:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\nPlease fix these errors before running the app.")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED!")
    print("\nYour app should work correctly. Run it with:")
    print("  ./run_app.sh")
    print("  or")
    print("  python app.py")
    sys.exit(0)
