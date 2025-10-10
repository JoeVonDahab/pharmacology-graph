import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load data
print("Loading data...")
drug_nodes = pd.read_csv('drug_nodes.csv')
protein_nodes = pd.read_csv('protein_nodes_with_embeddings.csv')
drug_effects = pd.read_csv('drug_effects.csv')
drugs_interactions = pd.read_csv('drugs_interactions.csv')

# Load predictions
predicted_targets = pd.read_csv('top_50_predicted_drug_protein.csv')
predicted_effects = pd.read_csv('top_50_predicted_drug_effects.csv')

# Load embeddings
embeddings = np.load('graph_embeddings.npy')
node_to_idx = np.load('node_to_idx.npy', allow_pickle=True).item()

print(f"✓ Loaded {len(drug_nodes)} drugs, {len(protein_nodes)} proteins, {len(drug_effects)} effects")

# Create reverse mappings
internal_to_drug_info = dict(zip(drug_nodes['drug_internal_id'], 
                                 zip(drug_nodes['drug_id'], drug_nodes['drug_name'])))
protein_id_to_name = dict(zip(protein_nodes['protein_id'], protein_nodes['protein_name']))
effect_id_to_name = dict(zip(drug_effects['effect_id'], drug_effects['effect_name']))

def search_drugs(query):
    """Search for drugs by name or ChEMBL ID"""
    try:
        if not query:
            return []
        
        query = query.lower()
        matches = drug_nodes[
            drug_nodes['drug_name'].str.lower().str.contains(query, na=False) |
            drug_nodes['drug_id'].str.lower().str.contains(query, na=False)
        ]
        
        # Return list of tuples (display_name, drug_internal_id)
        results = [(f"{row['drug_name']} ({row['drug_id']})", row['drug_internal_id']) 
                   for _, row in matches.head(20).iterrows()]
        
        return results
    except Exception as e:
        print(f"Error in search_drugs: {e}")
        return []

def get_drug_info(drug_internal_id):
    """Get basic drug information"""
    if drug_internal_id is None:
        return "No drug selected"
    
    drug_row = drug_nodes[drug_nodes['drug_internal_id'] == drug_internal_id]
    if len(drug_row) == 0:
        return "Drug not found"
    
    drug_row = drug_row.iloc[0]
    
    info = f"""
## 💊 {drug_row['drug_name']}

**ChEMBL ID:** {drug_row['drug_id']}  
**SMILES:** `{drug_row['smile'][:100]}...`
"""
    return info

def get_known_targets(drug_internal_id):
    """Get known protein targets for a drug"""
    if drug_internal_id is None:
        return pd.DataFrame()
    
    # Get ChEMBL ID
    drug_row = drug_nodes[drug_nodes['drug_internal_id'] == drug_internal_id]
    if len(drug_row) == 0:
        return pd.DataFrame()
    
    drug_chembl_id = drug_row.iloc[0]['drug_id']
    
    # Get known interactions
    known = drugs_interactions[drugs_interactions['drug_id'] == drug_chembl_id].copy()
    
    if len(known) == 0:
        return pd.DataFrame(columns=['Target Protein', 'pChEMBL (max)', 'Measurements'])
    
    # Format for display
    display_df = pd.DataFrame({
        'Target Protein': known['protein_name'].values,
        'ChEMBL ID': known['protein_id'].values,
        'pChEMBL (max)': known['pchembl_max'].values,
        'pChEMBL (avg)': known['pchembl_avg'].values,
        'Measurements': known['num_measurements'].values
    })
    
    return display_df.sort_values('pChEMBL (max)', ascending=False)

def get_known_effects(drug_internal_id):
    """Get known clinical effects for a drug"""
    if drug_internal_id is None:
        return pd.DataFrame()
    
    # Get known effects
    known = drug_effects[drug_effects['drug_internal_id'] == drug_internal_id].copy()
    
    if len(known) == 0:
        return pd.DataFrame(columns=['Clinical Effect', 'Phase', 'References'])
    
    # Format for display
    display_df = pd.DataFrame({
        'Clinical Effect': known['effect_name'].values,
        'MeSH ID': known['effect_id'].values,
        'Phase': known['indication_phase'].values,
        'References': known['num_references'].values
    })
    
    return display_df.sort_values('Phase', ascending=False)

def get_predicted_targets(drug_internal_id, top_k=10):
    """Get predicted novel protein targets"""
    if drug_internal_id is None:
        return pd.DataFrame()
    
    # Get ChEMBL ID
    drug_row = drug_nodes[drug_nodes['drug_internal_id'] == drug_internal_id]
    if len(drug_row) == 0:
        return pd.DataFrame()
    
    drug_chembl_id = drug_row.iloc[0]['drug_id']
    
    # Get known targets to filter out
    known_targets = set(drugs_interactions[
        drugs_interactions['drug_id'] == drug_chembl_id
    ]['protein_id'].values)
    
    # Check if drug is in embedding space
    if drug_internal_id not in node_to_idx:
        return pd.DataFrame(columns=['Predicted Target', 'Similarity', 'Confidence'])
    
    # Get drug embedding
    drug_idx = node_to_idx[drug_internal_id]
    drug_emb = embeddings[drug_idx].reshape(1, -1)
    
    # Compute similarities to all proteins
    predictions = []
    for _, protein_row in protein_nodes.iterrows():
        protein_id = protein_row['protein_id']
        protein_name = protein_row['protein_name']
        
        # Skip known targets
        if protein_id in known_targets:
            continue
        
        # Get protein embedding
        if protein_id not in node_to_idx:
            continue
        
        protein_idx = node_to_idx[protein_id]
        protein_emb = embeddings[protein_idx].reshape(1, -1)
        
        # Calculate similarity
        sim = cosine_similarity(drug_emb, protein_emb)[0][0]
        
        predictions.append({
            'Predicted Target': protein_name,
            'ChEMBL ID': protein_id,
            'Similarity': f"{sim:.4f}",
            'Confidence': 'High' if sim > 0.5 else 'Medium' if sim > 0.45 else 'Low'
        })
    
    # Sort and return top-k
    predictions_df = pd.DataFrame(predictions)
    if len(predictions_df) == 0:
        return pd.DataFrame(columns=['Predicted Target', 'Similarity', 'Confidence'])
    
    predictions_df['Similarity_Float'] = predictions_df['Similarity'].astype(float)
    predictions_df = predictions_df.sort_values('Similarity_Float', ascending=False).head(top_k)
    
    return predictions_df[['Predicted Target', 'ChEMBL ID', 'Similarity', 'Confidence']]

def get_predicted_effects(drug_internal_id, top_k=10):
    """Get predicted novel clinical effects"""
    if drug_internal_id is None:
        return pd.DataFrame()
    
    # Get known effects to filter out
    known_effects = set(drug_effects[
        drug_effects['drug_internal_id'] == drug_internal_id
    ]['effect_id'].values)
    
    # Check if drug is in embedding space
    if drug_internal_id not in node_to_idx:
        return pd.DataFrame(columns=['Predicted Effect', 'Similarity', 'Confidence'])
    
    # Get drug embedding
    drug_idx = node_to_idx[drug_internal_id]
    drug_emb = embeddings[drug_idx].reshape(1, -1)
    
    # Compute similarities to all effects
    predictions = []
    unique_effects = drug_effects[['effect_id', 'effect_name']].drop_duplicates()
    
    for _, effect_row in unique_effects.iterrows():
        effect_id = effect_row['effect_id']
        effect_name = effect_row['effect_name']
        
        # Skip known effects
        if effect_id in known_effects:
            continue
        
        # Get effect embedding
        if effect_id not in node_to_idx:
            continue
        
        effect_idx = node_to_idx[effect_id]
        effect_emb = embeddings[effect_idx].reshape(1, -1)
        
        # Calculate similarity
        sim = cosine_similarity(drug_emb, effect_emb)[0][0]
        
        predictions.append({
            'Predicted Effect': effect_name,
            'MeSH ID': effect_id,
            'Similarity': f"{sim:.4f}",
            'Confidence': 'High' if sim > 0.5 else 'Medium' if sim > 0.45 else 'Low'
        })
    
    # Sort and return top-k
    predictions_df = pd.DataFrame(predictions)
    if len(predictions_df) == 0:
        return pd.DataFrame(columns=['Predicted Effect', 'Similarity', 'Confidence'])
    
    predictions_df['Similarity_Float'] = predictions_df['Similarity'].astype(float)
    predictions_df = predictions_df.sort_values('Similarity_Float', ascending=False).head(top_k)
    
    return predictions_df[['Predicted Effect', 'MeSH ID', 'Similarity', 'Confidence']]

def create_network_visualization(drug_internal_id, show_known=True, show_predicted=True, max_nodes=20):
    """Create interactive network visualization using Plotly"""
    if drug_internal_id is None:
        return None
    
    # Get drug info
    drug_row = drug_nodes[drug_nodes['drug_internal_id'] == drug_internal_id]
    if len(drug_row) == 0:
        return None
    
    drug_name = drug_row.iloc[0]['drug_name']
    drug_chembl_id = drug_row.iloc[0]['drug_id']
    
    # Create network
    G = nx.Graph()
    
    # Add central drug node
    G.add_node(drug_name, node_type='drug', color='#FF6B6B', size=30)
    
    edge_traces = []
    node_texts = []
    
    # Add known targets
    if show_known:
        known_targets_df = get_known_targets(drug_internal_id)
        for idx, row in known_targets_df.head(max_nodes // 2).iterrows():
            target_name = row['Target Protein'][:30]
            G.add_node(target_name, node_type='protein_known', color='#4ECDC4', size=15)
            G.add_edge(drug_name, target_name, edge_type='known', color='#95E1D3', width=3)
    
    # Add predicted targets
    if show_predicted:
        predicted_targets_df = get_predicted_targets(drug_internal_id, top_k=max_nodes // 2)
        for idx, row in predicted_targets_df.iterrows():
            target_name = row['Predicted Target'][:30]
            similarity = float(row['Similarity'])
            G.add_node(target_name, node_type='protein_predicted', color='#A8E6CF', size=10)
            G.add_edge(drug_name, target_name, edge_type='predicted', 
                      color='#DCEDC8', width=1.5, similarity=similarity)
    
    # Add known effects
    if show_known:
        known_effects_df = get_known_effects(drug_internal_id)
        for idx, row in known_effects_df.head(max_nodes // 3).iterrows():
            effect_name = row['Clinical Effect'][:30]
            G.add_node(effect_name, node_type='effect_known', color='#FFD93D', size=15)
            G.add_edge(drug_name, effect_name, edge_type='known_effect', color='#FCF4A3', width=3)
    
    # Add predicted effects
    if show_predicted:
        predicted_effects_df = get_predicted_effects(drug_internal_id, top_k=max_nodes // 3)
        for idx, row in predicted_effects_df.iterrows():
            effect_name = row['Predicted Effect'][:30]
            similarity = float(row['Similarity'])
            G.add_node(effect_name, node_type='effect_predicted', color='#FFF9A3', size=10)
            G.add_edge(drug_name, effect_name, edge_type='predicted_effect', 
                      color='#FFFACD', width=1.5, similarity=similarity)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge[2].get('width', 1),
                color=edge[2].get('color', '#888')
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_color.append(node[1].get('color', '#888'))
        node_size.append(node[1].get('size', 10))
        
        # Create hover text
        node_type = node[1].get('node_type', 'unknown')
        if node_type == 'drug':
            text = f"<b>{node[0]}</b><br>Type: Drug<br>ChEMBL: {drug_chembl_id}"
        elif 'protein' in node_type:
            text = f"<b>{node[0]}</b><br>Type: Protein Target<br>{'Known' if 'known' in node_type else 'Predicted'}"
        elif 'effect' in node_type:
            text = f"<b>{node[0]}</b><br>Type: Clinical Effect<br>{'Known' if 'known' in node_type else 'Predicted'}"
        else:
            text = f"<b>{node[0]}</b>"
        
        node_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node[0] for node in G.nodes()],
        textposition='top center',
        textfont=dict(size=8),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=dict(
            text=f"Knowledge Graph for {drug_name}",
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240,240,240,0.9)',
        height=700
    )
    
    # Add legend manually
    fig.add_annotation(
        text="<b>Legend:</b><br>" +
             "🔴 Drug | 🔵 Known Target | 🟢 Predicted Target<br>" +
             "🟡 Known Effect | 🟨 Predicted Effect",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def analyze_drug(search_query, drug_selection, show_known, show_predicted, max_nodes):
    """Main analysis function"""
    try:
        if drug_selection is None:
            return (
                "Please search and select a drug from the dropdown",
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None
            )
        
        # Extract drug_internal_id from selection
        drug_internal_id = drug_selection
        
        # Get all data with error handling
        info = get_drug_info(drug_internal_id)
        known_targets = get_known_targets(drug_internal_id)
        known_effects = get_known_effects(drug_internal_id)
        pred_targets = get_predicted_targets(drug_internal_id, top_k=15)
        pred_effects = get_predicted_effects(drug_internal_id, top_k=15)
        network = create_network_visualization(drug_internal_id, show_known, show_predicted, max_nodes)
        
        return info, known_targets, known_effects, pred_targets, pred_effects, network
    
    except Exception as e:
        error_msg = f"⚠️ Error analyzing drug: {str(e)}\n\nPlease try selecting a different drug or check the console for details."
        print(f"Error in analyze_drug: {e}")
        import traceback
        traceback.print_exc()
        
        return (
            error_msg,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None
        )

# Create Gradio interface
with gr.Blocks(title="Pharmacology Knowledge Graph Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 💊 Pharmacology Knowledge Graph Explorer
        
        Explore drug-target-effect relationships using AI-powered predictions from a TransE knowledge graph model.
        
        **Features:**
        - 🔍 Search 800+ approved drugs
        - 🎯 View known and predicted protein targets
        - 💉 Discover potential therapeutic uses (drug repurposing)
        - 🕸️ Interactive network visualization
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Drug Search")
            search_box = gr.Textbox(
                label="Search by drug name or ChEMBL ID",
                placeholder="e.g., Aspirin, Morphine, CHEMBL25",
                info="Type to search, then select from dropdown"
            )
            
            drug_dropdown = gr.Dropdown(
                label="Select Drug",
                choices=[],
                interactive=True
            )
            
            gr.Markdown("### ⚙️ Visualization Settings")
            show_known_checkbox = gr.Checkbox(label="Show known interactions", value=True)
            show_predicted_checkbox = gr.Checkbox(label="Show predicted interactions", value=True)
            max_nodes_slider = gr.Slider(
                minimum=10, maximum=50, value=20, step=5,
                label="Max nodes to display"
            )
            
            analyze_btn = gr.Button("🔬 Analyze Drug", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            drug_info = gr.Markdown("### Select a drug to begin analysis")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Known Protein Targets")
            known_targets_table = gr.Dataframe(
                headers=['Target Protein', 'ChEMBL ID', 'pChEMBL (max)', 'pChEMBL (avg)', 'Measurements'],
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("### 🔮 Predicted Novel Targets")
            predicted_targets_table = gr.Dataframe(
                headers=['Predicted Target', 'ChEMBL ID', 'Similarity', 'Confidence'],
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💉 Known Clinical Effects")
            known_effects_table = gr.Dataframe(
                headers=['Clinical Effect', 'MeSH ID', 'Phase', 'References'],
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("### 💡 Predicted Novel Effects (Repurposing)")
            predicted_effects_table = gr.Dataframe(
                headers=['Predicted Effect', 'MeSH ID', 'Similarity', 'Confidence'],
                interactive=False
            )
    
    gr.Markdown("---")
    
    gr.Markdown("### 🕸️ Interactive Knowledge Graph")
    network_plot = gr.Plot()
    
    gr.Markdown(
        """
        ---
        
        ### 📊 About the Model
        
        This app uses a **TransE knowledge graph embedding model** trained on:
        - 800+ FDA-approved drugs
        - 200+ human protein targets (with ESM-2 embeddings)
        - 400+ clinical effects and indications
        
        **Prediction method:** Cosine similarity in learned embedding space  
        **Model performance:** ~90% precision on top-50 predictions
        
        **Citation:** VonDahab, J. (2025). Pharmacology Knowledge Graph. [GitHub](https://github.com/JoeVonDahab/pharmacology-graph)
        
        ---
        **Disclaimer:** This is a research tool for exploratory analysis only. Predictions should be validated experimentally. Not for clinical use.
        """
    )
    
    # Event handlers
    # Store mapping globally to avoid re-searching
    drug_display_to_id = gr.State({})
    selected_drug_id = gr.State()
    
    def update_dropdown(search_query):
        if not search_query or len(search_query) < 1:
            return gr.update(choices=[]), {}
        
        results = search_drugs(search_query)
        print(f"Search '{search_query}' found {len(results)} results")  # Debug output
        
        # Create mapping: display_name -> internal_id
        mapping = {display: internal_id for display, internal_id in results}
        choices = list(mapping.keys())
        
        print(f"Returning choices: {choices[:3]}..." if len(choices) > 3 else f"Returning choices: {choices}")  # Debug
        return gr.update(choices=choices, value=None), mapping
    
    def store_selection(dropdown_value, mapping):
        if not dropdown_value or not mapping:
            return None
        
        # Get internal_id from mapping
        return mapping.get(dropdown_value, None)
    
    search_box.change(
        fn=update_dropdown,
        inputs=[search_box],
        outputs=[drug_dropdown, drug_display_to_id]
    )
    
    drug_dropdown.change(
        fn=store_selection,
        inputs=[drug_dropdown, drug_display_to_id],
        outputs=[selected_drug_id]
    )
    
    analyze_btn.click(
        fn=analyze_drug,
        inputs=[
            search_box,
            selected_drug_id,
            show_known_checkbox,
            show_predicted_checkbox,
            max_nodes_slider
        ],
        outputs=[
            drug_info,
            known_targets_table,
            known_effects_table,
            predicted_targets_table,
            predicted_effects_table,
            network_plot
        ]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
