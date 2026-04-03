# Phase 2: Protein Interaction Network Construction

This module constructs a high-fidelity biological graph by querying the STRING Protein-Protein Interaction (PPI) database.

## Biological Network Construction

*   **STRING API Integration**: Connects to the STRING (v12.0) database to fetch protein-protein interactions.
*   **Gene Precision**: Focuses purely on the 2,000 top-variance genes identified in Phase 1.
*   **Confidence Filtering**: Retains only edges with a high confidence score (combined_score >= 700).

## Graph-Feature Integrity

*   **Node Pruning**: Removes nodes without any connections (isolated nodes) to ensure the GNN trains only on connected biological pathways.
*   **Structural Alignment**: Exports a final, ordered list of genes (`final_graph_nodes.json`) to guarantee 1-to-1 consistency with the feature matrix dimensions in Phase 3.

## Deliverables

- `data/graph.json`: Frontend-compatible JSON for D3.js or Cytoscape.js.
- `data/adjacency.pt`: PyTorch sparse tensor representation of the interaction network.
- `data/final_graph_nodes.json`: The aligned node list used for GNN tensor mapping.
