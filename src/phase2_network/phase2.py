import os
import json
import requests
import networkx as nx
import torch
from src.utils import setup_logger

logger = setup_logger("Phase2")

STRING_API_URL = "https://string-db.org/api/json/network"

def query_string_api(genes, species_id=9606, min_score=700):
    logger.info(f"Querying STRING API for {len(genes)} genes...")
    params = {
        "identifiers": "\r".join(genes),
        "species": species_id,
        "required_score": min_score,
        "caller_identity": "onconexus_pipeline"
    }

    try:
        response = requests.post(STRING_API_URL, data=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to query STRING API: {e}")
        return []

def build_network(genes, interactions):
    G = nx.Graph()
    G.add_nodes_from(genes)

    edges_added = 0
    for interaction in interactions:
        gene1 = interaction.get("preferredName_A")
        gene2 = interaction.get("preferredName_B")
        score = interaction.get("score")
        
        if gene1 in genes and gene2 in genes:
            G.add_edge(gene1, gene2, weight=score)
            edges_added += 1

    logger.info(f"Added {edges_added} high-confidence edges to network.")

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    logger.info(f"Removed {len(isolated)} isolated nodes. Remaining nodes: {G.number_of_nodes()}")

    return G

def export_frontend_graph(G, out_path):
    data = {
        "nodes": [{"id": n, "label": n} for n in G.nodes()],
        "edges": [{"source": u, "target": v, "weight": d['weight']} for u, v, d in G.edges(data=True)]
    }
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Frontend graph exported.")

def export_pyg_adjacency(G, sorted_genes, out_path):
    gene_to_idx = {gene: i for i, gene in enumerate(sorted_genes)}
    
    edge_index_u = []
    edge_index_v = []
    edge_attr = []
    
    for u, v, d in G.edges(data=True):
        u_id = gene_to_idx[u]
        v_id = gene_to_idx[v]
        
        edge_index_u.extend([u_id, v_id])
        edge_index_v.extend([v_id, u_id])
        edge_attr.extend([d['weight'], d['weight']])

    edge_index = torch.tensor([edge_index_u, edge_index_v], dtype=torch.long)
    torch.save(edge_index, out_path)
    logger.info(f"PyTorch sparse adjacency exported.")

def run_phase2(data_dir: str = "data"):
    logger.info("Starting Phase 2: Protein Interaction Network Construction")
    
    pre_ppi_path = os.path.join(data_dir, "pre_ppi_gene_list.json")
    if not os.path.exists(pre_ppi_path):
        raise FileNotFoundError(f"{pre_ppi_path} not found.")
        
    with open(pre_ppi_path, "r") as f:
        genes = json.load(f)
        
    interactions = query_string_api(genes, min_score=700)
            
    G = build_network(genes, interactions)
    
    # CRITICAL FIX 3: Export the FINAL ALIGNED list of genes
    final_nodes = list(G.nodes())
    # Sort them alphanumerically to guarantee deterministic geometric index maps in Phase 3
    final_nodes = sorted(final_nodes)
    
    with open(os.path.join(data_dir, "final_graph_nodes.json"), "w") as f:
        json.dump(final_nodes, f)
    
    export_frontend_graph(G, os.path.join(data_dir, "graph.json"))
    export_pyg_adjacency(G, final_nodes, os.path.join(data_dir, "adjacency.pt"))
    
    logger.info("Phase 2 complete! Graph Feature Alignment successfully tracked.")
