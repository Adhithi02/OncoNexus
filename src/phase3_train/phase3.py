import os
import json
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, DeepGraphInfomax
import pandas as pd
from src.utils import setup_logger

logger = setup_logger("Phase3")

class GATEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, out_channels=128):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))

def run_phase3(data_dir: str = "data", models_dir: str = "models", logs_dir: str = "logs"):
    logger.info("Starting Phase 3: Hardware-Optimized DGI Training")
    
    # CRITICAL FIX 6: GPU Logging & Failsafe
    if torch.cuda.is_available():
        device = torch.device('cuda')
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"HARDWARE ACQUIRED: CUDA / RTX GPU detected ({vram:.2f} GB VRAM). Training will be ultra-fast.")
    else:
        device = torch.device('cpu')
        logger.warning(
            "CRITICAL HARDWARE WARNING: torch.cuda.is_available() returned False!\n"
            "You are heavily wasting your RTX 2050 capability! PyTorch is defaulting to CPU.\n"
            "Fix: pip uninstall torch && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    
    features_path = os.path.join(data_dir, "gtex_train.csv")
    graph_path = os.path.join(data_dir, "adjacency.pt")
    nodes_path = os.path.join(data_dir, "final_graph_nodes.json")
    
    gtex_df = pd.read_csv(features_path, index_col=0)
    edge_index = torch.load(graph_path)
    
    with open(nodes_path, "r") as f:
        final_nodes = json.load(f)
        
    # CRITICAL FIX 3: Feature-Graph Tensor Alignment
    logger.info(f"Aligning the feature matrix exactly to the {len(final_nodes)} surviving STRING network nodes.")
    gtex_df = gtex_df[final_nodes]
    
    logger.info(f"Training Data geometry - Nodes: {gtex_df.shape[1]}, Samples: {gtex_df.shape[0]}")
    
    encoder = GATEncoder(in_channels=1, hidden_channels=256, out_channels=128)
    model = DeepGraphInfomax(hidden_channels=128, encoder=encoder, summary=summary, corruption=corruption).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    logger.info("Engaging Micro-Batch DGI Training...")
    model.train()
    
    edge_index = edge_index.to(device)
    samples_x = [torch.tensor(row.values, dtype=torch.float).unsqueeze(1) for _, row in gtex_df.iterrows()]
    
    epochs = 150
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        for x_i in samples_x:
            optimizer.zero_grad()
            
            x_i_device = x_i.to(device)
            pos_z, neg_z, g_summary = model(x_i_device, edge_index)
            loss = model.loss(pos_z, neg_z, g_summary)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(samples_x)
        loss_history.append(avg_loss)
        
        if epoch % 20 == 0:
            mem = torch.cuda.memory_allocated()/1024**2 if device.type=='cuda' else 0
            logger.info(f"[GPU Mem: {mem:.2f} MB] Epoch {epoch:03d}/{epochs:03d}, Avg DGI Loss: {avg_loss:.4f}")
            
    logger.info("Training finished. Exporting structural weights...")
    torch.save(model.state_dict(), os.path.join(models_dir, "gae_weights.pt"))
    
    with open(os.path.join(logs_dir, "training_loss.json"), "w") as f:
        json.dump(loss_history, f)
        
    logger.info("Phase 3 complete!")
