# Phase 3: Hardware-Optimized DGI Training

This module trains a Graph Attention-based Neural Network to learn the structural fingerprint of healthy human mammary tissue.

## Graph Attention Architecture (GATv2)

*   **Deep Graph Infomax (DGI)**: Learns self-supervised node embeddings by maximizing the Mutual Information between local nodes and global graph features.
*   **Graph Attention (GATv2)**: Employs Attention mechanisms to dynamically learn which protein links are biologically more significant in the context of the network.
*   **Parametric Non-linearity (PReLU)**: Improves feature abstraction and prevents gradient dissipation during backpropagation.

## Hardware Acceleration (RTX 2050 Optimized)

*   **CUDA Natively Supported**: Optimized for execution on NVIDIA RTX 2050 GPUs via CUDA.
*   **Memory Efficiency**: Utilizes $O(N)$ DGI discriminators instead of $O(N^2)$ Autoencoder decoders, maintaining a small VRAM footprint.
*   **Micro-batching**: Sequentialized GPU-flushing to process hundreds of patients within 4GB VRAM limits.

## Deliverables

- `models/gae_weights.pt`: Trained GAT model weights.
- `logs/training_loss.json`: Loss history used for convergence visualization.
