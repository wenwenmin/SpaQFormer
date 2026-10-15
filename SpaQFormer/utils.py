import torch


def neighbor_mean(h, edge_index):
    """
    Aggregate incoming neighbor embeddings with PyG-style edge_index.

    edge_index[0] is treated as source nodes and edge_index[1] as target nodes.
    Nodes without incoming neighbors receive a zero vector.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E].")

    row, col = edge_index
    out = torch.zeros_like(h)
    count = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)

    out.index_add_(0, col, h[row])
    count.index_add_(0, col, torch.ones(col.size(0), dtype=h.dtype, device=h.device))

    count = count.clamp(min=1.0).unsqueeze(1)
    return out / count
