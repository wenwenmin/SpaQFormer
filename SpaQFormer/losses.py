import torch
import torch.nn.functional as F


def known_matching_loss(scores, soft_targets, hard_targets, ce_weight=0.3):
    """
    Combine soft pseudo-label KL loss with hard pseudo-label CE loss.
    """
    loss_kl = F.kl_div(
        F.log_softmax(scores, dim=1),
        soft_targets,
        reduction="batchmean",
    )
    loss_ce = F.cross_entropy(scores, hard_targets)
    return loss_kl + ce_weight * loss_ce


def energy_known_loss(energy, margin):
    """
    Penalize known samples whose energy is higher than the known margin.
    """
    return F.relu(energy - margin).mean()


def energy_unknown_loss(energy, margin):
    """
    Penalize unknown candidates whose energy is lower than the unknown margin.
    """
    return F.relu(margin - energy).mean()


def spatial_consistency_loss(scores, edge_index):
    """
    Smooth known-type probabilities over spatial graph edges.
    """
    row, col = edge_index
    probs = F.softmax(scores, dim=1)
    return torch.mean(torch.sum((probs[row] - probs[col]) ** 2, dim=1))


def query_diversity_loss(type_queries):
    """
    Encourage per-class queries to capture different evidence.

    type_queries has shape [C, R, d].
    """
    queries = F.normalize(type_queries, dim=2)
    gram = torch.matmul(queries, queries.transpose(1, 2))
    identity = torch.eye(
        type_queries.size(1),
        dtype=type_queries.dtype,
        device=type_queries.device,
    )
    return torch.mean((gram - identity.unsqueeze(0)) ** 2)
