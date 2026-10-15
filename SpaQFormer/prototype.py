import torch
import torch.nn.functional as F


def compute_reference_prototypes(embeddings, labels, num_classes, normalize=True):
    """
    Compute one scRNA reference prototype per known cell type.
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [N, d].")
    if labels.dim() != 1:
        raise ValueError("labels must have shape [N].")
    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels must have the same length.")

    prototypes = torch.zeros(
        num_classes,
        embeddings.size(1),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    counts = torch.zeros(
        num_classes,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )

    prototypes.index_add_(0, labels, embeddings)
    counts.index_add_(
        0,
        labels,
        torch.ones(labels.size(0), dtype=embeddings.dtype, device=embeddings.device),
    )

    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).tolist()
        raise ValueError(f"missing embeddings for classes: {missing}")

    prototypes = prototypes / counts.unsqueeze(1)
    if normalize:
        prototypes = F.normalize(prototypes, dim=1)

    return prototypes


def build_reference_prototypes(encoder, sc_x, sc_y, num_classes, device=None):
    """
    Encode scRNA expression and compute class prototypes.
    """
    was_training = encoder.training
    encoder.eval()

    if device is not None:
        sc_x = sc_x.to(device)
        sc_y = sc_y.to(device)
        encoder = encoder.to(device)

    with torch.no_grad():
        embeddings = encoder(sc_x)

    if was_training:
        encoder.train()

    return compute_reference_prototypes(embeddings, sc_y, num_classes)
