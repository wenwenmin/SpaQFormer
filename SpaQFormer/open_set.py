import torch


def energy_score(scores, temperature=1.0):
    """
    Compute energy scores from known-type matching scores.

    Lower energy means stronger matching to at least one known cell type.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    return -temperature * torch.logsumexp(scores / temperature, dim=1)


def estimate_energy_threshold(energy, quantile=0.95):
    """
    Estimate an open-set threshold from known-seed energy scores.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    return torch.quantile(energy.detach(), quantile)


def predict_open_set(scores, threshold, temperature=1.0):
    """
    Predict known labels and unknown mask from matching scores.
    """
    energy = energy_score(scores, temperature=temperature)
    known_ids = torch.argmax(scores, dim=1)
    unknown_mask = energy > threshold
    return known_ids, unknown_mask, energy
