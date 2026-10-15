import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def compute_label_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "n_eval": int(y_true.shape[0]),
    }


def valid_label_mask(labels, exclude_label="unlabeled"):
    """
    Return a mask that excludes labels such as 'unlabeled'.
    """
    labels = np.asarray(labels).astype(str)
    if exclude_label is None:
        return np.ones(labels.shape[0], dtype=bool)
    return np.array([label.lower() != exclude_label.lower() for label in labels])


def label_summary(labels, exclude_label="unlabeled"):
    """
    Summarize total/evaluable labels before metric computation.
    """
    labels = np.asarray(labels).astype(str)
    valid_mask = valid_label_mask(labels, exclude_label=exclude_label)
    invalid_count = int((~valid_mask).sum())
    valid_labels = labels[valid_mask]

    return {
        "total_spots": int(labels.shape[0]),
        "excluded_label": exclude_label,
        "excluded_spots": invalid_count,
        "valid_spots": int(valid_mask.sum()),
        "n_types_with_excluded": int(len(np.unique(labels))),
        "n_valid_types": int(len(np.unique(valid_labels))),
        "valid_types": sorted(np.unique(valid_labels).tolist()),
    }


def per_class_metrics(y_true, y_pred):
    """
    Compute per-class precision, recall, F1, and support.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    support = pd.Series(y_true).value_counts().reindex(labels, fill_value=0)
    pred_count = pd.Series(y_pred).value_counts().reindex(labels, fill_value=0)

    return pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.to_numpy(dtype=int),
            "pred_count": pred_count.to_numpy(dtype=int),
        }
    )


def evaluate_spaqformer_predictions(y_true, known_pred, open_set_pred):
    """
    Evaluate SpaQFormer outputs in three complementary ways.

    closed_set:
        Uses known predictions before unknown rejection. This is the fair
        comparison against closed-set baselines.
    open_set:
        Uses labels after unknown rejection. If the dataset has no true unknown
        label, predicted unknown spots are counted as wrong.
    known_only:
        Filters out predicted unknown spots and evaluates only accepted spots.
        Report coverage alongside the metrics.
    """
    y_true = np.asarray(y_true).astype(str)
    known_pred = np.asarray(known_pred).astype(str)
    open_set_pred = np.asarray(open_set_pred).astype(str)

    if not (len(y_true) == len(known_pred) == len(open_set_pred)):
        raise ValueError("All prediction arrays must have the same length.")

    unknown_mask = open_set_pred == "unknown"
    known_mask = ~unknown_mask

    metrics = {
        "closed_set": compute_label_metrics(y_true, known_pred),
        "open_set": compute_label_metrics(y_true, open_set_pred),
    }
    metrics["closed_set"]["coverage"] = 1.0
    metrics["open_set"]["coverage"] = 1.0
    metrics["open_set"]["unknown_rate"] = float(unknown_mask.mean())

    if known_mask.any():
        metrics["known_only"] = compute_label_metrics(
            y_true[known_mask],
            known_pred[known_mask],
        )
        metrics["known_only"]["coverage"] = float(known_mask.mean())
        metrics["known_only"]["n_rejected_unknown"] = int(unknown_mask.sum())
    else:
        metrics["known_only"] = {
            "coverage": 0.0,
            "n_eval": 0,
            "n_rejected_unknown": int(unknown_mask.sum()),
        }

    return metrics
