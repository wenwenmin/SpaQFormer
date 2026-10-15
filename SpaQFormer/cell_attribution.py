from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .explain import (
    SpaQFormerCellTypeScore,
    _normalize_cell_attribution,
    get_cached_explain_loader,
    select_cells_for_type,
)


def borda_fuse_scores(
    scores_by_method: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fuse method-specific gene scores with the existing Borda rule."""
    if not scores_by_method:
        raise ValueError("scores_by_method is empty.")

    arrays = [np.asarray(scores, dtype=np.float64) for scores in scores_by_method.values()]
    reference_shape = arrays[0].shape
    if any(scores.shape != reference_shape for scores in arrays):
        raise ValueError("All method score arrays must have the same shape.")
    if len(reference_shape) not in (1, 2):
        raise ValueError("Scores must have shape [G] or [N, G].")

    squeeze = len(reference_shape) == 1
    if squeeze:
        arrays = [scores[None, :] for scores in arrays]

    n_rows, n_genes = arrays[0].shape
    fused = np.zeros((n_rows, n_genes), dtype=np.float64)
    for scores in arrays:
        order = np.argsort(-scores, axis=1)
        ranks = np.empty_like(order)
        row_ids = np.arange(n_rows)[:, None]
        ranks[row_ids, order] = np.arange(1, n_genes + 1)[None, :]
        fused += n_genes - ranks + 1

    fused /= float(len(arrays) * n_genes)
    fused = fused.astype(np.float32)
    return fused[0] if squeeze else fused


def compute_single_cell_attributions(
    spaqformer,
    cell_type,
    node_ids: Optional[np.ndarray] = None,
    max_cells: Optional[int] = 50,
    methods: Sequence[str] = ("ig", "gs", "oc"),
    num_parts: int = 50,
    ig_steps: int = 16,
    ig_internal_batch_size: Optional[int] = 4,
    gs_samples: int = 8,
    gs_stdevs: float = 0.01,
    perturbations_per_eval: int = 16,
    input_dropout_rate: float = 0.0,
    random_seed: int = 2026,
    explain_loader=None,
) -> Dict:
    """Return selected-cell gene attributions using the existing IG/GS/OC design.

    The same target-logit wrapper, zero baseline, absolute attribution, and
    per-cell L2 normalization as ``SpaQFormer.explain`` are retained. Borda is
    computed as a rank fusion of the requested attribution methods.
    """
    try:
        from captum.attr import GradientShap, IntegratedGradients, Occlusion
    except ImportError as exc:
        raise ImportError(
            "Captum is required. Install it with `pip install captum`."
        ) from exc

    methods = tuple(dict.fromkeys(methods))
    invalid = set(methods) - {"ig", "gs", "oc"}
    if invalid:
        raise ValueError(f"Unsupported attribution methods: {sorted(invalid)}")
    if not methods:
        raise ValueError("At least one attribution method is required.")

    input_dropout_rate = float(input_dropout_rate)
    if not 0.0 <= input_dropout_rate < 1.0:
        raise ValueError("input_dropout_rate must be in [0, 1).")

    if node_ids is None:
        node_ids, target_idx, target_name = select_cells_for_type(
            spaqformer,
            cell_type=cell_type,
            max_cells=max_cells,
        )
    else:
        target_name = str(cell_type)
        if target_name not in spaqformer.cell_type_dict:
            raise KeyError(f"Unknown cell type: {target_name}")
        target_idx = int(spaqformer.cell_type_dict[target_name])
        node_ids = np.asarray(node_ids, dtype=np.int64)

    if explain_loader is None:
        explain_loader = get_cached_explain_loader(
            spaqformer,
            num_parts=num_parts,
        )

    selected_set = set(np.asarray(node_ids, dtype=np.int64).tolist())
    collected_ids = []
    collected_scores = {method: [] for method in methods}
    rng = np.random.default_rng(int(random_seed))
    prototypes = spaqformer.reference_prototypes.to(spaqformer.device)
    spaqformer.model2.eval()

    for data in explain_loader:
        global_ids = data.index.cpu().numpy().astype(np.int64)
        local_ids = np.asarray(
            [
                local_id
                for local_id, global_id in enumerate(global_ids)
                if int(global_id) in selected_set
            ],
            dtype=np.int64,
        )
        if local_ids.size == 0:
            continue

        data = data.to(spaqformer.device)
        inputs = data.x.unsqueeze(0).clone()
        if input_dropout_rate > 0:
            keep = rng.random((local_ids.size, inputs.shape[-1]))
            keep = torch.as_tensor(
                keep >= input_dropout_rate,
                dtype=inputs.dtype,
                device=spaqformer.device,
            )
            inputs[0, local_ids] = inputs[0, local_ids] * keep

        baseline = torch.zeros_like(inputs)
        wrapper = SpaQFormerCellTypeScore(
            model=spaqformer.model2,
            edge_index=data.edge_index,
            prototypes=prototypes,
            node_ids=local_ids,
            target_idx=target_idx,
        ).to(spaqformer.device)
        wrapper.eval()
        torch.manual_seed(int(random_seed))

        method_results = {}
        if "ig" in methods:
            method_results["ig"] = IntegratedGradients(wrapper).attribute(
                inputs,
                baselines=baseline,
                n_steps=int(ig_steps),
                internal_batch_size=ig_internal_batch_size,
            )
        if "gs" in methods:
            method_results["gs"] = GradientShap(wrapper).attribute(
                inputs,
                baselines=baseline,
                n_samples=int(gs_samples),
                stdevs=float(gs_stdevs),
            )
        if "oc" in methods:
            method_results["oc"] = Occlusion(wrapper).attribute(
                inputs,
                baselines=baseline,
                sliding_window_shapes=(data.x.shape[0], 1),
                strides=(data.x.shape[0], 1),
                perturbations_per_eval=int(perturbations_per_eval),
            )

        for method, attribution in method_results.items():
            attribution = attribution.detach().cpu().squeeze(0).numpy()
            normalized = _normalize_cell_attribution(attribution[local_ids])
            collected_scores[method].append(normalized.astype(np.float32))
        collected_ids.append(global_ids[local_ids])

        del data, inputs, baseline, wrapper, method_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not collected_ids:
        raise RuntimeError(f"No selected cells were attributed for {target_name}.")

    recovered_ids = np.concatenate(collected_ids).astype(np.int64)
    position = {
        int(global_id): index
        for index, global_id in enumerate(recovered_ids)
    }
    missing_ids = [
        int(global_id)
        for global_id in node_ids
        if int(global_id) not in position
    ]
    if missing_ids:
        raise RuntimeError(
            f"Attribution rows were not recovered for cell ids: {missing_ids[:10]}"
        )
    order = np.asarray(
        [position[int(global_id)] for global_id in node_ids],
        dtype=np.int64,
    )
    cell_scores = {
        method: np.vstack(collected_scores[method])[order]
        for method in methods
    }
    mean_scores = {
        method: scores.mean(axis=0).astype(np.float32)
        for method, scores in cell_scores.items()
    }
    cell_scores["borda"] = borda_fuse_scores(cell_scores)
    mean_scores["borda"] = borda_fuse_scores(mean_scores)

    return {
        "cell_type": target_name,
        "target_idx": target_idx,
        "selected_cell_ids": np.asarray(node_ids, dtype=np.int64),
        "cell_scores": cell_scores,
        "mean_scores": mean_scores,
    }


class SpaQFormerSingleCellExplainer:
    """Cached single-cell attribution extension for SpaQFormer."""

    def __init__(self, spaqformer, num_parts: int = 50):
        self.spaqformer = spaqformer
        self.num_parts = int(num_parts)
        self.explain_loader = get_cached_explain_loader(
            spaqformer,
            num_parts=self.num_parts,
        )

    def explain_cell_type(self, cell_type, **kwargs) -> Dict:
        return compute_single_cell_attributions(
            self.spaqformer,
            cell_type=cell_type,
            num_parts=self.num_parts,
            explain_loader=self.explain_loader,
            **kwargs,
        )

    def explain_all_cell_types(self, **kwargs) -> Dict[str, Dict]:
        results = {}
        for target_idx in sorted(self.spaqformer.inverse_dict):
            target_name = str(self.spaqformer.inverse_dict[target_idx])
            try:
                results[target_name] = self.explain_cell_type(
                    target_name,
                    **kwargs,
                )
            except Exception as exc:
                print(f"Skip {target_name}: {exc}")
        return results
