from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn


CellType = Union[str, int]


def _require_after_annotate(spaqformer):
    required = (
        "model2",
        "reference_prototypes",
        "annotation_probs",
        "annotation_ids",
        "srt_x",
        "srt_edges",
    )
    missing = [
        name
        for name in required
        if not hasattr(spaqformer, name) or getattr(spaqformer, name) is None
    ]
    if missing:
        raise RuntimeError(
            "Run pretrain(), pseudo_label(), build_prototypes(), and annotate() "
            f"before explanation. Missing: {missing}"
        )


def _safe_gene_names(spaqformer) -> np.ndarray:
    n_genes = int(spaqformer.srt_x.shape[1])
    gene_names = getattr(spaqformer, "gene_names", None)
    if gene_names is not None:
        gene_names = np.asarray(gene_names).astype(str)
        if len(gene_names) == n_genes:
            return gene_names
    return np.asarray([f"feature_{i}" for i in range(n_genes)])


def _resolve_cell_type(spaqformer, cell_type: CellType) -> Tuple[str, int]:
    if isinstance(cell_type, (int, np.integer)):
        target_idx = int(cell_type)
        if target_idx not in spaqformer.inverse_dict:
            raise KeyError(f"Unknown cell type index: {target_idx}")
        return str(spaqformer.inverse_dict[target_idx]), target_idx

    target_name = str(cell_type)
    if target_name not in spaqformer.cell_type_dict:
        raise KeyError(
            f"Unknown cell type: {target_name}. "
            f"Available types: {list(spaqformer.cell_type_dict)}"
        )
    return target_name, int(spaqformer.cell_type_dict[target_name])


def select_cells_for_type(
    spaqformer,
    cell_type: CellType,
    max_cells: Optional[int] = 50,
) -> Tuple[np.ndarray, int, str]:
    """Select the most confident cells predicted as one cell type."""
    target_name, target_idx = _resolve_cell_type(spaqformer, cell_type)
    prediction = np.asarray(spaqformer.annotation_ids)
    probability = np.asarray(spaqformer.annotation_probs)
    selected = np.where(prediction == target_idx)[0]
    if selected.size == 0:
        raise ValueError(f"No cells were predicted as {target_name}.")

    if max_cells is not None and selected.size > max_cells:
        order = np.argsort(probability[selected, target_idx])[::-1]
        selected = selected[order[:max_cells]]
    return selected.astype(np.int64), target_idx, target_name


def _edge_index_tensor(edges) -> torch.Tensor:
    edge_index = torch.as_tensor(edges, dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError("srt_edges must be a two-dimensional array.")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError("srt_edges must have shape [2, E] or [E, 2].")
    return edge_index


def build_explain_loader(spaqformer, num_parts: int = 50):
    """Build deterministic METIS subgraphs for attribution."""
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import ClusterData, ClusterLoader
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required for SpaQFormer explanation."
        ) from exc

    data = Data(
        x=torch.as_tensor(spaqformer.srt_x, dtype=torch.float32),
        edge_index=_edge_index_tensor(spaqformer.srt_edges),
    )
    data.index = torch.arange(data.x.shape[0], dtype=torch.long)
    dataset = ClusterData(data, num_parts=int(num_parts), recursive=False)
    return ClusterLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def get_cached_explain_loader(spaqformer, num_parts: int = 50):
    """Reuse METIS partitions while the SRT arrays and partition count are unchanged."""
    cache = getattr(spaqformer, "_explain_loader_cache", None)
    if cache is None:
        cache = {}
        setattr(spaqformer, "_explain_loader_cache", cache)

    cache_key = (
        id(spaqformer.srt_x),
        id(spaqformer.srt_edges),
        int(num_parts),
    )
    if cache_key not in cache:
        cache[cache_key] = build_explain_loader(
            spaqformer,
            num_parts=num_parts,
        )
    return cache[cache_key]


def clear_explain_cache(spaqformer):
    """Clear cached METIS loaders after replacing SRT data."""
    cache = getattr(spaqformer, "_explain_loader_cache", None)
    if cache is not None:
        cache.clear()


class SpaQFormerCellTypeScore(nn.Module):
    """Return the mean target-cell-type logit for selected nodes."""

    def __init__(
        self,
        model,
        edge_index,
        prototypes,
        node_ids,
        target_idx: int,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("edge_index", _edge_index_tensor(edge_index))
        self.register_buffer(
            "prototypes", torch.as_tensor(prototypes, dtype=torch.float32)
        )
        self.register_buffer(
            "node_ids", torch.as_tensor(node_ids, dtype=torch.long)
        )
        self.target_idx = int(target_idx)

    def _forward_one(self, x):
        logits, _, _ = self.model(x, self.edge_index, self.prototypes)
        return logits[self.node_ids, self.target_idx].mean()

    def forward(self, x):
        if x.ndim == 2:
            return self._forward_one(x).view(1)
        if x.ndim == 3:
            return torch.stack([self._forward_one(item) for item in x])
        raise ValueError("Input x must have shape [N, G] or [B, N, G].")


def _normalize_cell_attribution(attribution: np.ndarray) -> np.ndarray:
    """Apply the per-cell L2 normalization used by river.py."""
    attribution = np.abs(np.asarray(attribution, dtype=np.float64))
    norm = np.linalg.norm(attribution, axis=1, keepdims=True)
    norm = np.maximum(norm, np.finfo(np.float64).eps)
    return attribution / norm


def compute_cell_type_attributions(
    spaqformer,
    node_ids: np.ndarray,
    target_idx: int,
    methods: Sequence[str] = ("ig", "gs", "oc"),
    num_parts: int = 50,
    ig_steps: int = 16,
    gs_samples: int = 8,
    gs_stdevs: float = 0.01,
    perturbations_per_eval: int = 16,
    explain_loader=None,
) -> Dict[str, np.ndarray]:
    """Compute raw IG, GradientShap, and Occlusion gene scores for one cell type."""
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

    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_genes = int(spaqformer.srt_x.shape[1])
    score_sums = {
        method: np.zeros(n_genes, dtype=np.float64)
        for method in methods
    }
    selected_count = {method: 0 for method in methods}
    prototypes = spaqformer.reference_prototypes.to(spaqformer.device)
    spaqformer.model2.eval()

    if explain_loader is None:
        explain_loader = get_cached_explain_loader(
            spaqformer,
            num_parts=num_parts,
        )

    for data in explain_loader:
        global_ids = data.index.cpu().numpy().astype(np.int64)
        local_ids = np.where(np.isin(global_ids, node_ids))[0]
        if local_ids.size == 0:
            continue

        data = data.to(spaqformer.device)
        inputs = data.x.unsqueeze(0)
        baseline = torch.zeros_like(inputs)
        wrapper = SpaQFormerCellTypeScore(
            model=spaqformer.model2,
            edge_index=data.edge_index,
            prototypes=prototypes,
            node_ids=local_ids,
            target_idx=target_idx,
        ).to(spaqformer.device)
        wrapper.eval()

        local_results = {}
        if "ig" in methods:
            local_results["ig"] = IntegratedGradients(wrapper).attribute(
                inputs,
                baselines=baseline,
                n_steps=ig_steps,
            )
        if "gs" in methods:
            local_results["gs"] = GradientShap(wrapper).attribute(
                inputs,
                baselines=baseline,
                n_samples=gs_samples,
                stdevs=gs_stdevs,
            )
        if "oc" in methods:
            local_results["oc"] = Occlusion(wrapper).attribute(
                inputs,
                baselines=baseline,
                sliding_window_shapes=(data.x.shape[0], 1),
                strides=(data.x.shape[0], 1),
                perturbations_per_eval=perturbations_per_eval,
            )

        for method, attribution in local_results.items():
            attribution = attribution.detach().cpu().squeeze(0).numpy()
            normalized = _normalize_cell_attribution(attribution[local_ids])
            score_sums[method] += normalized.sum(axis=0)
            selected_count[method] += int(local_ids.size)

        del data, inputs, baseline, wrapper, local_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [
        method
        for method, count in selected_count.items()
        if count == 0
    ]
    if missing:
        raise RuntimeError(f"No selected cells were attributed for: {missing}")

    return {
        method: (score_sums[method] / selected_count[method]).astype(np.float32)
        for method in methods
    }


def summarize_cell_type_attributions(
    spaqformer,
    attribution_scores: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rank genes per method and fuse the ranks with a simple Borda score."""
    if not attribution_scores:
        raise ValueError("attribution_scores is empty.")

    gene_names = _safe_gene_names(spaqformer)
    n_genes = len(gene_names)
    result = pd.DataFrame({"gene": gene_names})
    borda_score = np.zeros(n_genes, dtype=np.float64)

    for method, score in attribution_scores.items():
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (n_genes,):
            raise ValueError(
                f"{method} score has shape {score.shape}; expected {(n_genes,)}."
            )
        order = np.argsort(-score)
        rank = np.empty(n_genes, dtype=np.int64)
        rank[order] = np.arange(1, n_genes + 1)
        result[f"{method}_score"] = score
        result[f"{method}_rank"] = rank
        borda_score += n_genes - rank + 1

    final_order = np.argsort(-borda_score)
    final_rank = np.empty(n_genes, dtype=np.int64)
    final_rank[final_order] = np.arange(1, n_genes + 1)
    result["borda_score"] = borda_score
    result["final_rank"] = final_rank
    return result.sort_values("final_rank").reset_index(drop=True)


class SpaQFormerExplainer:
    """Simple cell-type-level IG, GradientShap, and Occlusion explainer."""

    def __init__(self, spaqformer, num_parts: int = 50):
        self.spaqformer = spaqformer
        self.num_parts = int(num_parts)
        _require_after_annotate(spaqformer)
        self.explain_loader = get_cached_explain_loader(
            spaqformer,
            num_parts=self.num_parts,
        )

    def explain_cell_type(
        self,
        cell_type: CellType,
        max_cells: Optional[int] = 50,
        methods: Sequence[str] = ("ig", "gs", "oc"),
        top_k: int = 200,
        **attribution_kwargs,
    ) -> Dict:
        selected_ids, target_idx, target_name = select_cells_for_type(
            self.spaqformer,
            cell_type=cell_type,
            max_cells=max_cells,
        )
        attribution_scores = compute_cell_type_attributions(
            self.spaqformer,
            node_ids=selected_ids,
            target_idx=target_idx,
            methods=methods,
            num_parts=self.num_parts,
            explain_loader=self.explain_loader,
            **attribution_kwargs,
        )
        gene_scores = summarize_cell_type_attributions(
            self.spaqformer,
            attribution_scores,
        )
        top_k = min(max(int(top_k), 1), len(gene_scores))
        return {
            "cell_type": target_name,
            "target_idx": target_idx,
            "selected_cell_ids": selected_ids,
            "gene_scores": gene_scores,
            "top_genes": gene_scores.head(top_k).copy(),
            "raw_attribution_scores": attribution_scores,
        }

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
