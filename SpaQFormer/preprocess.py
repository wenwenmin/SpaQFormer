import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _to_dense_array(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _row_standardize(x, epsilon=1e-8):
    x = np.nan_to_num(x, nan=0.0)
    means = np.mean(x, axis=1, keepdims=True)
    stds = np.std(x, axis=1, keepdims=True)
    stds[stds == 0] = epsilon
    return ((x - means) / stds).astype(np.float32)


def _build_knn_edges_on_pos(pos, Kn):
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape [num_spots, 2].")
    if Kn < 1:
        raise ValueError("Kn must be >= 1.")
    if Kn >= len(pos):
        raise ValueError("Kn must be smaller than the number of spots.")

    nbrs = NearestNeighbors(n_neighbors=Kn + 1, algorithm="auto").fit(pos)
    _, indices = nbrs.kneighbors(pos)
    return np.array(
        [
            [i, indices[i, j]]
            for i in range(len(pos))
            for j in range(1, Kn + 1)
        ],
        dtype=np.int64,
    )


def get_knn_edge_index(pos, Kn=3):
    return _build_knn_edges_on_pos(pos, Kn)


def get_knn_edge_index_per_slice(pos, slice_ids, Kn=3):
    pos = np.asarray(pos, dtype=np.float32)
    slice_ids = np.asarray(slice_ids)
    all_edges = []

    for slice_id in np.unique(slice_ids):
        mask = slice_ids == slice_id
        global_ids = np.where(mask)[0]
        local_edges = _build_knn_edges_on_pos(pos[mask], Kn=Kn)
        all_edges.append(global_ids[local_edges])

    return np.concatenate(all_edges, axis=0).astype(np.int64)


def load_data(scfilename, srtfilename, Kn=3, slice_col=None):
    import scanpy

    sc_adata = scanpy.read_h5ad(scfilename)
    srt_adata = scanpy.read_h5ad(srtfilename)

    if "label" not in sc_adata.obs.columns:
        raise KeyError("scRNA AnnData must contain obs['label'].")
    if "x" not in srt_adata.obs.columns or "y" not in srt_adata.obs.columns:
        raise KeyError("SRT AnnData must contain obs['x'] and obs['y'].")

    gene_names = np.asarray(srt_adata.var_names.astype(str))
    sc_obs_names = np.asarray(sc_adata.obs_names.astype(str))
    srt_obs_names = np.asarray(srt_adata.obs_names.astype(str))
    srt_coords = srt_adata.obs[["x", "y"]].values.astype(np.float32)

    cell_types = sorted(list(set(sc_adata.obs["label"])))
    cell_type_dict = {cell_type: i for i, cell_type in enumerate(cell_types)}
    inverse_dict = {i: cell_type for cell_type, i in cell_type_dict.items()}

    sc_x = _to_dense_array(sc_adata.X)
    srt_x = _to_dense_array(srt_adata.X)
    sc_y_raw = list(sc_adata.obs["label"])
    sc_y = np.array([cell_type_dict[label] for label in sc_y_raw], dtype=np.int64)

    if slice_col is not None:
        if slice_col not in srt_adata.obs.columns:
            raise KeyError(f"slice_col='{slice_col}' not found in srt_adata.obs.")
        srt_edges = get_knn_edge_index_per_slice(
            srt_coords,
            srt_adata.obs[slice_col].values,
            Kn=Kn,
        )
    else:
        srt_edges = get_knn_edge_index(srt_coords, Kn=Kn)

    undirected_edges = []
    for u, v in srt_edges:
        undirected_edges.append([u, v])
        undirected_edges.append([v, u])

    return (
        _row_standardize(sc_x),
        sc_y,
        _row_standardize(srt_x),
        np.asarray(undirected_edges, dtype=np.int64),
        cell_type_dict,
        inverse_dict,
        gene_names,
        sc_obs_names,
        srt_obs_names,
        srt_coords,
    )
