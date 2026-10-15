import os
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
import novosparc as nc
from scipy import stats
import tangram as tg
from scipy.spatial.distance import cdist
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
import sys
import time


def batch_process_gromov_wasserstein(cost_marker_genes, cost_expression, cost_locations,
                                     alpha_linear, p_expression, p_location, batch_size=5000):
    total_rows = cost_marker_genes.shape[0]
    result = []
    for start_row in range(0, total_rows, batch_size):
        end_row = min(start_row + batch_size, total_rows)
        batch_marker_genes = cost_marker_genes[start_row:end_row, :]
        batch_expression = cost_expression[start_row:end_row, :]
        batch_locations = cost_locations[start_row:end_row, :]
        batch_result = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(
            batch_marker_genes, batch_expression, batch_locations,
            alpha_linear, p_expression, p_location, 'square_loss', epsilon=5e-3, verbose=True
        )
        result.append(batch_result)
        del batch_marker_genes, batch_expression, batch_locations, batch_result
    final_result = np.concatenate(result, axis=0)
    return final_result


"""https://github.com/YangLabHKUST/SpatialScope/blob/master/compared_methods/novoSpaRc_pipeline.py"""
def main(sc_file, st_file, output_dir, cell_key='celltype', x='x', y='y', seed=None):
    start_time = time.time()
    resolved_seed = 2026 if seed is None else int(seed)
    np.random.seed(resolved_seed)
    
    output_file_path = os.path.join(output_dir, 'novoSpaRc')
    os.makedirs(output_file_path, exist_ok=True)

    ad_sc = sc.read(sc_file)
    ad_sp_full = sc.read(st_file)

    # Use every reference cell through deterministic cell-type prototypes.
    ref_cells = ad_sc.to_df()
    ref_labels = ad_sc.obs[cell_key].astype(str)
    ref_df = ref_cells.groupby(ref_labels, observed=True).mean()
    ref_obs = pd.DataFrame(
        {cell_key: ref_df.index.astype(str)}, index=ref_df.index.astype(str)
    )
    ad_sc = sc.AnnData(ref_df.to_numpy(dtype=np.float32), obs=ref_obs,
                       var=pd.DataFrame(index=ref_df.columns))

    # Exact novoSpaRc requires a dense target-target GW cost matrix. For this
    # dataset that matrix alone is ~55 GiB, so build spatial landmarks from all
    # target cells, run the same GW objective, then broadcast landmark scores.
    n_landmarks = min(2048, ad_sp_full.n_obs)
    coords = ad_sp_full.obs[[x, y]].to_numpy(dtype=np.float32)
    coord_mean = coords.mean(axis=0, keepdims=True)
    coord_scale = coords.std(axis=0, keepdims=True)
    coords_scaled = (coords - coord_mean) / np.maximum(coord_scale, 1e-6)
    landmark_id = MiniBatchKMeans(
        n_clusters=n_landmarks, random_state=resolved_seed, batch_size=8192,
        n_init=3, max_iter=100
    ).fit_predict(coords_scaled)
    counts = np.bincount(landmark_id, minlength=n_landmarks).astype(np.float32)
    membership = sparse.csr_matrix(
        (np.ones(ad_sp_full.n_obs, dtype=np.float32),
         (landmark_id, np.arange(ad_sp_full.n_obs))),
        shape=(n_landmarks, ad_sp_full.n_obs),
    )
    landmark_x = membership @ ad_sp_full.X
    if sparse.issparse(landmark_x):
        landmark_x = landmark_x.toarray()
    landmark_x = np.asarray(landmark_x, dtype=np.float32) / counts[:, None]
    landmark_coords = membership @ coords
    landmark_coords = np.asarray(landmark_coords, dtype=np.float32) / counts[:, None]
    landmark_obs = pd.DataFrame(
        {x: landmark_coords[:, 0], y: landmark_coords[:, 1]},
        index=[f"landmark_{i}" for i in range(n_landmarks)],
    )
    ad_sp = sc.AnnData(landmark_x, obs=landmark_obs, var=ad_sp_full.var.copy())
    print(
        f"Using all {ref_cells.shape[0]} reference cells as {ad_sc.n_obs} prototypes; "
        f"all {ad_sp_full.n_obs} target cells as {n_landmarks} spatial landmarks"
    )

    gene_names = np.array(ad_sc.var.index.values)
    dge = ad_sc.to_df().values
    num_cells = dge.shape[0]
    hvg = np.argsort(np.divide(np.var(dge, axis=0), np.mean(dge, axis=0) + 0.0001))
    dge_hvg = dge[:, hvg[-2000:]]

    locations = ad_sp.obs[[x, y]].values
    num_locations = locations.shape[0]

    p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
    cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg, locations, num_neighbors_source=5, num_neighbors_target=5)

    gene_is = ad_sp.var.index.tolist()
    gene_sc = ad_sc.var.index.tolist()
    insitu_genes = list(set(gene_is).intersection(gene_sc))

    markers_in_sc = np.array([], dtype='int')
    for marker in insitu_genes:
        marker_index = np.where(gene_names == marker)[0]
        if len(marker_index) > 0:
            markers_in_sc = np.append(markers_in_sc, marker_index[0])

    insitu_matrix = np.array(ad_sp.to_df()[insitu_genes])
    cost_marker_genes = cdist(dge[:, markers_in_sc] / np.amax(dge[:, markers_in_sc]),
                              insitu_matrix / np.amax(insitu_matrix))

    alpha_linear = 0.5
    gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                                            alpha_linear, p_expression, p_location, 'square_loss',
                                                            epsilon=5e-3, verbose=True)
    gamma = gw
    for j in range(gamma.shape[1]):
        gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])

    ad_map = sc.AnnData(gamma, obs=ad_sc.obs, var=ad_sp.obs)
    tg.project_cell_annotations(ad_map, ad_sp, annotation=cell_key)
    landmark_scores = ad_sp.obsm['tangram_ct_pred']
    full_scores = landmark_scores.iloc[landmark_id].copy()
    full_scores.index = ad_sp_full.obs_names
    full_scores.to_csv(os.path.join(output_file_path, 'novoSpaRc_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"novoSpaRC Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype', x='x', y='y')
