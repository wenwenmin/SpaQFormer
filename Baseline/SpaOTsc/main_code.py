# basic imports
import os
import time
import scanpy as sc
import pandas as pd
import numpy as np
from Baseline.SpaOTsc.spaotsc import SpaOTsc
from scipy import stats
import tangram as tg
from scipy.sparse import csr_matrix
import sys


def batch_spearmanr(a, b, batch_size=5000):
    m, n_a = a.shape
    p, n_b = b.shape

    if n_a != n_b:
        raise ValueError(f"The number of columns in a ({n_a}) must match the number of columns in b ({n_b}).")

    rho_list = []
    pval_list = []

    if isinstance(a, pd.DataFrame):
        a = a.values
    if isinstance(b, pd.DataFrame):
        b = b.values

    a_split = np.array_split(a, np.ceil(m / batch_size), axis=0)
    print(f"Total number of batches: {len(a_split)}")

    for i, batch in enumerate(a_split, start=1):
        print(f"Batch {i} started. batch.shape={batch.shape}, b.shape={b.shape}")

        batch_rho = []
        batch_pval = []
        j=1
        for row_a in batch:
            batch_rho_row = []
            batch_pval_row = []
            for row_b in b:
                print(j)
                j=j+1
                rho, pval = stats.spearmanr(row_a, row_b)
                batch_rho_row.append(rho)
                batch_pval_row.append(pval)
            batch_rho.append(batch_rho_row)
            batch_pval.append(batch_pval_row)

        batch_rho = np.array(batch_rho)
        batch_pval = np.array(batch_pval)
        print(f"Batch {i} rho.shape={batch_rho.shape}, pval.shape={batch_pval.shape}")

        rho_list.append(batch_rho)
        pval_list.append(batch_pval)

        print(f"Batch {i} finished.")

    rho_final = np.concatenate(rho_list, axis=0)
    pval_final = np.concatenate(pval_list, axis=0)

    print(f"Final rho_final.shape={rho_final.shape}, pval_final.shape={pval_final.shape}")

    return rho_final, pval_final


def main(sc_file, st_file, output_dir, cell_key='celltype', x='x', y='y', seed=None):
    start_time = time.time()
    if seed is not None:
        np.random.seed(int(seed))
    
    output_file_path = os.path.join(output_dir, 'SpaOTsc')
    os.makedirs(output_file_path, exist_ok=True)

    ad_sc = sc.read(sc_file)
    ad_sp = sc.read(st_file)
    celltype_key = cell_key
    df_sc_cells = ad_sc.to_df()
    df_IS = ad_sp.to_df()

    # Alpha is zero below, so SpaOTsc does not use the spatial distance matrix.
    # Avoid materializing an 85k x 85k matrix and retain every reference cell
    # through deterministic cell-type means.
    labels = ad_sc.obs[celltype_key].astype(str)
    df_sc = df_sc_cells.groupby(labels, observed=True).mean()
    prototype_obs = pd.DataFrame(
        {celltype_key: df_sc.index.astype(str)}, index=df_sc.index.astype(str)
    )
    print(f"Using all {ad_sc.n_obs} reference cells as {df_sc.shape[0]} cell-type prototypes")

    df_is = df_IS

    gene_is = df_is.columns.tolist()
    gene_sc = df_sc.columns.tolist()
    gene_overloap = list(set(gene_is).intersection(gene_sc))
    a = df_is[gene_overloap]
    b = df_sc[gene_overloap]
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    print(a.shape, b.shape)

    # Compute only prototype-to-target Spearman correlations. scipy.spearmanr
    # on the concatenated matrix also creates the unused target-to-target block.
    rank_is = stats.rankdata(a.to_numpy(), axis=1).astype(np.float32)
    rank_sc = stats.rankdata(b.to_numpy(), axis=1).astype(np.float32)
    rank_is -= rank_is.mean(axis=1, keepdims=True)
    rank_sc -= rank_sc.mean(axis=1, keepdims=True)
    rank_is /= np.linalg.norm(rank_is, axis=1, keepdims=True).clip(min=1e-12)
    rank_sc /= np.linalg.norm(rank_sc, axis=1, keepdims=True).clip(min=1e-12)
    mcc = rank_sc @ rank_is.T
    np.nan_to_num(mcc, copy=False)
    C = np.exp(1 - mcc)

    issc = SpaOTsc.spatial_sc(sc_data=df_sc, is_data=df_is)

    issc.transport_plan(C ** 2, alpha=0, rho=1.0, epsilon=1.0, cor_matrix=mcc, scaling=False)
    gamma = issc.gamma_mapping
    for j in range(gamma.shape[1]):
        gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])
    ad_map = sc.AnnData(gamma, obs=prototype_obs, var=ad_sp.obs)
    tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
    ad_sp.obsm['tangram_ct_pred'].to_csv(os.path.join(output_file_path, 'SpaOTsc_result.csv'))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"SpaOTsc Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype')
