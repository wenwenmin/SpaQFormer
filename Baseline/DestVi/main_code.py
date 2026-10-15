"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/DestVI_pipeline.py"""
import hashlib
import copy

import scanpy as sc
import numpy as np
import pandas as pd
import time
import scvi
from scvi.model import CondSCVI, DestVI
import anndata as ad
import sys
import os


def main(sc_file, st_file, output_dir, cell_key='celltype', smoke=False):
    start_time = time.time()
    
    st_adata1 = ad.read_h5ad(st_file)
    sc_adata1 = ad.read_h5ad(sc_file)
    sc_adata = copy.deepcopy(sc_adata1)
    st_adata = copy.deepcopy(st_adata1)
    outpath = os.path.join(output_dir, 'DestVi')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    st_adata = st_adata[:, intersect].copy()
    sc_adata = sc_adata[:, intersect].copy()
    G = len(intersect)
    sc.pp.filter_genes(sc_adata, min_counts=10)

    sc_adata.layers["counts"] = sc_adata.X.copy()

    sc.pp.highly_variable_genes(
        sc_adata,
        n_top_genes=G,
        subset=True,
        layer="counts",
        flavor="seurat_v3"
    )

    sc.pp.normalize_total(sc_adata, target_sum=10e4)
    sc.pp.log1p(sc_adata)
    sc_adata.raw = sc_adata
    st_adata.layers["counts"] = st_adata.X.copy()
    sc.pp.normalize_total(st_adata, target_sum=10e4)
    sc.pp.log1p(st_adata)
    st_adata.raw = st_adata
    intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    st_adata = st_adata[:, intersect].copy()
    sc_adata = sc_adata[:, intersect].copy()
    print(st_adata)
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key=cell_key)

    stat = os.stat(sc_file)
    signature = f"{os.path.abspath(sc_file)}|{stat.st_size}|{stat.st_mtime_ns}|{cell_key}|{smoke}"
    cache_key = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(output_dir), '_shared', 'DestVI', cache_key)
    if os.path.isfile(os.path.join(cache_dir, 'model.pt')):
        print(f"Reusing shared CondSCVI reference model: {cache_dir}")
        sc_model = CondSCVI.load(cache_dir, adata=sc_adata)
    else:
        sc_model = CondSCVI(sc_adata, weight_obs=True)
        sc_model.train(max_epochs=250, lr=0.001 if smoke else 0.0001, batch_size=1024)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        sc_model.save(cache_dir, overwrite=True)

    if hasattr(st_adata.X, 'toarray'):
        dense_matrix = st_adata.X.toarray()
    else:
        dense_matrix = st_adata.X

    dense_matrix = dense_matrix.astype(np.float64)
    row_sums = np.sum(dense_matrix, axis=1)
    non_zero_rows = row_sums != 0
    stadata_filtered = st_adata[non_zero_rows, :].copy()
    filtered_file = os.path.join(outpath, 'DestVI_Spatial.h5ad')
    stadata_filtered.write_h5ad(filtered_file)
    scvi.model.DestVI.setup_anndata(stadata_filtered, layer="counts")
    st_model = DestVI.from_rna_model(stadata_filtered, sc_model)
    st_model.train(max_epochs=500, batch_size=1024)
    proportions = st_model.get_proportions()
    full_proportions = proportions.reindex(st_adata1.obs_names.astype(str))
    prior = sc_adata1.obs[cell_key].astype(str).value_counts(normalize=True)
    prior = prior.reindex(full_proportions.columns, fill_value=0.0)
    if prior.sum() == 0:
        prior[:] = 1.0 / len(prior)
    else:
        prior /= prior.sum()
    missing = full_proportions.isna().all(axis=1)
    full_proportions.loc[missing, :] = prior.to_numpy()
    full_proportions.to_csv(os.path.join(outpath, 'DestVI_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"DestVI Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype')
