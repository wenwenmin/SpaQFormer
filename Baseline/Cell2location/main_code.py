import hashlib
import sys
import time

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import subprocess

import os

import cell2location
import scvi

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
import seaborn as sns
from scipy.sparse import csr_matrix
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel


def get_reference_signatures(sc_file, output_dir, celltype_key):
    stat = os.stat(sc_file)
    signature = f"{os.path.abspath(sc_file)}|{stat.st_size}|{stat.st_mtime_ns}|{celltype_key}"
    cache_key = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(output_dir), '_shared', 'Cell2location', cache_key)
    cache_file = os.path.join(cache_dir, 'reference_signatures.csv')
    if os.path.isfile(cache_file):
        print(f"Reusing shared Cell2location reference signatures: {cache_file}")
        return pd.read_csv(cache_file, index_col=0)

    os.makedirs(cache_dir, exist_ok=True)
    adata_snrna_raw = sc.read_h5ad(sc_file)
    adata_snrna_raw.X = csr_matrix(adata_snrna_raw.X)
    print(adata_snrna_raw)
    adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isin(np.array(
        adata_snrna_raw.obs[celltype_key].value_counts()[
            adata_snrna_raw.obs[celltype_key].value_counts() <= 1].index))]
    sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
    sc.pp.filter_cells(adata_snrna_raw, min_genes=1)
    adata_snrna_raw.obs[celltype_key] = pd.Categorical(adata_snrna_raw.obs[celltype_key])
    adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isna(), :]
    selected = filter_genes(
        adata_snrna_raw,
        cell_count_cutoff=5,
        cell_percentage_cutoff2=0.03,
        nonz_mean_cutoff=1.12,
    )
    adata_snrna_raw = adata_snrna_raw[:, selected].copy()
    RegressionModel.setup_anndata(adata=adata_snrna_raw, labels_key=celltype_key)
    if hasattr(adata_snrna_raw.X, 'toarray'):
        adata_snrna_raw.X = csr_matrix(np.round(adata_snrna_raw.X.toarray()).astype(int))
    else:
        adata_snrna_raw.X = np.round(adata_snrna_raw.X).astype(int)
    model = RegressionModel(adata_snrna_raw)
    model.train(max_epochs=300, batch_size=2500, train_size=1, lr=0.002)
    adata_snrna_raw = model.export_posterior(
        adata_snrna_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
    )
    if 'means_per_cluster_mu_fg' in adata_snrna_raw.varm.keys():
        inf_aver = adata_snrna_raw.varm['means_per_cluster_mu_fg'][
            [f'means_per_cluster_mu_fg_{i}' for i in adata_snrna_raw.uns['mod']['factor_names']]
        ].copy()
    else:
        inf_aver = adata_snrna_raw.var[
            [f'means_per_cluster_mu_fg_{i}' for i in adata_snrna_raw.uns['mod']['factor_names']]
        ].copy()
    inf_aver.columns = adata_snrna_raw.uns['mod']['factor_names']
    inf_aver.to_csv(cache_file)
    return inf_aver


def main(sc_file, st_file, output_dir, cell_key='celltype'):
    start_time = time.time()
    
    celltype_key = cell_key
    output_file_path = os.path.join(output_dir, 'Cell2location')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    adata_vis = sc.read_h5ad(st_file)
    adata_vis.X = csr_matrix(adata_vis.X)
    inf_aver = get_reference_signatures(sc_file, output_dir, celltype_key)

    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    if hasattr(adata_vis.X, 'toarray'):
        counts_dense1 = adata_vis.X.toarray()
        counts_rounded1 = np.round(counts_dense1).astype(int)
        adata_vis.X = csr_matrix(counts_rounded1)
    else:
        adata_vis.X = np.round(adata_vis.X).astype(int)

    cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
    mod = cell2location.models.Cell2location(
        adata_vis, cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=200
    )

    mod.train(max_epochs=500, batch_size=None, train_size=1)

    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 200, 'batch_size': 1000}
    )
    adata_vis.obsm['q05_cell_abundance_w_sf'].to_csv(os.path.join(output_file_path, 'Cell2location_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"cell2location Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype')
