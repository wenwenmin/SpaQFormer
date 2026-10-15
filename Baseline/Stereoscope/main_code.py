"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/Stereoscope_pipeline.py"""
import copy
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import sys
import time
import os

import scvi
from scipy.sparse import csr_matrix
from scvi.external import RNAStereoscope, SpatialStereoscope


def check_and_convert_to_integers(tensor):
    if not torch.all(tensor == tensor.to(torch.int)):
        print("输入数据包含浮点数，正在转换为整数...")
        tensor = tensor.to(torch.int)
    return tensor


def main(sc_file, st_file, output_dir, cell_key='celltype'):
    start_time = time.time()
    
    celltype_key = cell_key
    output_path = os.path.join(output_dir, 'Stereoscope')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sc_adata1 = sc.read_h5ad(sc_file)
    st_adata1 = sc.read_h5ad(st_file)

    sc_adata = copy.deepcopy(sc_adata1)
    st_adata = copy.deepcopy(st_adata1)

    sc.pp.filter_genes(sc_adata, min_counts=10)

    non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
    sc_adata = sc_adata[:, non_mito_genes_list]

    sc_adata.layers["counts"] = sc_adata.X.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e5)
    sc.pp.log1p(sc_adata)
    sc_adata.raw = sc_adata

    # 动态调整n_top_genes，避免基因数不足报错
    n_genes = min(7000, sc_adata.n_vars - 1)
    if n_genes > 0:
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes=n_genes,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span=1
        )
    else:
        print(f"Warning: Not enough genes for HVG selection, using all {sc_adata.n_vars} genes")
    if len(set(st_adata.var_names)) != len(st_adata.var_names):
        print("Removing duplicate genes from st_ad")
        mask = ~st_adata.var_names.duplicated()
        st_adata = st_adata[:, mask].copy()
    intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    st_adata = st_adata[:, intersect].copy()
    sc_adata = sc_adata[:, intersect].copy()

    # Filter empty cells in SC data after intersection to preserve numerical stability
    print(f"SC Cells before filtering: {sc_adata.n_obs}")
    sc.pp.filter_cells(sc_adata, min_counts=1)
    print(f"SC Cells after filtering zero-count cells: {sc_adata.n_obs}")

    # Filter empty cells in SC data after intersection to preserve numerical stability
    print(f"SC Cells before filtering: {sc_adata.n_obs}")
    sc.pp.filter_cells(sc_adata, min_counts=1)
    print(f"SC Cells after filtering zero-count cells: {sc_adata.n_obs}")

    RNAStereoscope.setup_anndata(sc_adata, layer="counts", labels_key=celltype_key)

    # Ensure counts are integers and handle NaNs
    if hasattr(sc_adata.layers['counts'], 'toarray'):
        counts_dense = sc_adata.layers['counts'].toarray()
        if np.isnan(counts_dense).any():
            print("NaNs found in SC data, replacing with 0")
            counts_dense = np.nan_to_num(counts_dense)
        counts_rounded = np.round(counts_dense).astype(int)
        sc_adata.layers['counts'] = csr_matrix(counts_rounded)
    else:
        if np.isnan(sc_adata.layers['counts']).any():
             print("NaNs found in SC data, replacing with 0")
             sc_adata.layers['counts'] = np.nan_to_num(sc_adata.layers['counts'])
        sc_adata.layers['counts'] = sc_adata.layers['counts'].round().astype(int)

    # Re-setup anndata to ensure changes are registered
    RNAStereoscope.setup_anndata(sc_adata, layer="counts", labels_key=celltype_key)

    stereo_sc_model = RNAStereoscope(sc_adata)
    stereo_sc_model.train(max_epochs=50, batch_size=1024)

    st_adata.layers["counts"] = st_adata.X.copy()

    if hasattr(st_adata.layers["counts"], 'toarray'):
        counts_dense1 = st_adata.layers['counts'].toarray()
        if np.isnan(counts_dense1).any():
            print("NaNs found in ST data, replacing with 0")
            counts_dense1 = np.nan_to_num(counts_dense1)
        counts_rounded1 = np.round(counts_dense1).astype(int)
        st_adata.layers['counts'] = csr_matrix(counts_rounded1)
    else:
        if np.isnan(st_adata.layers["counts"]).any():
            print("NaNs found in ST data, replacing with 0")
            st_adata.layers["counts"] = np.nan_to_num(st_adata.layers["counts"])
        st_adata.layers['counts'] = np.round(st_adata.layers["counts"]).astype(int)

    scvi.external.SpatialStereoscope.setup_anndata(st_adata, layer="counts")
    spatial_model = SpatialStereoscope.from_rna_model(st_adata, stereo_sc_model)
    spatial_model.train(max_epochs=350, batch_size=1024)
    spatial_model.get_proportions().to_csv(os.path.join(output_path, 'Stereoscope_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Stereoscope Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype')
