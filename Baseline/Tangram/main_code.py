"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/Tangram_pipeline.py#L9"""
import copy
import os

import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import tangram as tg
import time


def main(sc_file, st_file, output_dir, cell_key='celltype', seed=None):
    import torch
    import random
    import numpy as np
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 注意: 设置deterministic可能会降低性能
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    start_time = time.time()
    
    output_file_path = os.path.join(output_dir, 'Tangram')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    celltype_key = cell_key
    ad_sc1 = sc.read_h5ad(sc_file)
    ad_sp1 = sc.read_h5ad(st_file)
    ad_sc = copy.deepcopy(ad_sc1)
    ad_sp = copy.deepcopy(ad_sp1)
    
    # use raw count both of scrna and spatial
    sc.pp.normalize_total(ad_sc)
    celltype_counts = ad_sc.obs[celltype_key].value_counts()
    celltype_drop = celltype_counts.index[celltype_counts < 2]
    print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
    ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()
    sc.tl.rank_genes_groups(ad_sc, groupby=celltype_key, use_raw=False)
    n_markers = min(300, len(ad_sc.uns["rank_genes_groups"]["names"]))
    markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:n_markers, :]
    genes_sc = np.unique(markers_df.melt().value.values)
    genes_st = ad_sp.var_names.values
    genes = list(set(genes_sc).intersection(set(genes_st)))
    tg.pp_adatas(ad_sc, ad_sp, genes=genes)
    ad_map = tg.map_cells_to_space(
        ad_sc,
        ad_sp,
        num_epochs=500,
        mode='clusters',
        cluster_label=celltype_key,
        device='cuda:0' if torch.cuda.is_available() else 'cpu')
    tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
    celltype_density = ad_sp.obsm['tangram_ct_pred']
    celltype_density = (celltype_density.T / celltype_density.sum(axis=1)).T

    celltype_density.to_csv(os.path.join(output_file_path, 'Tangram_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tangram Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    main('Datasets/scRNA_commmon_genes.h5ad', 'Datasets/Spatial_ST1_commmon_genes.h5ad', 'Results', 'celltype')
