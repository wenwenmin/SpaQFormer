from collections import Counter
import os
import sys
import copy
import time

import anndata
import anndata as ad
import pandas as pd

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from Baseline.SpaDAMA.SpaDAMA_model import SpaDAMA
from Baseline.SpaDAMA.data_prepare import *


def main(sc_file, st_file, output_dir, cell_key='celltype', seed=None, smoke=False):
    start_time = time.time()
    resolved_seed = 42 if seed is None else int(seed)
    
    st_data1 = ad.read_h5ad(st_file)
    sc_data1 = ad.read_h5ad(sc_file)
    sc_data = copy.deepcopy(sc_data1)
    st_data = copy.deepcopy(st_data1)
    
    outfile = os.path.join(output_dir, 'SpaDAMA')
    datafile = os.path.join(outfile, 'preproced_data')
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    if not os.path.exists(datafile):
        os.makedirs(datafile)

    """数据处理"""
    data_prepare(sc_ad=sc_data, st_ad=st_data, celltype_key=cell_key,
                 h5ad_file_path=outfile, data_file_path=datafile,
                 n_layers=2, n_latent=128 if smoke else 2048,
                 sm_size=5000 if smoke else 20000,
                 scvi_max_epochs=50 if smoke else 500,
                 seed=resolved_seed)

    sc_adata = anndata.read_h5ad(os.path.join(datafile, 'sm_scvi_ad.h5ad'))
    st_adata = anndata.read_h5ad(os.path.join(datafile, 'st_scvi_ad.h5ad'))
    real_sc_adata = anndata.read_h5ad(os.path.join(datafile, 'Scdata_filter.h5ad'))
    sm_labelad = anndata.read_h5ad(os.path.join(datafile, 'Sm_STdata_filter.h5ad'))

    sm_data = pd.DataFrame(data=sc_adata.X, columns=sc_adata.var_names)
    sm_label = sm_labelad.obsm['label']
    st_data = pd.DataFrame(data=st_adata.X, columns=st_adata.var_names)
    
    count_ct_dict = Counter(list(real_sc_adata.obs[cell_key]))
    celltypenum = len(count_ct_dict)
    
    print("------Start Running Stage------")
    model_da = SpaDAMA(celltypenum, outdirfile=outfile,
                    used_features=list(sm_data.columns), num_epochs=100 if smoke else 300,
                    seed=resolved_seed)
    model_da.double_train(sm_data=sm_data, st_data=st_data, sm_label=sm_label, mask_retio=0.2)
    
    final_preds_target = model_da.prediction()
    final_preds_target.columns = sm_label.columns.tolist()
    final_preds_target.index = st_adata.obs_names.astype(str)
    final_preds_target.to_csv(os.path.join(outfile, 'final_pro1.csv'))
    full_predictions = final_preds_target.reindex(st_data1.obs_names.astype(str))
    prior = sc_data1.obs[cell_key].astype(str).value_counts(normalize=True)
    prior = prior.reindex(full_predictions.columns, fill_value=0.0)
    if prior.sum() == 0:
        prior[:] = 1.0 / len(prior)
    else:
        prior /= prior.sum()
    missing = full_predictions.isna().all(axis=1)
    full_predictions.loc[missing, :] = prior.to_numpy()
    full_predictions.to_csv(os.path.join(outfile, 'SpaDAMA_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"SpaDAMA Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    sc_file = os.path.join(project_root, 'Datasets', 'scRNA_commmon_genes.h5ad')
    st_file = os.path.join(project_root, 'Datasets', 'Spatial_ST1_commmon_genes.h5ad')
    main(sc_file, st_file, 'Results', 'celltype')
