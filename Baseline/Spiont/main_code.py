"""读取数据"""
import copy
import multiprocessing as mp
import anndata as ad
import torch
import os
import sys
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from Baseline.Spiont.model import init_model
import Baseline.Spiont.data_utils

import torch
print("cuda is available:    ",torch.cuda.is_available())


def main(sc_file, st_file, output_dir, cell_key='celltype', smoke=False):
    start_time = time.time()
    
    st_ad1 = ad.read_h5ad(st_file)
    sc_data1 = ad.read_h5ad(sc_file)
    st_ad = copy.deepcopy(st_ad1)
    sc_data = copy.deepcopy(sc_data1)
    
    output_path = os.path.join(output_dir, 'Spiont')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    model = init_model(sc_ad=sc_data,
                       st_ad=st_ad,
                       celltype_key=cell_key,
                       n_top_markers=500,
                       n_top_hvg=2500,
                       sm_size=2000 if smoke else 10000,
                       use_gpu=True,
                       )
    
    model.model_train(
        sm_lr=0.01,
        st_lr=0.01,
        max_steps=100 if smoke else 3000,
        scvi_max_epochs=10 if smoke else 100,
    )
    pre = model.deconv_spatial()
    full_predictions = pre.reindex(st_ad1.obs_names.astype(str))
    prior = sc_data1.obs[cell_key].astype(str).value_counts(normalize=True)
    prior = prior.reindex(full_predictions.columns, fill_value=0.0)
    if prior.sum() == 0:
        prior[:] = 1.0 / len(prior)
    else:
        prior /= prior.sum()
    missing = full_predictions.isna().all(axis=1)
    full_predictions.loc[missing, :] = prior.to_numpy()
    full_predictions.to_csv(os.path.join(output_path, 'Spiont_result.csv'))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Spiont Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    sc_file = os.path.join(project_root, 'Datasets', 'scRNA_commmon_genes.h5ad')
    st_file = os.path.join(project_root, 'Datasets', 'Spatial_ST1_commmon_genes.h5ad')
    main(sc_file, st_file, 'Results', 'celltype')
