import copy
import os
import sys
import numpy as np
import scipy.sparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from Baseline.scp.scpDeconv_main.model.DANN_model import *
from Baseline.scp.scpDeconv_main.model.utils import *
from Baseline.scp.scpDeconv_main.model.refer_mixup import ReferMixup
import anndata
from collections import Counter
import time


def main(sc_file, st_file, output_dir, cell_key='celltype'):
    start_time = time.time()
    
    # 读取数据
    sc_adata = anndata.read_h5ad(sc_file)
    st_adata = anndata.read_h5ad(st_file)
    
    # 获取细胞类型列表
    cell_types = list(set(sc_adata.obs[cell_key].tolist()))
    celltypenum = len(cell_types)
    print(f"Found {celltypenum} cell types: {cell_types}")
    
    # 输出目录
    outfile = os.path.join(output_dir, 'scpDeconv')
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    
    # 配置 ReferMixup 选项
    option_list = {
        'data_dir': '',
        'ref_dataset_name': os.path.abspath(sc_file),
        'ref_metadata_name': None,
        'target_dataset_name': os.path.abspath(st_file),
        'target_metadata_name': None,
        'random_type': cell_key,
        'type_list': cell_types,
        'ref_sample_num': 5000,
        'sample_size': 10,
        'HVP_num': 0,
        'target_type': 'real',
        'target_sample_num': 500,
        'SaveResultsDir': outfile,
    }
    
    # 使用 ReferMixup 生成模拟混合数据
    print("------Start Running Stage 1 & 2: Generating simulated mixtures------")
    mixup = ReferMixup(option_list)
    train_data, target_data = mixup.mixup()
    print(f"Generated training data shape: {train_data.shape}")
    print(f"Target data shape: {target_data.shape}")
    
    # 准备训练数据
    if scipy.sparse.issparse(train_data.X):
        sm_data = pd.DataFrame(data=train_data.X.toarray(), columns=train_data.var_names)
    else:
        sm_data = pd.DataFrame(data=train_data.X, columns=train_data.var_names)
    
    sm_label = train_data.obs[cell_types].copy()
    
    # 准备目标数据
    if scipy.sparse.issparse(target_data.X):
        st_data = pd.DataFrame(data=target_data.X.toarray(), columns=target_data.var_names, index=target_data.obs_names)
    else:
        st_data = pd.DataFrame(data=target_data.X, columns=target_data.var_names, index=target_data.obs_names)
    
    ### Run Stage 3 ###
    print("------Start Running Stage 3: Training DANN model------")
    model_da = DANN(celltype_num=celltypenum, outdirfile=outfile)
    print(f"st_data shape: {st_data.shape}, sm_data shape: {sm_data.shape}, sm_label shape: {sm_label.shape}")
    model_da.train(st_data=st_data, sm_data=sm_data, sm_label=sm_label)
    print("Stage 3: DANN model training finished!")

    ### Run Stage 4 ###
    print("------Start Running Stage 4: Inference for target data------")
    final_preds_target = model_da.prediction()
    final_preds_target.columns = cell_types
    # 设置正确的索引（使用原始ST数据的cell ID），确保为字符串类型
    final_preds_target.index = st_data.index.astype(str)
    final_preds_target.to_csv(os.path.join(outfile, 'scpDeconv_result.csv'))
    print("Stage 4: Inference for target data finished!")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"scpDeconv Total time taken: {total_time:.2f} seconds")


if __name__ == '__main__':
    sc_file = os.path.join(project_root, 'Datasets', 'scRNA_commmon_genes.h5ad')
    st_file = os.path.join(project_root, 'Datasets', 'Spatial_ST1_commmon_genes.h5ad')
    main(sc_file, st_file, 'Results', 'celltype')
