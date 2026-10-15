import os
import sys
import tempfile
import pandas as pd
import numpy as np
import scanpy as sc

def main(sc_file, st_file, output_dir, cell_key='celltype', seed=None):
    """
    scDOT主函数
    
    scDOT uses deterministic optimal transport on the complete reference data.
    
    参数:
        seed: Random seed for deterministic execution.
    """
    import random
    import numpy as np
    
    # 设置随机种子用于采样
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 将当前的 scDOT 目录加入到 sys.path 以便导入 scDOT 方法
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 从你的本地脚本导入该函数
    from scDOT import scDOT
    
    output_file_path = os.path.join(output_dir, 'scDOT')
    os.makedirs(output_file_path, exist_ok=True)
    
    # 按照外部对齐的承诺原样加载数据
    ref_adata = sc.read_h5ad(sc_file)
    query_adata = sc.read_h5ad(st_file)
    
    print(f"  使用完整参考集: {ref_adata.n_obs} cells")
    
    # 隔离落地文件，运行完毕自动销毁
    with tempfile.TemporaryDirectory(prefix='scdot_tmp_') as tmp_dir:
        # scDOT 要求传入的 loc 字符串拼接时要有后缀斜杠，Windows 和 Mac 要通用可以强加 '/'
        loc = tmp_dir.replace('\\', '/') + '/'
        
        # 将 scipy 稀疏矩阵转换，给 Pandas 使用
        def to_dense(mat):
            import scipy
            if isinstance(mat, scipy.sparse.csr_matrix) or isinstance(mat, scipy.sparse.csc_matrix):
                return mat.todense()
            return mat
            
        print("正在将参考集转录为指定的 CSV...")
        ref_X = to_dense(ref_adata.X)
        df_ref_cell = pd.DataFrame(ref_X, index=ref_adata.obs_names, columns=ref_adata.var_names)
        df_ref_cell.to_csv(loc + "ref_cell.csv")
        
        # 写入真实标签 (scDOT 读取第一个值，而且包含 index)
        df_ref_label = pd.DataFrame({cell_key: ref_adata.obs[cell_key].values}, index=ref_adata.obs_names)
        df_ref_label.to_csv(loc + "ref_label.csv")
        
        print("正在将查询集转录为指定的 CSV...")
        query_X = to_dense(query_adata.X)
        df_query_cell = pd.DataFrame(query_X, index=query_adata.obs_names, columns=query_adata.var_names)
        df_query_cell.to_csv(loc + "query_cell.csv")
        
        print("执行 scDOT 最优传输算法计算推断...")
        # 调用 scDOT 寻找结果，不关心 OT 底层逻辑
        y_hat, score, y_hat_with_unseen = scDOT(
            loc=loc, 
            ref_name=['ref'],   # 我们刚生成的参考集前缀就叫 ref
            query_name='query'  # 查询集叫 query
        )
        
    print("获取并保存推断结果...")
    # 保存结果
    result_df = pd.DataFrame(index=query_adata.obs_names)
    result_df['Prediction'] = y_hat
    result_df['Probability'] = score
    result_df['Prediction_with_unseen'] = y_hat_with_unseen
    
    result_df.to_csv(os.path.join(output_file_path, 'scDOT_result.csv'))
    print("scDOT 评估与生成保存完毕。")
