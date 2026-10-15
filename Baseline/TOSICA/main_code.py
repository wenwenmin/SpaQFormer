import hashlib
import os
import sys
import pandas as pd
import scanpy as sc

def main(sc_file, st_file, output_dir, cell_key='celltype', seed=None, smoke=False):
    # 将当前的 TOSICA 目录加入到 sys.path 以便导入 TOSICA 模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    import TOSICA
    
    output_file_path = os.path.join(output_dir, 'TOSICA')
    os.makedirs(output_file_path, exist_ok=True)
    
    # 加载数据（由于外部已经对齐，直接读取即可）
    ref_adata = sc.read_h5ad(sc_file)
    query_adata = sc.read_h5ad(st_file)

    # TOSICA expects library-size-normalized, log-transformed expression.
    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)
    sc.pp.normalize_total(query_adata, target_sum=1e4)
    sc.pp.log1p(query_adata)
    
    # Cache reference-only training artifacts across slices of the same dataset.
    epochs = 30 if smoke else 10
    project_name = 'tosica_proj'
    gmt_path = 'mouse_gobp' if 'mouse' in str(sc_file).lower() else 'human_gobp'
    stat = os.stat(sc_file)
    signature = f"{os.path.abspath(sc_file)}|{stat.st_size}|{stat.st_mtime_ns}|{cell_key}|{epochs}|{gmt_path}"
    cache_key = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:12]
    cache_root = os.path.join(os.path.dirname(output_dir), '_shared', 'TOSICA', cache_key)
    os.makedirs(cache_root, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(cache_root)
    try:
        model_weight_path = f'./{project_name}/model-{epochs-1}.pth'
        if not os.path.isfile(model_weight_path):
            print(f"Training shared TOSICA reference model in: {cache_root}")
            TOSICA.train(
                ref_adata,
                gmt_path=gmt_path,
                label_name=cell_key,
                epochs=epochs,
                project=project_name,
                seed=seed,
            )
        else:
            print(f"Reusing shared TOSICA reference model: {model_weight_path}")

        new_adata = TOSICA.pre(
            query_adata,
            model_weight_path=model_weight_path,
            project=project_name,
            cutoff=0.0,
        )

        import shutil
        weight_csv = f'./{project_name}/gene2token_weights.csv'
        if os.path.exists(weight_csv):
            shutil.copy(weight_csv, os.path.join(output_file_path, 'gene2token_weights.csv'))
    finally:
        os.chdir(original_cwd)
            
    # 从返回的 adata 提取结果并保存，直接输出预测标签和概率
    result_df = pd.DataFrame(index=new_adata.obs_names)
    if 'Prediction' in new_adata.obs.columns:
        result_df['Prediction'] = new_adata.obs['Prediction']
    if 'Probability' in new_adata.obs.columns:
        result_df['Probability'] = new_adata.obs['Probability']
            
    result_df.to_csv(os.path.join(output_file_path, 'TOSICA_result.csv'))
