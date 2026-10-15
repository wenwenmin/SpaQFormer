import os
import warnings
import TOSICA
import scanpy as sc
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

def main():
    print("--- 1. Loading Data ---")
    ref_adata = sc.read('test/demo_train.h5ad')
    ref_adata = ref_adata[:, ref_adata.var_names]
    

    query_adata = sc.read('test/demo_test.h5ad')
    query_adata = query_adata[:, ref_adata.var_names]
    
    print("--- 2. Training Model ---")
    project_name = 'hGOBP_demo_script'
    TOSICA.train(
        ref_adata, 
        gmt_path='human_gobp',
        label_name='Celltype',
        epochs=3,
        project=project_name
    )

    print("--- 3. Predicting ---")
    model_weight_path = f'./{project_name}/model-0.pth'
    new_adata = TOSICA.pre(
        query_adata, 
        model_weight_path=model_weight_path, 
        project=project_name
    )
    
    output_file = 'demo_attn_script.h5ad'
    new_adata.write(output_file)

if __name__ == '__main__':
    main()
