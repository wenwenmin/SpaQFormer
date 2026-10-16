# SpaQFormer: Prototype-Initialized Query Matching for Cell-Type Annotation and Gene Attribution in Spatial Transcriptomics

## Overview

Accurate cell-type annotation in cell-level spatial transcriptomics (ST) is essential for resolving tissue organization and cellular interactions. Annotated single-cell RNA sequencing (scRNA-seq) data provide a natural reference, but differences in gene coverage, sparsity, measurement noise, tissue context, and cellular state make direct knowledge transfer difficult. Existing deconvolution, mapping, and classifier-based approaches mainly rely on aggregate abundance, global correspondence, or decision functions shared across target cells, which can be unreliable when expression evidence is ambiguous.

**SpaQFormer** is a reference-guided framework that reformulates spatial cell annotation as **prototype-initialized cell-type query matching**. A reference model first learns discriminative representations from annotated scRNA-seq data. The resulting class prototypes initialize learnable cell-type queries, while the pretrained reference model selects high-confidence pseudo-labels from the target ST data. For every target cell and candidate type, SpaQFormer then uses cross-attention to retrieve and combine cell-intrinsic expression, graph-derived spatial expression, and explicit neighborhood evidence. This produces a cell-specific matching score instead of applying a static class vector to every target cell.

The matching scores also provide differentiable targets for cell-type-level gene attribution. SpaQFormer combines Integrated Gradients, GradientShap, and Occlusion through Borda rank fusion to identify genes that support each predicted cell type. The supplied experiments cover seven paired scRNA-seq and cell-level ST datasets generated with CosMx SMI, Xenium, and MERFISH.

## Figure 1

![Figure 1: Overview of SpaQFormer](fig1.png)

Figure 1 summarizes the complete framework:

- **A. Reference pretraining:** an autoencoder and classifier learn cell-type-related latent representations from annotated scRNA-seq data, and class prototypes are computed for query initialization.
- **B. Pseudo-label selection:** the frozen reference model predicts target cells and retains the lowest-entropy fraction within each predicted cell type.
- **C. Query-guided annotation:** expression, graph-derived spatial expression, and neighborhood context are encoded as complementary evidence for every target cell.
- **D. QFormer matching:** prototype-initialized cell-type queries retrieve cell-specific evidence through multi-head cross-attention and produce candidate-type matching scores.
- **E. Gene attribution:** IG, GS, and OC attribute cell-type matching scores to input genes, followed by Borda rank fusion.
- **F. Downstream analysis:** the resulting annotations and gene rankings support spatial visualization and biological interpretation.

## Installation

The current experiments run in the Conda environment `SpaQFormer` with Python 3.10, PyTorch 2.5.1, CUDA 12.1, and PyTorch Geometric 2.6.1.

### 1. Create the virtual environment

```bash
conda create --name SpaQFormer python=3.10
```

### 2. Activate the environment

```bash
conda activate SpaQFormer
```

### 3. Install dependencies

Run the command from the repository root:

```bash
pip install -r requirements.txt
```

The supplied `requirements.txt` uses the CUDA 12.1 wheels matching the current `SpaQFormer` environment. For a CPU-only system or a different CUDA version, replace the PyTorch and PyTorch Geometric wheel sources with versions compatible with that system.

## Repository structure

```text
SpaQFormer/
├── Config/                  # Dataset-specific best-parameter YAML files
├── SpaQFormer/              # Model, training, preprocessing, evaluation, and explanation code
├── Baseline/                # Baseline methods for comparison
├── Tutorial/                # Jupyter notebooks for tutorials
├── requirements.txt         # Dependencies for the SpaQFormer environment
└── README.md
```

The `logs/` directory is created by the notebooks at runtime.

## Input data format

SpaQFormer requires two AnnData (`.h5ad`) files:

- **Annotated scRNA-seq reference**
  - expression matrix in `.X`;
  - cell-type labels in `.obs["label"]`.
- **Target spatial transcriptomics data**
  - expression matrix in `.X`;
  - spatial coordinates in `.obs["x"]` and `.obs["y"]`.

The reference and target matrices must contain the same genes in the same order. SpaQFormer performs cell-wise z-score normalization separately for the two datasets and constructs a spatial K-nearest-neighbor graph from the target coordinates. For multi-slice data, `slice_col` can be supplied to prevent edges from being constructed between slices.

Ground-truth columns such as `.obs["label"]` or `.obs["label_aligned"]` are used only by the evaluation cells in the notebooks. Dataset-specific fields such as `fov`, `slice_id`, or `Bregma` are used only for visualization or per-slice graph construction.

## NSCLC tutorial

The Human NSCLC Lung9_1 dataset is used as the main running example.

### 1. Download the data

Download the paired **Human NSCLC (Lung9_1)** reference and target datasets from [Zenodo](https://zenodo.org/records/23390067).

Place the two files in a local data directory:

```text
data/
├── NSCLC_9_1_reference.h5ad
└── NSCLC_9_1_target.h5ad
```

### 2. Set the data paths

Open [`Tutorial/Run_NSCLC.ipynb`](Tutorial/Run_NSCLC.ipynb) and update the two paths in the **Dataset paths and configuration** cell:

```python
SC_FILE = Path("path/to/data/NSCLC_9_1_reference.h5ad")
SRT_FILE = Path("path/to/data/NSCLC_9_1_target.h5ad")
CONFIG_FILE = Path("Config") / "Config_NSCLC.yaml"
```

### 3. Run the notebook

Start Jupyter from the repository root:

```bash
jupyter notebook Tutorial/Run_NSCLC.ipynb
```

The notebooks set their working directory to the repository root so that package imports, configuration paths, and output paths resolve correctly.

Execute the cells from top to bottom. The notebook performs:

1. configuration loading and SpaQFormer initialization;
2. reference representation pretraining;
3. target pseudo-label selection;
4. reference prototype construction and query initialization;
5. target annotation;
6. result export, metric reporting, and spatial visualization;
7. cell-type-level gene attribution.

The corresponding API sequence is:

```python
from pathlib import Path

import torch
import yaml

from SpaQFormer.trainer import SpaQFormer

with Path("Config/Config_NSCLC.yaml").open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpaQFormer(
    sc_file="path/to/data/NSCLC_9_1_reference.h5ad",
    srt_file="path/to/data/NSCLC_9_1_target.h5ad",
    device=device,
    **config,
)

model.pretrain()
model.pseudo_label()
model.build_prototypes()
annotation = model.annotate(
    use_cluster=True,
    num_parts=10,
    cluster_batch_size=1,
    open_set=False,
)
```

## Cell-type gene attribution example

Run attribution after `model.annotate(...)`, while the trained `model` remains in memory. The following NSCLC example computes cell-type-level gene importance with Integrated Gradients (`ig`), GradientShap (`gs`), and Occlusion (`oc`), and saves the complete gene rankings and the top 20 genes for every reference cell type:

```python
import pandas as pd

from SpaQFormer import SpaQFormerExplainer


explain_output_dir = Path("logs/spaqformer_nsclc_best/explain")
explain_output_dir.mkdir(parents=True, exist_ok=True)

explainer = SpaQFormerExplainer(
    model,
    num_parts=50,
)

all_results = explainer.explain_all_cell_types(
    max_cells=50,
    methods=("ig", "gs", "oc"),
    top_k=20,
    ig_steps=16,
    gs_samples=8,
    perturbations_per_eval=16,
)

all_gene_score_tables = []
all_top_gene_tables = []

for cell_type, explanation in all_results.items():
    gene_scores = explanation["gene_scores"].copy()
    gene_scores.insert(0, "cell_type", cell_type)
    all_gene_score_tables.append(gene_scores)

    top_genes = explanation["top_genes"].copy()
    top_genes.insert(0, "cell_type", cell_type)
    all_top_gene_tables.append(top_genes)

all_gene_scores_df = pd.concat(all_gene_score_tables, ignore_index=True)
all_top_genes_df = pd.concat(all_top_gene_tables, ignore_index=True)
```

The main attribution parameters are:

- `max_cells`: maximum number of high-confidence predicted cells explained for each cell type;
- `methods`: attribution methods included in the Borda rank fusion;
- `top_k`: number of genes retained in each cell type's top-gene table;
- `ig_steps`: number of Integrated Gradients approximation steps;
- `gs_samples`: number of GradientShap samples;
- `perturbations_per_eval`: number of Occlusion perturbations evaluated together.

Attribution keeps the trained spatial graph fixed and explains the target cell-type matching score with respect to the input genes. Ground-truth labels are not required to generate these gene rankings.

## Outputs

The NSCLC notebook saves the main result to:

```text
logs/spaqformer_nsclc_best/spaqformer_nsclc_best_results.h5ad
```

This result-only AnnData file contains:

- the original target `.obs` table;
- `.obs["SpaQFormer_annotation"]`, containing the closed-set cell-type prediction;
- `.obsm["X_SpaQFormer"]`, containing the learned target embedding.

The attribution stage saves two files under `logs/spaqformer_nsclc_best/explain/`:

```text
SpaQFormer_NSCLC_all_cell_types_all_gene_scores.csv
SpaQFormer_NSCLC_all_cell_types_top20_genes.csv
```

The remaining dataset notebooks use the same workflow and place results in dataset- and target-specific subdirectories under `logs/`.

## Notes

- Reference pretraining is reused across targets in notebooks that define multiple target items.
- METIS-partitioned subgraph training is enabled for large spatial graphs.
- Cell-type pairs are processed in chunks during cross-attention; if GPU memory is limited, reduce `qformer_chunk_size` before changing the software environment.


## Contact details

If you have any questions, please contact huangxu_bzcd@stu.ynu.edu.cn and minwenwen@yun.edu.cn.
