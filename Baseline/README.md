# Baseline methods

This directory contains 13 baseline methods. The existing source code and directory structure have been preserved. No methods have been run, and no algorithm code has been modified.

## Method list

| Method | Directory / entry point | Description |
| --- | --- | --- |
| Cell2location | `Cell2location/main_code.py` | Estimates spatial cell-type abundances using reference expression signatures. Requires the external `cell2location` package. |
| DestVI | `DestVi/main_code.py` | Performs spatial transcriptomics deconvolution using variational models, specifically CondSCVI / DestVI from `scvi-tools`. |
| DSCT | `DSCT-master/main_code.py` | Transfers reference cell-type annotations to spatial data. Model and data-processing code is located in `src/`. |
| novoSpaRc | `novoSpaRc/main.py` | Performs expression and spatial mapping using optimal transport. Requires the external `novosparc` package. |
| scpDeconv | `scp/scpDeconv_main/main.py` | Implements cell-type deconvolution with reference data mixing, expression imputation, and domain adaptation models. |
| SpaDAMA | `SpaDAMA/main_code.py` | Implements cell-type annotation with simulated spatial data, data augmentation, and domain adaptation models. |
| SpaOTsc | `SpaOTsc/main_code.py` | Connects single-cell expression with spatial data using optimal transport. The core source code is located in `spaotsc/`. |
| Spoint | `Spiont/main_code.py` | Implements cell-type mapping with data simulation, augmentation, and model training. The original directory name, `Spiont`, is retained. |
| Stereoscope | `Stereoscope/main_code.py` | Estimates spatial cell-type proportions using reference single-cell data and the Stereoscope models from `scvi-tools`. |
| Tangram | `Tangram/main_code.py` | Maps reference single cells to spatial locations and projects cell-type labels onto them. Requires the external `tangram` package. |
| TACCO | `TACCO/main_code.py` | Transfers annotations using the external `tacco` package. The current file contains only an example call to `tc.tl.annotate`. |
| TOSICA | `TOSICA/main_code.py` | Performs cell-type annotation using a gene-set-constrained Transformer. Source code is located in the `TOSICA/` subdirectory; gene-set resources are not included. |
| scDOT | `scDOT/main_code.py` | Performs cell-type mapping using optimal transport. The core implementation is in `scDOT.py` and `utils.py`. |

## Usage notes

Some methods require third-party packages to be installed separately via pip or conda. This directory is not a complete source mirror of all third-party libraries. Dependencies have not been reinstalled or verified.

Most adapted entry points accept reference and spatial AnnData file paths (`sc_file`, `st_file`), an output directory (`output_dir`), and a cell-type column name (`cell_key`). Refer to each file for its exact parameters. For datasets commonly used in this project, the cell-type column is typically `label`, and the spatial coordinate columns are `x` and `y`. Some entry points still use `celltype` as the default column name.

Use this code from the project root, `C:\Code\SpaQFormer`, so that `Baseline.*` imports resolve correctly. DSCT, TOSICA, and scDOT have additional local module import requirements; follow their respective entry-point files.
