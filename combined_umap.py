# Core scverse libraries
from __future__ import annotations

import anndata as ad

# Data retrieval
import pooch
import scanpy as sc
import muon as mu
from pathlib import Path
import harmonypy

print(sc.__version__)
print(harmonypy.__version__)

# directory containing your local h5 files
DATA_DIR = Path(r"C:\RNAseq data\\")

#preset figure formatting
sc.settings.set_figure_params(dpi=50, facecolor="white")

#read data files
mdata1=mu.read_10x_h5(DATA_DIR / "SC_1_15_outs\SC_1_15_filtered_feature_bc_matrix.h5")
mdata2=mu.read_10x_h5(DATA_DIR / "SC_2_16_outs\SC_2_16_filtered_feature_bc_matrix.h5")

#make variable names unique
mdata1.mod['rna'].var_names_make_unique()
mdata1.mod['atac'].var_names_make_unique()

mdata2.mod['rna'].var_names_make_unique()
mdata2.mod['atac'].var_names_make_unique()

#pull out rna data into anndata objects
rna_1=mdata1.mod['rna']
rna_2=mdata2.mod['rna']


# basic filtering
sc.pp.filter_cells(rna_1, min_genes=100)
sc.pp.filter_genes(rna_1, min_cells=3)
sc.pp.filter_cells(rna_2, min_genes=100)
sc.pp.filter_genes(rna_2, min_cells=3)

rna_data=[rna_1, rna_2]
combined_data=ad.concat(rna_data, label='sample',keys=["SC_1_15", "SC_2_16"],
                        join='outer', fill_value=0)

# normalization and log transform
sc.pp.normalize_total(combined_data, target_sum=1e4)
sc.pp.log1p(combined_data)

sc.pp.highly_variable_genes(combined_data, n_top_genes=2000, batch_key="sample")

# Subset to HVGs
combined_data = combined_data[:, combined_data.var.highly_variable].copy()

#scale and run PCA
sc.pp.scale(combined_data, max_value=10)
sc.tl.pca(combined_data, n_comps=50, svd_solver="arpack")

# Sanity check (IMPORTANT)
print(combined_data.obsm.keys())
# should print: (n_cells, 50)

#run batch correction
sc.external.pp.harmony_integrate(combined_data, key='sample')

sc.pp.neighbors(combined_data, use_rep="X_pca_harmony", n_neighbors=15)
sc.tl.umap(combined_data)
sc.tl.leiden(combined_data, flavor="igraph") #cluster using leiden algorithm
sc.pl.umap(combined_data, color=['leiden', 'sample']) #generate UMAP with clusters