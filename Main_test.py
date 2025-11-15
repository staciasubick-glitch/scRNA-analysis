#this script runs through the preprocessing and clustering tutorial on Scanpy.io

# Core scverse libraries
from __future__ import annotations

import anndata as ad

# Data retrieval
import pooch
import scanpy as sc

sc.settings.set_figure_params(dpi=50, facecolor="white")

#Download sample data for tutorial
EXAMPLE_DATA = pooch.create(
    path=pooch.os_cache("scverse_tutorials"),
    base_url="doi:10.6084/m9.figshare.22716739.v1/",
)
EXAMPLE_DATA.load_registry_from_doi()

samples = {
    "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
    "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
}
adatas = {}

#read sample data into an AnnData object
#uses scanpy fxns to read in data and assign
#https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html
for sample_id, filename in samples.items():
    path = EXAMPLE_DATA.fetch(filename) #path to hdf5 file
    sample_adata = sc.read_10x_h5(path) #reads a 10x-genomics-formatted hdf5 file
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
print(adata.obs["sample"].value_counts())
adata

#define gene populations of focus to pass into qc_metrics fxn
#standard prefixes in the variable names allow us tosort for specific variables
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

#computes quality control metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

#generate violin plots for QCs of interest:
# "n_genes_by_counts": number of genes expressed in count matrix
# "total_counts": total counts per cell
# "pct_counts_mt": percentage of counts in mitochondrial genes
# https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html#scanpy.pp.calculate_qc_metrics
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)
#scatter plot colored by percentage of counts in mitochondrial genes
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

#filter out cells with too few genes expressed and genes expressed in too few cells
#can also filter out cells expressing too many mitochondrial genes at this point
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

#run doublet detection algorithm (doublets can cause problems in downstream analysis)
#scanpy fxn scrublet adds a "doublet_score" and "predicted_doublet" to .obs in data object
sc.pp.scrublet(adata, batch_key="sample")

#Normalization
# Saving count data
adata.layers["counts"] = adata.X.copy()
# Normalizing to median total counts
#size factor can be altered by changing variable target_sum in pp.normalize_total
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

#feature selection (only include most important genes)
#can change "flavor" to mimic seurat and cell ranger functionality. default is seurat
#https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
sc.pl.highly_variable_genes(adata)