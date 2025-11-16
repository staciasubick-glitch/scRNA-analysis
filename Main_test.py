#this script runs through the preprocessing and clustering tutorial on Scanpy.io

# Core scverse libraries
from __future__ import annotations

import anndata as ad

# Data retrieval
import pooch
import scanpy as sc
from matplotlib.text import Annotation

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

#use PCA to reduce dimensionality of data
sc.tl.pca(adata)
#display contributions of principal components to variance
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
#plot individual PCs
sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)

#nearest neighbors distance matrix of cells using PCA representation
#plot as UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=2,
)
#if batch effects are observed in UMAP can use scanorama or scvi-tools for batch integration

#clustering
#Scanpy recommends using Leiden graph-clustering method similar to Seurat
# Using the igraph implementation and a fixed number of iterations can be significantly faster,
# especially for larger datasets
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
sc.pl.umap(adata, color=["leiden"])

#check filtering by looking at QC metrics using UMAPS
sc.pl.umap(
    adata,
    color=["leiden", "predicted_doublet", "doublet_score"],
    # increase horizontal space between panels
    wspace=0.5,
    size=3,
)
sc.pl.umap(
    adata,
    color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
)

#Manual Cell-Type Annotation
#Cluster cells and annotate marker genes that correspond to different cell types
#use Leiden clustering alg to extract cell groups from nearest neighbors graph (created previously)
for res in [0.02, 0.5, 2.0]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")

#create UMAP
#we can examine these plots to determine which resolution factor to use
#in this example, 2 is over-clustered
sc.pl.umap(
    adata,
    color=["leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00"],
    legend_loc="on data",
)

#define a set of marker genes for main cell types were expecting
#this set is adapted from "Single Cell Best Practices"
#https://www.sc-best-practices.org/cellular_structure/annotation.html
marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    # Note: DMXL2 should be negative
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    # Note HBM and GYPA are negative markers
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    # Note IGHD and IGHM are negative markers
    "B cells": [
        "MS4A1",
        "ITGB1",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
    ],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    # Note PAX5 is a negative marker
    "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
}
#plot patterns of expression for each marker
sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.02", standard_scale="var")

#label broad clusters
adata.obs["cell_type_lvl1"] = adata.obs["leiden_res_0.02"].map(
    {
        "0": "Lymphocytes",
        "1": "Monocytes",
        "2": "Erythroid",
        "3": "B Cells",
    }
)

#lets repeat with higher resolution
sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.50", standard_scale="var")