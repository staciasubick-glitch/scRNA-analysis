# Core scverse libraries
from __future__ import annotations

import anndata as ad

# Data retrieval
import pooch
import scanpy as sc
import muon as mu
from pathlib import Path
from matplotlib.text import Annotation

# Import data from local computer:

# directory containing your local h5 files
DATA_DIR = Path(r"C:\RNAseq data\SC_2_16_outs")
SAMPLE_ID = "SC_2_16"

#preset figure formatting
sc.settings.set_figure_params(dpi=50, facecolor="white")
#set directory to save figures to
sc.settings.figdir = r"C:\RNAseq data\scanpy_analysis_files\\" + SAMPLE_ID + "_analysis"

mdata=mu.read_10x_h5(DATA_DIR / (SAMPLE_ID + "_filtered_feature_bc_matrix.h5"))
mdata.mod['rna'].var_names_make_unique()
mdata.mod['atac'].var_names_make_unique()
print(mdata)

# split data into rna and atac objects for separate analysis first
#rna object
rna_data=mdata.mod['rna']

# separate out mitochondrial genes for qc metrics
rna_data.var['mt']=rna_data.var_names.str.upper().str.startswith('MT-')
sc.pp.calculate_qc_metrics(rna_data, qc_vars=['mt'],inplace=True)

#violin plot for rna data
sc.pl.violin(
    rna_data,
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    jitter=0.4,
    multi_panel=True,
    save='_' + SAMPLE_ID +'_RNA.png'
)

#filter out cells with too few genes expressed and genes expressed in too few cells
#can also filter out cells expressing too many mitochondrial genes at this point
sc.pp.filter_cells(rna_data, min_genes=100)
sc.pp.filter_genes(rna_data, min_cells=3)

#run doublet detection algorithm (doublets are when barcodes tag multiple cells by accident. can cause problems in downstream analysis)
#scanpy fxn scrublet adds a "doublet_score" and "predicted_doublet" to .obs in data object
#sc.pp.scrublet(rna_data, batch_key="sample")

#Normalization
# Saving count data
rna_data.layers["counts"] = rna_data.X.copy()
# Normalizing to median total counts
#size factor can be altered by changing variable target_sum in pp.normalize_total
sc.pp.normalize_total(rna_data)
# Logarithmize the data
sc.pp.log1p(rna_data)

#feature selection (only include most important genes)
#can change "flavor" to mimic seurat and cell ranger functionality. default is seurat
#https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html
sc.pp.highly_variable_genes(rna_data, n_top_genes=2000)
sc.pl.highly_variable_genes(rna_data, save='_' + SAMPLE_ID + '_RNA.png')

#use PCA to reduce dimensionality of data
sc.tl.pca(rna_data)
#display contributions of principal components to variance
sc.pl.pca_variance_ratio(rna_data, n_pcs=50, log=True, save= '_' + SAMPLE_ID +'_RNA.png')
#sc.tl.leiden(rna_data)

sc.pp.neighbors(rna_data)
sc.tl.umap(rna_data)
sc.tl.leiden(rna_data, flavor="igraph") #cluster using leiden algorithm
sc.pl.umap(rna_data, color='leiden', save='_' + SAMPLE_ID +'_RNA_leiden.png', title=SAMPLE_ID + ' GEX') #generate UMAP with clusters

# --------------
# ATAC ANALYSIS
# --------------

#atac object
atac_data=mdata.mod['atac']

#data reduction  steps for atac
mu.atac.pp.tfidf(atac_data)
mu.atac.tl.lsi(atac_data)

sc.pp.neighbors(atac_data, use_rep='X_lsi')
sc.tl.leiden(atac_data)
sc.tl.umap(atac_data)
sc.pl.umap(atac_data, color='leiden', save='_' + SAMPLE_ID +'_ATAC_leiden.png', title=SAMPLE_ID + ' ATAC')
