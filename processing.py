import numpy as np
import pandas as pd
import scanpy as sc

def create_barcode_map(cellBC, geneBC):
    BCmap = cellBC.copy()
    BCmap['SampleType']=""
    for row in BCmap.index:
        BCmap.loc[row,'SampleType'] = geneBC.loc[BCmap.loc[row]['multiseq barcode']]['sample']
    BCmap=BCmap.set_index('cell barcode')  
    
    return BCmap


def read_matrix(dr):
    adata = sc.read_10x_mtx(
    dr,  # the directory with the `.mtx` file
    var_names='gene_symbols',                  # use gene symbols for the variable names (variables-axis index)
    cache=True)
    
    return adata


def add_cell_type_to_obs(adata, BCmap):
    adata.obs_names=pd.Series(adata.obs_names).apply(lambda x: x.split('-')[0])
    cell_order=list(adata.obs_names)
    BCmap=BCmap.reindex(cell_order)
    adata.obs['SampleType']=BCmap['SampleType']
    adata.obs['quality']=BCmap['quality']
    
    return adata

def filter_minGenes_minCells(adata, min_genes, min_cells):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def calc_percentMito_nCounts(adata):
    mito_genes = adata.var_names.str.startswith('MT-')
    # for each cell compute fraction of counts in mito genes vs. all genes
    # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
    adata.obs['percent_mito'] = np.sum(
        adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    # add the total counts per cell as observations-annotation to adata
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    
    return adata


def keep_real_cells(adata):
    adata.obs.SampleType = adata.obs.SampleType.astype('str')
    adata = adata[adata.obs.SampleType != 'nan', :]
    return adata


def filter_genes_mito(adata, genes, mito):
    adata = adata[adata.obs.n_genes < genes, :]
    adata = adata[adata.obs.percent_mito < mito, :]
    
    return adata


def normalize_and_log(adata):
    raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    return adata, raw


def regress_scale_pca(adata):
    sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    return adata


def log_ranked_genes(adata):
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    rank_genes=pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'logfoldchanges']})
    dfs = []

    for i in range((len(rank_genes.columns))//2):
        dfs.append(pd.DataFrame())

    col_n=0
    col_l=2
    for i in range(0,len(dfs)):
        dfs[i] = rank_genes[rank_genes.columns[col_n:col_l]]
        col_n+=2
        col_l+=2
    for i in range(len(dfs)):
        dfs[i] = dfs[i].fillna(0)
        dfs[i]=dfs[i].sort_values(by=dfs[i].columns[1], ascending=False)
        dfs[i].reset_index(drop=True, inplace=True)
        
    log_ranks=pd.concat(dfs, axis=1)
    
    return log_ranks
