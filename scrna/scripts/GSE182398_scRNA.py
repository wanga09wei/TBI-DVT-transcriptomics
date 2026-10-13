# -*- coding: utf-8 -*-
"""
GSE182398 scRNA 管线（人 TBI 切除脑组织 CD45-CD31+ 内皮细胞）
6 样本 = 3 患者 (R1/R2/R3) x (injury center / injury peripheral)
10x v3 三件套，33,538 基因。QC + 归一化 + PCA/UMAP/Leiden + 内皮标记评分。
输出：scrna/output/GSE182398_*.h5ad/png/csv（300 dpi，英文标注）
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scanpy as sc
import anndata as ad
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRNA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SCRNA, 'data', 'GSE182398')
OUT = os.path.join(SCRNA, 'output'); os.makedirs(OUT, exist_ok=True)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=110, facecolor='white', format='png')

# ---------- 1) 读取 6 样本 ----------
adatas = []
for d in sorted(os.listdir(DATA)):
    p = os.path.join(DATA, d)
    if not os.path.isdir(p): continue
    patient, region = d.split('_')[0], d.split('_')[1]      # R1, injury
    reg = 'center' if 'center' in d else 'peripheral'
    a = sc.read_10x_mtx(p, var_names='gene_symbols', make_unique=True, cache=False)
    a.obs['sample'] = d
    a.obs['patient'] = patient
    a.obs['region'] = reg
    print(f'{d}: {a.shape[0]} cells')
    adatas.append(a)
A = ad.concat(adatas, index_unique='-')
print(f'合并: {A.shape[0]} cells x {A.shape[1]} genes')

# ---------- 2) QC ----------
A.var['mt'] = A.var_names.str.upper().str.startswith('MT-')
A.var['ribo'] = A.var_names.str.upper().str.startswith(('RPS', 'RPL'))
sc.pp.calculate_qc_metrics(A, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
print(A.obs.groupby('sample')[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']]
      .median().round(1).to_string())
g_lo, g_hi = np.percentile(A.obs['n_genes_by_counts'], [1, 99])
B = A[(A.obs.n_genes_by_counts > max(200, g_lo)) & (A.obs.n_genes_by_counts < g_hi)
      & (A.obs.pct_counts_mt < 15)].copy()
print(f'QC 后: {B.shape[0]} cells (保留 {100*B.shape[0]/A.shape[0]:.1f}%)')

# ---------- 3) 归一化 + 降维 + 聚类 ----------
B.layers['counts'] = B.X.copy()
sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
sc.pp.highly_variable_genes(B, n_top_genes=3000, batch_key='sample')
sc.pp.scale(B, max_value=10)
sc.tl.pca(B, n_comps=40)
sc.pp.neighbors(B, n_neighbors=15, n_pcs=30)
sc.tl.leiden(B, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
sc.tl.umap(B)

# ---------- 4) 内皮/周细胞标记评分 ----------
MARKERS = {
    'EC': ['PECAM1', 'VWF', 'CLDN5', 'KDR', 'FLT1', 'CDH5'],
    'Arterial': ['GJA5', 'HEY1', 'SEMA3G', 'EFNB2'],
    'Capillary': ['RGCC', 'CA4', 'ESAM'],
    'Venous': ['NR2F2', 'ACKR1'],
    'Tip cell': ['APOD', 'KCNE4', 'RAMP3'],
    'Pericyte': ['PDGFRB', 'RGS5', 'NOTCH3'],
    'SMC': ['ACTA2', 'TAGLN', 'MYH11'],
    'Neuron': ['RBFOX3', 'SNAP25'],
    'Astrocyte': ['GFAP', 'AQP4'],
    'Microglia': ['P2RY12', 'TMEM119', 'CX3CR1'],
}
for k, genes in MARKERS.items():
    gl = [g for g in genes if g in B.var_names]
    if gl:
        sc.tl.score_genes(B, gl, score_name=f'score_{k}', use_raw=False)
score_cols = [c for c in B.obs.columns if c.startswith('score_')]
print(B.obs.groupby('leiden')[score_cols].mean().round(2).to_string())
B.obs.groupby('leiden')[score_cols].mean().round(3).to_csv(
    os.path.join(OUT, 'GSE182398_marker_scores_per_leiden.csv'), encoding='utf-8-sig')
comp = B.obs.groupby(['leiden', 'region']).size().unstack(fill_value=0)
print(comp.to_string())
comp.to_csv(os.path.join(OUT, 'GSE182398_region_composition.csv'), encoding='utf-8-sig')

# ---------- 5) 图 ----------
B.obs['sample'] = B.obs['sample'].astype('category')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, col, t in zip(axes, ['sample', 'region', 'leiden'],
                      ['Samples', 'Injury region', 'Leiden clusters']):
    sc.pl.umap(B, color=col, ax=ax, show=False, title=t, frameon=False)
fig.suptitle('GSE182398 human TBI CD45-CD31+ endothelial scRNA (n=6)', y=1.02, fontsize=11)
fig.savefig(os.path.join(OUT, 'GSE182398_umap_overview.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

fig = sc.pl.umap(B, color=score_cols[:9], ncols=3, return_fig=True)
fig.savefig(os.path.join(OUT, 'GSE182398_umap_marker_scores.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

B.write(os.path.join(OUT, 'GSE182398_qc_clustered.h5ad', ), compression='gzip')
print('[saved] GSE182398_qc_clustered.h5ad + umap_overview/umap_marker_scores.png + composition/scores csv')
