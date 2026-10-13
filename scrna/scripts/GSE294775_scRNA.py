# -*- coding: utf-8 -*-
"""
GSE294775 scRNA 管线（人 TBI 尸检脑组织，7 TBI / 7 对照，14 样本）
Nat Commun 2025，MDM/小胶质命运图谱。10x 矩阵包 16,974 基因 x 98,492 细胞 + metadata.csv。
QC + 归一化 + PCA/UMAP/Leiden + 髓系/小胶质标记评分概览。
输出：scrna/output/GSE294775_*.h5ad/png/csv（300 dpi，英文标注）
"""
import os, warnings, gzip, shutil
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRNA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SCRNA, 'data', 'GSE294775')
OUT = os.path.join(SCRNA, 'output'); os.makedirs(OUT, exist_ok=True)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=110, facecolor='white', format='png')

# ---------- 0) 归一化文件名以便 read_10x_mtx ----------
need = {'GSE294775_matrix.mtx.gz': 'matrix.mtx.gz',
        'GSE294775_barcodes.tsv.gz': 'barcodes.tsv.gz',
        'GSE294775_features.tsv.gz': 'features.tsv.gz'}
for src, dst in need.items():
    s, d = os.path.join(DATA, src), os.path.join(DATA, dst)
    if not os.path.exists(d):
        try: os.link(s, d)          # 同卷硬链接，不占空间
        except OSError: shutil.copy2(s, d)
        print(f'[link] {src} -> {dst}')

# ---------- 1) 读取矩阵（npz 分块转换产物，防 MemoryError） ----------
import scipy.sparse as sp
npz_path = os.path.join(DATA, 'GSE294775_raw_counts.npz')
if not os.path.exists(npz_path):
    raise SystemExit('[abort] 请先运行 GSE294775_mtx_to_npz.py 生成 npz')
z = np.load(npz_path)
M = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']), dtype=np.float32)
A = sc.AnnData(M.T.tocsr())                      # cells x genes
del z, M
feat = pd.read_csv(os.path.join(DATA, 'features.tsv.gz'), sep='\t', header=None)
A.var_names = feat[1].values
A.var_names_make_unique()
bcs = pd.read_csv(os.path.join(DATA, 'barcodes.tsv.gz'), sep='\t', header=None)[0].astype(str)
assert len(bcs) == A.shape[0], f'barcodes {len(bcs)} != cells {A.shape[0]}'
A.obs_names = bcs.values
print(f'矩阵: {A.shape[0]} cells x {A.shape[1]} genes; barcode 示例: {bcs.iloc[0]}')

meta = pd.read_csv(os.path.join(DATA, 'GSE294775_metadata.csv.gz'))
print('metadata 列:', list(meta.columns)[:12])
# metadata 首列即完整 cell id（与 obs_names 一致，含 -1 后缀）
bc_col = meta.columns[0]
meta = meta.set_index(bc_col)
hit = pd.Series(A.obs_names.isin(meta.index), index=A.obs_names)
print(f'metadata 匹配: {hit.sum()}/{A.shape[0]}')
for col in meta.columns:
    if meta[col].nunique() <= 60:
        A.obs[col] = meta.loc[A.obs_names, col].values
tb_col = next((c for c in A.obs.columns if c.lower() in ('tbi',) or 'tbi' in c.lower()), None)
if tb_col: A.obs['group'] = A.obs[tb_col].astype(str).map(lambda v: 'TBI' if v.strip().lower() in ('yes', 'true', '1') else ('Control' if v.strip().lower() in ('no', 'false', '0') else v))
print('分组:', A.obs['group'].value_counts().to_dict() if 'group' in A.obs else 'n/a')

# ---------- 2) QC ----------
A.var['mt'] = A.var_names.str.upper().str.startswith('MT-')
A.var['ribo'] = A.var_names.str.upper().str.startswith(('RPS', 'RPL'))
sc.pp.calculate_qc_metrics(A, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
grp_col = 'group' if 'group' in A.obs else None
cols_q = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
print(A.obs.groupby(grp_col)[cols_q].median().round(1).to_string() if grp_col else A.obs[cols_q].median().round(1).to_string())
g_lo, g_hi = np.percentile(A.obs['n_genes_by_counts'], [1, 99])
B = A[(A.obs.n_genes_by_counts > max(200, g_lo)) & (A.obs.n_genes_by_counts < g_hi)
      & (A.obs.pct_counts_mt < 15)].copy()
print(f'QC 后: {B.shape[0]} cells (保留 {100*B.shape[0]/A.shape[0]:.1f}%)')
del A

# ---------- 3) 归一化 + 降维 + 聚类 ----------
sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
B.raw = B
n_hvg = 3000
sc.pp.highly_variable_genes(B, n_top_genes=n_hvg)
B = B[:, B.var.highly_variable].copy()
sc.pp.scale(B, max_value=10)
sc.tl.pca(B, n_comps=50)
sc.pp.neighbors(B, n_neighbors=15, n_pcs=40)
sc.tl.leiden(B, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
sc.tl.umap(B)

# ---------- 4) 髓系/细胞类型标记评分 ----------
MARKERS = {
    'Microglia_homeostatic': ['P2RY12', 'TMEM119', 'SALL1', 'CX3CR1'],
    'Microglia_DAM': ['APOE', 'TREM2', 'SPP1', 'CD9'],
    'MDM_monocyte': ['S100A8', 'S100A9', 'LYZ', 'VCAN', 'FCN1'],
    'Macrophage': ['CD68', 'CD163', 'MSR1', 'MRC1'],
    'T_cell': ['CD3D', 'CD3E', 'TRAC'],
    'EC': ['PECAM1', 'VWF', 'CLDN5'],
    'Astrocyte': ['GFAP', 'AQP4', 'ALDH1L1'],
    'Oligodendrocyte': ['MBP', 'PLP1', 'MAG'],
    'Neuron': ['RBFOX3', 'SNAP25'],
}
for k, genes in MARKERS.items():
    gl = [g for g in genes if g in B.raw.var_names]
    if gl:
        sc.tl.score_genes(B, gl, score_name=f'score_{k}', use_raw=True)
score_cols = [c for c in B.obs.columns if c.startswith('score_')]
sm = B.obs.groupby('leiden')[score_cols].mean().round(2)
print(sm.to_string())
sm.to_csv(os.path.join(OUT, 'GSE294775_marker_scores_per_leiden.csv'), encoding='utf-8-sig')
by_grp = ['group'] if grp_col else []
if by_grp:
    comp = B.obs.groupby(by_grp + ['leiden']).size().unstack(fill_value=0)
    print(comp.to_string())
    comp.to_csv(os.path.join(OUT, 'GSE294775_group_composition.csv'), encoding='utf-8-sig')

# ---------- 5) 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
titles = ['Groups' if grp_col else '', 'Leiden clusters', 'QC (genes)']
cols = [c for c in [grp_col, 'leiden', 'n_genes'] if c in B.obs.columns]
for ax, col, t in zip(axes, cols, titles):
    sc.pl.umap(B, color=col, ax=ax, show=False, title=t, frameon=False)
fig.suptitle('GSE294775 human TBI post-mortem brain (7 TBI / 7 control)', y=1.02, fontsize=11)
fig.savefig(os.path.join(OUT, 'GSE294775_umap_overview.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

fig = sc.pl.umap(B, color=score_cols[:9], ncols=3, return_fig=True)
fig.savefig(os.path.join(OUT, 'GSE294775_umap_marker_scores.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

B.write(os.path.join(OUT, 'GSE294775_qc_clustered.h5ad'), compression='gzip')
print('[saved] GSE294775_qc_clustered.h5ad + umap png + composition/scores csv')
