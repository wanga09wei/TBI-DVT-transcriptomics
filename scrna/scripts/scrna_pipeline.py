# -*- coding: utf-8 -*-
"""
GSE223245 单细胞管线（Step 1: 加载 + QC + 归一化 + 降维 + 聚类 + 注释）
数据：人 TBI 创伤后凝血病（PTC）PBMC scRNA-seq，16 样本
      Mi1-4 / Mo1-4 / Se1-4（轻/中/重 TBI）+ Co1-4（健康对照）
原始文件：scrna/data/GSM69438xx_*.txt.gz（每样本 ~25MB 文本矩阵）

用法：python scrna_pipeline.py [--sample-one]   # --sample-one 只跑首个样本（管线验证）
输出：scrna/output/ 下的 h5ad、QC 图、聚类图、细胞类型构成表
"""
import os, sys, glob, gzip, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scanpy as sc
import anndata as ad
import matplotlib; matplotlib.use('Agg')

BASE = os.path.dirname(os.path.abspath(__file__))
SCRNA = os.path.dirname(BASE)                       # scrna/（脚本位于 scrna/scripts/ 下）
DATA = os.path.join(SCRNA, 'data')
OUT = os.path.join(SCRNA, 'output')
os.makedirs(OUT, exist_ok=True)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=110, facecolor='white', format='png')

# ---------- 1) 样本分组 ----------
GROUP_MAP = {}
for f in glob.glob(os.path.join(DATA, 'GSM*.txt.gz')):
    stem = os.path.basename(f).replace('.txt.gz', '')
    tag = stem.split('_')[-1][:2].lower()       # Mi / Mo / Se / Co
    grp = {'mi': 'Mild', 'mo': 'Moderate', 'se': 'Severe', 'co': 'Control'}.get(tag, 'Unknown')
    GROUP_MAP[stem] = grp
print('样本分组:', GROUP_MAP)
if not GROUP_MAP:
    sys.exit('[abort] scrna/data 下未发现 GSM*.txt.gz')

if '--sample-one' in sys.argv:
    keep = sorted(GROUP_MAP)[0]
    GROUP_MAP = {k: v for k, v in GROUP_MAP.items() if k == keep}
    print(f'[single-sample mode] {keep}')

# ---------- 2) 自适应加载器 ----------
def load_one(path, sample_id):
    """逐行读取 gz 文本矩阵，自动识别分隔符/表头方向。返回 AnnData（cells x genes）。"""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as fh:
        head = [next(fh) for _ in range(3)]
    delim = '\t' if head[0].count('\t') >= head[0].count(',') else ','
    ncols = head[0].rstrip('\n').split(delim)
    # 判定方向：第一列是基因名（行=基因）还是条码（行=细胞）
    first_cell = head[1].split(delim)[0]
    looks_like_gene = bool(re.match(r'^[A-Za-z][A-Za-z0-9_.\-]*$', first_cell)) and not first_cell.startswith('AA')
    if looks_like_gene:                        # 行=基因, 列=细胞
        df = pd.read_csv(path, sep=delim, index_col=0)
        X = df.T                               # -> cells x genes
    else:                                      # 行=细胞, 列=基因
        df = pd.read_csv(path, sep=delim, index_col=0)
        X = df
    a = ad.AnnData(X.astype(np.float32))
    a.obs['sample'] = sample_id
    a.obs['group'] = GROUP_MAP[sample_id]
    a.var_names_make_unique()
    print(f'  {sample_id}: {a.shape[0]} cells x {a.shape[1]} genes')
    return a

print('\n=== 读取样本 ===')
adatas = []
for sid in GROUP_MAP:
    f = glob.glob(os.path.join(DATA, f'{sid}*.txt.gz'))[0]
    adatas.append(load_one(f, sid))
A = ad.concat(adatas, label='sample', keys=list(GROUP_MAP.keys()), index_unique='-')
A.obs['group'] = A.obs['sample'].map(lambda s: GROUP_MAP[s])
print(f'\n合并后：{A.shape[0]} cells x {A.shape[1]} genes, {A.obs["group"].value_counts().to_dict()}')

# ---------- 3) QC ----------
A.var['mt'] = A.var_names.str.upper().str.startswith(('MT-', 'MT.'))
A.var['ribo'] = A.var_names.str.upper().str.startswith(('RPS', 'RPL'))
sc.pp.calculate_qc_metrics(A, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
print(A.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']].describe().round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, lab in zip(axes, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                        ['Genes per cell', 'UMIs per cell', '% mitochondrial']):
    for g, sub in A.obs.groupby('group'):
        ax.hist(sub[col], bins=50, alpha=0.5, label=g, density=True)
    ax.set_xlabel(lab); ax.set_ylabel('density'); ax.set_yscale('log')
    ax.legend(fontsize=7, frameon=False)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.suptitle('QC distributions by group (pre-filter)', y=1.02, fontsize=10, fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'QC_pre_filter.png'), bbox_inches='tight'); plt.close(fig)

# 过滤阈值：先用分位数自动定，最终以正式分析为准
g_lo, g_hi = np.percentile(A.obs['n_genes_by_counts'], [1, 99])
print(f'QC 阈值(自动): genes {g_lo:.0f}-{g_hi:.0f}; mt<15%; 将在正式分析中再固定')
B = A[(A.obs['n_genes_by_counts'] > max(200, g_lo)) &
      (A.obs['n_genes_by_counts'] < g_hi) &
      (A.obs['pct_counts_mt'] < 15)].copy()
print(f'QC 过滤后：{B.shape[0]} cells（保留 {100*B.shape[0]/A.shape[0]:.1f}%）')

# ---------- 4) 归一化 + HVG + 降维 ----------
sc.pp.normalize_total(B, target_sum=1e4)
sc.pp.log1p(B)
B.raw = B
sc.pp.highly_variable_genes(B, n_top_genes=3000, batch_key='sample', flavor='seurat_v3'
                            if False else 'seurat')
B = B[:, B.var['highly_variable']].copy()
sc.pp.scale(B, max_value=10)
sc.tl.pca(B, n_comps=40, svd_solver='arpack')
sc.pp.neighbors(B, n_neighbors=15, n_pcs=30)
sc.tl.umap(B)
sc.tl.leiden(B, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
B.write(os.path.join(OUT, 'gse223245_qc_clustered.h5ad'))
print('\n细胞数/群：')
print(B.obs.groupby(['group', 'leiden']).size().unstack(fill_value=0).to_string())

# ---------- 5) 初步标记基因评分（免疫血栓轴相关） ----------
MARKERS = {
 'T cell (CD3D)': ['CD3D', 'CD3E', 'TRAC'],
 'CD8 cytotoxic': ['CD8A', 'GZMK', 'GZMB', 'NKG7'],
 'CD4/Treg': ['IL7R', 'FOXP3', 'TIGIT'],
 'NK': ['NCAM1', 'KLRD1', 'GNLY'],
 'B cell': ['MS4A1', 'CD79A', 'CD79B'],
 'Plasma': ['MZB1', 'JCHAIN', 'XBP1'],
 'Monocyte/CD14': ['CD14', 'LYZ', 'S100A8', 'S100A9'],
 'Monocyte/FCGR3A': ['FCGR3A', 'MS4A7', 'CDKN1C'],
 'Dendritic': ['FCER1A', 'CLEC10A', 'CD1C'],
 'Platelet': ['PPBP', 'PF4', 'ITGA2B', 'GP9'],
 'Neutrophil': ['CSF3R', 'FCGR3B', 'CXCR2', 'S100A12'],
 'Megakaryocyte': ['ITGA2B', 'PF4', 'VWF'],
}
for nm, gl in MARKERS.items():
    gl = [g for g in gl if g in B.raw.var_names]
    if gl:
        sc.tl.score_genes(B, gl, score_name=f'score_{nm}', use_raw=True)
score_cols = [c for c in B.obs.columns if c.startswith('score_')]
sc.pl.umap(B, color=['group', 'leiden'] + score_cols[:8], ncols=4,
           save='_markers.png', show=False, vmax='p99')
print('\n各群平均标记得分（top marker per leiden）:')
sm = B.obs.groupby('leiden')[score_cols].mean()
print((sm.T == sm.max(0)).sum().to_string(), '\n')
print(sm.round(2).to_string())
B.obs[score_cols].to_csv(os.path.join(OUT, 'marker_scores_per_cell.csv'))
sm.to_csv(os.path.join(OUT, 'marker_scores_per_leiden.csv'))
print('\n[saved] output/gse223245_qc_clustered.h5ad + QC_pre_filter.png + umap_markers.png + marker_scores_*.csv')
