# -*- coding: utf-8 -*-
"""
S2: GSE182398 内皮/血管壁细胞亚型精细化（v2 修正版）
关键修正：
  - A.X 在 S1 中是 scale 后的值 → 本脚本从 A.layers['counts'] 恢复原始计数重新归一化
  - 标记评分必须在**全基因集**上计算（HVG 子集会让大多数标记基因缺失，导致亚型误判）
  - 补 matplotlib.pyplot 导入
流程：恢复计数 → normalize/log1p → 血管壁细胞筛选 → 标记评分 → HVG → scale → PCA/UMAP/Leiden → 亚型命名 → 区域 DEG → PAGA
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:/TBI_DVT_Study/scrna/output'
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=110, facecolor='white', format='png')

A = sc.read_h5ad(os.path.join(OUT, 'GSE182398_qc_clustered.h5ad'))
print('载入:', A.shape, '| layers:', list(A.layers.keys()))
if 'counts' in A.layers:
    A.X = A.layers['counts'].copy()
print('X 是否原始计数: min=%.1f max=%.1f' % (A.X.min(), A.X.max()))

# ---------- 1) 血管壁细胞筛选（沿用 S1 评分） ----------
keep = ((A.obs['score_EC'] > 0.3) | (A.obs['score_Pericyte'] > 0.3) | (A.obs['score_SMC'] > 0.3))
contam = ((A.obs['score_Neuron'] > 0.3) | (A.obs['score_Astrocyte'] > 0.3) | (A.obs['score_Microglia'] > 0.2)) & (A.obs['score_EC'] < 0.3)
sel = keep & ~contam
print(f'血管壁细胞保留: {sel.sum()}/{A.shape[0]} ({100*sel.sum()/A.shape[0]:.1f}%)')
B = A[sel].copy()

# ---------- 2) 重新归一化（全基因集） ----------
sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
B.raw = B

MARK = {
    'Capillary': ['RGCC', 'CA4', 'ESAM', 'CD36'],
    'Arterial': ['GJA5', 'HEY1', 'SEMA3G', 'EFNB2', 'DKK2'],
    'Venous': ['NR2F2', 'ACKR1', 'SELP', 'VCAM1'],
    'Tip_stalk': ['APOD', 'KCNE4', 'RAMP3', 'ESM1', 'ANGPT2'],
    'Pericyte': ['RGS5', 'PDGFRB', 'NOTCH3', 'ABCC9'],
    'SMC': ['ACTA2', 'TAGLN', 'MYH11', 'CNN1'],
}
for k, gs in MARK.items():
    gl = [g for g in gs if g in B.raw.var_names]
    print(f'  {k}: {len(gl)}/{len(gs)} 基因可用 {gl}')
    if gl: sc.tl.score_genes(B, gl, score_name=f'm_{k}', use_raw=True)

# ---------- 3) HVG + 降维聚类 ----------
sc.pp.highly_variable_genes(B, n_top_genes=2000, batch_key='sample')
Bh = B[:, B.var.highly_variable].copy()
sc.pp.scale(Bh, max_value=10)
sc.tl.pca(Bh, n_comps=30)
sc.pp.neighbors(Bh, n_neighbors=15, n_pcs=25)
sc.tl.leiden(Bh, resolution=0.8, key_added='ec_cluster', flavor='igraph', n_iterations=2, directed=False)
sc.tl.umap(Bh)
sc.tl.paga(Bh, groups='ec_cluster')

# ---------- 4) 亚型命名（在全基因集上评分的 obs 已带入 Bh） ----------
mcols = [c for c in Bh.obs.columns if c.startswith('m_')]
mean_sc = Bh.obs.groupby('ec_cluster')[mcols].mean()
print(mean_sc.round(3).to_string())
assign = mean_sc.idxmax(axis=1).str.replace('m_', '', regex=False)
Bh.obs['subtype'] = Bh.obs['ec_cluster'].map(assign).astype(str)
print('\n亚型分配:', Bh.obs['subtype'].value_counts().to_dict())
comp = Bh.obs.groupby(['subtype', 'region']).size().unstack(fill_value=0)
print(comp.to_string())
comp.to_csv(os.path.join(OUT, 'GSE182398_S2_subtype_region.csv'), encoding='utf-8-sig')
mean_sc.round(4).to_csv(os.path.join(OUT, 'GSE182398_S2_subtype_marker_scores.csv'), encoding='utf-8-sig')

# ---------- 5) 亚型内 center vs peripheral DEG（Wilcoxon，log 归一化层） ----------
B_sub = B.copy(); B_sub.obs['subtype'] = Bh.obs['subtype'].values
rows = []
for st in B_sub.obs['subtype'].unique():
    sub = B_sub[B_sub.obs['subtype'] == st]
    n1 = int((sub.obs.region == 'center').sum()); n2 = int((sub.obs.region == 'peripheral').sum())
    if min(n1, n2) < 20: continue
    sc.tl.rank_genes_groups(sub, 'region', groups=['center'], reference='peripheral',
                            method='wilcoxon', use_raw=True)
    df = sc.get.rank_genes_groups_df(sub, group='center')
    df['subtype'] = st; df['n_center'] = n1; df['n_peripheral'] = n2
    rows.append(df.head(80))
    print(f'{st}: center {n1} vs peripheral {n2} | top up: {list(df.head(5).names)}')
if rows:
    deg = pd.concat(rows, ignore_index=True)
    deg.to_csv(os.path.join(OUT, 'GSE182398_S2_subtype_region_DEG.csv'), index=False, encoding='utf-8-sig')

# ---------- 6) 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, col, t in zip(axes, ['subtype', 'region', 'ec_cluster'],
                      ['Endothelial subtypes', 'Injury region', 'Subclusters']):
    sc.pl.umap(Bh, color=col, ax=ax, show=False, title=t, frameon=False)
fig.suptitle('GSE182398 S2: vascular-wall cells re-clustered (n=%d)' % Bh.shape[0], y=1.02, fontsize=11)
fig.savefig(os.path.join(OUT, 'GSE182398_S2_subtypes_umap.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

sc.pl.paga(Bh, color=['subtype'], show=False, frameon=False, title='PAGA connectivity of vascular subtypes')
plt.savefig(os.path.join(OUT, 'GSE182398_S2_paga.png'), dpi=300, bbox_inches='tight')
plt.close('all')

fig, axes = plt.subplots(1, len(mcols), figsize=(3.2 * len(mcols), 3.6))
for ax, col in zip(axes, mcols):
    sc.pl.umap(Bh, color=col, ax=ax, show=False, title=col.replace('m_', ''), frameon=False, vmin=-0.5, vmax=1.5)
fig.suptitle('Marker scores per subtype (GSE182398 S2)', y=1.03, fontsize=11)
fig.savefig(os.path.join(OUT, 'GSE182398_S2_marker_umaps.png'), dpi=300, bbox_inches='tight')
plt.close('all')

Bh.write(os.path.join(OUT, 'GSE182398_S2_subtypes.h5ad'), compression='gzip')
print('[saved] GSE182398_S2_subtypes.h5ad + subtype csv + umap/paga/marker png')
