# -*- coding: utf-8 -*-
"""
S8: 内皮亚型 PAGA 拓扑 + 扩散拟时序（GSE182398 血管壁细胞）
 (a) 亚型水平 PAGA（含 region 构成权重）
 (b) 扩散图 + DPT 拟时序：root = 损伤中心 tip/stalk 细胞（最"激活"的内皮态）
 (c) 沿拟时序的内皮功能基因趋势（F3/ANXA2/S100A10/ACKR1/VWF/CLDN5/MFSD2A/EMP1/HSPA1A/SELE）
 (d) 亚型 × 区域 × 拟时序分布 → 重建"动脉→毛细血管→静脉/tip"的损伤梯度
输入：GSE182398_S2_subtypes.h5ad（已聚类）+ GSE182398_qc_clustered.h5ad（全基因表达）
输出：figures/Fig8_endothelial_pseudotime.png、GSE182398_S8_*.csv
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scipy.sparse as sp
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'D:/TBI_DVT_Study'
OUT = os.path.join(ROOT, 'scrna', 'output')
FIG = os.path.join(ROOT, 'figures')
sc.settings.verbosity = 1
LOG = open(os.path.join(OUT, 'S8_run.log'), 'a', encoding='utf-8')
def log(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); LOG.write(m + '\n'); LOG.flush()

# ---------- 1) 载入 S2 亚型对象（HVG、scaled） ----------
Bh = sc.read_h5ad(os.path.join(OUT, 'GSE182398_S2_subtypes.h5ad'))
log(f'[load] S2 亚型对象 {Bh.shape} | obs: {[c for c in Bh.obs.columns if not c.startswith("m_")][:12]}')
if 'ec_cluster' not in Bh.obs:
    raise SystemExit('缺少 ec_cluster')
sub_of = Bh.obs['subtype'].astype(str)
region_of = Bh.obs['region'].astype(str)

# ---------- 2) 仅内皮细胞 + 扩散图 + DPT ----------
# 关键：壁细胞（SMC/周细胞）与内皮在扩散空间天然分离，会形成"伪梯度" → 拟时序只在 EC 内部做
EC_SUB = ['Capillary', 'Arterial', 'Venous', 'Tip_stalk']
is_ec = sub_of.isin(EC_SUB)
log(f'[subset] 内皮亚型细胞 {int(is_ec.sum())}/{len(is_ec)}（剔除壁细胞，避免伪梯度）')
Bh = Bh[is_ec.values].copy()
sub_of = Bh.obs['subtype'].astype(str)
region_of = Bh.obs['region'].astype(str)

sc.pp.neighbors(Bh, n_neighbors=15, n_pcs=25, use_rep='X_pca')
sc.tl.diffmap(Bh, n_comps=15)
# root：损伤中心中 tip/stalk 评分最高者（血管新生最活跃端）
mcols = [c for c in Bh.obs.columns if c.startswith('m_')]
root_col = 'm_Tip_stalk' if 'm_Tip_stalk' in mcols else mcols[0]
cand = Bh.obs[(Bh.obs.region == 'center')]
if len(cand) == 0:
    cand = Bh.obs
root_idx = cand[root_col].astype(float).idxmax()
Bh.uns['iroot'] = int(np.where(Bh.obs_names == root_idx)[0][0])
log(f'[DPT] root = {root_idx} (subtype={sub_of[root_idx]}, {root_col}={Bh.obs.loc[root_idx, root_col]:.3f})')
sc.tl.dpt(Bh)
pt = Bh.obs['dpt_pseudotime'].astype(float)

# ---------- 3) PAGA on subtypes ----------
sc.tl.paga(Bh, groups='subtype')
conn = pd.DataFrame(Bh.uns['paga']['connectivities'].todense()
                    if sp.issparse(Bh.uns['paga']['connectivities']) else Bh.uns['paga']['connectivities'],
                    index=Bh.obs['subtype'].cat.categories if hasattr(Bh.obs['subtype'], 'cat') else sorted(sub_of.unique()),
                    columns=Bh.obs['subtype'].cat.categories if hasattr(Bh.obs['subtype'], 'cat') else sorted(sub_of.unique()))
conn.to_csv(os.path.join(OUT, 'GSE182398_S8_paga_connectivity.csv'), encoding='utf-8-sig')
log('\n[PAGA] 亚型连通性:\n' + conn.round(3).to_string())

# 亚型拟时序统计
tab = pd.DataFrame({'subtype': sub_of, 'region': region_of, 'pseudotime': pt})
summ = tab.groupby(['subtype', 'region'])['pseudotime'].agg(['count', 'median']).reset_index()
summ.to_csv(os.path.join(OUT, 'GSE182398_S8_pseudotime_by_subtype_region.csv'), index=False, encoding='utf-8-sig')
log('\n[DPT] 亚型×区域 拟时序中位:\n' + summ.to_string(index=False))

# 区域效应的统计检验（每个亚型内 center vs peripheral）
from scipy import stats as sst
rows = []
for st in sorted(tab.subtype.unique()):
    s = tab[tab.subtype == st]
    c = s.loc[s.region == 'center', 'pseudotime']; p = s.loc[s.region == 'peripheral', 'pseudotime']
    if len(c) >= 10 and len(p) >= 10:
        u, pv = sst.mannwhitneyu(c, p, alternative='two-sided')
        rows.append({'subtype': st, 'n_center': len(c), 'n_periph': len(p),
                     'median_center': round(float(c.median()), 4), 'median_periph': round(float(p.median()), 4),
                     'P_center_vs_periph': float(pv)})
reg = pd.DataFrame(rows)
if len(reg):
    reg.to_csv(os.path.join(OUT, 'GSE182398_S8_region_pseudotime_test.csv'), index=False, encoding='utf-8-sig')
    log('\n[DPT] 亚型内 中心 vs 外周 拟时序检验:\n' + reg.to_string(index=False))

# ---------- 4) 全基因 log 表达（用于基因趋势） ----------
A = sc.read_h5ad(os.path.join(OUT, 'GSE182398_qc_clustered.h5ad'))
keep = ((A.obs['score_EC'] > 0.3) | (A.obs['score_Pericyte'] > 0.3) | (A.obs['score_SMC'] > 0.3))
contam = ((A.obs['score_Neuron'] > 0.3) | (A.obs['score_Astrocyte'] > 0.3) | (A.obs['score_Microglia'] > 0.2)) & (A.obs['score_EC'] < 0.3)
F = A[keep & ~contam].copy(); F.X = F.layers['counts'].copy()
sc.pp.normalize_total(F, target_sum=1e4); sc.pp.log1p(F)
common = F.obs_names.intersection(Bh.obs_names)
log(f'\n[expr] 全基因对象 {F.shape} | 与亚型对象交集 {len(common)}')
F = F[common].copy()
F.obs['subtype'] = sub_of.loc[common].values
F.obs['region'] = region_of.loc[common].values
F.obs['pseudotime'] = pt.loc[common].values
# 用同一扩散坐标做基因趋势（避免重复计算）
pt_v = F.obs['pseudotime'].values

GENE_TREND = ['F3', 'ANXA2', 'S100A10', 'ACKR1', 'VWF', 'CLDN5', 'MFSD2A',
              'EMP1', 'HSPA1A', 'SELE', 'VCAM1', 'TIMP1', 'APOLD1', 'EMP2']
GENE_TREND = [g for g in GENE_TREND if g in F.var_names]
nb = 12
edges = np.quantile(pt_v, np.linspace(0, 1, nb + 1))
edges[0] -= 1e-9
bidx = np.clip(np.digitize(pt_v, edges) - 1, 0, nb - 1)
trend = {}
for g in GENE_TREND:
    x = F[:, g].X
    x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    trend[g] = pd.Series(x).groupby(bidx).mean().reindex(range(nb)).values
tr = pd.DataFrame(trend, index=np.round([pt_v[bidx == i].mean() if (bidx == i).any() else np.nan for i in range(nb)], 3))
tr.index.name = 'pseudotime'
tr.to_csv(os.path.join(OUT, 'GSE182398_S8_gene_trend_pseudotime.csv'), encoding='utf-8-sig')
log('\n[trend] 基因沿拟时序趋势表已存')

# ---------- 5) 图 ----------
fig = plt.figure(figsize=(15.5, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# A: UMAP 亚型
ax = fig.add_subplot(gs[0, 0])
sc.pl.umap(Bh, color='subtype', ax=ax, show=False, frameon=False, title='A  Vascular subtypes', legend_loc='on data', legend_fontsize=6)
# B: UMAP 拟时序
ax = fig.add_subplot(gs[0, 1])
sc.pl.umap(Bh, color='dpt_pseudotime', ax=ax, show=False, frameon=False, cmap='viridis',
           title='B  Diffusion pseudotime (root = injury-centre tip/stalk EC)')
# C: 亚型 vs 拟时序 箱线
ax = fig.add_subplot(gs[0, 2])
order = tab.groupby('subtype')['pseudotime'].median().sort_values().index.tolist()
data = [tab.loc[tab.subtype == s, 'pseudotime'].values for s in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, widths=.6)
for p, c in zip(bp['boxes'], plt.get_cmap('Set2')(np.linspace(0, 1, len(order)))):
    p.set_facecolor(c); p.set_alpha(.85)
ax.set_ylabel('Pseudotime'); ax.tick_params(axis='x', rotation=25)
ax.set_title('C  Pseudotime by subtype', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# D: 基因趋势（小倍数）
ax = fig.add_subplot(gs[1, 0:2])
cmap = plt.get_cmap('tab20')
for i, g in enumerate(GENE_TREND):
    v = tr[g].values
    v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    ax.plot(tr.index.values, v, '-o', ms=3, lw=1.6, color=cmap(i % 20), label=g)
ax.axhline(0, color='k', lw=.5)
ax.set_xlabel('Pseudotime'); ax.set_ylabel('Expression (z-score)')
ax.legend(fontsize=7, ncol=4, frameon=False, loc='upper left')
ax.set_title('D  Endothelial effector genes along the injury pseudotime axis', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# E: PAGA 图
ax = fig.add_subplot(gs[1, 2])
try:
    sc.pl.paga(Bh, color='subtype', ax=ax, show=False, frameon=False,
               title='E  PAGA topology', node_size_scale=1.6, fontsize=7)
except Exception as e:
    sc.pl.paga(Bh, ax=ax, show=False, frameon=False, title='E  PAGA topology')
    log(f'[figE] fallback: {e}')

fig.suptitle('Fig 8  Injury gradient of human TBI endothelium: PAGA topology and diffusion pseudotime (GSE182398)',
             fontsize=11, y=0.98)
p8 = os.path.join(FIG, 'Fig8_endothelial_pseudotime.png')
fig.savefig(p8, dpi=300, bbox_inches='tight'); plt.close(fig)
log('\n[saved] ' + p8)

# 保存带拟时序的对象
Bh.write(os.path.join(OUT, 'GSE182398_S8_pseudotime.h5ad'), compression='gzip')
log('[saved] GSE182398_S8_pseudotime.h5ad')
LOG.close()
