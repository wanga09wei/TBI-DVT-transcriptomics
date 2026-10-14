# -*- coding: utf-8 -*-
"""
S4: GSE294775 深挖 + 跨数据集整合
  S4a 官方注释下的 TBI vs CTR 细胞类型比例（χ² + FDR）与逐类型 DEG（Wilcoxon，BH）
  S4b 跨数据集整合图（Fig 5）：四联证据链
       (a) 本院队列轨迹表型 C1/C2/C3（血小板）
       (b) GSE223245 bulk 芯片中性粒轴 z-score 趋势
       (c) GSE182398 内皮亚型 中心/外周构成
       (d) GSE294775 TBI 特异性胶质/髓系簇（log2 OR）
"""
import os, warnings, json
import numpy as np, pandas as pd
from scipy import stats, sparse
warnings.filterwarnings('ignore')
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values.

    Returns the adjusted values in the SAME order as the input (not sorted),
    capped at 1.0. Non-finite inputs are returned as NaN.

    NOTE (2026-10-14): the previous inline implementation
        q = np.minimum.accumulate((np.sort(p) * n / (np.arange(n)[::-1] + 1))[::-1])
        out = q[np.argsort(np.argsort(p))]
    was wrong twice over: the missing trailing [::-1] left q ordered by
    *descending* p, and np.argsort(np.argsort(p)) is not the inverse
    permutation that maps sorted values back to the original rows. The result
    was an FDR column mis-aligned with its p-values (values > 1 appeared).
    """
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    if m.sum() == 0:
        return q
    pv = p[m]
    n = pv.size
    order = np.argsort(pv, kind='mergesort')
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[m] = out
    return q


OUT = r'D:/TBI_DVT_Study/scrna/output'
FIGDIR = r'D:/TBI_DVT_Study/figures'
sc.settings.verbosity = 0

# ================= S4a =================
A = sc.read_h5ad(os.path.join(OUT, 'GSE294775_qc_clustered.h5ad'))
print('载入:', A.shape, '| raw:', None if A.raw is None else A.raw.shape)

def celltype_shift(key, min_n=50):
    tab = A.obs.groupby([key, 'group']).size().unstack(fill_value=0)
    tot_t, tot_c = tab.get('TBI', pd.Series(0, index=tab.index)).sum(), tab.get('CTR', pd.Series(0, index=tab.index)).sum()
    rows = []
    for ct in tab.index:
        t, c = int(tab.loc[ct].get('TBI', 0)), int(tab.loc[ct].get('CTR', 0))
        if t + c < min_n: continue
        chi2, p, _, _ = stats.chi2_contingency([[t, tot_t - t], [c, tot_c - c]])
        OR = (t + .5) * (tot_c - c + .5) / ((tot_t - t + .5) * (c + .5))
        rows.append({key: ct, 'n_TBI': t, 'n_CTR': c,
                     'pct_TBI': 100 * t / tot_t, 'pct_CTR': 100 * c / tot_c,
                     'log2OR': np.log2(OR), 'P_chi2': p})
    df = pd.DataFrame(rows).sort_values('P_chi2')
    df['FDR'] = bh_fdr(df.P_chi2.values)
    return df

for key in ['celltype_l1', 'celltype_l2']:
    if key in A.obs.columns:
        d = celltype_shift(key)
        d.to_csv(os.path.join(OUT, f'S4_celltype_shift_{key}.csv'), index=False, encoding='utf-8-sig')
        print(f'\n=== {key} TBI vs CTR（top10 变化）===')
        print(pd.concat([d.head(6), d.tail(6)]).round(3).to_string(index=False))

# 逐细胞类型 DEG（TBI vs CTR）
if A.raw is not None:
    Xlog = A.raw.X
    var = list(A.raw.var_names)
else:
    Xlog, var = A.X, list(A.var_names)
gi = {g: i for i, g in enumerate(var)}
groups = A.obs['group'].astype(str).values
TYPES = ['Astro', 'Oligo', 'Micro', 'MDM', 'OPC', 'Endo', 'Peri', 'Neuro', 'Tcell']
TYPES = [t for t in TYPES if t in set(A.obs.get('celltype_l1', pd.Series(dtype=str)).astype(str))]
if not TYPES and 'celltype_l1' in A.obs:
    TYPES = list(pd.unique(A.obs['celltype_l1'].astype(str)))[:8]
ct_l1 = A.obs['celltype_l1'].astype(str).values if 'celltype_l1' in A.obs else np.array(['NA'] * A.shape[0])
rows = []
for ct in TYPES:
    m = ct_l1 == ct
    if m.sum() < 100: continue
    t_mask, c_mask = m & (groups == 'TBI'), m & (groups == 'CTR')
    if min(t_mask.sum(), c_mask.sum()) < 30: continue
    for g in ['F3', 'VWF', 'CLDN5', 'PECAM1', 'HMGB1', 'PADI4', 'S100B', 'GFAP', 'AQP4', 'VIM', 'CD44', 'SPP1', 'APOE', 'TREM2', 'SERPINE1', 'PLAUR', 'TFPI', 'THBD', 'PROCR']:
        if g not in gi: continue
        col = Xlog[:, gi[g]]
        col = np.asarray(col.todense()).ravel() if sparse.issparse(col) else np.asarray(col).ravel()
        t, c = col[t_mask], col[c_mask]
        u, p = stats.mannwhitneyu(t, c, alternative='two-sided')
        rows.append({'celltype_l1': ct, 'gene': g, 'n_TBI': int(t_mask.sum()), 'n_CTR': int(c_mask.sum()),
                     'mean_TBI': t.mean(), 'mean_CTR': c.mean(),
                     'log2ratio': np.log2((t.mean() + 1e-3) / (c.mean() + 1e-3)), 'P_MWU': p})
deg = pd.DataFrame(rows)
deg['FDR'] = bh_fdr(deg['P_MWU'].values)
deg.to_csv(os.path.join(OUT, 'S4_percelltype_DEG_TBI_vs_CTR.csv'), index=False, encoding='utf-8-sig')
sig = deg[(deg.FDR < 0.05) & (deg.log2ratio.abs() > 0.3)]
print(f'\n=== 逐类型 DEG 显著 {len(sig)}/{len(deg)} ===')
print(sig.sort_values(['celltype_l1', 'FDR']).round(3).head(30).to_string(index=False))

# ================= S4b 跨数据集整合图 =================
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

# (a) 队列轨迹表型（真实数据，从 cohort_summary.json 解析）
ax = fig.add_subplot(gs[0, 0])
try:
    cc = json.load(open(r'D:/TBI_DVT_Study/analysis/cohort_summary.json', encoding='utf-8'))
    traj = cc['轨迹表型(GBTM)']
    import re
    series = {}
    NAME_EN = {'C1': 'C1 low-burden', 'C2': 'C2 intermediate', 'C3': 'C3 high-burden peak-fall'}
    for k, v in traj.items():
        if not isinstance(v, str): continue
        m = re.search(r'n=(\d+),\s*(?:血小板)?([\d.]+)->([\d.]+)->([\d.]+)', v)
        if m and k.startswith(('C1', 'C2', 'C3')):
            tag = k.split('_')[0]
            series[tag] = (int(m.group(1)), [float(m.group(i)) for i in (2, 3, 4)])
    for (k, (n, vals)), col in zip(sorted(series.items()), ['#27ae60', '#e67e22', '#c0392b']):
        ax.plot([0, 1, 2], vals, '-o', color=col, lw=2, ms=5,
                label=f'{NAME_EN.get(k, k)} (n={n})')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['T0 admission', 'T1 thrombosis\ndetection', 'T2 after 2-wk\nanticoagulation'], fontsize=8)
    ax.set_ylabel('Platelet count (×10⁹/L)')
    ax.legend(fontsize=7.5, frameon=False)
except Exception as e:
    ax.text(.5, .5, f'cohort trajectory unavailable:\n{e}', ha='center', va='center', fontsize=8, color='grey')
ax.set_title('A  Our cohort: platelet trajectory phenotypes (GBTM, n=213)', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# (b) bulk 中性粒轴（ρ 取探针级 fig4_summary 口径）
ax = fig.add_subplot(gs[0, 1])
z = pd.read_csv(os.path.join(OUT, 'fig4_data_zscores.csv'), index_col=0)
rho_tab = pd.read_csv(os.path.join(OUT, 'fig4_summary.csv'))
rho_tab = rho_tab.set_index('gene') if 'gene' in rho_tab.columns else rho_tab
if z.shape[1] == 16:
    zmeans = {g: [z.loc[g].iloc[i*4:(i+1)*4].mean() for i in range(4)] for g in z.index}
    genes = [g for g in ['S100A12', 'S100A9', 'MMP9', 'PADI4', 'S100A8'] if g in zmeans]
    for g, col in zip(genes, plt.get_cmap('YlOrRd')(np.linspace(0.35, 0.95, len(genes)))):
        rho = float(rho_tab.loc[g, 'rho_severity']) if g in rho_tab.index else np.nan
        ax.plot(range(4), zmeans[g], '-o', color=col, lw=2, ms=4, label=f'{g} (ρ={rho:.2f})')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Control', 'Mild', 'Moderate', 'Severe'], fontsize=8)
    ax.set_ylabel('Expression (z-score)'); ax.legend(fontsize=7.5, frameon=False)
ax.set_title('B  GSE223245 whole blood: neutrophil/DAMP-NET axis grades with severity', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# (c) 内皮亚型构成
ax = fig.add_subplot(gs[1, 0])
comp = pd.read_csv(os.path.join(OUT, 'GSE182398_S2_subtype_region.csv'), index_col=0)
pct = comp.div(comp.sum(axis=1), axis=0) * 100
x = np.arange(len(pct)); w = 0.38
ax.bar(x - w/2, pct.get('center', 0), w, label='Injury centre', color='#c0392b')
ax.bar(x + w/2, pct.get('peripheral', 0), w, label='Peri-injury', color='#7f8c8d')
ax.set_xticks(x); ax.set_xticklabels(pct.index, rotation=20)
ax.set_ylabel('% of cells in region'); ax.legend(fontsize=8, frameon=False)
ax.set_title('C  GSE182398 endothelial subtypes: centre vs peri-injury', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# (d) TBI 特异性胶质/髓系簇
ax = fig.add_subplot(gs[1, 1])
enr = pd.read_csv(os.path.join(OUT, 'GSE294775_cluster_enrichment.csv'))
col = 'log2OR' if 'log2OR' in enr.columns else None
if col is None:
    enr['log2OR'] = np.log2(enr['OR_TBI_vs_CTR'].clip(lower=1e-3))
    col = 'log2OR'
top = enr.sort_values(col, ascending=False).head(12)
labels = [f'c{int(c)}' for c in top.leiden]
# 用标记评分给簇加标签
ms = pd.read_csv(os.path.join(OUT, 'GSE294775_marker_scores_per_leiden.csv'), index_col=0)
mcols = [c for c in ms.columns if c.startswith('score_')]
lab2 = {}
for cl in top.leiden:
    if cl in ms.index:
        s = ms.loc[cl, mcols]
        lab2[cl] = s.idxmax().replace('score_', '').replace('Microglia_', 'MG-').replace('MDM_monocyte', 'MDM')
labels = [f'c{int(c)}, {lab2.get(c, "")}' for c in top.leiden]
ax.barh(range(len(top))[::-1], top[col], color='#2980b9')
ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(labels, fontsize=7.5)
ax.axvline(0, color='k', lw=.6)
ax.set_xlabel('log2 OR (TBI vs control)')
ax.set_title('D  GSE294775: TBI-enriched glial/myeloid clusters', fontsize=10, loc='left')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Fig 5  Multi-modal evidence chain: clinical trajectory → blood biomarker axis → vascular subsets → CNS glia/myeloid',
             fontsize=11, y=0.98)
p = os.path.join(FIGDIR, 'Fig5_multimodal_evidence.png')
fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
print('[saved]', p)
