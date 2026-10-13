# -*- coding: utf-8 -*-
"""
GSE294775 后分析：
A) leiden 簇 TBI vs CTR 富集（chi2 + OR）
B) 候选基因（S100A8/A9/A12, PADI4, MMP9, F3, VWF, CLDN5, PECAM1）在官方 celltype_l1 注释下
   的 TBI vs CTR Wilcoxon 表达差异（raw counts 层，负二项不做，报均值+秩检验）
输出：scrna/output/GSE294775_cluster_enrichment.csv / GSE294775_candidate_gene_diff.csv
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
import scanpy as sc

OUT = r'D:/TBI_DVT_Study/scrna/output'
A = sc.read_h5ad(os.path.join(OUT, 'GSE294775_qc_clustered.h5ad'))
print(A.shape, A.obs.columns.tolist())

# ---------- A) 簇富集 ----------
tab = A.obs.groupby(['leiden', 'group']).size().unstack(fill_value=0)
rows = []
tot_tbi = tab.get('TBI', 0).sum(); tot_ctr = tab.get('CTR', 0).sum()
for cl in tab.index:
    tbi = tab.loc[cl, 'TBI'] if 'TBI' in tab.columns else 0
    ctr = tab.loc[cl, 'CTR'] if 'CTR' in tab.columns else 0
    t2 = [[tbi, tot_tbi - tbi], [ctr, tot_ctr - ctr]]
    chi2, p, _, _ = stats.chi2_contingency(t2)
    OR = (tbi + 0.5) * (tot_ctr - ctr + 0.5) / ((tot_tbi - tbi + 0.5) * (ctr + 0.5))
    rows.append({'leiden': cl, 'TBI': tbi, 'CTR': ctr,
                 'pct_TBI': 100 * tbi / tot_tbi, 'pct_CTR': 100 * ctr / tot_ctr,
                 'OR_TBI_vs_CTR': OR, 'P_chi2': p})
enr = pd.DataFrame(rows).sort_values('P_chi2')
enr.to_csv(os.path.join(OUT, 'GSE294775_cluster_enrichment.csv'),
           index=False, encoding='utf-8-sig')
print('\n=== TBI 富集 top10 ===')
print(enr.head(10).round(3).to_string(index=False))
print('\n=== CTR 富集 top5 ===')
print(enr.tail(5).round(3).to_string(index=False))

# ---------- B) 候选基因差异（TBI vs CTR，按 celltype_l1） ----------
CAND = ['S100A8', 'S100A9', 'S100A12', 'PADI4', 'MMP9', 'F3', 'VWF', 'PECAM1', 'CLDN5', 'HMGB1']
present = [g for g in CAND if g in A.raw.var_names] if A.raw is not None else [g for g in CAND if g in A.var_names]
print('\n候选基因在矩阵中:', present)

# 用 log-normalized 层（scale 前的 raw）
Xlog = A.raw.X if A.raw is not None else A.X
gidx = {g: i for i, g in enumerate((A.raw.var_names if A.raw is not None else A.var_names))}
isl1 = 'celltype_l1' in A.obs.columns
rows2 = []
groups = A.obs['group'].astype(str).values
ctypes = A.obs['celltype_l1'].astype(str).values if isl1 else np.array(['NA'] * A.shape[0])
from scipy import sparse
for g in present:
    col = Xlog[:, gidx[g]]
    col = np.asarray(col.todense()).ravel() if sparse.issparse(col) else np.asarray(col).ravel()
    for ct in pd.unique(ctypes):
        m = ctypes == ct
        t, c = col[m & (groups == 'TBI')], col[m & (groups == 'CTR')]
        if min(len(t), len(c)) < 10: continue
        u, p = stats.mannwhitneyu(t, c, alternative='two-sided')
        rows2.append({'gene': g, 'celltype_l1': ct, 'n_TBI': len(t), 'n_CTR': len(c),
                      'mean_TBI': t.mean(), 'mean_CTR': c.mean(),
                      'log2ratio': np.log2((t.mean() + 1e-3) / (c.mean() + 1e-3)),
                      'P_MWU': p})
diff = pd.DataFrame(rows2)
# BH 校正
ok = diff['P_MWU'].notna().values
q = np.minimum.accumulate((np.sort(diff.loc[ok, 'P_MWU'].values) * ok.sum() /
                           (np.arange(ok.sum())[::-1] + 1))[::-1])
diff.loc[ok, 'FDR'] = q[np.argsort(np.argsort(diff.loc[ok, 'P_MWU'].values))]
diff.to_csv(os.path.join(OUT, 'GSE294775_candidate_gene_diff.csv'),
            index=False, encoding='utf-8-sig')
sig = diff[(diff.FDR < 0.05) & (diff.log2ratio.abs() > 0.25)]
print(f'\n显著（FDR<0.05 & |log2ratio|>0.25）: {len(sig)}/{len(diff)}')
print(sig.sort_values('gene').round(3).to_string(index=False))
