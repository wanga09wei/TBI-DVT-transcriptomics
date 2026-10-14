# -*- coding: utf-8 -*-
"""
GSE223245 bulk 分析（重大更正后的正确用法）：
GEO 上该系列仅有 Agilent-086188 ceRNA 芯片（GPL33038），不含 scRNA-seq/TCR-seq。
本脚本用 series matrix 自带的 log 表达值做探针级差异表达（探索性）：
  组 1：TBI 全体（Mi/Mo/Se，n=12）  vs  Control（n=4）
  辅助：severity 线性趋势（Co=0, Mi=1, Mo=2, Se=3, Spearman）
输出：scrna/output/GSE223245_bulk_*.csv + 候选枢纽基因表
注意：n 极小（12 vs 4），结果为探索性，仅供与公共验证数据集交叉，不得单独作为主文证据。
"""
import gzip, os, re, csv
import numpy as np, pandas as pd
from scipy import stats


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values, returned in the input order and
    capped at 1.0 (NaN preserved).

    NOTE (2026-10-14): replaces an inline implementation that (i) omitted the
    trailing [::-1] on the cumulative minimum, leaving the adjusted values
    ordered by descending p, and (ii) un-sorted them with
    np.argsort(np.argsort(p)), which is not the inverse permutation. The
    published FDR column was therefore mis-aligned with its p-values and
    contained values greater than 1.
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


SCRNA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SCRNA, 'data'); OUT = os.path.join(SCRNA, 'output')
os.makedirs(OUT, exist_ok=True)

# ---------- 1) 读取表达矩阵 ----------
sm = os.path.join(DATA, 'GSE223245_series_matrix.txt.gz')
lines = gzip.open(sm, 'rt', encoding='utf-8', errors='replace').read().split('\n')
def meta(sec):
    for l in lines:
        if l.startswith('!' + sec):
            return [v.strip().strip('"') for v in l.split('\t')[1:]]
    return []
gsm_acc = meta('Sample_geo_accession'); smp_title = meta('Sample_title')
title_map = dict(zip(gsm_acc, smp_title))
print('title 映射示例:', list(title_map.items())[:3])
samples, expr = None, []
for i, l in enumerate(lines):
    if l.startswith('!series_matrix_table_begin'):
        header = lines[i+1].strip('"').split('\t')
        samples = header[1:]
        for row in lines[i+2:]:
            if row.startswith('!series_matrix_table_end'): break
            p = row.rstrip('"').split('\t')
            expr.append([p[0].strip('"')] + [float(x) if x not in ('', 'null') else np.nan for x in p[1:]])
        break
X = pd.DataFrame([r[1:] for r in expr], columns=[s.strip('"') for s in samples],
                 index=[r[0] for r in expr]).astype(float)
print('表达矩阵:', X.shape, '| 样本:', list(X.columns))

# ---------- 2) 注释 ----------
ann = {}
with open(os.path.join(DATA, 'GPL33038_table.txt'), encoding='utf-8') as f:
    in_tbl = False
    for row in csv.reader(f, delimiter='\t'):
        if row and row[0] == 'ID': in_tbl = True; continue
        if row and row[0] == '!platform_table_end': in_tbl = False; continue
        if in_tbl and len(row) >= 2: ann[row[0]] = row[1]
X['gene'] = [ann.get(p, p) for p in X.index]
print('注释覆盖:', sum(1 for p in X.index if p in ann), '/', len(X))

# ---------- 3) 分组 ----------
def grp(s):
    t = title_map.get(s, s)
    m = re.match(r'^(Mi|Mo|Se|Co)', t, re.I)
    if not m: return 'NA'
    return {'mi': 'TBI', 'mo': 'TBI', 'se': 'TBI', 'co': 'Control'}[m.group(1).lower()]
groups = np.array([grp(c) for c in X.columns if c != 'gene'])
cols = [c for c in X.columns if c != 'gene']
sev = np.array([{'co': 0, 'mi': 1, 'mo': 2, 'se': 3}[re.match(r'^(Mi|Mo|Se|Co)', title_map.get(c, c), re.I).group(1).lower()] for c in cols])
Xv = X[cols]
print('分组:', pd.Series(groups).value_counts().to_dict())

# ---------- 4) 差异表达（Welch t + BH） ----------
tb = Xv.loc[:, groups == 'TBI'].values
ct = Xv.loc[:, groups == 'Control'].values
t_, p_ = stats.ttest_ind(tb, ct, axis=1, equal_var=False, nan_policy='omit')
lfc = tb.mean(axis=1) - ct.mean(axis=1)
res = pd.DataFrame({'probe': X.index, 'gene': X['gene'].values, 'log2FC_TBI_vs_Ctrl': lfc,
                    'P': p_, 'mean_TBI': tb.mean(axis=1), 'mean_Ctrl': ct.mean(axis=1)})
# severity 趋势
rho = [stats.spearmanr(Xv.iloc[i].values, sev, nan_policy='omit')[0] for i in range(Xv.shape[0])]
psp = [stats.spearmanr(Xv.iloc[i].values, sev, nan_policy='omit')[1] for i in range(Xv.shape[0])]
res['rho_severity'] = rho; res['P_severity'] = psp
res['FDR'] = bh_fdr(res['P'].values)
res = res.sort_values('P')
sig = res[(res.FDR < 0.05) & (res.log2FC_TBI_vs_Ctrl.abs() > 0.58)]
print(f'显著探针（FDR<0.05 & |log2FC|>0.58）: {len(sig)}')
print(sig.head(25).round(4).to_string(index=False))

res.to_csv(os.path.join(OUT, 'GSE223245_bulk_all_probes.csv'), index=False, encoding='utf-8-sig')
sig.to_csv(os.path.join(OUT, 'GSE223245_bulk_sig_probes.csv'), index=False, encoding='utf-8-sig')

# ---------- 5) 候选枢纽基因（免疫血栓轴） ----------
CAND = ['HMGB1','F3','PADI4','S100B','S100A8','S100A9','MMP9','PF4','PPBP','SELP','VWF',
        'TREM1','ELANE','MPO','CSF3R','CXCR2','CXCL2','IL1B','TNF','TFPI','SERPINE1',
        'ITGA2B','NLRP3','TLR4','TLR2','C5AR1','S100A12','RETN','LCN2','PADI2']
cand = res[res.gene.isin(CAND)].copy()
cand.to_csv(os.path.join(OUT, 'GSE223245_bulk_candidate_hub_genes.csv'), index=False, encoding='utf-8-sig')
print('\n=== 候选枢纽基因在芯片上的表现 ===')
print(cand.head(30).round(4).to_string(index=False))

# ---------- 6) 轨迹候选基因（与我们队列三时点轨迹表型呼应） ----------
print('\n=== 汇总 ===')
print('TBI(n=12) vs Control(n=4)：探针级 Welch t；显著 %d 个（FDR<0.05, |log2FC|>0.58）' % len(sig))
print('候选枢纽基因命中：', cand.gene.nunique(), '个基因 /', len(cand), '个探针')
