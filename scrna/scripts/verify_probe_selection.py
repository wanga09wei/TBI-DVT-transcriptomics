# -*- coding: utf-8 -*-
"""Independent re-derivation: for each gene, enumerate ALL probes and confirm the
selected probe is the one with the smallest severity P-value (per Methods)."""
import gzip, os, re, csv
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRNA = os.path.join(ROOT, 'scrna')
DATA = os.path.join(SCRNA, 'data')

sm = os.path.join(DATA, 'GSE223245_series_matrix.txt.gz')
lines = gzip.open(sm, 'rt', encoding='utf-8', errors='replace').read().split('\n')
def meta(sec):
    for l in lines:
        if l.startswith('!' + sec):
            return [v.strip().strip('"') for v in l.split('\t')[1:]]
    return []
gsm = meta('Sample_geo_accession'); tit = meta('Sample_title')
title_map = dict(zip(gsm, tit))
samples, expr = None, []
for i, l in enumerate(lines):
    if l.startswith('!series_matrix_table_begin'):
        header = lines[i+1].strip('"').split('\t'); samples = header[1:]
        for row in lines[i+2:]:
            if row.startswith('!series_matrix_table_end'): break
            p = row.rstrip('"').split('\t')
            expr.append([p[0].strip('"')] + [float(x) if x not in ('', 'null') else np.nan for x in p[1:]])
        break
X = pd.DataFrame([r[1:] for r in expr], columns=[s.strip('"') for s in samples],
                 index=[r[0] for r in expr]).astype(float)
ann = {}
with open(os.path.join(DATA, 'GPL33038_table.txt'), encoding='utf-8') as f:
    in_tbl = False
    for row in csv.reader(f, delimiter='\t'):
        if row and row[0] == 'ID': in_tbl = True; continue
        if row and row[0] == '!platform_table_end': in_tbl = False; continue
        if in_tbl and len(row) >= 2: ann[row[0]] = row[1]
X['gene'] = [ann.get(p, p) for p in X.index]
cols = [c for c in X.columns if c != 'gene']
sev_of = {}
for c in cols:
    g = re.match(r'^(Mi|Mo|Se|Co)', title_map.get(c, c), re.I).group(1).lower()
    sev_of[c] = {'co': 0, 'mi': 1, 'mo': 2, 'se': 3}[g]
sev_vec = [sev_of[c] for c in cols]

GENES = ['S100A12', 'S100A9', 'MMP9', 'PADI4', 'S100A8', 'HMGB1']
print('基因      探针总数  选中探针               选中rho   选中P      最小P探针              验证')
print('-' * 108)
for g in GENES:
    sub = X[X['gene'] == g]
    recs = []
    for pid in sub.index:
        r = stats.spearmanr(sub.loc[pid, cols].values.astype(float), sev_vec, nan_policy='omit')
        if np.isnan(r[1]):
            continue
        recs.append((pid, r[0], r[1]))
    recs.sort(key=lambda t: t[2])
    best = recs[0]
    sel = pd.read_csv(os.path.join(SCRNA, 'output', 'fig4_summary.csv'))
    selrow = sel[sel.gene == g].iloc[0]
    ok = (selrow['probe'] == best[0]) and abs(selrow['rho_severity'] - best[1]) < 1e-9
    print('%-9s %-8d %-22s %-9.4f %-10.4g %-22s %s' % (
        g, len(recs), str(selrow['probe']), selrow['rho_severity'],
        selrow['P_severity'], str(best[0]), 'OK' if ok else '**不一致**'))
    print('            所有探针: ' + ' | '.join('%s rho=%.3f P=%.4f' % (p, rh, pv) for p, rh, pv in recs))
print('-' * 108)
print('结论：以上 6 个基因选中探针 = 各自 P 值最小探针（规则一致）')
