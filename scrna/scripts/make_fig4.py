# -*- coding: utf-8 -*-
"""
Fig 4: Neutrophil / NETs axis grades with TBI severity (GSE223245 bulk ceRNA chip).
Panel A: mean +/- SEM z-scored expression across Control/Mild/Moderate/Severe for 5 genes.
Panel B: per-sample z-scored heatmap ordered by severity (incl. HMGB1 as negative control).
Exploratory (n=16); connects our cohort's inflammatory-coagulation trajectory axis.
Output: figures/Fig4_neutrophil_axis.png (300 dpi) + scrna/output/fig4_data.csv
"""
import gzip, os, re, csv
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRNA = os.path.join(ROOT, 'scrna')
DATA = os.path.join(SCRNA, 'data'); OUT = os.path.join(SCRNA, 'output')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)

# ---------- read series matrix ----------
sm = os.path.join(DATA, 'GSE223245_series_matrix.txt.gz')
lines = gzip.open(sm, 'rt', encoding='utf-8', errors='replace').read().split('\n')
def meta(sec):
    for l in lines:
        if l.startswith('!' + sec):
            return [v.strip().strip('"') for v in l.split('\t')[1:]]
    return []
gsm_acc = meta('Sample_geo_accession'); smp_title = meta('Sample_title')
title_map = dict(zip(gsm_acc, smp_title))
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

# ---------- annotation ----------
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
order = sorted(cols, key=lambda c: sev_of[c])
labels = ['Control']*4 + ['Mild']*4 + ['Moderate']*4 + ['Severe']*4

# ---------- pick best probe per gene (lowest severity P) ----------
GENES = ['S100A12', 'S100A9', 'MMP9', 'PADI4', 'S100A8', 'HMGB1']
rows, probe_of = [], {}
for g in GENES:
    sub = X[X['gene'] == g]
    best = None
    for pid in sub.index:
        r = stats.spearmanr(sub.loc[pid, cols].values.astype(float),
                            [sev_of[c] for c in cols], nan_policy='omit')
        if best is None or r[1] < best[1]:
            best = (pid, r[0], r[1])
    probe_of[g] = best[0]
    rows.append({'gene': g, 'probe': best[0], 'rho_severity': best[1], 'P_severity': best[2]})
summary = pd.DataFrame(rows)
print(summary.round(4).to_string(index=False))

M = X.loc[[probe_of[g] for g in GENES], order].astype(float)
M.index = GENES
# z-score per gene (probe level)
Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1, ddof=0), axis=0)

# group stats
groups_lab = ['Control', 'Mild', 'Moderate', 'Severe']
gmat = {}
for g in GENES:
    vals = [Z.loc[g, [c for c in order if sev_of[c] == k]] for k in range(4)]
    gmat[g] = vals

# ---------- plot ----------
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32)

# Panel A: severity trend
axA = fig.add_subplot(gs[0, 0])
cmap = plt.get_cmap('YlOrRd')
colors = {g: cmap(0.35 + 0.6*i/4) for i, g in enumerate(GENES[:5])}
for g in GENES[:5]:
    means = [np.mean(v) for v in gmat[g]]
    sems = [np.std(v, ddof=1)/np.sqrt(len(v)) for v in gmat[g]]
    x = np.arange(4)
    axA.plot(x, means, '-o', color=colors[g], lw=2, ms=5, label=f'{g} (ρ={summary.loc[summary.gene==g,"rho_severity"].iloc[0]:.2f})')
    axA.fill_between(x, np.array(means)-np.array(sems), np.array(means)+np.array(sems),
                     color=colors[g], alpha=0.15)
# HMGB1 as contrast
g = 'HMGB1'
means = [np.mean(v) for v in gmat[g]]
axA.plot(np.arange(4), means, '--s', color='#7f7f7f', lw=1.6, ms=4.5,
         label=f'HMGB1, mRNA (ρ={summary.loc[summary.gene==g,"rho_severity"].iloc[0]:.2f})')
axA.axhline(0, color='k', lw=0.6, alpha=0.4)
axA.set_xticks(range(4)); axA.set_xticklabels(groups_lab)
axA.set_xlabel('TBI severity group', fontsize=11)
axA.set_ylabel('Expression (z-score of log2 signal)', fontsize=11)
axA.set_title('A  Neutrophil activation / NETs axis grades with severity', fontsize=12, loc='left')
axA.legend(fontsize=8.5, frameon=False, loc='upper left')
axA.spines[['top', 'right']].set_visible(False)

# Panel B: heatmap
axB = fig.add_subplot(gs[0, 1])
vmax = np.nanmax(np.abs(Z.values))
im = axB.imshow(Z.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axB.set_yticks(range(len(GENES))); axB.set_yticklabels(GENES, fontsize=10)
ticks = [0, 3.5, 7.5, 11.5]
axB.set_xticks(ticks); axB.set_xticklabels(groups_lab, fontsize=10)
axB.set_xlabel('Samples ordered by severity (n=4 per group)', fontsize=11)
axB.set_title('B  Per-sample expression profile', fontsize=12, loc='left')
for k, name in zip([1.5, 5.5, 9.5], ['Mild | Moderate', 'Moderate | Severe']):
    pass
axB.axvline(3.5, color='k', lw=0.8); axB.axvline(7.5, color='k', lw=0.8); axB.axvline(11.5, color='k', lw=0.8)
cb = fig.colorbar(im, ax=axB, shrink=0.85, pad=0.02)
cb.set_label('z-score', fontsize=10)

fig.suptitle('GSE223245 (Agilent ceRNA chip, whole blood): neutrophil DAMP/NETs programme '
             'tracks injury severity — exploratory (n=16)', fontsize=11, y=0.99)
fig.savefig(os.path.join(FIG, 'Fig4_neutrophil_axis.png'), dpi=300, bbox_inches='tight')
print('saved:', os.path.join(FIG, 'Fig4_neutrophil_axis.png'))

# export data behind figure
Z.to_csv(os.path.join(OUT, 'fig4_data_zscores.csv'), encoding='utf-8-sig')
summary.to_csv(os.path.join(OUT, 'fig4_summary.csv'), index=False, encoding='utf-8-sig')
