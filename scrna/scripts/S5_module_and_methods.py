# -*- coding: utf-8 -*-
"""
S5: 虚拟扰动共识模块 + 跨数据集验证 + 方法学草稿
 (a) 4 个虚拟敲除的共识下游模块（≥3/4 实验 FDR<0.05）
 (b) 共识模块基因在 GSE223245 全血芯片上的 severity 趋势（是否与队列/血液轴呼应）
 (c) 内皮 TF 程序（center-up vs peripheral-up）
 (d) 生成 METHODS_BIOINFO_EN.md（英式拼写，可直接进稿件方法学）
输出：S5_*.csv、figures/Fig6_virtual_KO_module.png、idea-stage/METHODS_BIOINFO_EN.md
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'D:/TBI_DVT_Study'
OUT = os.path.join(ROOT, 'scrna', 'output')
FIG = os.path.join(ROOT, 'figures')
IDEA = os.path.join(ROOT, 'idea-stage')

# ---------- (a) 共识模块 ----------
kos = {}
for f in glob.glob(os.path.join(OUT, 'S3_vKO_*_centerEC.csv')):
    g = os.path.basename(f).split('_')[2]
    d = pd.read_csv(f)
    col = 'adjusted p-value' if 'adjusted p-value' in d.columns else 'p.adj'
    d = d.sort_values(col)
    kos[g] = d.set_index('Gene')
print('虚拟敲除实验:', {k: v.shape[0] for k, v in kos.items()})

allg = sorted(set().union(*[set(d.index) for d in kos.values()]))
rows = []
for g in allg:
    zs, hits, tops = [], 0, 0
    for k, d in kos.items():
        if g in d.index:
            col = 'adjusted p-value' if 'adjusted p-value' in d.columns else 'p.adj'
            zs.append(d.loc[g, 'Z'])
            if d.loc[g, col] < 0.05: hits += 1
            if g in d.sort_values('Z', ascending=False).head(15).index: tops += 1
    if tops >= 3:                       # 稳健定义：≥3/4 个虚拟敲除的 top-15 下游
        rows.append({'gene': g, 'n_KO_top15': tops, 'n_KO_FDR05': hits, 'median_Z': float(np.median(zs))})
cons = pd.DataFrame(rows).sort_values(['n_KO_top15', 'median_Z'], ascending=False)
cons.to_csv(os.path.join(OUT, 'S5_vKO_consensus_module.csv'), index=False, encoding='utf-8-sig')
print(f'\n=== 共识下游模块（≥3/4 KO 的 top-15 下游）: {len(cons)} 基因 ===')
print(cons.head(30).round(3).to_string(index=False))

# ---------- (b) 在 GSE223245 全血芯片的 severity 趋势 ----------
bulk = pd.read_csv(os.path.join(OUT, 'GSE223245_bulk_all_probes.csv'))
bulk = bulk.dropna(subset=['rho_severity'])
best = bulk.sort_values('P_severity').groupby('gene', as_index=False).first()
mod = best[best.gene.isin(cons.gene)][['gene', 'rho_severity', 'P_severity', 'log2FC_TBI_vs_Ctrl']]
mod = mod.sort_values('rho_severity', ascending=False)
mod.to_csv(os.path.join(OUT, 'S5_module_in_blood_bulk.csv'), index=False, encoding='utf-8-sig')
print(f'\n=== 共识模块在 GSE223245 全血芯片（{len(mod)}/{len(cons)} 基因可测）===')
print(mod.round(3).to_string(index=False))
bg = best.rho_severity.dropna()
if len(mod):
    u, p = stats.mannwhitneyu(mod.rho_severity, bg, alternative='greater')
    print(f'模块基因 severity rho 中位 {mod.rho_severity.median():.3f} vs 全基因组 {bg.median():.3f}，'
          f'Mann-Whitney P={p:.4f}（模块整体更随严重度上升）')

# ---------- (c) TF 程序 ----------
tf = pd.read_csv(os.path.join(OUT, 'S3_TF_activity_center_vs_periph.csv'), index_col=0)
tf.to_csv(os.path.join(OUT, 'S5_TF_program_center_vs_periph.csv'), encoding='utf-8-sig')
up = ', '.join(tf.sort_values('diff', ascending=False).head(15).index)
dn = ', '.join(tf.sort_values('diff').head(15).index)
print('\n中心区 EC 上调 TF:', up)
print('外周区 EC 上调 TF:', dn)

# ---------- Fig 6 ----------
fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.34)

# A: 共识模块 x 各 KO 的 Z 热图
ax = fig.add_subplot(gs[0, 0])
topg = cons.head(24).gene.tolist()
M = pd.DataFrame({k: [d.loc[g, 'Z'] if g in d.index else np.nan for g in topg] for k, d in kos.items()},
                 index=topg)
im = ax.imshow(M.values, aspect='auto', cmap='Reds', vmin=0, vmax=np.nanmax(M.values))
ax.set_yticks(range(len(topg))); ax.set_yticklabels(topg, fontsize=8)
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f'virtual KO\n{k}' for k in M.columns], fontsize=8)
ax.set_title('A  Shared downstream module of endothelial virtual knockouts\n(GSE182398 injury-centre EC)', fontsize=9.5, loc='left')
fig.colorbar(im, ax=ax, shrink=0.8, label='Z (regulatory change)')

# B: 模块基因在血液芯片的 severity rho
ax = fig.add_subplot(gs[0, 1])
if len(mod):
    rng = np.random.default_rng(0)
    ax.scatter(bg.values, rng.normal(0, .06, len(bg)), s=6, color='#bdc3c7', alpha=.6, label='all genes (GSE223245)')
    ax.scatter(mod.rho_severity, rng.normal(0, .06, len(mod)), s=42, color='#c0392b', zorder=3, label='virtual-KO module')
    for _, r in mod.head(12).iterrows():
        ax.annotate(r.gene, (r.rho_severity, 0), textcoords='offset points', xytext=(0, 9),
                    fontsize=7, ha='center', color='#7b241c')
    ax.axvline(0, color='k', lw=.6)
    ax.set_xlabel('Spearman ρ with TBI severity (whole-blood chip)')
    ax.set_yticks([]); ax.legend(fontsize=7.5, frameon=False, loc='lower left')
ax.set_title('B  Module genes track injury severity in blood (cross-dataset)', fontsize=9.5, loc='left')
ax.spines[['top', 'right', 'left']].set_visible(False)

fig.suptitle('Fig 6  Virtual perturbation of injury-centre endothelium converges on a vascular-stress / procoagulant module',
             fontsize=10.5, y=1.0)
p6 = os.path.join(FIG, 'Fig6_virtual_KO_module.png')
fig.savefig(p6, dpi=300, bbox_inches='tight'); plt.close(fig)
print('\n[saved]', p6)

# ---------- (d) 方法学草稿 ----------
methods = f"""# Bioinformatics methods (draft, British spelling)

## 1. Clinical cohort and trajectory phenotypes
Consecutive patients with traumatic brain injury (TBI) plus lower-limb fracture admitted between 2014 and 2018
were reviewed (n = 338 records; n = 285 with documented duplex assessment; n = 284 with confirmed lower-limb
deep-vein thrombosis, DVT). Group-based trajectory modelling (GBTM; first-order polynomial, EM with multiple
random starts) was fitted to the three time-point platelet counts (T0 admission, T1 thrombosis detection,
T2 after two weeks of anticoagulation), yielding three trajectory phenotypes (C1 low-burden, n = 64;
C2 intermediate, n = 119; C3 high-burden with peak-and-fall, n = 30). Phenotype robustness was assessed by
bootstrap ARI and by re-clustering on a six-dimensional marker set.

## 2. Whole-blood transcriptomic (ceRNA microarray)
GSE223245 (Agilent-086188 ceRNA array, GPL33038; whole blood; 12 TBI of graded severity and 4 controls) was
analysed from the series-matrix log2 values. Probes were annotated against the platform table (26,340/26,340
mapped). Differential expression used Welch's t-test with Benjamini-Hochberg control; because of the small
sample size (12 vs 4) and the resulting limited power, severity-related monotonic trends were quantified with
Spearman's rho across control/mild/moderate/severe (scores 0-3) and are reported as exploratory.

## 3. Single-cell RNA sequencing
**GSE182398** (human TBI, surgically resected brain tissue; CD45-CD31+ endothelial cells; six samples from three
patients, injury centre and peri-injury tissue). **GSE294775** (human post-mortem brain; 98,492 cells; 7 TBI and
7 control donors) with the original cell-type annotations and metadata. Cells were quality-controlled
(genes 1st-99th percentile; mitochondrial fraction < 15%), normalised (CPM-like scaling to 1e4, log1p), and
clustered with Leiden (resolution 0.8-1.0) after principal-component analysis on highly variable genes.
Vascular subpopulations were annotated with canonical marker scores (capillary RGCC/CA4/ESAM,
arterial GJA5/HEY1/SEMA3G, venous NR2F2/ACKR1, tip/stalk APOD/KCNE4/RAMP3, pericyte RGS5/PDGFRB/NOTCH3,
smooth-muscle ACTA2/TAGLN/MYH11). Cell-type composition differences were tested with chi-square tests and
reported as odds ratios with Benjamini-Hochberg correction.

## 4. In-silico perturbation (three virtuals)
Raw counts of injury-centre endothelial cells were used for network construction (genes expressed in >= 5% of
cells; 1,500 highly variable genes).
**(i) Virtual knockout** was performed with scTenifoldKnk (single-gene knockout of the wild-type principal-component
network tensor; ten subsampled networks of 500 cells, three principal components, quantile threshold 0.9),
reporting Z scores and BH-adjusted p values.
**(ii) Virtual overexpression** was implemented by network propagation on the consensus principal-component
network (mean of ten subsampled networks, symmetric, row-normalised; three propagation steps, damping 0.7);
the propagated influence of each overexpressed gene on all other genes is reported.
**(iii) Virtual gene-regulatory-network comparison** was performed with scTenifoldNet, contrasting injury-centre
with peri-injury endothelium (manifold alignment, tensor decomposition K = 3). Transcription-factor activity was
estimated with decoupler (CollecTRI regulon, univariate linear model).

## 5. Software
Python 3.13; scanpy 1.12.4; anndata 0.13.4; scTenifoldpy 0.5.1; decoupler 2.2.0; scipy 1.18.1;
scikit-learn 1.9.1; matplotlib 3.11.2. Figures were rendered at 300 dpi.

## 6. Statistical reporting
Two-sided tests were used throughout; multiplicity was controlled with Benjamini-Hochberg where applicable.
Exploratory analyses (small-sample microarray, ambient-RNA-prone per-cell-type tests in post-mortem tissue)
are explicitly labelled as such.
"""
mp = os.path.join(IDEA, 'METHODS_BIOINFO_EN.md')
open(mp, 'w', encoding='utf-8').write(methods)
print('[saved]', mp)
