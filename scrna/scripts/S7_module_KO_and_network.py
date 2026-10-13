# -*- coding: utf-8 -*-
"""
S7: 共识模块基因补做虚拟敲除 + 模块内交叉调控网络 + 上游特异性判定
 (a) 对 S5 共识模块基因（SAT1/NR4A1/TIMP1/ACKR1/NNMT/EMP1）在损伤中心 EC 上做 scTenifoldKnk 虚拟敲除
 (b) 构建"KO 基因 × 下游模块基因"的 Z 矩阵 → 模块内定向调控网络（Fig 7）
 (c) 上游特异性判定：每个 KO 的 top-15 下游与模块的重叠度、以及模块内其他基因是否被下调/上调
 (d) 核查 NSFC 轴基因（MFSD2A / NEDD4L / CTNNB1 / ANXA2 / S100A10 / CDH5 / TJP1 / OCLN）是否出现在扰动下游
数据与基因面板与 S3 完全一致（保证可比），输出 S7_vKO_*.csv / S7_*.csv / figures/Fig7_module_network.png
"""
import os, time, glob, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scipy.sparse as sp
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'D:/TBI_DVT_Study'
OUT = os.path.join(ROOT, 'scrna', 'output')
FIG = os.path.join(ROOT, 'figures')
LOG = open(os.path.join(OUT, 'S7_run.log'), 'a', encoding='utf-8')
def log(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); LOG.write(m + '\n'); LOG.flush()

t0 = time.time()
def el(): return f'{(time.time()-t0)/60:.1f} min'

# ============ 数据（与 S3 同口径） ============
A = sc.read_h5ad(os.path.join(OUT, 'GSE182398_qc_clustered.h5ad'))
ec = (A.obs['score_EC'] > 0.3) & (A.obs['score_Neuron'] < 0.3) & (A.obs['score_Astrocyte'] < 0.3)
Ln = A[ec].copy(); Ln.X = Ln.layers['counts'].copy()
sc.pp.normalize_total(Ln, target_sum=1e4); sc.pp.log1p(Ln)
sc.pp.highly_variable_genes(Ln, n_top_genes=1500, batch_key='sample')
CAND = ['F3', 'SELP', 'VCAM1', 'ANGPT2', 'ESM1', 'CLDN5', 'VWF', 'PECAM1', 'ICAM1', 'SELE',
        'THBD', 'PROCR', 'PLAT', 'SERPINE1', 'CDH5', 'KDR', 'FLT1', 'TFPI', 'MMP14', 'PLAUR', 'MFSD2A']
panel = sorted(set(Ln.var_names[Ln.var.highly_variable]) | {g for g in CAND if g in Ln.var_names})
log(f'\n[data] EC {Ln.shape[0]} cells | 面板 {len(panel)} 基因 | {el()}')

Cnt = A[ec].copy(); Cnt.X = Cnt.layers['counts'].copy()
def counts_df(ad):
    M = ad[:, panel].X
    M = np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)
    return pd.DataFrame(M.T.astype(float), index=panel, columns=[f'c{i}' for i in range(M.shape[0])])
Dc = counts_df(Cnt[Cnt.obs.region == 'center'])
log(f'[data] counts center {Dc.shape}')

NC = dict(n_nets=10, n_samp_cells=500, q=0.9)

# ============ (a) 模块基因虚拟敲除 ============
cons = pd.read_csv(os.path.join(OUT, 'S5_vKO_consensus_module.csv'))
MODULE = list(cons.gene)
log(f'\n[JobA] 共识模块 {len(MODULE)} 基因: {MODULE}')
tgt = [g for g in MODULE if g in panel and (Dc.loc[g] > 0).mean() >= 0.10]
log(f'[JobA] 可敲除目标（面板内且中心 EC 表达率≥10%）: {tgt}')
try:
    from scTenifold import scTenifoldKnk
    for g in tgt:
        out_csv = os.path.join(OUT, f'S7_vKO_{g}_centerEC.csv')
        if os.path.exists(out_csv):
            log(f'[JobA] {g} 已存在跳过'); continue
        t1 = time.time(); log(f'[JobA] KO {g} …')
        try:
            res = scTenifoldKnk(Dc, ko_genes=[g], nc_kws=NC, td_kws={'K': 3}).build()
        except Exception as e:
            log(f'[JobA] {g} 跳过（{type(e).__name__}: {e}）'); continue
        res['ko_gene'] = g
        res.to_csv(out_csv, index=False, encoding='utf-8-sig')
        col = 'adjusted p-value'
        # 该 KO 对模块内其他基因的影响
        inter = res[res.Gene.isin(MODULE) & (res.Gene != g)][['Gene', 'Z', col]].sort_values('Z', ascending=False)
        log(f'[JobA] {g} 完成 ({(time.time()-t1)/60:.1f} min) | 对模块内基因影响 top: '
            + ', '.join(f'{r.Gene}(Z={r.Z:.2f})' for r in inter.head(6).itertuples()))
        log(f'[JobA] {el()}')
except Exception as e:
    log(f'[JobA][ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}')

# ============ (b) 模块内交叉调控矩阵 ============
kos = {}
for f in glob.glob(os.path.join(OUT, 'S7_vKO_*_centerEC.csv')):
    g = os.path.basename(f).split('_')[2]
    d = pd.read_csv(f).set_index('Gene')
    kos[g] = d
# 并入 S3 的 F3/ANGPT2/CLDN5/VWF 以扩大网络
for f in glob.glob(os.path.join(OUT, 'S3_vKO_*_centerEC.csv')):
    g = os.path.basename(f).split('_')[2]
    if g not in kos:
        kos[g] = pd.read_csv(f).set_index('Gene')
log(f'\n[JobB] 合并虚拟敲除实验 {len(kos)} 个: {list(kos)}')

mod_present = [g for g in MODULE if any(g in d.index for d in kos.values())]
Z = pd.DataFrame(index=mod_present, columns=list(kos.keys()), dtype=float)
Q = pd.DataFrame(index=mod_present, columns=list(kos.keys()), dtype=float)
col = 'adjusted p-value'
for ko, d in kos.items():
    for g in mod_present:
        if g in d.index:
            Z.loc[g, ko] = d.loc[g, 'Z']; Q.loc[g, ko] = d.loc[g, col]
Z.to_csv(os.path.join(OUT, 'S7_module_crossregulation_Z.csv'), encoding='utf-8-sig')
Q.to_csv(os.path.join(OUT, 'S7_module_crossregulation_FDR.csv'), encoding='utf-8-sig')
log('\n[JobB] 模块基因 × KO 的 Z 矩阵:\n' + Z.round(2).to_string())

# ============ (c) 上游特异性判定 ============
rows = []
for ko, d in kos.items():
    col = 'adjusted p-value'
    top15 = set(d.sort_values('Z', ascending=False).head(15).index)
    sig05 = set(d.index[d[col] < 0.05])
    rows.append({
        'KO_gene': ko,
        'top15_overlap_module': len(top15 & set(MODULE)),
        'FDR05_overlap_module': len(sig05 & set(MODULE)),
        'n_FDR05_total': len(sig05),
        'downstream_module_top': ', '.join([g for g in MODULE if g in top15 and g != ko]),
    })
spec = pd.DataFrame(rows).sort_values('top15_overlap_module', ascending=False)
spec.to_csv(os.path.join(OUT, 'S7_upstream_specificity.csv'), index=False, encoding='utf-8-sig')
log('\n[JobC] 上游特异性（各 KO 的 top-15 下游与模块重叠）:\n' + spec.to_string(index=False))

# ============ (d) NSFC 轴基因核查 ============
NSFC = ['MFSD2A', 'NEDD4L', 'NEDD4', 'CTNNB1', 'ANXA2', 'S100A10', 'CDH5', 'TJP1', 'OCLN',
        'CLDN5', 'F3', 'PLAUR', 'SERPINE1', 'TIMP1', 'MMP9', 'MFSD2B', 'SLC2A1', 'LEF1']
ns_rows = []
for ko, d in kos.items():
    for g in NSFC:
        if g in d.index:
            ns_rows.append({'KO_gene': ko, 'gene': g, 'Z': d.loc[g, 'Z'], 'FDR': d.loc[g, col]})
ns = pd.DataFrame(ns_rows)
if len(ns):
    ns.to_csv(os.path.join(OUT, 'S7_NSFC_axis_in_downstream.csv'), index=False, encoding='utf-8-sig')
    piv = ns.pivot_table(index='gene', columns='KO_gene', values='Z')
    log('\n[JobD] NSFC 轴基因在各虚拟敲除中的 Z:\n' + piv.round(2).to_string())

# ============ Fig 7 ============
fig = plt.figure(figsize=(13.5, 5.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.36)

# A: 有向网络（KO → 模块基因），仅绘 top 边
ax = fig.add_subplot(gs[0, 0])
edges = []
for ko, d in kos.items():
    for g in Z.index:
        if ko == g or pd.isna(Z.loc[g, ko]):
            continue
        if ko in d.index and d.loc[g, col] < 0.05:
            edges.append((ko, g, float(Z.loc[g, ko])))
try:
    import networkx as nx
    G = nx.DiGraph()
    for u, v, z in edges:
        G.add_edge(u, v, weight=z)
    pos = nx.spring_layout(G, k=0.9, seed=7, iterations=200)
    ws = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#c0392b',
                           width=[0.6 + 1.6 * (w / max(ws)) for w in ws], alpha=.55,
                           arrowsize=9, connectionstyle='arc3,rad=0.12')
    node_col = ['#8e44ad' if n in MODULE and n not in ('F3', 'ANGPT2', 'CLDN5', 'VWF') else '#2c3e50' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_col, node_size=430, alpha=.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color='white')
    ax.set_axis_off()
    ax.set_title('A  Directed module network: virtual KO → FDR<0.05 downstream module gene\n'
                 '(GSE182398 injury-centre EC; purple = consensus-module targets, dark = vascular targets)',
                 fontsize=9, loc='left')
except Exception as e:
    ax.text(.5, .5, f'network skipped: {e}', ha='center'); ax.axis('off')

# B: Z 热图
ax = fig.add_subplot(gs[0, 1])
M = Z.astype(float)
order_r = M.mean(axis=1).sort_values(ascending=False).index
order_c = M.mean(axis=0).sort_values(ascending=False).index
im = ax.imshow(M.loc[order_r, order_c].values, aspect='auto', cmap='Reds', vmin=0, vmax=np.nanmax(M.values))
ax.set_yticks(range(len(order_r))); ax.set_yticklabels(order_r, fontsize=8)
ax.set_xticks(range(len(order_c))); ax.set_xticklabels([f'KO\n{c}' for c in order_c], fontsize=7.5)
ax.set_title('B  Regulatory Z of each module gene under each virtual knockout', fontsize=9, loc='left')
fig.colorbar(im, ax=ax, shrink=0.85, label='Z')
fig.suptitle('Fig 7  Consensus vascular-stress module: internal regulatory structure and upstream specificity',
             fontsize=10.5, y=1.005)
p7 = os.path.join(FIG, 'Fig7_module_network.png')
fig.savefig(p7, dpi=300, bbox_inches='tight'); plt.close(fig)
log('\n[saved] ' + p7)
log(f'=== S7 结束，总耗时 {el()} ===')
LOG.close()
