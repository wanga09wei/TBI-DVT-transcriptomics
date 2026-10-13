# -*- coding: utf-8 -*-
"""
S3 v2: 三虚拟扰动（GSE182398 人 TBI 脑内皮细胞）
关键修正：
  1) scTenifoldNet/Knk 的 QC 按**原始计数**设计（lib≥1000、MT 比例过滤）→ Net/Knk 一律喂 counts
  2) 基因面板：在 log 数据上选 HVG（seurat flavor），再回 counts 取值
  3) Job2 修复：coo_matrix 必须 .toarray()（np.asarray 会得到 0 维 object 数组）
  4) Job4 decoupler 2.x API：dc.op.collectri()
任务：① 虚拟敲除 Knk ② 虚拟过表达 网络传播 ③ 虚拟 GRN 差分调控 Net ④ TF 活性
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scipy.sparse as sp
import scanpy as sc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:/TBI_DVT_Study/scrna/output'
LOG = open(os.path.join(OUT, 'S3_run.log'), 'a', encoding='utf-8')
def log(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); LOG.write(m + '\n'); LOG.flush()

t0 = time.time()
def el(): return f'{(time.time()-t0)/60:.1f} min'

# ============ 数据 ============
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
Dp = counts_df(Cnt[Cnt.obs.region == 'peripheral'])
# log 版（Job2 自研方法用）
Lg = Ln[:, panel].copy()
def log_df(ad):
    M = ad.X; M = np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)
    return pd.DataFrame(M.T.astype(float), index=panel, columns=[f'l{i}' for i in range(M.shape[0])])
Lc = log_df(Lg[Lg.obs.region == 'center'])
log(f'[data] counts center {Dc.shape} / periph {Dp.shape}；log center {Lc.shape}')

NC = dict(n_nets=10, n_samp_cells=500, q=0.9)

# ============ ③ 虚拟 GRN（Net 差分调控） ============
try:
    from scTenifold import scTenifoldNet
    log('\n[Job3] scTenifoldNet 差分调控：损伤中心 vs 外周 EC …')
    t1 = time.time()
    net = scTenifoldNet(Dc, Dp, x_label='injury_center', y_label='peripheral',
                        nc_kws=NC, td_kws={'K': 3})
    dreg = net.build()
    dreg.to_csv(os.path.join(OUT, 'S3_vGRN_diff_regulation_center_vs_periph.csv'),
                index=False, encoding='utf-8-sig')
    log(f'[Job3] 完成 {dreg.shape} ({(time.time()-t1)/60:.1f} min) | {el()}')
    log('[Job3] top10:\n' + dreg.sort_values('adjusted p-value').head(10).to_string(index=False))
except Exception as e:
    log(f'[Job3][ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}')

# ============ ① 虚拟敲除（Knk） ============
KO_TARGETS = [g for g in ['F3', 'ANGPT2', 'CLDN5', 'VWF', 'MFSD2A', 'TIMP1', 'ACKR1']
              if g in panel and (Dc.loc[g] > 0).mean() >= 0.15][:4]
log(f'\n[Job1] 虚拟敲除目标（QC 前预筛表达率≥15%）: {KO_TARGETS}')
try:
    from scTenifold import scTenifoldKnk
    for g in KO_TARGETS:
        out_csv = os.path.join(OUT, f'S3_vKO_{g}_centerEC.csv')
        if os.path.exists(out_csv):
            log(f'[Job1] {g} 已存在跳过'); continue
        t1 = time.time(); log(f'[Job1] KO {g} …')
        try:
            res = scTenifoldKnk(Dc, ko_genes=[g], nc_kws=NC, td_kws={'K': 3}).build()
        except Exception as e:
            log(f'[Job1] {g} 跳过（{type(e).__name__}: {e}）'); continue
        res['ko_gene'] = g
        res.to_csv(out_csv, index=False, encoding='utf-8-sig')
        top = res.sort_values('adjusted p-value').head(15)
        sig_names = [f'{r.Gene}(Z={r.Z:.2f})' for r in top.itertuples() if r._6 < 0.05]
        log(f'[Job1] {g} 完成 ({(time.time()-t1)/60:.1f} min) | FDR<0.05 下游: '
            + (', '.join(sig_names) if sig_names else '（无）'))
        log(f'[Job1] {g} top15:\n' + top[['Gene', 'Z', 'FC', 'adjusted p-value']].to_string(index=False))
        log(f'[Job1] {el()}')
except Exception as e:
    log(f'[Job1][ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}')

# ============ ② 虚拟过表达（网络传播，自研） ============
log('\n[Job2] 虚拟过表达（PCNet 网络传播，log 数据）…')
try:
    from scTenifold.core import make_networks
    nets = make_networks(Lc, n_nets=10, n_samp_cells=500, n_comp=3, q=0.9, random_state=1)
    Adj = np.mean([np.asarray(n.toarray(), dtype=float) for n in nets], axis=0)
    Adj = (Adj + Adj.T) / 2; np.fill_diagonal(Adj, 0)
    genes = list(Lc.index); gi = {g: i for i, g in enumerate(genes)}
    Arow = Adj / (np.abs(Adj).sum(axis=1, keepdims=True) + 1e-10)

    def propagate(gene, alpha=0.7, steps=3):
        s = np.zeros(len(genes)); s[gi[gene]] = 1.0
        cur, acc = s.copy(), np.zeros(len(genes))
        for _ in range(steps):
            cur = alpha * (Arow.T @ cur); acc += cur
        acc[gi[gene]] = 0
        return pd.Series(acc, index=genes)

    OE_TARGETS = [g for g in ['ANGPT2', 'F3', 'ESM1', 'SELP', 'VWF'] if g in genes][:3]
    oe = pd.concat([propagate(g).rename(g) for g in OE_TARGETS], axis=1)
    oe.to_csv(os.path.join(OUT, 'S3_vOE_network_propagation.csv'), encoding='utf-8-sig')
    for g in OE_TARGETS:
        log(f'[Job2] OE {g} → top 下游: {list(oe[g].sort_values(ascending=False).head(10).index)}')
    topg = oe.abs().mean(axis=1).sort_values(ascending=False).head(25).index
    V = oe.loc[topg].values
    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(V, aspect='auto', cmap='RdBu_r', vmin=-np.abs(V).max(), vmax=np.abs(V).max())
    ax.set_yticks(range(len(topg))); ax.set_yticklabels(topg, fontsize=8)
    ax.set_xticks(range(oe.shape[1])); ax.set_xticklabels(oe.columns, rotation=45, ha='right', fontsize=9)
    ax.set_title('Virtual overexpression: propagated influence\n(GSE182398 injury-centre endothelium, PCNet)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6, label='propagated influence')
    fig.savefig(os.path.join(OUT, 'S3_vOE_heatmap.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
    log(f'[Job2] 完成 | {el()}')
except Exception as e:
    log(f'[Job2][ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}')

# ============ ④ TF 活性（decoupler 2.x） ============
log('\n[Job4] decoupler TF 活性（CollecTRI）…')
try:
    import decoupler as dc
    net_tf = dc.op.collectri(organism='human')
    Lfull = A[ec].copy(); Lfull.X = Lfull.layers['counts'].copy()
    sc.pp.normalize_total(Lfull, target_sum=1e4); sc.pp.log1p(Lfull)
    dc.mt.ulm(Lfull, net=net_tf, verbose=False)
    est = Lfull.obsm['score_ulm'] if 'score_ulm' in Lfull.obsm else list(Lfull.obsm.values())[0]
    tf = pd.DataFrame(est, index=Lfull.obs_names, columns=net_tf['source'].unique())
    tf['region'] = Lfull.obs['region'].values
    m = tf.groupby('region').mean().T
    if {'center', 'peripheral'} <= set(m.columns):
        m['diff'] = m['center'] - m['peripheral']
        m.sort_values('diff', ascending=False).to_csv(
            os.path.join(OUT, 'S3_TF_activity_center_vs_periph.csv'), encoding='utf-8-sig')
        log('[Job4] center-up TF top10: ' + ', '.join(m.sort_values('diff', ascending=False).head(10).index))
        log('[Job4] periph-up TF top10: ' + ', '.join(m.sort_values('diff').head(10).index))
    log(f'[Job4] 完成 | {el()}')
except Exception as e:
    log(f'[Job4][WARN] {type(e).__name__}: {e}')

log(f'\n=== S3 结束，总耗时 {el()} ===')
LOG.close()
