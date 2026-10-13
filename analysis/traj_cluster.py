# -*- coding: utf-8 -*-
"""
促凝轨迹表型分型（Group-Based Trajectory Modelling, GBTM）
数据：analysis/merged.pkl（真实临床队列 2014-2018，TBI + 下肢骨折，n=285）
主变量：血小板 三期（T0 入院 / T1 血栓检出 / T2 抗凝2周后），完整 n=214
辅助：WBC（同期完整 n=214）；FIB 子集验证

模型：有限混合，类内均值 = 时间的一阶多项式（GBTM 标准形式），
      类内残差方差同质（对角结构），EM 多起点估计，BIC 选型。
"""
import pandas as pd, numpy as np, warnings, json
from scipy.special import logsumexp
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
np.random.seed(0)

D = pd.read_pickle('merged.pkl')
E = D[D['血栓0_1'].notna()].copy()

def num(x):
    s = pd.Series(x).astype(str).str.replace('——', '', regex=False).str.replace('不详', '', regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

def build(lab, t2alt=None):
    cols = []
    for t in ['T0', 'T1', 'T2']:
        for cand in ([f'{t}_{t2alt}', f'{t}_{lab}'] if t2alt else [f'{t}_{lab}']):
            if cand in E.columns:
                cols.append(cand); break
        else:
            cols.append(None)
    M = pd.DataFrame({t: (num(E[c]) if c else np.nan) for t, c in zip(['T0','T1','T2'], cols)},
                        index=E.index)
    M = M.replace(0, np.nan)          # 血小板/WBC/FIB 不可能为 0 → 录入缺失
    return M

PL = build('血小板'); WB = build('WBC'); FB = build('FIB')
for nm, M in [('血小板', PL), ('WBC', WB), ('FIB', FB)]:
    v = M.values.ravel(); v = v[~np.isnan(v)]
    print(f'{nm:5s} 各期 n={[int(M[c].notna().sum()) for c in M.columns]} 三期完整={int(M.notna().all(axis=1).sum())}'
          f' | min={v.min():.1f} p1={np.percentile(v,1):.1f} med={np.median(v):.1f} '
          f'p99={np.percentile(v,99):.1f} max={v.max():.1f}')

# ================= 主分析样本 =================
mask = PL.notna().all(axis=1)
P = PL[mask].copy(); Wij = WB.loc[mask].copy(); Fij = FB.loc[mask].copy()
Y = P.values.astype(float)                      # n x 3
TGRID = np.array([0.0, 1.0, 2.0])
print(f'\n主分析样本 n = {len(P)}（血小板三期完整）')

def gbtm(Y, K, order=1, n_init=60, max_iter=1000, tol=1e-9):
    """有限混合 + 多项式均值（对角残差方差），EM 多起点。"""
    n, T = Y.shape
    Xd = np.vander(TGRID, order + 1, increasing=True)     # T x (order+1)
    p = Xd.shape[1]
    best = None
    for init in range(n_init):
        lab = KMeans(n_clusters=K, n_init=1, random_state=init).fit(Y).labels_
        beta = np.zeros((K, p)); sig = np.ones(K)
        for j in range(K):
            m = lab == j
            if m.sum() > p:
                beta[j] = np.linalg.lstsq(Xd, Y[m].mean(axis=0), rcond=None)[0]
            sig[j] = max(float(Y[m].var()) if m.sum() > 1 else 1.0, 1e-6)
        pi = np.clip(np.array([np.mean(lab == j) for j in range(K)]), 1e-3, None); pi /= pi.sum()
        prev = -np.inf
        for it in range(max_iter):
            logp = np.empty((n, K))
            for j in range(K):
                mu = Xd @ beta[j]                        # 长度 T
                r = Y - mu                               # n x T
                logp[:, j] = (np.log(pi[j]) - 0.5*T*np.log(2*np.pi*sig[j])
                              - 0.5*np.sum(r**2, axis=1)/sig[j])
            ll = float(logsumexp(logp, axis=1).sum())
            w = np.exp(logp - logp.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            Nk = w.sum(axis=0)
            for j in range(K):
                W = w[:, j]
                # 加权正规方程（时间点对所有人相同 → 设计矩阵的加权外积 = W_Σ · XᵀX）
                A = Xd.T @ Xd + 1e-9*np.eye(p)
                ybar = (Y * W[:, None]).sum(axis=0) / max(Nk[j], 1e-9)
                beta[j] = np.linalg.solve(A, Xd.T @ ybar)
                mu = Xd @ beta[j]
                sig[j] = max(float(np.sum(W[:, None]*(Y - mu)**2)/(Nk[j]*T)), 1e-6)
            pi = Nk / n
            if abs(ll - prev) < tol: break
            prev = ll
        if best is None or ll > best['ll']:
            best = dict(ll=ll, beta=beta.copy(), sig=sig.copy(), pi=pi.copy(),
                        w=w.copy(), iters=it)
    return best

def summarize(fit, Y, K, order):
    n = len(Y); p = order + 1
    npar = K*p + K + (K - 1)
    w = fit['w']; lc = w.argmax(axis=1)
    AvePP = float(w.max(axis=1).mean())
    OCC = []
    for j in range(K):
        idx = lc == j
        a = float(w[idx, j].mean()) if idx.sum() else np.nan
        pj = float(fit['pi'][j])
        OCC.append(float((a/(1-a))/(pj/(1-pj))) if (idx.sum() and 0 < a < 1 and 0 < pj < 1) else np.nan)
    ent = float(-np.sum(w*np.log(w + 1e-300))/n)
    return dict(K=K, order=order, ll=fit['ll'], npar=npar,
                BIC=-2*fit['ll'] + npar*np.log(n),
                AIC=-2*fit['ll'] + 2*npar,
                AvePP=AvePP, entropy=ent, OCC=OCC,
                size=[int((lc == j).sum()) for j in range(K)],
                pi=[float(x) for x in fit['pi']],
                beta=fit['beta'].tolist(), sigma=fit['sig'].tolist(),
                labels=lc.tolist(), posterior=w.tolist())

# ---- K=1 基准 ----
Xd = np.vander(TGRID, 2, increasing=True)
b = np.linalg.lstsq(Xd, Y.mean(axis=0), rcond=None)[0]
mu = Xd @ b; sig = float(np.mean((Y - mu)**2))
ll1 = float(np.sum(-0.5*3*np.log(2*np.pi*sig) - 0.5*np.sum((Y - mu)**2, axis=1)/sig))
sel = {1: dict(K=1, order=1, ll=ll1, npar=3, BIC=-2*ll1 + 3*np.log(len(Y)),
               AIC=-2*ll1 + 6, AvePP=1.0, entropy=0.0, OCC=[np.nan],
               size=[len(Y)], pi=[1.0], beta=[b.tolist()], sigma=[sig])}

print('\n=== GBTM 选型（order=1；仅 3 个时点，高阶不可识别）===')
print(f"{'K':>2} {'logLik':>11} {'npar':>5} {'BIC':>11} {'ΔBIC':>9} {'AvePP':>7} {'Entropy':>8} {'min%':>6}  sizes")
for K in range(2, 6):
    fit = gbtm(Y, K)
    sel[K] = summarize(fit, Y, K, 1)
    print(f"{K:>2} {sel[K]['ll']:>11.1f} {sel[K]['npar']:>5} {sel[K]['BIC']:>11.1f} "
          f"{sel[K]['BIC']-sel[1]['BIC']:>9.1f} {sel[K]['AvePP']:>7.3f} {sel[K]['entropy']:>8.3f} "
          f"{100*min(sel[K]['size'])/len(Y):>6.1f}  {sel[K]['size']}")
print(f"{1:>2} {sel[1]['ll']:>11.1f} {sel[1]['npar']:>5} {sel[1]['BIC']:>11.1f} {'0.0':>9}  (baseline, single class)")

json.dump({str(k): {kk: vv for kk, vv in v.items()
                    if kk not in ('labels', 'posterior')} for k, v in sel.items()},
          open('gbtm_selection.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# 保存最优 K 的分型结果（供后续关联分析与出图）——先按 BIC 选，再看临床可解释性
bic_best = min(sel, key=lambda k: sel[k]['BIC'])
print(f"\nBIC 最优 K = {bic_best}")
out = pd.DataFrame({'T0': P['T0'].values, 'T1': P['T1'].values, 'T2': P['T2'].values},
                   index=P.index)
if 'labels' in sel[bic_best]:
    out['class_BICbest'] = sel[bic_best]['labels']
    out['posterior_max'] = np.max(sel[bic_best]['posterior'], axis=1)
else:
    out['class_BICbest'] = 0
    out['posterior_max'] = 1.0
for k in range(2, 6):
    if 'labels' in sel[k]:
        out[f'class_K{k}'] = sel[k]['labels']
out.to_csv('traj_classes.csv', encoding='utf-8-sig')
print('[saved] gbtm_selection.json / traj_classes.csv')
print('\n各类轨迹均值（原始单位，血小板 ×10^9/L）:')
for k in range(2, 6):
    if 'labels' not in sel[k]: continue
    lb = np.array(sel[k]['labels'])
    print(f"  K={k}: " + " | ".join(
        f"C{j+1}(n={int((lb==j).sum())}) T0/1/2=" +
        "/".join(f"{Y[lb==j][:,t].mean():.0f}" for t in range(3)) for j in range(k)))
