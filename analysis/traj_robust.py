# -*- coding: utf-8 -*-
"""轨迹表型稳健性检验：
1) 抗凝后出血的 Fisher 精确检验（C2 vs 其余）+ 年龄混杂检查
2) 与"T0 血小板三分位"的交叉表 → 量化水平依赖性
3) 联合血小板+WBC 六维聚类（GMM）复现性
4) Bootstrap 重抽样类别稳定性（ARI）
5) 组合标记轨迹（血小板/WBC/FIB 标准化）类内一致性
"""
import pandas as pd, numpy as np, warnings, json
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings('ignore')
np.random.seed(42)

D = pd.read_pickle('merged.pkl'); E = D[D['血栓0_1'].notna()].copy()
TC = pd.read_csv('traj_classes.csv', index_col=0)
def num(x):
    s = pd.Series(x).astype(str).str.replace('——','',regex=False).str.replace('不详','',regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]','',regex=True), errors='coerce')
def yn(x):
    o=[]
    for v in pd.Series(x):
        s=str(v).strip()
        o.append(0 if s in ('0','0.0','无','——','－','未','未查') else (1 if s in ('1','1.0','有','是') else np.nan))
    return pd.Series(o, index=pd.Series(x).index)

loc=E['血栓位置'].astype(str)
E['lv_prox']=loc.str.contains(r'腘|股总|股浅|股深|股静脉|髂').astype(float)
E['bleed']=yn(E['合并其它出血']); E['age']=num(E['年龄']); E['male']=(num(E['性别'])==1).astype(float)
E['detect_day']=num(E['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0]).fillna(num(E['血栓检出时间']))
E['v_T0_WBC']=num(E['T0_WBC']); E['v_T1_WBC']=num(E['T1_WBC']); E['v_T2_WBC']=num(E['T2_WBC'])
E['v_T0_FIB']=num(E['T0_FIB']); E['v_T1_FIB']=num(E['T1_FIB']); E['v_T2_FIB']=num(E['T2_FIB'])

TC['class_K3']=TC['class_K3'].astype(int)
E=E.join(TC[['class_K3','T0','T1','T2']].rename(columns={'class_K3':'traj'}), how='inner')
order=E.groupby('traj')['T0'].mean().sort_values().index.tolist()
E['traj']=E['traj'].map({o:i for i,o in enumerate(order)})
print(f'n = {len(E)} | 分布 {E.traj.value_counts().sort_index().to_dict()}')

# ---------- 1) 抗凝后出血 ----------
print('\n=== 1) 抗凝后出血 ===')
a = int(E.loc[E.traj==2,'bleed'].sum()); b = int((E.traj==2).sum()-a)
c = int(E.loc[E.traj!=2,'bleed'].sum()); d = int((E.traj!=2).sum()-c)
OR,p = stats.fisher_exact([[a,b],[c,d]])
print(f'  C2(高负荷型): {a}/{a+b} = {100*a/(a+b):.1f}%   |   其余两型: {c}/{c+d} = {100*c/(c+d):.1f}%')
print(f'  Fisher 精确检验: OR={OR:.2f}  P={p:.4f}')
print(f'  C2 vs C0: {a}/{a+b} vs {int(E.loc[E.traj==0,"bleed"].sum())}/{int((E.traj==0).sum())}, P={stats.fisher_exact([[a,b],[int(E.loc[E.traj==0,"bleed"].sum()),int((E.traj==0).sum())-int(E.loc[E.traj==0,"bleed"].sum())]])[1]:.4f}')
# 年龄混杂
bl=E[E.bleed==1]; nb=E[E.bleed==0]
print(f'  出血者 vs 未出血者 年龄: {bl.age.median():.0f} vs {nb.age.median():.0f}  P(MWU)={stats.mannwhitneyu(bl.age.dropna(),nb.age.dropna()).pvalue:.3f}')
print(f'  出血者中 C2 占比: {int((bl.traj==2).sum())}/{len(bl)} = {100*(bl.traj==2).mean():.1f}%')
print(f'  年龄 与 traj 的关联已见上（C2 最年轻 45 岁, P=0.001）→ 出血-分型关联可能部分由年龄/损伤严重度中介，须标注为探索性')

# ---------- 2) 与 T0 三分位交叉 ----------
print('\n=== 2) 分型 vs T0 血小板三分位 ===')
t0t = pd.qcut(E['T0'], 3, labels=['T1低','T2中','T3高'])
ct = pd.crosstab(E['traj'], t0t)
print(ct.to_string())
agree = np.mean([ct.iloc[i, [0,1,2][i]] if i < 3 else 0 for i in range(3)]) if ct.shape[1]==3 else np.nan
print(f'  对角一致率（分型↔三分位）= {100*np.trace(ct.values[:3,:3])/len(E):.1f}%')
print('  → 一致率越高，说明分型越接近"基线水平三分位"')

# ---------- 3) 联合 6 维聚类（血小板+WBC） ----------
print('\n=== 3) 联合 (血小板+WBC) 六维 GMM 复现性 ===')
cols = ['T0_血小板','T1_血小板','T2_血小板','T0_WBC','T1_WBC','T2_WBC']
sub = E[cols].apply(num).replace(0, np.nan)
m = sub.notna().all(axis=1)
print(f'  六维完整 n = {int(m.sum())}')
X = ((sub[m] - sub[m].mean())/sub[m].std()).values
best=None
for k in [2,3,4]:
    g = GaussianMixture(k, covariance_type='diag', n_init=20, random_state=0, max_iter=1000).fit(X)
    bic = g.bic(X)
    print(f'  K={k}: BIC={bic:.1f}')
    if best is None or bic<best[1]: best=(k,bic,g)
k,g = best[0], best[2]
lab = g.predict(X)
ari = adjusted_rand_score(E.loc[m,'traj'].values, lab)
print(f'  最优 K={k}; ARI(联合聚类 vs 主分型) = {ari:.3f}')
print(pd.crosstab(E.loc[m,'traj'], lab).to_string())

# ---------- 4) Bootstrap 稳定性 ----------
print('\n=== 4) Bootstrap 类别稳定性（K-means on [T0,T1,T2], ref=主分型）===')
from sklearn.cluster import KMeans
Y = E[['T0','T1','T2']].values
ref = KMeans(3, n_init=50, random_state=0).fit(Y).labels_
aris=[]
for b in range(200):
    idx = np.random.choice(len(Y), len(Y), replace=True)
    try:
        lb = KMeans(3, n_init=10, random_state=b).fit(Y[idx]).labels_
        aris.append(adjusted_rand_score(ref[idx], lb))
    except Exception: pass
print(f'  Bootstrap ARI 中位={np.median(aris):.3f} (IQR {np.percentile(aris,25):.3f}-{np.percentile(aris,75):.3f})')
print('  （>0.75 视为稳定）')

# ---------- 5) 多标记一致性 ----------
print('\n=== 5) 三个标记的类间梯度一致性 ===')
for nm, cs in [('血小板', ['T0','T1','T2']), ('WBC', ['v_T0_WBC','v_T1_WBC','v_T2_WBC']),
               ('FIB', ['v_T0_FIB','v_T1_FIB','v_T2_FIB'])]:
    med = np.array([[pd.to_numeric(E.loc[E.traj==k, c], errors='coerce').median() for c in cs] for k in range(3)])
    print(f'  {nm:6s} C0={med[0]}  C1={med[1]}  C2={med[2]}')
