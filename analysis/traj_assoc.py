# -*- coding: utf-8 -*-
"""
轨迹表型的严格性检验 + 临床关联分析
1) 水平 vs 形态分解：类间 T0 水平差异 + 类内归一化形态
2) 形态专用聚类（个体内 z 标准化后聚类）与主分型的一致性（ARI/kappa）
3) 与真实结局关联：近端/主干累及、IVC 滤器、抗凝后出血、血栓并发症、检出天数、住院天数、骨折/颅脑损伤构成
4) WBC 与 FIB 轨迹按主分型验证
"""
import pandas as pd, numpy as np, warnings, json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings('ignore')
np.random.seed(0)

D = pd.read_pickle('merged.pkl')
E = D[D['血栓0_1'].notna()].copy()
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

# ---------- 结局变量 ----------
loc = E['血栓位置'].astype(str)
E['lv_prox']  = loc.str.contains(r'腘|股总|股浅|股深|股静脉|髂').astype(float)
E['lv_calf']  = loc.str.contains(r'肌间|胫后|胫前|腓').astype(float)
E['lv_sup']   = loc.str.contains(r'大隐|小隐').astype(float)
E['lv_bilat'] = loc.str.contains('双').astype(float)
E['age']=num(E['年龄']); E['los']=num(E['住院天数']); E['male']=(num(E['性别'])==1).astype(float)
for c,n in [('吸烟史','smoke'),('高血压','htn'),('糖尿病','dm'),('静脉曲张病史','varicose'),
            ('合并其它出血','bleed'),('IVC滤器','ivcf'),('血栓并发症','comp'),('脂质斑块','plaque')]:
    E[n]=yn(E[c])
sd=E['首次手术天数'].astype(str); E['opday']=np.where(sd.str.contains('当天'),0,num(sd))
E['detect_day']=num(E['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0]).fillna(num(E['血栓检出时间']))
bt=E['脑出血类型'].astype(str)
for c,n in [('1','EDH'),('2','SDH'),('3','SAH'),('4','IPH')]: E['ICH_'+n]=bt.str.contains(c).astype(float)
fx=E['下肢骨折类型'].astype(str)
E['fx_femur']=fx.str.contains('股骨|髋|髌').astype(float); E['fx_tibia']=fx.str.contains('胫|腓|踝|足').astype(float)
rv=E['出院前复查'].astype(str)
E['resolved']=rv.str.contains('消失').astype(float); E['rechecked']=(~rv.isin(['未查','未检查','nan',''])).astype(float)
for t,l in [('T0','血红蛋白'),('T1','血红蛋白'),('T2','血色素')]:
    E[f'v_{t}_HB']=num(E[f'{t}_{l}']) if f'{t}_{l}' in E.columns else np.nan
E['v_T0_WBC']=num(E['T0_WBC']); E['v_T1_WBC']=num(E['T1_WBC']); E['v_T2_WBC']=num(E['T2_WBC'])
E['v_T0_FIB']=num(E['T0_FIB']); E['v_T1_FIB']=num(E['T1_FIB']); E['v_T2_FIB']=num(E['T2_FIB'])
E['v_T0_DD']=num(E['T0_D-二聚体']); E['v_T1_DD']=num(E['T1_D-二聚体']); E['v_T2_DD']=num(E['T2_D-二聚体'])

CL = 'class_K3'                     # BIC 最优主分型
TC[CL] = TC[CL].astype(int)
E = E.join(TC[[CL, 'T0', 'T1', 'T2']].rename(columns={CL: 'traj'}), how='inner')
print(f'主分型样本 n = {len(E)}  | 类别分布: {E["traj"].value_counts().sort_index().to_dict()}')

# 类标签按 T0 水平排序：0=低起、1=中起、2=高起
order = E.groupby('traj')['T0'].mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
E['traj'] = E['traj'].map(remap)
LBL = {0: 'Low-baseline sustained rise', 1: 'Mid-baseline sustained rise', 2: 'High-baseline peak-and-fall'}
print('类别重排:', {k: f'{LBL[k]} (n={int((E["traj"]==k).sum())})' for k in sorted(E['traj'].unique())})

# ================= 1) 水平 vs 形态分解 =================
print('\n=== 1) 水平 vs 形态分解 ===')
R = E[['T0','T1','T2']].values
print('类间 T0 水平: ' + ' | '.join(
    f'C{k}={E.loc[E.traj==k,"T0"].mean():.0f}' for k in sorted(E.traj.unique())))
print('ANOVA(类 x T0): F=%.1f, P=%.2e' % stats.f_oneway(*[E.loc[E.traj==k,'T0'] for k in sorted(E.traj.unique())]))
print('Kruskal(类 x T0): H=%.1f, P=%.2e' % stats.kruskal(*[E.loc[E.traj==k,'T0'] for k in sorted(E.traj.unique())]))
# 类内归一化形态（除以自身 T0）
shape_ratio = R / R[:, [0]]
print('\n类内 T1/T0 与 T2/T0 比值（形态指标，与水平无关）:')
for k in sorted(E.traj.unique()):
    m = (E.traj == k).values
    r1 = shape_ratio[m, 1]; r2 = shape_ratio[m, 2]
    print(f'  C{k} n={m.sum():3d}  T1/T0={np.median(r1):.2f} (IQR {np.percentile(r1,25):.2f}-{np.percentile(r1,75):.2f})'
          f'  T2/T0={np.median(r2):.2f}')
print('Kruskal(类 x T1/T0): H=%.1f P=%.2e' % stats.kruskal(*[shape_ratio[(E.traj==k).values,1] for k in sorted(E.traj.unique())]))
print('Kruskal(类 x T2/T0): H=%.1f P=%.2e' % stats.kruskal(*[shape_ratio[(E.traj==k).values,2] for k in sorted(E.traj.unique())]))

# ================= 2) 形态专用聚类（个体内 z 标准化） =================
print('\n=== 2) 形态专用聚类（个体内 z 标准化，剔除水平信息）===')
Z = (R - R.mean(axis=1, keepdims=True)) / (R.std(axis=1, keepdims=True) + 1e-9)
for K in [2, 3]:
    lab = KMeans(n_clusters=K, n_init=50, random_state=0).fit(Z).labels_
    ari = adjusted_rand_score(E['traj'].values, lab)
    ct = pd.crosstab(E['traj'], lab)
    print(f'  shape-only K={K}: ARI vs 主分型 = {ari:.3f}')
    print(ct.to_string())
print('  → ARI 低 = 主分型主要由"水平"驱动；ARI 高 = 存在真实的"形态"分型')
print('\n个体内标准化后的类均值（应全部 ≈ 0，仅看形状）:')
Zdf = pd.DataFrame(Z, columns=['T0','T1','T2'], index=E.index); Zdf['traj'] = E['traj'].values
print(Zdf.groupby('traj').mean().round(3).to_string())

# ================= 3) 临床关联 =================
print('\n=== 3) 轨迹表型的临床关联 ===')
def cont(v):
    g = [pd.to_numeric(E.loc[E.traj==k, v], errors='coerce').dropna() for k in sorted(E.traj.unique())]
    g = [x for x in g if len(x) > 2]
    if len(g) < 2: return ''
    return f'P(Kruskal)={stats.kruskal(*g).pvalue:.3f}'
def catf(v, den=None):
    m = (pd.to_numeric(E[den],errors='coerce')==1) if den else pd.Series(True, index=E.index)
    rows=[]
    for k in sorted(E.traj.unique()):
        s = pd.to_numeric(E.loc[E.traj==k, v], errors='coerce'); mm = m.loc[E.traj==k]
        kk = int(s[mm].sum()); nn = int(s[mm].notna().sum())
        rows.append((int(kk), int(nn)))
    tab = [[r[0], r[1]-r[0]] for r in rows]
    try:
        p = stats.chi2_contingency(tab)[1]
        if stats.chi2_contingency(tab)[3].min() < 5:
            p = stats.fisher_exact([tab[0], tab[1]])[1] if len(tab)==2 else p
    except Exception:
        p = np.nan
    return ' | '.join(f'C{k}: {rows[k][0]}/{rows[k][1]} ({100*rows[k][0]/max(rows[k][1],1):.1f}%)' for k in range(len(rows))) + f'  P={p:.3f}' if p==p else '—'

for v, lab in [('lv_prox','近端/主干累及'), ('lv_calf','小腿深静脉'), ('lv_sup','浅静脉'),
               ('lv_bilat','双侧'), ('ivcf','IVC 滤器'), ('bleed','抗凝后出血'),
               ('comp','血栓并发症'), ('rechecked','出院前复查'), ('resolved','血栓消失(复查者)'),
               ('fx_femur','股骨/髋骨折'), ('fx_tibia','胫腓/踝/足骨折'),
               ('ICH_EDH','EDH'), ('ICH_SDH','SDH'), ('ICH_SAH','SAH'), ('ICH_IPH','IPH'),
               ('male','男性'), ('smoke','吸烟史'), ('htn','高血压')]:
    print(f'{lab:16s} {catf(v, "rechecked" if v=="resolved" else None)}')
for v, lab in [('age','年龄'), ('los','住院天数'), ('opday','首次手术天数'), ('detect_day','血栓检出天数'),
               ('v_T0_HB','T0 血红蛋白'), ('v_T1_HB','T1 血红蛋白')]:
    print(f'{lab:16s} ' + ' '.join(f'C{k}={pd.to_numeric(E.loc[E.traj==k,v],errors="coerce").median():.1f}'
          for k in sorted(E.traj.unique())) + f'  {cont(v)}')

# ================= 4) WBC / FIB 验证 =================
print('\n=== 4) WBC 与 FIB 轨迹按主分型验证 ===')
for lab, cols, unit in [('WBC', ['v_T0_WBC','v_T1_WBC','v_T2_WBC'], '×10⁹/L'),
                        ('FIB', ['v_T0_FIB','v_T1_FIB','v_T2_FIB'], 'g/L'),
                        ('D-dimer', ['v_T0_DD','v_T1_DD','v_T2_DD'], 'mg/L')]:
    print(f'  {lab} ({unit}):')
    for k in sorted(E.traj.unique()):
        s = E[E.traj==k]
        vals = [pd.to_numeric(s[c],errors='coerce') for c in cols]
        print(f'    C{k} n={len(s):3d}  ' + ' / '.join(f'{v.median():.2f}' for v in vals) +
              '   (n=' + ','.join(str(int(v.notna().sum())) for v in vals) + ')')

E[['traj','T0','T1','T2','lv_prox','lv_bilat','ivcf','bleed','comp','detect_day','age','los']]\
  .to_csv('traj_phenotype.csv', encoding='utf-8-sig')
print('\n[saved] traj_phenotype.csv')
