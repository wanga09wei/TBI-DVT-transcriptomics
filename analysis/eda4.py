import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
D=pd.read_pickle('merged.pkl')
def num(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
loc=D['血栓位置'].astype(str)
D['prox']=loc.str.contains('腘静脉|股静脉|髂|腓静脉|胫后|胫前').astype(int)
D['distal']=loc.str.contains('肌间').astype(int)
ev=D[D['血栓0_1'].notna()].copy()
print("=== 亚组: 累及近端(n=%d) vs 仅远端(n=%d) ==="%(ev['prox'].sum(),((ev['distal']==1)&(ev['prox']==0)).sum()))
for L in ['年龄','住院天数','T0_WBC','T0_血小板','T0_D-二聚体','T0_FIB','T1_血小板','T1_FIB','T2_血小板','T2_FIB']:
    if L not in ev.columns: continue
    v=num(ev[L]); g1=v[ev['prox']==1].dropna(); g0=v[(ev['distal']==1)&(ev['prox']==0)].dropna()
    print(f"{L:16s} 近端 med={g1.median():>8} (n={len(g1):3d}) | 仅远端 med={g0.median():>8} (n={len(g0):3d})")
print("\n=== 血栓检出时间 vs 近端累及 ===")
dt=pd.to_numeric(ev['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0],errors='coerce')
print("近端累及 median 天数=%.1f  仅远端 median=%.1f"%(dt[ev['prox']==1].median(), dt[(ev['distal']==1)&(ev['prox']==0)].median()))
print("\n=== 三时点配对变化(Wilcoxon近似, 仅描述) ===")
for L in ['血小板','FIB','D-二聚体','WBC']:
    a=num(D[f'T0_{L}']); b=num(D[f'T1_{L}']); c=num(D[f'T2_{L}'])
    print(f"{L:8s} T0={a.median():.2f} T1={b.median():.2f} T2={c.median():.2f}  (T0->T1 n={ (a.notna()&b.notna()).sum() }, T1->T2 n={ (b.notna()&c.notna()).sum() })")
print("\n=== 下肢骨折部位(大腿/股骨 vs 小腿/胫腓) 简易分层 ===")
fr=D['下肢骨折类型'].astype(str)
D['femur']=fr.str.contains('股骨|髋|髌').astype(int)
D['tibia']=fr.str.contains('胫|腓|踝|足').astype(int)
D['none_fx']=fr.str.contains('无').astype(int)
print("股骨/髋受累 n=%d, 胫腓/踝/足 n=%d, 记录为'无' n=%d"%(D['femur'].sum(),D['tibia'].sum(),D['none_fx'].sum()))
print("\n=== 脑出血类型分布 ===")
bt=D['脑出血类型'].astype(str)
for k,lab in [('1','硬膜外'),('2','硬膜下'),('3','蛛网膜下腔'),('4','脑实质')]:
    print(f"  {lab}: n={bt.str.contains(k).sum()}")
print("  记录为0/无: n=",((bt=='0')|(bt=='0.0')).sum())
print("\n=== 各年度入组 ===")
print(D.groupby('year').size().to_string())
print("\n=== 完整度: 至少有T0检验值 的病例数 ===")
t0cols=[c for c in D.columns if c.startswith('T0_')]
D['nt0']=D[t0cols].notna().sum(axis=1)
print("T0检验项>=5 的病例:",(D['nt0']>=5).sum(),"/",len(D))
