# -*- coding: utf-8 -*-
"""最终版：Table 1 / Table 2 / 分层诊断（临床标准口径，可复现）"""
import pandas as pd, numpy as np, warnings, json
from scipy import stats
warnings.filterwarnings('ignore')

D = pd.read_pickle('merged.pkl')
E = D[D['血栓0_1'].notna()].copy()
loc = E['血栓位置'].astype(str)

# ===== 临床标准分层（关键词已补全"股总/股浅/股深"）=====
PROX = r'腘|股总|股浅|股深|股静脉|髂'
CALF = r'肌间|胫后|胫前|腓'
SUP  = r'大隐|小隐'
E['lv_prox']  = loc.str.contains(PROX).astype(int)
E['lv_calf']  = loc.str.contains(CALF).astype(int)
E['lv_sup']   = loc.str.contains(SUP).astype(int)
E['lv_bilat'] = loc.str.contains('双').astype(int)
def g(r):
    if r['lv_prox']: return 'Proximal/trunk'
    if r['lv_calf']: return 'Distal calf only'
    if r['lv_sup']:  return 'Superficial only'
    return 'Unspecified'
E['group']=E.apply(g,axis=1)
print("=== 最终分层 ===")
print(E['group'].value_counts().to_string())
print("双侧:",int(E['lv_bilat'].sum())," | 累及小腿深静脉(含近端):",int(E['lv_calf'].sum()))
A=E[E['group']=='Proximal/trunk']; B=E[E['group']=='Distal calf only']
print(f"Proximal/trunk n={len(A)} | Distal calf only n={len(B)} | Other n={len(E)-len(A)-len(B)}")

# ===== 派生变量 =====
def num(x):
    s=pd.Series(x).astype(str).str.replace('——','',regex=False).str.replace('不详','',regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
def yn(x):
    o=[]
    for v in pd.Series(x):
        s=str(v).strip()
        o.append(0 if s in ('0','0.0','无','——','－','未','未查') else (1 if s in ('1','1.0','有','是') else np.nan))
    return pd.Series(o,index=pd.Series(x).index)

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
E['fx_none']=fx.str.contains('^无$|^0').astype(float)
mech=E['致伤机制'].astype(str)
E['mech_traffic']=mech.str.contains('交通|车祸').astype(float); E['mech_fall']=mech.str.contains('坠落|摔|砸').astype(float)
rv=E['出院前复查'].astype(str)
E['resolved']=rv.str.contains('消失').astype(float)
E['rechecked']=(~rv.isin(['未查','未检查','nan',''])).astype(float)

LABS=[('WBC','WBC (×10⁹/L)'),('血小板','Platelet (×10⁹/L)'),('PT','PT (s)'),('INR','INR'),
      ('FIB','Fibrinogen (g/L)'),('部分凝血','APTT (s)'),('D-二聚体','D-dimer (mg/L)'),
      ('胆固醇','Cholesterol (mmol/L)'),('甘油三酯','Triglyceride (mmol/L)'),
      ('高密度脂蛋白','HDL-C (mmol/L)'),('低密度脂蛋白','LDL-C (mmol/L)'),('血红蛋白','Haemoglobin (g/L)')]
HB={'T0':'血红蛋白','T1':'血红蛋白','T2':'血色素'}
for t in ['T0','T1','T2']:
    for k,_ in LABS:
        for c in [f'{t}_{k}', f'{t}_{HB.get(t,k)}']:
            if c in E.columns: E[f'v_{t}_{k}']=num(E[c]); break

# ===== 统计 =====
def fmt(s,d=1): return '—' if len(s)==0 else f"{s.median():.{d}f} ({s.quantile(.25):.{d}f}–{s.quantile(.75):.{d}f})"
def cont(v,lab,d=1):
    ov=pd.to_numeric(E[v],errors='coerce').dropna()
    a=pd.to_numeric(E.loc[A.index,v],errors='coerce').dropna(); b=pd.to_numeric(E.loc[B.index,v],errors='coerce').dropna()
    p='—'
    if len(a)>=3 and len(b)>=3:
        try: p=f"{stats.mannwhitneyu(a,b,alternative='two-sided').pvalue:.3f}"
        except Exception: pass
    return [lab,fmt(ov,d),fmt(a,d),fmt(b,d),str(len(ov)),p]
def cat(v,lab,den=None):
    m=(pd.to_numeric(E[den],errors='coerce')==1) if den else pd.Series(True,index=E.index)
    def rk(idx):
        s=pd.to_numeric(E.loc[idx,v],errors='coerce'); mm=m.loc[idx]
        return int(s[mm].sum()), int(s[mm].notna().sum())
    tm=m.reindex(E.index).fillna(False); t=pd.to_numeric(E[v],errors='coerce')
    tn=int(t[tm].notna().sum()); tk=int(t[tm].sum()); k1,n1=rk(A.index); k0,n0=rk(B.index)
    pc=lambda k,n: f"{k} ({100*k/n:.1f}%)" if n>0 else '—'
    p='—'
    if n1>0 and n0>0:
        tab=[[k1,n1-k1],[k0,n0-k0]]
        try:
            exp=stats.chi2_contingency(tab)[3]
            p=f"{stats.fisher_exact(tab)[1]:.3f}" if exp.min()<5 else f"{stats.chi2_contingency(tab)[0:2][1]:.3f}"
        except Exception: pass
    return [lab, pc(tk,tn) if tn else '—', pc(k1,n1), pc(k0,n0), str(tn), p]

rows=[cont('age','Age, years',0),cat('male','Male sex'),
 cat('mech_traffic','Mechanism: traffic accident'),cat('mech_fall','Mechanism: fall/falling object'),
 cat('ICH_EDH','Cranial EDH'),cat('ICH_SDH','Cranial SDH'),cat('ICH_SAH','Cranial SAH'),cat('ICH_IPH','Cranial IPH'),
 cat('fx_femur','Fracture: femur/hip/patella'),cat('fx_tibia','Fracture: tibia/fibula/ankle/foot'),
 cat('smoke','Smoking history'),cat('htn','Hypertension'),cat('dm','Diabetes mellitus'),cat('varicose','Varicose veins'),
 cont('opday','Time to first surgery, days',0),cont('los','Length of hospital stay, days',0),
 cont('detect_day','Time to VTE detection, days post-injury',0),
 cat('lv_bilat','Bilateral involvement'),cat('lv_sup','Superficial vein (GSV/SSV) involvement'),
 cat('plaque','Vascular lipid plaque on ultrasound'),cat('ivcf','IVC filter placement'),
 cat('bleed','Bleeding after anticoagulation'),cat('comp','Thrombotic complications (PE etc.)'),
 cat('rechecked','Ultrasound re-checked before discharge'),
 cat('resolved','Thrombus resolution',den='rechecked')]
T1=pd.DataFrame(rows,columns=['Variable',f'Overall (N={len(E)})',f'Proximal/trunk (n={len(A)})',f'Distal calf only (n={len(B)})','n','P'])
print("\n"+T1.to_string(index=False))
T1.to_csv('table1_baseline.csv',index=False,encoding='utf-8-sig')
with open('table1_baseline.md','w',encoding='utf-8') as f:
    f.write('| '+' | '.join(T1.columns)+' |\n|'+'---|'*len(T1.columns)+'\n')
    for _,r in T1.iterrows(): f.write('| '+' | '.join(str(x) for x in r.values)+' |\n')

tr=[]
for k,lab in LABS:
    d={'Laboratory variable':lab}
    for t,tl in [('T0','Admission'),('T1','VTE detection'),('T2','2 wk post-anticoag')]:
        c=f'v_{t}_{k}'
        if c in E.columns:
            v=E[c]
            d[tl]=f"{v.median():.2f} ({v.quantile(.25):.2f}–{v.quantile(.75):.2f})" if v.notna().sum() else '—'
            d[tl+' n']=int(v.notna().sum())
    a=E.get('v_T0_'+k); b=E.get('v_T1_'+k)
    if a is not None and b is not None:
        mm=a.notna()&b.notna(); d['paired n']=int(mm.sum())
        if mm.sum()>=5:
            try: d['P (T0 vs T1)']=f"{stats.wilcoxon(a[mm],b[mm]).pvalue:.1e}"
            except Exception: d['P (T0 vs T1)']='—'
    tr.append(d)
T2=pd.DataFrame(tr); print("\n"+T2.to_string(index=False))
T2.to_csv('table2_trajectory.csv',index=False,encoding='utf-8-sig')
with open('table2_trajectory.md','w',encoding='utf-8') as f:
    f.write('| '+' | '.join(T2.columns)+' |\n|'+'---|'*len(T2.columns)+'\n')
    for _,r in T2.iterrows(): f.write('| '+' | '.join(str(x) for x in r.values)+' |\n')

json.dump({'group_counts':{k:int(v) for k,v in E['group'].value_counts().items()},
 'bilateral':int(E['lv_bilat'].sum()),'superficial':int(E['lv_sup'].sum()),
 'proximal':len(A),'distal_calf_only':len(B)},open('layers.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print("\n[saved] table1/table2 + layers.json")
