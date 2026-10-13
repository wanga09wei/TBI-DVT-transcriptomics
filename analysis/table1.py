# -*- coding: utf-8 -*-
"""Table 1/2 生成器 v2（修正 P 值、分组口径、分母）"""
import pandas as pd, numpy as np, warnings, json
from scipy import stats
warnings.filterwarnings('ignore')

D = pd.read_pickle('merged.pkl')
E = D[D['血栓0_1'].notna()].copy()          # 分析人群 = 285

def num(x):
    s = pd.Series(x).astype(str)
    s = s.str.replace('——','',regex=False).str.replace('不详','',regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]','',regex=True), errors='coerce')

def yn(x):
    out=[]
    for v in pd.Series(x):
        s=str(v).strip()
        if s in ('0','0.0','无','——','－','未','未查'): out.append(0)
        elif s in ('1','1.0','有','是'): out.append(1)
        else: out.append(np.nan)
    return pd.Series(out, index=pd.Series(x).index)

# ===== 分组：临床标准化 =====
loc = E['血栓位置'].astype(str)
E['v_prox_deep'] = loc.str.contains('腘静脉|股静脉|股总静脉|髂').astype(int)          # 近端深静脉
E['v_distal']    = loc.str.contains('肌间|胫后静脉|胫前静脉|腓静脉').astype(int)      # 小腿深静脉
E['v_superf']    = loc.str.contains('大隐静脉|小隐静脉').astype(int)                 # 浅静脉
E['v_bilateral'] = loc.str.contains('双').astype(float)
def grp(r):
    if r['v_prox_deep']: return 'Proximal DVT'
    if r['v_distal']:    return 'Distal (calf) DVT'
    return 'Other/unspecified'
E['group']=E.apply(grp,axis=1)
print("=== 分组 ==="); print(E['group'].value_counts().to_string())
print("近端深静脉:",E['v_prox_deep'].sum(),"| 小腿深静脉:",E['v_distal'].sum(),
      "| 浅静脉(大/小隐):",E['v_superf'].sum(),"| 双侧:",int(E['v_bilateral'].sum()))
A = E[E['group']=='Proximal DVT']; B = E[E['group']=='Distal (calf) DVT']

# ===== 派生 =====
E['age']=num(E['年龄']); E['los']=num(E['住院天数']); E['male']=(num(E['性别'])==1).astype(float)
E['smoke']=yn(E['吸烟史']); E['htn']=yn(E['高血压']); E['dm']=yn(E['糖尿病']); E['varicose']=yn(E['静脉曲张病史'])
E['bleed']=yn(E['合并其它出血']); E['ivcf']=yn(E['IVC滤器']); E['comp']=yn(E['血栓并发症']); E['plaque']=yn(E['脂质斑块'])
sd=E['首次手术天数'].astype(str); E['opday']=np.where(sd.str.contains('当天'),0,num(sd))
E['detect_day']=num(E['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0]).fillna(num(E['血栓检出时间']))
bt=E['脑出血类型'].astype(str)
for c,n in [('1','EDH'),('2','SDH'),('3','SAH'),('4','IPH')]: E[f'ICH_{n}']=bt.str.contains(c).astype(float)
fx=E['下肢骨折类型'].astype(str); E['fx_femur']=fx.str.contains('股骨|髋|髌').astype(float); E['fx_tibia']=fx.str.contains('胫|腓|踝|足').astype(float)
mech=E['致伤机制'].astype(str); E['mech_traffic']=mech.str.contains('交通|车祸').astype(float); E['mech_fall']=mech.str.contains('坠落|摔').astype(float)
rv=E['出院前复查'].astype(str)
E['resolved']=rv.str.contains('消失').astype(float)
E['rechecked']=(~rv.isin(['未查','未检查','nan',''])).astype(float)
# 三时点（修正 T2 血红蛋白=血色素）
LABS=[('WBC','WBC (×10⁹/L)'),('血小板','Platelet (×10⁹/L)'),('PT','PT (s)'),('INR','INR'),
      ('FIB','Fibrinogen (g/L)'),('部分凝血','APTT (s)'),('D-二聚体','D-dimer (mg/L)'),
      ('胆固醇','Cholesterol (mmol/L)'),('甘油三酯','Triglyceride (mmol/L)'),
      ('高密度脂蛋白','HDL-C (mmol/L)'),('低密度脂蛋白','LDL-C (mmol/L)')]
T2HB={'T0':'血红蛋白','T1':'血红蛋白','T2':'血色素'}
for t in ['T0','T1','T2']:
    for k,_ in LABS+[('血红蛋白','')]:
        for cand in [f'{t}_{k}', f'{t}_{T2HB.get(t,k)}']:
            if cand in E.columns: E[f'v_{t}_{k}']=num(E[cand]); break

# ===== 统计 =====
def fmt(s,d=1):
    return '—' if len(s)==0 else f"{s.median():.{d}f} ({s.quantile(.25):.{d}f}–{s.quantile(.75):.{d}f})"
def cont(var,lab,d=1):
    ov=A0=pd.to_numeric(E[var],errors='coerce').dropna()
    a=pd.to_numeric(E.loc[A.index,var],errors='coerce').dropna(); b=pd.to_numeric(E.loc[B.index,var],errors='coerce').dropna()
    p='—'
    if len(a)>=3 and len(b)>=3:
        try: p=f"{stats.mannwhitneyu(a,b,alternative='two-sided').pvalue:.3f}"
        except Exception: pass
    return [lab,fmt(ov,d),fmt(a,d),fmt(b,d),str(len(ov)),p]
def cat(var,lab,den=None):
    den_num = pd.to_numeric(E[den],errors='coerce') if den else None
    mask = (den_num==1) if den else pd.Series(True,index=E.index)
    def rk(idx):
        s=pd.to_numeric(E.loc[idx,var],errors='coerce'); m=mask.loc[idx]
        return int(s[m].sum()), int(s[m].notna().sum())
    tot=E[var].astype(float); tm=mask.reindex(E.index).fillna(False)
    tn=int(tot[tm].notna().sum()); tk=int(tot[tm].sum())
    k1,n1=rk(A.index); k0,n0=rk(B.index)
    pc=lambda k,n: f"{k} ({100*k/n:.1f}%)" if n>0 else '—'
    p='—'
    if n1>0 and n0>0:
        tab=[[k1,n1-k1],[k0,n0-k0]]
        try:
            exp=stats.chi2_contingency(tab)[3]
            p=f"{stats.chi2_contingency(tab,correction=(exp.min()<5))[1]:.3f}"
            if exp.min()<5: p=f"{stats.fisher_exact(tab)[1]:.3f}"
        except Exception: pass
    return [lab, pc(tk,tn) if tn else '—', pc(k1,n1), pc(k0,n0), str(tn), p]

rows=[cont('age','Age, years',0), cat('male','Male sex'),
 cat('mech_traffic','Mechanism: traffic accident'), cat('mech_fall','Mechanism: fall/falling object'),
 cat('ICH_EDH','Cranial EDH'), cat('ICH_SDH','Cranial SDH'), cat('ICH_SAH','Cranial SAH'), cat('ICH_IPH','Cranial IPH'),
 cat('fx_femur','Fracture: femur/hip/patella'), cat('fx_tibia','Fracture: tibia/fibula/ankle/foot'),
 cat('smoke','Smoking history'), cat('htn','Hypertension'), cat('dm','Diabetes mellitus'), cat('varicose','Varicose veins'),
 cont('opday','Time to first surgery, days',0), cont('los','Length of hospital stay, days',0),
 cont('detect_day','Time to VTE detection, days post-injury',0),
 cat('v_bilateral','Bilateral involvement'), cat('v_superf','Superficial vein (GSV/SSV) involvement'),
 cat('plaque','Vascular lipid plaque on ultrasound'),
 cat('ivcf','IVC filter placement'), cat('bleed','Bleeding after anticoagulation'), cat('comp','Thrombotic complications (PE etc.)'),
 cat('rechecked','Ultrasound re-checked before discharge'),
 cat('resolved','Thrombus resolution', den='rechecked')]
T1=pd.DataFrame(rows,columns=['Variable',f'Overall (N={len(E)})',f'Proximal DVT (n={len(A)})',f'Distal calf DVT (n={len(B)})','n available','P value'])
print("\n"+T1.to_string(index=False))
T1.to_csv('table1_baseline.csv',index=False,encoding='utf-8-sig')

tr=[]
for k,lab in LABS+[('血红蛋白','Haemoglobin (g/L)')]:
    d={'Laboratory variable':lab}
    for t,tl in [('T0','Admission'),('T1','VTE detection'),('T2','2 wk post-anticoag')]:
        c=f'v_{t}_{k}'
        if c in E.columns:
            v=E[c]; d[tl]=f"{v.median():.2f} ({v.quantile(.25):.2f}–{v.quantile(.75):.2f})" if v.notna().sum() else '—'
            d[tl+' n']=int(v.notna().sum())
    a=E.get(f'v_T0_{k}'); b=E.get(f'v_T1_{k}')
    if a is not None and b is not None:
        m=a.notna()&b.notna(); d['paired n']=int(m.sum())
        if m.sum()>=5:
            try: d['P (T0 vs T1)']=f"{stats.wilcoxon(a[m],b[m]).pvalue:.2e}"
            except Exception: d['P (T0 vs T1)']='—'
    tr.append(d)
T2=pd.DataFrame(tr); print("\n"+T2.to_string(index=False))
T2.to_csv('table2_trajectory.csv',index=False,encoding='utf-8-sig')
print("\n[saved]")
