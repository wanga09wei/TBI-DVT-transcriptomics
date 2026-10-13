import pandas as pd, numpy as np, warnings, re
warnings.filterwarnings('ignore')
D=pd.read_pickle('merged.pkl')
def num(x):
    s=pd.Series(x).astype(str).str.replace(r'[^0-9.\-]','',regex=True)
    return pd.to_numeric(s,errors='coerce')
E=D[D['year']<2018].copy() if False else D.copy()
ev=E[E['血栓0_1'].notna()]
print("=== 队列构成 ===")
print("总记录:",len(D)," 有血栓评估:",len(ev))
# DVT 近端 vs 远端
loc=ev['血栓位置'].astype(str)
prox_kw='腘静脉|股静脉|髂|腓静脉|胫后静脉|胫前静脉|大隐静脉|小隐静脉|下肢静脉'
ev['loc_distal']=loc.str.contains('肌间')
ev['loc_prox']=loc.str.contains('腘静脉|股静脉|髂|腓静脉')
print("\n=== 血栓位置分类 ===")
print("仅肌间(远端):", (ev['loc_distal'] & ~ev['loc_prox']).sum())
print("累及近端/主干:", ev['loc_prox'].sum())
print("双侧:", loc.str.contains('双').sum())
print("\n=== 检出血栓时间(伤后天) ===")
dt=num(ev['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0])
dt2=pd.to_numeric(ev['血栓检出时间'].astype(str).str.replace(r'[^0-9]','',regex=True),errors='coerce')
tt=dt.fillna(dt2)
print(tt.describe().round(2).to_string())
print("中位数(天):",tt.median()," IQR:",tt.quantile(.25),"-",tt.quantile(.75))
print("\n=== 三时点实验室轨迹 (median, n) ===")
labs=['WBC','血小板','PT','INR','FIB','部分凝血','D-二聚体','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
rows=[]
for L in labs:
    r={'lab':L}
    for t in ['T0','T1','T2']:
        for cand in [f'{t}_{L}', f'{t}_血色素' if L=='血红蛋白' else f'{t}_{L}']:
            if cand in D.columns:
                v=num(D[cand]); r[t+'_n']=v.notna().sum(); r[t+'_med']=round(v.median(),2) if v.notna().sum()>0 else np.nan; break
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\n=== 临床关键指标 ===")
def cnt(c,lab):
    v=num(D[c]); print(f"{lab:22s} n={v.notna().sum():3d}  有/1={int((v==1).sum()):3d} ({(v==1).mean()*100 if v.notna().sum() else 0:.1f}%)")
for c,l in [('合并其它出血','抗凝后其它出血'),('IVC滤器','下腔静脉滤器'),('血栓并发症','血栓并发症(PE等)'),
            ('高血压','高血压'),('糖尿病','糖尿病'),('吸烟史','吸烟史'),('静脉曲张病史','静脉曲张病史'),
            ('下肢肿胀疼痛','下肢肿胀/疼痛'),('动脉硬化','动脉硬化'),('脂质斑块','血管脂质斑块')]:
    cnt(c,l)
print("\n=== 年龄/性别/住院天数/手术 ===")
for c in ['年龄','住院天数','首次手术天数','手术时间min']:
    v=num(D[c]); print(f"{c:12s} n={v.notna().sum():3d} median={v.median()} IQR={v.quantile(.25)}-{v.quantile(.75)}")
print("性别 男:",int((num(D['性别'])==1).sum())," 女:",int((num(D['性别'])==2).sum()))
print("\n=== 抗凝时机分布(前8) ===")
print(D['抗凝时机'].value_counts().head(8).to_string())
print("\n=== 2018出院前血栓复查结果 分布(前12) ===")
print(D['出院前复查'].value_counts().head(12).to_string())
