import pandas as pd, numpy as np, warnings, re, json, os
warnings.filterwarnings('ignore')
# 数据源：优先项目内 raw_clinical/（已归档），回退原始桌面路径
_Here=os.path.dirname(os.path.abspath(__file__))
_LOCAL=os.path.join(_Here,'..','raw_clinical')
_DESK=r"C:/Users/Administrator/Desktop/病例数据"
SRC=_LOCAL if os.path.isdir(_LOCAL) else _DESK
print("数据源目录:",SRC)
F={2014:os.path.join(SRC,"颅脑外伤合并下肢骨折 病例收集(2014).xlsx"),
   2015:os.path.join(SRC,"颅脑外伤合并下肢骨折 病例收集(2015).xlsx"),
   2016:os.path.join(SRC,"颅脑外伤合并下肢骨折 病例收集(2016).xlsx"),
   2017:os.path.join(SRC,"颅脑外伤合并下肢骨折 病例收集(2017).xlsx"),
   2018:os.path.join(SRC,"颅脑外伤合并下肢骨折 病例收集2018【1-5】.xlsx")}
LAB=['WBC','血红蛋白','血色素','血小板','PT','活动度','INR','FIB','部分凝血','D-二聚体','MA值','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
allrows=[]
for y,p in F.items():
    df=pd.read_excel(p,header=None); h0=df.iloc[0].tolist(); h1=df.iloc[1].tolist()
    b=df.iloc[2:].copy(); b=b[b[0].notna() & (b[0].astype(str).str.strip()!='样本范例')]
    # block starts from h1
    st={}
    for i,s in enumerate(h1):
        s=str(s)
        if '入院首次' in s and 'T0' not in st: st['T0']=i-1
        if '血栓首次发现' in s: st['T1']=i-1
        if '抗凝2周后' in s: st['T2']=i-1
    # also row0 '检验'
    for i,s in enumerate(h0):
        if str(s).strip()=='检验': st['T0']=i
    if 'T1' not in st: st['T1']=st['T0']+13 if y==2014 else st['T0']+14
    if 'T2' not in st: st['T2']=st['T1']+13 if y==2014 else st['T1']+14
    rec={}
    base={0:'病例',1:'住院号',2:'住院天数',3:'性别',4:'年龄',5:'致伤机制',6:'脑出血类型',7:'下肢骨折类型',8:'其他损伤',
          9:'既往史',10:'吸烟史',11:'高血压',12:'糖尿病',13:'静脉曲张病史',14:'其它既往因素',15:'手术',
          16:'首次手术天数',17:'手术部位',18:'手术时间min',20:'血栓0_1',21:'血栓位置',22:'血栓大小mm',23:'血栓检出时间',
          24:'合并其它出血',25:'IVC滤器',26:'血栓并发症',27:'出院前复查',28:'抗凝药',29:'抗凝时机',30:'药物',31:'量',32:'周期天',
          33:'阿司匹林',34:'双嘧达莫',35:'前列地尔',36:'华法林',37:'瑞恤定',38:'阿加曲班利伐沙班',
          80:'下肢肿胀疼痛',81:'入院血管损伤',82:'动脉硬化',83:'脂质斑块'}
    for i,nm in base.items():
        rec[nm]=b[i].values if i<b.shape[1] else np.nan
    for tag,stt in st.items():
        for j in range(0,15):
            ci=stt+j
            if ci>=b.shape[1]: break
            nm=str(h0[ci]).strip()
            if nm in LAB:
                key=f"{tag}_{nm}"
                if key not in rec: rec[key]=b[ci].values
    rec['year']=np.full(len(b),y)
    allrows.append(pd.DataFrame(rec))
D=pd.concat(allrows,ignore_index=True)
print("合并队列 N =",len(D))
def numf(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.replace('——','',regex=False).str.replace('未','',regex=False)
                         .str.replace('不详','',regex=False).str.replace('无','',regex=False).str.replace('－','',regex=False)
                         .replace({'':np.nan,'nan':np.nan,'None':np.nan}),errors='coerce')
print("\n### 血栓发生率")
t=D['血栓0_1'].copy()
def hv(v):
    s=str(v).strip()
    if s in ('1','1.0'): return 1
    if s in ('0','0.0'): return 0
    return np.nan
D['DVT']=D['血栓0_1'].map(hv)
D.loc[D['血栓0_1'].astype(str).str.contains('有',na=False),'DVT']=1
D.loc[D['血栓0_1'].astype(str).str.contains('无',na=False),'DVT']=0
print(D.groupby('year')['DVT'].agg(['count','sum','mean']))
print("总体 DVT: %d/%d = %.1f%%"%(D['DVT'].sum(),D['DVT'].notna().sum(),100*D['DVT'].mean()))
D.to_pickle('merged.pkl')
print("\n### 关键变量缺失/分布")
for c in ['年龄','性别','住院天数','脑出血类型','下肢骨折类型','高血压','糖尿病','吸烟史','静脉曲张病史','首次手术天数','下肢肿胀疼痛','入院血管损伤','动脉硬化','脂质斑块']:
    s=D[c]; print(f"{c:12s} nonnull={s.notna().sum():3d}  值样例={list(pd.Series(s.dropna().unique())[:6])}")
print("\n### 三时点D-二聚体")
for k in ['T0_D-二聚体','T1_D-二聚体','T2_D-二聚体']:
    if k in D: print(k, "nonnull=",numf(D[k]).notna().sum(), "median=",round(numf(D[k]).median(),2) if numf(D[k]).notna().sum() else None)
print("\n### 可用列总数:", D.shape[1])
print(list(D.columns))
