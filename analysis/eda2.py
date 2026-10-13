import pandas as pd, numpy as np, warnings, re
warnings.filterwarnings('ignore')
def rd(p):
    return pd.read_excel(p, header=None)
F={2014:r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2014).xlsx",
   2015:r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2015).xlsx",
   2016:r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2016).xlsx",
   2017:r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2017).xlsx",
   2018:r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集2018【1-5】.xlsx"}
LABS=['WBC','血色素','血红蛋白','血小板','PT','活动度','INR','FIB','部分凝血','D-二聚体','MA值','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
rows=[]
for y,p in F.items():
    df=rd(p); h0=df.iloc[0].tolist(); h1=df.iloc[1].tolist()
    body=df.iloc[2:].copy()
    body=body[body[0].notna() & (body[0].astype(str).str.strip()!='样本范例')]
    # locate block markers in row0/row1
    marks={}
    for i,h in enumerate(h0):
        s=str(h)
        if '入院首次' in s or ('检验' in s and i<45): marks.setdefault('T0',i)
        if '血栓首次发现' in s: marks['T1']=i
        if '抗凝2周后' in s: marks['T2']=i
    # 检验 label col
    rec={'year':y,'n':len(body)}
    def grab(block_start):
        out={}
        if block_start is None: return out
        for i in range(block_start, min(block_start+16, len(h0))):
            nm=str(h0[i]).strip()
            if i+1 < len(h0) and pd.isna(h0[i]) is False and nm in LABS: pass
            key=str(h0[i]).strip()
            key=key if key!='nan' else None
            if key is None: continue
            if key in LABS and key not in out: out[key]=i
        return out
    blk={k:grab(v) for k,v in marks.items()}
    # simpler: derive from T0 marker
    t0 = marks.get('T0')
    if t0 is not None:
        names0=[str(x).strip() for x in h0[t0:t0+17]]
        # map sequential
        seq=['WBC','血红蛋白','血小板','PT','活动度','INR','FIB','部分凝血','D-二聚体','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
        seq15=['WBC','血红蛋白','血小板','PT','活动度','INR','FIB','部分凝血','D-二聚体','MA值','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
        seq16=['WBC','血色素','血小板','PT','活动度','INR','FIB','部分凝血','D-二聚体','胆固醇','甘油三酯','高密度脂蛋白','低密度脂蛋白']
        for nm,seqn in [('T0',seq),('T1',seq),('T2',seq16)]:
            pass
    rec['body']=body; rec['h0']=h0; rec['marks']=marks
    rows.append(rec)
for r in rows:
    print(r['year'],'n=',r['n'],'block markers=',r['marks'])
    print('   row0 around T0..:', [f"{i}:{r['h0'][i]}" for i in range(max(0,r['marks'].get('T0',0)-1), min(len(r['h0']),r['marks'].get('T0',0)+18))])
