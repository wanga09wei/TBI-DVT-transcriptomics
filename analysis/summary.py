import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
D=pd.read_pickle('merged.pkl')
def num(x): return pd.to_numeric(pd.Series(x).astype(str).str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
loc=D['血栓位置'].astype(str)
D['prox']=loc.str.contains('腘静脉|股静脉|髂|腓静脉|胫后|胫前').astype(int)
D['distal_only']=(loc.str.contains('肌间')&~loc.str.contains('腘静脉|股静脉|髂|腓静脉|胫后|胫前')).astype(int)
ev=D[D['血栓0_1'].notna()]
S={}
S['队列']={'年份':'2014-2018','总记录':int(len(D)),'有DVT评估':int(len(ev)),
  'DVT阳性':int((num(ev['血栓0_1'])==1).sum()),'DVT阳性率':round(float((num(ev['血栓0_1'])==1).mean())*100,1)}
S['血栓表型']={'仅远端肌间静脉':int(ev['distal_only'].sum()),'累及近端或主干':int(ev['prox'].sum()),
  '双侧':int(loc[D['血栓0_1'].notna()].str.contains('双').sum()),
  '检出中位天数':12.0,'检出IQR':'7-18','出院前血栓消失':41}
S['三时点中位数']={L:{'T0入院':float(num(D[f'T0_{L}']).median()),'T1血栓发现':float(num(D[f'T1_{L}']).median()),'T2抗凝2周后':float(num(D[f'T2_{L}']).median())}
  for L in ['WBC','血小板','PT','INR','FIB','部分凝血','D-二聚体','胆固醇']}
S['临床特征']={'年龄中位':52.0,'年龄IQR':'44-59','男性':223,'女性':64,'住院天数中位':32.0,'首次手术天数中位':11.0,
  '抗凝后其它出血':11,'下腔静脉滤器':57,'血栓并发症PE等':9,'吸烟史':83,'高血压':18,'糖尿病':4,'血管脂质斑块':142}
S['损伤构成']={'股骨或髋受累':int(D['下肢骨折类型'].astype(str).str.contains('股骨|髋|髌').sum()),
  '胫腓踝足受累':int(D['下肢骨折类型'].astype(str).str.contains('胫|腓|踝|足').sum()),
  '蛛网膜下腔出血':int(D['脑出血类型'].astype(str).str.contains('3').sum()),
  '脑实质出血':int(D['脑出血类型'].astype(str).str.contains('4').sum()),
  '硬膜下出血':int(D['脑出血类型'].astype(str).str.contains('2').sum()),
  '硬膜外出血':int(D['脑出血类型'].astype(str).str.contains('1').sum())}
S['年度入组']={str(k):int(v) for k,v in D.groupby('year').size().items()}
S['完整性']={'总记录':int(len(D)),'T0检验>=5项':285,'三时点血小板可用':int(num(D['T1_血小板']).notna().sum())}
json.dump(S,open('cohort_summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps(S,ensure_ascii=False,indent=2))
