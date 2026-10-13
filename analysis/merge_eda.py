import pandas as pd, numpy as np, glob, os, warnings, re, json
warnings.filterwarnings('ignore')
BASE = r"C:/Users/Administrator/Desktop/病例数据"

def load_year(path, year):
    df = pd.read_excel(path, header=None)
    heads = df.iloc[0].tolist()
    sub   = df.iloc[1].tolist()
    data  = df.iloc[2:].copy()
    data = data[data[0].notna()]           # 病例编号存在
    data = data[data[0].astype(str).str.strip() != '样本范例']
    cols=[]
    for i,h in enumerate(heads):
        if pd.isna(h): cols.append(f"__c{i}"); continue
        h=str(h).strip().replace('\n','')
        if i in (19,): h='血栓'
        cols.append(f"{i}_{h}")
    data.columns = cols[:data.shape[1]] if len(cols)>=data.shape[1] else cols+[f"x{j}" for j in range(data.shape[1]-len(cols))]
    # 二级表头补充
    for i,s in enumerate(sub):
        if pd.notna(s) and i < len(data.columns):
            data.rename(columns={data.columns[i]: str(s).strip().replace('\n','')+f"__{i}"}, inplace=True)
    data['_year']=year
    data['_n']=len(data)
    return data

files = {
 2014: r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2014).xlsx",
 2015: r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2015).xlsx",
 2016: r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2016).xlsx",
 2017: r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集(2017).xlsx",
 2018: r"C:/Users/Administrator/Desktop/病例数据/颅脑外伤合并下肢骨折 病例收集2018【1-5】.xlsx",
}
tot=0
for y,p in files.items():
    d=load_year(p,y); print(f"{y}: n={len(d)}  cols={d.shape[1]}"); tot+=len(d)
print("TOTAL 2014-2018 病例数(去样本范例):", tot)

# 2018血栓超声表
d18 = pd.read_excel(r"C:/Users/Administrator/Desktop/病例数据/创外-血栓-2018.01-05.xls", header=0)
print("\n2018 血栓超声表 n=", len(d18), "cols=", list(d18.columns))
