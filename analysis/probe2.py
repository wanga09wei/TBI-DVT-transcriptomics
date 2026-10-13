import pandas as pd, glob, os, warnings, json
warnings.filterwarnings('ignore')
BASE = r"C:/Users/Administrator/Desktop/病例数据"
files = sorted(glob.glob(os.path.join(BASE, "*.xls*")))
out={}
for f in files:
    bn=os.path.basename(f)
    xl = pd.ExcelFile(f)
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None)
        if df.dropna(how='all').empty: continue
        print("="*110)
        print(f"FILE: {bn} | SHEET: {sh} | raw rows={df.shape[0]} cols={df.shape[1]}")
        hdr0 = df.iloc[0].tolist()
        hdr1 = df.iloc[1].tolist()
        print("\n-- ROW0 (一级表头) --")
        for i,h in enumerate(hdr0):
            if pd.notna(h): print(f"  [{i}] {h}")
        print("\n-- ROW1 (二级表头) --")
        for i,h in enumerate(hdr1):
            if pd.notna(h): print(f"  [{i}] {h}")
        data=df.iloc[2:].dropna(how='all')
        print(f"\n-- DATA rows (含样本范例?): {data.shape[0]}")
