import pandas as pd, glob, os, warnings
warnings.filterwarnings('ignore')
BASE = r"C:/Users/Administrator/Desktop/病例数据"
files = sorted(glob.glob(os.path.join(BASE, "*.xls*")))
for f in files:
    print("="*100)
    print("FILE:", os.path.basename(f))
    try:
        xl = pd.ExcelFile(f)
        print("SHEETS:", xl.sheet_names)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=None, nrows=8)
            print("-"*80)
            print(f"  SHEET: {sh}  raw_shape(first rows)")
            print(df.to_string(max_cols=40))
    except Exception as e:
        print("ERR", e)
