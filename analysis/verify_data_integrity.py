# -*- coding: utf-8 -*-
"""数据完整性核验：确认重建后的 merged.pkl 与既有统计基线（cohort_summary.json）一致。

背景：merged.pkl 原由较新版本 pandas 落盘，在 pandas 2.3.3 下读取报
NotImplementedError（StringDtype.__setstate__ 格式变更）。修复方式为以
raw_clinical/ 原始年度 Excel 重跑 build.py 重新落盘，本脚本用于证明重建
前后统计量逐一等价。

口径说明（与 summary.py 保持一致）：
  - 年龄/性别/住院天数/首次手术天数 分母 = 全体记录中的非缺失值（n=287）
  - 血栓表型相关分母 = 有 DVT 标注者（n=285）
  - 双侧判定用 substring '双'（兼容"双下肢小腿段肌间静脉"等写法）
"""
import json, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')

D = pd.read_pickle('merged.pkl')
J = json.load(open('cohort_summary.json', encoding='utf-8'))
def num(x): return pd.to_numeric(pd.Series(x).astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

ok, bad = [], []
def chk(name, got, exp, tol=0.0):
    good = abs(float(got) - float(exp)) <= tol if isinstance(exp, (int, float)) else str(got) == str(exp)
    (ok if good else bad).append((name, got, exp))

ev = D[D['血栓0_1'].notna()]
lines = ['pandas %s | merged.pkl 重建核验' % pd.__version__,
         '总记录 %d | 有DVT标注 %d' % (len(D), len(ev)), '']
chk('总记录', len(D), 338); chk('有DVT评估', len(ev), 285)
chk('DVT阳性', int(D['DVT'].sum()), 284)
for y, e in [(2014, 68), (2015, 70), (2016, 70), (2017, 90), (2018, 40)]:
    chk('年度入组-%d' % y, int((D['year'] == y).sum()), e)

for lab in ['WBC', '血小板', 'PT', 'INR', 'FIB', '部分凝血', 'D-二聚体', '胆固醇']:
    for i, t in enumerate(['T0', 'T1', 'T2']):
        k = 'T0入院' if t == 'T0' else ('T1血栓发现' if t == 'T1' else 'T2抗凝2周后')
        if '三时点中位数' in J and lab in J['三时点中位数']:
            chk('%s-%s' % (lab, t), round(float(num(D['%s_%s' % (t, lab)]).median()), 3),
                J['三时点中位数'][lab][k], tol=0.01)

loc = ev['血栓位置'].astype(str)
PROX = r'腘|股总|股浅|股深|股静脉|髂'; CALF = r'肌间|胫后|胫前|腓'; SUP = r'大隐|小隐'
g = pd.Series(np.where(loc.str.contains(PROX), 'P', np.where(loc.str.contains(CALF), 'D', np.where(loc.str.contains(SUP), 'S', 'U'))), index=ev.index)
chk('仅小腿深静脉', int((g.eq('D') & ~g.eq('P')).sum()), 240)
chk('近端/主干', int(g.eq('P').sum()), 33)
chk('仅浅静脉', int((g.eq('S') & ~g.eq('P') & ~g.eq('D')).sum()), 3)
chk('位置不明确', int(g.eq('U').sum()), 9)
chk('双侧', int(loc.str.contains('双').sum()), 66)

chk('年龄中位', float(num(D['年龄']).median()), 52.0, tol=0.01)
chk('男性', int(num(D['性别']).eq(1).sum()), 223)
chk('女性', int(num(D['性别']).eq(2).sum()), 64)
chk('住院天数中位', float(num(D['住院天数']).median()), 32.0, tol=0.01)
chk('首次手术天数中位', float(num(D['首次手术天数']).median()), 11.0, tol=0.01)
for c, exp, nm in [('合并其它出血', 11, '抗凝后其它出血'), ('IVC滤器', 57, 'IVC滤器率分子'),
                   ('血栓并发症', 9, '血栓并发症'), ('吸烟史', 83, '吸烟史'),
                   ('高血压', 18, '高血压'), ('糖尿病', 4, '糖尿病'),
                   ('脂质斑块', 142, '血管脂质斑块')]:
    chk(nm, int(D[c].astype(str).str.strip().isin(['1', '1.0', '有', '是']).sum()), exp)

lines += ['一致项 %d / %d' % (len(ok), len(ok) + len(bad))]
lines += ['不一致项: ' + (', '.join('%s(got=%s exp=%s)' % b for b in bad) if bad else '无 ✓')]
lines += ['列数 %d | 判定: %s' % (D.shape[1], 'PASS' if not bad else 'FAIL')]
txt = '\n'.join(lines)
print(txt)
open('data_integrity_check.txt', 'w', encoding='utf-8').write(txt + '\n')
