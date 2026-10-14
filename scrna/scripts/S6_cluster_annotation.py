# -*- coding: utf-8 -*-
"""
S6: GSE294775 TBI 特异性簇深注释（补 §14.2 遗留的 c12/c14 等未定簇）
用官方 celltype_l3 注释占位 + Wilcoxon 标记基因，给出每个 TBI 富集簇的身份判定。
输出：S6_TBI_enriched_cluster_annotation.csv / .md
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import scanpy as sc

OUT = r'D:/TBI_DVT_Study/scrna/output'
A = sc.read_h5ad(os.path.join(OUT, 'GSE294775_qc_clustered.h5ad'))

enr = pd.read_csv(os.path.join(OUT, 'GSE294775_cluster_enrichment.csv'))
enr['log2OR'] = np.log2(enr['OR_TBI_vs_CTR'].clip(lower=1e-3))
targets = enr.sort_values('log2OR', ascending=False).head(8).leiden.tolist()

# 官方注释在这批 cells 上的分布
key = 'celltype_l3' if 'celltype_l3' in A.obs.columns else ('celltype_l2' if 'celltype_l2' in A.obs else None)
rows = []
for cl in targets:
    m = A.obs.leiden == str(cl)
    sub = A[m]
    top = sub.obs[key].astype(str).value_counts() if key else pd.Series(dtype=int)
    # 标记基因（在 raw log 层上做）
    try:
        sc.tl.rank_genes_groups(sub, 'group', groups=['TBI'], reference='CTR', method='wilcoxon', use_raw=True)
        df = sc.get.rank_genes_groups_df(sub, group='TBI')
        up = ', '.join(df.sort_values('logfoldchanges', ascending=False).head(12).names)
    except Exception as e:
        # Clusters with 0 or 1 control cells cannot support a two-group test.
        # Emit an English, publication-safe note (the raw exception text used to
        # be written verbatim into the released CSV).
        n_ctr = int((sub.obs.group == 'CTR').sum())
        up = 'not estimable (%d control cell%s in this cluster)' % (n_ctr, '' if n_ctr == 1 else 's')
    rows.append({
        'leiden': cl,
        'n_TBI': int((sub.obs.group == 'TBI').sum()),
        'n_CTR': int((sub.obs.group == 'CTR').sum()),
        'log2OR': float(enr.loc[enr.leiden == cl, 'log2OR'].iloc[0]),
        'official_label_top3': '; '.join(f'{k}({v})' for k, v in top.head(3).items()) if len(top) else 'n/a',
        'TBI_up_markers': up,
    })

ann = pd.DataFrame(rows)
ann.to_csv(os.path.join(OUT, 'S6_TBI_enriched_cluster_annotation.csv'), index=False, encoding='utf-8-sig')

md = ['# GSE294775 TBI 富集簇深注释（官方注释 + 组间标记基因）\n']
for r in ann.itertuples():
    md.append(f'## cluster {r.leiden}  (TBI {r.n_TBI} / CTR {r.n_CTR}, log2OR={r.log2OR:.2f})\n')
    md.append(f'- 官方注释主体: {r.official_label_top3}')
    md.append(f'- TBI 组上调标记: {r.TBI_up_markers}\n')
open(os.path.join(OUT, 'S6_TBI_enriched_cluster_annotation.md'), 'w', encoding='utf-8').write('\n'.join(md))
print(ann.to_string(index=False)[:3000])
print('\n[saved] S6_TBI_enriched_cluster_annotation.csv/.md')
