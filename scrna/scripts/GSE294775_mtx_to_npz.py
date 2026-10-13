# -*- coding: utf-8 -*-
"""
GSE294775 matrix.mtx.gz 分块转换器（16GB 内存机器防 MemoryError）
逐块（4000 万条目/块）读入 COO -> CSR -> vstack，落盘 npz + features/barcodes。
输出: GSE294775/GSE294775_raw_counts.npz（csr: genes x cells）
"""
import os, gzip, numpy as np, pandas as pd
import scipy.sparse as sp

DATA = os.path.dirname(os.path.abspath(__file__)) if False else r'D:/TBI_DVT_Study/scrna/data/GSE294775'
mtx = os.path.join(DATA, 'GSE294775_matrix.mtx.gz')
out = os.path.join(DATA, 'GSE294775_raw_counts.npz')

# 头部维度
with gzip.open(mtx, 'rt') as f:
    for line in f:
        if not line.startswith('%'):
            nr, nc, nnz = map(int, line.split())
            break
print(f'dims: {nr} genes x {nc} cells, {nnz:,} entries')

CHUNK = 40_000_000
reader = pd.read_csv(mtx, sep=' ', comment='%', header=None,
                     names=['r', 'c', 'v'],
                     dtype={'r': np.int32, 'c': np.int32, 'v': np.float32},
                     chunksize=CHUNK)
mats_sum = None
total = 0
for i, ch in enumerate(reader):
    m = sp.coo_matrix((ch.v.values, (ch.r.values - 1, ch.c.values - 1)),
                      shape=(nr, nc), dtype=np.float32).tocsr()
    mats_sum = m if mats_sum is None else mats_sum + m
    total += len(ch)
    print(f'chunk {i}: {len(ch):,} rows (running nnz={total:,})')
    del ch

M = mats_sum
M.sum_duplicates()
M.sort_indices()
assert M.shape == (nr, nc), f'shape mismatch: {M.shape} vs {(nr, nc)}'
print(f'final: {M.shape}, nnz={M.nnz:,}')
np.savez_compressed(out, data=M.data, indices=M.indices, indptr=M.indptr,
                    shape=np.array(M.shape))
print('[saved]', out, f'{os.path.getsize(out)/1e6:.0f} MB')
