# -*- coding: utf-8 -*-
"""fix_fdr_ordering.py — repair mis-computed Benjamini-Hochberg FDR columns.

Background
----------
Three pipeline scripts (GSE223245_bulk_DEG.py, GSE294775_post_analysis.py,
S4_GSE294775_deepdive_and_integration.py) contained the same defective
inline BH implementation:

    q = np.minimum.accumulate((np.sort(p) * n / (np.arange(n)[::-1] + 1))[::-1])
    out = q[np.argsort(np.argsort(p))]

Two independent errors:

  1. the trailing ``[::-1]`` on the cumulative minimum was missing, so ``q``
     was ordered by *descending* p rather than ascending p;
  2. ``np.argsort(np.argsort(p))`` is the rank of p, not the permutation that
     maps sorted values back to the original row order (the inverse of
     ``np.argsort(p)``), so the adjusted values were written to the wrong rows.

Net effect: the published ``FDR`` column was mis-aligned with its p-values and
contained impossible values (> 1, e.g. 26344 for a probe with P = 4.7e-06).

This script recomputes the FDR column in place, from the p-value column
already present in each output file, using a correct BH implementation. It is
idempotent and does NOT touch any other column, so no reported statistic
changes. It is provided so that the released CSV outputs match the corrected
pipeline scripts without re-running the (heavy) upstream analyses.

Usage:  python fix_fdr_ordering.py [output_dir ...]
        (default: ../output relative to this script)
"""
import os
import sys
import numpy as np
import pandas as pd

# file -> name of the raw p-value column used by that script
TARGETS = {
    'GSE223245_bulk_all_probes.csv': 'P',
    'GSE223245_bulk_candidate_hub_genes.csv': 'P',
    'GSE223245_bulk_sig_probes.csv': 'P',
    'GSE294775_candidate_gene_diff.csv': 'P_MWU',
    'S4_celltype_shift_celltype_l1.csv': 'P_chi2',
    'S4_celltype_shift_celltype_l2.csv': 'P_chi2',
    'S4_percelltype_DEG_TBI_vs_CTR.csv': 'P_MWU',
}


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values in the input order, capped at 1."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    if m.sum() == 0:
        return q
    pv = p[m]
    n = pv.size
    order = np.argsort(pv, kind='mergesort')
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[m] = out
    return q


def repair(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    base = os.path.basename(path)
    pcol = TARGETS.get(base)
    if pcol is None or pcol not in df.columns or 'FDR' not in df.columns:
        return None
    before = df['FDR'].copy()
    df['FDR'] = bh_fdr(df[pcol].values)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    bad_before = int((before > 1).sum()) if before.notna().any() else 0
    bad_after = int((df['FDR'] > 1).sum())
    return base, len(df), bad_before, bad_after


def rebuild_sig_probes(outdir):
    """GSE223245_bulk_sig_probes.csv was empty because the selector used the
    broken FDR column. Regenerate it with the documented rule
    (FDR < 0.05 and |log2FC| > 0.58), sorted by P."""
    allf = os.path.join(outdir, 'GSE223245_bulk_all_probes.csv')
    if not os.path.exists(allf):
        return None
    df = pd.read_csv(allf, encoding='utf-8-sig')
    sig = df[(df.FDR < 0.05) & (df.log2FC_TBI_vs_Ctrl.abs() > 0.58)].sort_values('P')
    dst = os.path.join(outdir, 'GSE223245_bulk_sig_probes.csv')
    sig.to_csv(dst, index=False, encoding='utf-8-sig')
    return len(sig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    dirs = sys.argv[1:] or [os.path.join(root, 'output'), os.path.join('D:/TBI_DVT_Study/scrna/output')]
    for d in dirs:
        print('=' * 72)
        print('output dir:', d)
        if not os.path.isdir(d):
            print('  (not found, skipped)')
            continue
        for name in sorted(TARGETS):
            r = repair(os.path.join(d, name))
            if r is None:
                print('  %-46s no FDR column / not present' % name)
            else:
                print('  %-46s rows=%d  FDR>1 before=%d  after=%d' % r)
        n = rebuild_sig_probes(d)
        if n is not None:
            print('  %-46s regenerated, %d rows' % ('GSE223245_bulk_sig_probes.csv', n))


if __name__ == '__main__':
    main()
