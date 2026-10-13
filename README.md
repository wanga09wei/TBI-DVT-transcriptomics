# Injury-centre endothelium shifts towards a procoagulant venous state in traumatic brain injury

Analysis code and de-identified summary tables accompanying the manuscript:

> **Injury-centre endothelium shifts towards a procoagulant venous state in traumatic brain injury: a clinical cohort, blood transcriptomic and single-cell virtual-perturbation study**
> Shanshan Sun, Hui Li, Tianyu Li, Yuchang Wang, Wei Wang (corresponding)

**Corresponding author:** Wei Wang, MD, PhD — Department of Traumatic Surgery, Tongji Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan 430030, P.R. China · wangwei@tjh.tjmu.edu.cn · ORCID [0000-0001-6971-3461](https://orcid.org/0000-0001-6971-3461)

---

## 1. What this repository contains

| Path | Contents |
|---|---|
| `analysis/` | Clinical cohort scripts (build, cohort description, group-based trajectory modelling, association testing, robustness checks, figure generation) and the resulting de-identified summary tables (`table1_baseline`, `table2_trajectory`, `table3_trajectory_phenotypes`) |
| `analysis/cohort_summary.json` | Every measured number quoted in the clinical Results section, in machine-readable form (traceability ledger) |
| `scrna/scripts/` | Single-cell and bulk transcriptomic pipeline: QC, clustering, subtype annotation, pseudotime, virtual knockout (scTenifoldKnk), differential regulatory networks (scTenifoldNet), virtual overexpression (network propagation), transcription-factor activity (decoupler/CollecTRI) and module-level analyses |
| `scrna/output/` | De-identified intermediate result tables (per-cluster marker scores, region composition, differential expression, virtual-perturbation matrices, module scoring in blood) and run logs |
| `figures/` | Publication figures 1–8 at 300 dpi |

## 2. What this repository does NOT contain

**No patient-level data of any kind.** The 2014–2018 clinical cohort tables are not deposited here, in any form, because they contain protected health information. `*.pkl` row-level analysis sets, raw case-record spreadsheets, single-cell object files (`*.h5ad`) and cached GEO download archives are likewise excluded (see `.gitignore`).

Requests for de-identified derived data should be directed to the corresponding author and are subject to institutional data-governance approval.

## 3. Data sources

All transcriptomic data are public and were obtained from the NCBI Gene Expression Omnibus (GEO):

| Accession | Description |
|---|---|
| [GSE223245](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223245) | Peripheral-blood transcriptomes across TBI severity grades (Agilent one-colour microarray). Sample annotation reads *whole blood*, whereas the series overall design reads *PBMC*; in the manuscript this discrepancy is handled by describing the material as peripheral blood and stating the annotation conflict explicitly in Methods |
| [GSE182398](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182398) | Human brain endothelium (CD45⁻CD31⁺) from injury centre and peri-injury tissue after TBI, single-cell RNA-seq |
| [GSE294775](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294775) | Human post-mortem brain single-cell RNA-seq, 7 TBI and 7 control donors |

Clinical cohort: retrospective review of 338 records (285 with duplex ultrasound assessment) from the Department of Traumatic Surgery, Tongji Hospital, 2014–2018.

## 4. Environment

Python 3.13 with the packages pinned in `requirements.txt`. Key versions used for the reported results: scanpy 1.12.4, anndata 0.13.4, scTenifoldpy 0.5.1, decoupler 2.2.0, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib (300 dpi rendering).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Reproducing the analysis

### 5.1 Clinical cohort

```bash
cd analysis
python build.py            # merge the five annual cohorts into the analysis set
python final_analysis.py   # Table 1 and Table 2
python traj_cluster.py     # group-based trajectory modelling, model selection
python traj_assoc.py       # trajectory phenotype x outcome association
python traj_robust.py      # robustness checks (bootstrap ARI, alternative K)
python table1.py
python make_figs.py        # Figures 1-3
```

`build.py` requires the restricted clinical source tables, which are not distributed here; running the remaining scripts against a permitted local copy reproduces every reported summary statistic.

### 5.2 Single-cell and bulk transcriptomics

Scripts are numbered by pipeline stage and must be run in order:

```bash
cd scrna
bash download_gse223245.sh                    # public data retrieval (single-instance lock, resumable)
python scripts/GSE182398_scRNA.py             # QC, clustering, annotation (injury centre vs peri-injury)
python scripts/S2_GSE182398_subtypes.py       # endothelial subtype definition
python scripts/S3_virtual_perturbations.py    # scTenifoldNet differential network + scTenifoldKnk virtual knockout
python scripts/S8_endothelial_pseudotime.py   # diffusion pseudotime within endothelium only
python scripts/GSE294775_mtx_to_npz.py        # chunked matrix conversion (memory-safe)
python scripts/GSE294775_scRNA.py
python scripts/GSE294775_post_analysis.py
python scripts/S4_GSE294775_deepdive_and_integration.py
python scripts/GSE223245_bulk_DEG.py          # blood transcriptome vs injury severity
python scripts/S5_module_and_methods.py
python scripts/S6_cluster_annotation.py
python scripts/S7_module_KO_and_network.py
```

## 6. Notes and caveats

- **Virtual knockout and network-differential analyses require raw counts.** scTenifoldNet/scTenifoldKnk results are invalid if log-normalised values are supplied; `S3_*` and `S7_*` scripts therefore read the count layer.
- **Pseudotime was computed within the endothelial compartment only.** Including smooth-muscle cells and pericytes produced a spurious gradient and was discarded; the root is set at injury-centre tip/stalk endothelium.
- **Gene-set scoring is performed on the full gene set before dimensionality reduction.** Scoring on a highly-variable-gene subset caused widespread misassignment of clusters and is not used.
- **Trajectory phenotypes are dominated by baseline substrate level, not by curve shape.** Shape-only clustering gave ARI 0.028–0.046 against the reported three-group solution; the manuscript states this explicitly and does not claim a purely morphological taxonomy.
- Chromosome-cluster identities reported in earlier internal drafts were corrected against the deposited cell-type annotations; the corrected labels (`celltype_l2`) are the ones used in the manuscript.
- The `S7_vKO_*.csv` and `S3_vKO_*.csv` files are full per-gene perturbation matrices for each virtual knockout, not filtered gene lists.

## 7. Citation and licence

Code and summary tables are released under the MIT licence (see `LICENSE`). This licence covers the analysis code and the de-identified summary tables only; it grants no right to the restricted clinical source records, which are not distributed here. Please cite the manuscript above; a `CITATION.cff` is provided.

> This repository is a research record, not a clinical tool. Nothing here is intended for diagnostic or therapeutic use.
