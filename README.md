# Calibrated Germline Architecture of 62 Hereditary Cancer Genes Across 9,768 Genomes and 82 Endogamous Indian Communities Reconciled with National Cancer Surveillance

**Authors:** Atanu Kumar Dutta ([ORCID: 0000-0002-8413-3454](https://orcid.org/0000-0002-8413-3454)), Soumyajit Mondal  
**Affiliation:** Department of Biochemistry, All India Institute of Medical Sciences (AIIMS), Kalyani, NH-34 Connector, Basantapur, Saguna, Kalyani 741245, West Bengal, India  
**Corresponding Author:** `atanu.biochem@aiimskalyani.edu.in`  
**Target Journal:** *Familial Cancer* (Springer Nature)

---

## 1. Deposit Overview

This Zenodo archive provides the complete curated datasets, variant-level adjudication catalogs, annotation caches, reproducible Python scripts, and 600-DPI publication figures supporting our 62-gene hereditary cancer audit of GenomeIndia Phase 1 (`n = 9,768` individuals; 82 endogamous communities) and ICMR–NCDIR *Cancer Samiksha* national cancer surveillance.

### Licensing
- **Source Code (`*.py`, `scripts/`):** MIT License
- **Curated Data Tables, Variant Catalogs & Figures (`data_cancer/`, `data_lynch/`, `tables/`, `figures_manuscript/`):** Creative Commons Attribution 4.0 International (CC BY 4.0)  
See `LICENSE` for full terms and third-party database acknowledgments.

---

## 2. Directory Structure

```text
Zenodo_Upload/
├── README.md                              # This documentation & reproduction guide
├── LICENSE                                # Dual MIT (code) + CC-BY-4.0 (data/figures) license
├── CITATION.cff                           # Machine-readable citation metadata
├── .zenodo.json                           # Zenodo deposit metadata schema
├── requirements.txt                       # Python dependencies
├── SHA256SUMS.txt                         # Cryptographic SHA-256 checksums for every file
├── population_mapping.py                  # Full decoded ethno-linguistic-geographic labels (82 communities)
├── run_master_familial_cancer_62genes.py  # Master analysis script generating Tables 1-7 & Figs 1-6
├── data_cancer/                           # Curated 62-gene variants, 82-pop tables, ClinVar/gnomAD/SpliceAI/GN097 data
│   └── mavedb/                            # Published MaveDB SGE/DMS functional CSVs (BRCA1/2, MSH2, TP53, VHL, PTEN)
├── data_lynch/                            # Exploratory Lynch syndrome subset (MLH1, MSH2, MSH6, PMS2)
├── tables/                                # All CSV/XLSX tables, ESM_1 (15 tabs), ESM_2 (6,277 variants), ESM_3 (.zip)
├── figures_manuscript/                    # Seven 600-DPI publication PNGs (Figs 1-7)
└── scripts/
    ├── 00_legacy_extraction/              # Initial GenomeIndia chromosome extraction & VEP/ClinVar scripts
    ├── 01_spdi_and_adjudication/          # Canonical SPDI verification, MANE PVS1 & exon boundary audit scripts
    ├── 02_functional_splicing_and_models/ # MaveDB, OpenCRAVAT SpliceAI, Wahlund regional F_k, PAF, GWAS & Fig 7 scripts
    └── 03_packaging_and_qa/               # Manuscript DOCX builder, integrity & leak audit scripts
```

---

## 3. Standalone Reproducibility Guide

All manuscript tables (`Table_1`–`Table_7`, `Table_M1`–`Table_M5`, `Table_S1`), workbooks (`ESM_1`), and figures (`Fig1`–`Fig7`) can be reproduced completely offline from the included extracted 62-gene datasets without re-downloading multi-gigabyte raw chromosome files.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Offline Reproduction Commands (Run Inside Deposit Root)
```bash
# 1. Regenerate Tables 1-7, Master Excel Workbook, and Figures 1-6 (600 DPI):
python run_master_familial_cancer_62genes.py

# 2. Regenerate Penetrance-Adjusted PAF, Wahlund Regional F_k / NNS, SAS Diaspora Bias & Gallbladder GWAS (Tables M3-M5):
python scripts/02_functional_splicing_and_models/run_module2_stratified_models.py

# 3. Regenerate Figure 7 (600 DPI 4-panel functional/PAF/Wahlund/GWAS synthesis):
python scripts/02_functional_splicing_and_models/generate_fig7.py
```

*(Note: Scripts querying external live REST/GraphQL endpoints—such as `reconcile_exact_spdi_matches.py`, `finalize_62gene_adjudication_and_gnomadv4.py`, `run_opencravat_spliceai.py`, and `finalize_module1_mave_splice_chip.py`—have their complete retrieved JSON/CSV outputs pre-cached under `data_cancer/` so downstream modeling and figure generation run offline cleanly.)*

---

## 4. Key Data Files & Schema Summary

| File | Description |
|---|---|
| `tables/ESM_1_Familial_Cancer_62Gene_Workbook.xlsx` | 15-tab master workbook (`Cover_Metadata`, `T1`–`T7`, `S2`–`S3`, `M1`–`M5`) |
| `tables/ESM_2_Complete_62Gene_Variant_Audit.csv` | Line-by-line adjudication across all 6,277 coding/ClinVar-retained rare variants (filtered from 15,988 total `<1%` AF calls) with GRCh38 coordinates, MANE Select transcripts, SPDI-verified ClinVar classifications, and gnomAD v4.1 frequencies |
| `tables/Table_1_62Gene_Prevalence_and_Syndrome_Burden.csv` | Pre-/post-calibration carrier frequencies and 95% Wilson CIs |
| `tables/Table_2_All62_Genes_Decomposition.csv` | Gene-by-gene decomposition across all 62 genes (`Core-33` + `Extra-29`) |
| `tables/Table_4_Curated_219_Variant_Catalog_with_gnomADv4.csv` | 216 curated qualifying variants + 2 low-penetrance risk modifiers with gnomAD v4.1 global, SAS, and NFE benchmarks |
| `tables/Table_M1_SpliceAI_High_Confidence_Candidates.csv` | All 21 high-confidence `SpliceAI >= 0.50` cryptic-splice candidates |
| `tables/Table_M2_MAVE_NonQualifying_VUS_Audit.csv` | 338 non-qualifying rare variants empirically assayed in MaveDB datasets |
| `tables/Table_M3_Penetrance_Adjusted_PAF_Models.csv` | Expected clinical shares and 2025 Levin PAFs for breast, ovarian, colorectal, and endometrial cancer |
| `tables/Table_M4_Wahlund_RegionalF_and_SAS_Bias.csv` | Community-level Wahlund `q_k^2`, regional `F_k`, targeted NNS, and gnomAD SAS blind-spot decomposition across all 82 communities |
| `tables/Table_M5_Gallbladder_GWAS_82Communities.csv` | GRCh38-polarized gallbladder cancer GWAS risk alleles (`ABCB4`, `ABCG8`) across Indian geographic regions |

---

## 5. External Upstream Data Sources (Not Redistributed Verbatim)
1. **GenomeIndia Phase 1 Aggregate Summary Releases (`n = 9,768` & `MAF_83populations.tsv`):** Indian Biological Data Centre (IBDC) GI-DB portal (`https://ibdc.dbt.gov.in/gidb/`). Full upstream raw chromosome-level TSVs (>10 GB) are omitted to maintain archive portability; all extracted rare variants (`<1%` AF) across the 62 target genes and target GBC GWAS SNPs are included.
2. **ICMR–NCDIR National Cancer Registry Programme (*Cancer Samiksha*):** `https://www.ncdirindia.org/cancersamiksha/DEFAULT.aspx`. Extracted 2016, 2020, and 2025 site-level projections used in analysis are stored in `data_cancer/cancer_samiksha_projections.csv`.
3. **ClinGen ENIGMA *BRCA1/2* VCEP v1.2 Specifications (`GN097`):** `https://cspec.genome.network/cspec/ui/svi/doc/GN097`.
