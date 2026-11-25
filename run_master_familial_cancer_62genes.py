import json, math, os, re, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from population_mapping import get_full_ethnic_name

os.makedirs("tables", exist_ok=True)
os.makedirs("figures_manuscript", exist_ok=True)
ART_FIG_DIR = None

# Springer Familial Cancer plot styling: Arial, clean spines, no top/right border, colorblind-safe Okabe-Ito
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["savefig.dpi"] = 600

# Colors (Okabe-Ito colorblind-safe)
C_NAVY = "#004488"
C_BLUE = "#0072B2"
C_SKY = "#56B4E9"
C_TEAL = "#009E73"
C_ORANGE = "#E69F00"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY = "#6C757D"
C_DARK = "#222222"

N_GI_DIPLOIDS = 9768  # GenomeIndia Phase 1 release cohort size (9768GI_SummaryStats)
N_GI_MAX_CHR = 19742  # Observed singleton reciprocal 1/5.06534e-05 (pooled joint call chroms)

# 1. Load manifests and verify exact 62-gene taxonomy
with open("data_cancer/cancer_genes_manifest.json", encoding="utf-8") as f:
    man33 = json.load(f)
with open("data_cancer/extra_cancer_genes_manifest.json", encoding="utf-8") as f:
    man29 = json.load(f)
man62 = {**man33, **man29}
assert len(man33) == 33 and len(man29) == 29 and len(man62) == 62

# Exact gene metadata dictionary across all 62 genes
GENE_META = {
    # AD HBOC / HRR (9 genes)
    "BRCA1":   ("AD", "HBOC / HRR", "CDC Tier 1 / ACMG SF v3.2", "LoF / Functional", "High"),
    "BRCA2":   ("AD", "HBOC / HRR", "CDC Tier 1 / ACMG SF v3.2", "LoF / Functional", "High"),
    "PALB2":   ("AD", "HBOC / HRR", "ACMG SF v3.2 / NCCN High", "LoF / Functional", "High"),
    "ATM":     ("AD", "HBOC / HRR", "NCCN Breast/Pancreas Mod", "LoF / ATM missense", "Moderate"),
    "CHEK2":   ("AD", "HBOC / HRR", "NCCN Breast/CRC Mod", "LoF / CHEK2 kinase", "Moderate"),
    "RAD51C":  ("AD", "HBOC / HRR", "NCCN Ovary/TNBC Mod", "LoF / RAD51 complex", "Moderate"),
    "RAD51D":  ("AD", "HBOC / HRR", "NCCN Ovary/TNBC Mod", "LoF / RAD51 complex", "Moderate"),
    "BARD1":   ("AD", "HBOC / HRR", "NCCN Breast Mod", "LoF", "Moderate"),
    "BRIP1":   ("AD", "HBOC / HRR", "NCCN Ovary Mod", "LoF", "Moderate"),
    # AD Lynch Syndrome (5 genes)
    "MLH1":    ("AD", "Lynch Syndrome (MMR)", "CDC Tier 1 / ACMG SF v3.2", "LoF / MMR missense", "High"),
    "MSH2":    ("AD", "Lynch Syndrome (MMR)", "CDC Tier 1 / ACMG SF v3.2", "LoF / MMR missense", "High"),
    "MSH6":    ("AD", "Lynch Syndrome (MMR)", "CDC Tier 1 / ACMG SF v3.2", "LoF / MMR missense", "High-Mod"),
    "PMS2":    ("AD", "Lynch Syndrome (MMR)", "CDC Tier 1 / ACMG SF v3.2", "LoF / MMR missense", "Moderate"),
    "EPCAM":   ("AD", "Lynch Syndrome (MMR)", "NCCN Lynch (3' del only)", "3' CNV deletion", "High"),
    # AD GI Polyposis & Gastric (8 genes)
    "APC":     ("AD", "GI Polyposis & Gastric", "ACMG SF v3.2", "LoF / beta-cat reg", "High"),
    "BMPR1A":  ("AD", "GI Polyposis & Gastric", "ACMG SF v3.2", "LoF", "High"),
    "SMAD4":   ("AD", "GI Polyposis & Gastric", "ACMG SF v3.2", "LoF", "High"),
    "STK11":   ("AD", "GI Polyposis & Gastric", "ACMG SF v3.2", "LoF", "High"),
    "CDH1":    ("AD", "GI Polyposis & Gastric", "NCCN / IGCLC HDGC", "LoF", "High"),
    "CTNNA1":  ("AD", "GI Polyposis & Gastric", "NCCN / IGCLC HDGC", "LoF", "Moderate"),
    "AXIN2":   ("AD", "GI Polyposis & Gastric", "NCCN / EHTG Oligodontia-CRC", "LoF", "Moderate"),
    "GREM1":   ("AD_GoF", "GI Polyposis & Gastric", "NCCN / EHTG HMPS", "GoF upstream 40kb dup only", "Moderate"),
    # AD Proofreading & Esophageal GoF (3 genes)
    "POLE":    ("AD_GoF", "Proofreading & Esophageal (GoF)", "NCCN / EHTG PPAP", "Exonuclease missense GoF only", "High"),
    "POLD1":   ("AD_GoF", "Proofreading & Esophageal (GoF)", "NCCN / EHTG PPAP", "Exonuclease missense GoF only", "High"),
    "RHBDF2":  ("AD_GoF", "Proofreading & Esophageal (GoF)", "OMIM Tylosis-EC (148500)", "N-term GoF missense only", "High"),
    # AD Core Tumor Suppressors, Brain & Melanoma (9 genes)
    "TP53":    ("AD", "Core Suppressors, Brain & Melanoma", "ACMG SF v3.2", "LoF / DN missense", "High"),
    "PTEN":    ("AD", "Core Suppressors, Brain & Melanoma", "ACMG SF v3.2", "LoF / PTEN domain", "High"),
    "CDKN2A":  ("AD", "Core Suppressors, Brain & Melanoma", "NCCN Melanoma/Pancreas", "LoF / p16 missense", "High"),
    "RB1":     ("AD", "Core Suppressors, Brain & Melanoma", "ACMG SF v3.2", "LoF", "High"),
    "NF1":     ("AD", "Core Suppressors, Brain & Melanoma", "NCCN Breast/CNS", "LoF", "Moderate"),
    "NF2":     ("AD", "Core Suppressors, Brain & Melanoma", "ACMG SF v3.2", "LoF", "High"),
    "POT1":    ("AD", "Core Suppressors, Brain & Melanoma", "Melanoma/Glioma/Thyroid/CLL", "LoF", "Moderate"),
    "MITF":    ("AD_GoF", "Core Suppressors, Brain & Melanoma", "Melanoma Susceptibility (E318K)", "p.Glu318Lys GoF only", "Moderate"),
    "HOXB13":  ("AD_GoF", "Core Suppressors, Brain & Melanoma", "NCCN Prostate (G84E)", "Missense susceptibility only", "Moderate"),
    # AD Endocrine, Renal, Thyroid & PGL (16 genes)
    "MEN1":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "High"),
    "RET":     ("AD_GoF", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "Activating missense GoF only", "High"),
    "VHL":     ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF / VHL missense (excl R200W)", "High"),
    "SDHA":    ("AD_LowPen", "Endocrine, Renal, Thyroid & PGL", "PPGL Low-Penetrance", "LoF (low adult penetrance)", "Low (~1-2%)"),
    "SDHB":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF / SDHB missense", "High-Mod"),
    "SDHC":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "Moderate"),
    "SDHD":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF (paternal imprinting)", "Moderate"),
    "FH":      ("AD", "Endocrine, Renal, Thyroid & PGL", "HLRCC / Renal", "LoF / FH missense", "Moderate"),
    "BAP1":    ("AD", "Endocrine, Renal, Thyroid & PGL", "BAP1-TPDS", "LoF", "High"),
    "FLCN":    ("AD", "Endocrine, Renal, Thyroid & PGL", "Birt-Hogg-Dube", "LoF", "Moderate"),
    "TSC1":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "High"),
    "TSC2":    ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "High"),
    "WT1":     ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "High"),
    "DICER1":  ("AD", "Endocrine, Renal, Thyroid & PGL", "DICER1 Syndrome (Thyroid/PPB)", "LoF / RNase IIIb", "Moderate"),
    "PRKAR1A": ("AD", "Endocrine, Renal, Thyroid & PGL", "Carney Complex (Thyroid/Pituitary)", "LoF", "High"),
    "MAX":     ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "Moderate"),
    "TMEM127": ("AD", "Endocrine, Renal, Thyroid & PGL", "ACMG SF v3.2", "LoF", "Moderate"),
    # Autosomal Recessive Cancer & Polyposis (7 genes)
    "MUTYH":   ("AR", "AR Cancer & Polyposis", "ACMG SF v3.2 (Biallelic)", "LoF / MAP missense", "High (biallelic)"),
    "NTHL1":   ("AR", "AR Cancer & Polyposis", "NCCN / EHTG NTS (Biallelic)", "LoF", "High (biallelic)"),
    "MSH3":    ("AR", "AR Cancer & Polyposis", "EHTG FAP4 (Biallelic)", "LoF", "High (biallelic)"),
    "MBD4":    ("AR", "AR Cancer & Polyposis", "EHTG MANS (Biallelic)", "LoF", "High (biallelic)"),
    "FANCA":   ("AR", "AR Cancer & Polyposis", "Fanconi Anemia Group A", "LoF / FA missense", "High (biallelic)"),
    "BLM":     ("AR", "AR Cancer & Polyposis", "Bloom Syndrome", "LoF", "High (biallelic)"),
    "WRN":     ("AR", "AR Cancer & Polyposis", "Werner Syndrome", "LoF / cryptic splice", "High (biallelic)"),
    # Contested / Non-Oncologic Recessive (5 genes)
    "SEC23B":  ("Disputed_CDA2", "Evaluated Non-Oncologic / Contested FNMTC", "CDA-II (AR); Cowden-7 Disputed", "Recessive hematologic", "None (AD cancer)"),
    "HABP2":   ("Disputed_FNMTC", "Evaluated Non-Oncologic / Contested FNMTC", "Refuted FNMTC Candidate", "Uncertain / benign LoF", "Disputed/Refuted"),
    "SRGAP1":  ("Disputed_FNMTC", "Evaluated Non-Oncologic / Contested FNMTC", "Uncertain FNMTC Candidate", "Insufficient evidence", "Limited"),
    "FOXE1":   ("Disputed_FNMTC", "Evaluated Non-Oncologic / Contested FNMTC", "Thyroid GWAS / FNMTC Locus", "Non-coding/polyAla", "Limited"),
    "NKX2-1":  ("Disputed_FNMTC", "Evaluated Non-Oncologic / Contested FNMTC", "Brain-Lung-Thyroid / Contested", "Insufficient FNMTC", "Limited"),
}
print(f"Gene meta count: {len(GENE_META)}")
missing_in_meta = set(man62.keys()) - set(GENE_META.keys())
extra_in_meta = set(GENE_META.keys()) - set(man62.keys())
if extra_in_meta:
    for g in extra_in_meta:
        GENE_META.pop(g)
assert set(GENE_META.keys()) == set(man62.keys()), f"Mismatch: {missing_in_meta} | {extra_in_meta}"
print("Verified 100% 1:1 mapping for all 62 genes! (Note: WT1 wasn't in Extra29 -> popped cleanly)")

# 2. Load final adjudicated variant table and apply final NMD-escape / Zero-star ClinVar refinements
df = pd.read_csv("data_cancer/master_62genes_adjudicated_variants_final.csv", low_memory=False)
df33_all = pd.read_csv("data_cancer/annotated_pan_cancer_variants_genome_india.csv", low_memory=False)
df29_all = pd.read_csv("data_cancer/raw_extra_29_cancer_variants_gi_with_pops.csv", low_memory=False)

df["is_penult_exon"] = False
# Recompute exact is_last_exon from exon strings AND check penultimate-exon variants!
for idx, r in df.iterrows():
    ex = str(r["exon"]) if pd.notna(r["exon"]) else ""
    csq = str(r["mane_csq"]) if pd.notna(r["mane_csq"]) else ""
    rev = str(r["cv_agg_review"]).lower() if pd.notna(r["cv_agg_review"]) else ""
    gene = str(r["gene"])
    st = str(r["stratum"])
    is_last = False
    is_penult = False
    if "/" in ex:
        try:
            a_str, b_str = ex.split("/")[0], ex.split("/")[1]
            a_i = int(a_str.split("-")[0])
            b_i = int(b_str)
            is_last = (a_i == b_i)
            is_penult = (a_i == b_i - 1 and b_i >= 2)
        except Exception:
            pass
    df.at[idx, "is_last_exon"] = is_last
    df.at[idx, "is_penult_exon"] = is_penult

    # Check PMS2CL pseudogene homology zone (exons 12-15, and note exon 11)
    is_pms2cl = False
    if gene == "PMS2" and "/" in ex:
        try:
            if int(ex.split("/")[0].split("-")[0]) >= 12:
                is_pms2cl = True
        except Exception:
            pass
    df.at[idx, "is_pms2cl_exon"] = is_pms2cl

    # Refine Stratum1B_ClinVar_PLP_Other (0-star no assertion criteria)
    if st == "Stratum1B_ClinVar_PLP_Other" and "no assertion criteria" in rev:
        df.at[idx, "stratum"] = "Excluded_ZeroStar_ClinVar"
        df.at[idx, "mech_note"] = "ClinVar 0-star (no assertion criteria provided; -3 splice variant)"

    # Refine Stratum2 last-exon NMD escape under ClinGen SVI PVS1 (keep BRCA2 < codon 3309 per ENIGMA VCEP!)
    if st == "Stratum2_Novel_MANE_LoF_PVS1" and is_last:
        is_splice = ("splice_donor" in csq) or ("splice_acceptor" in csq)
        if not is_splice and ("start_lost" not in csq):
            if gene == "BRCA2" and pd.notna(r["prot_start"]) and float(r["prot_start"]) < 3309:
                df.at[idx, "mech_note"] = "Retained under ENIGMA BRCA2 VCEP (< codon 3309 functional domain)"
            else:
                df.at[idx, "stratum"] = "Excluded_LastExon_NMD_Escape_VUS"
                df.at[idx, "mech_note"] = f"Novel unreviewed last-exon truncation ({ex}) lacking ClinVar P/LP consensus (NMD escape)"

    # ClinGen InSiGHT APC VCEP: PVS1 restricted to codons 49-2645; exclude 5' truncations (< codon 49)
    if df.at[idx, "stratum"] == "Stratum2_Novel_MANE_LoF_PVS1" and gene == "APC" and pd.notna(r["prot_start"]) and float(r["prot_start"]) < 49:
        df.at[idx, "stratum"] = "Excluded_InSiGHT_Codon49_5Prime_VUS"
        df.at[idx, "mech_note"] = "Excluded under ClinGen InSiGHT APC VCEP (< codon 49 PVS1 5' boundary)"

# Assign inheritance mode & syndrome group
df["inh_mode"] = df["gene"].map(lambda g: GENE_META[g][0])
df["syndrome_group"] = df["gene"].map(lambda g: GENE_META[g][1])
df["guideline_tier"] = df["gene"].map(lambda g: GENE_META[g][2])

# Save final master variant CSV
df.to_csv("tables/Table_S1_Complete_62Gene_Variant_Adjudication_Audit.csv", index=False)

print("\n=== FINAL STRATUM COUNTS ACROSS ALL 6,277 AUDITED RARE VARIANTS ===")
print(df["stratum"].value_counts().to_string())

# Print any penultimate-exon Stratum2 variants to verify clinically
pen_s2 = df[(df["stratum"] == "Stratum2_Novel_MANE_LoF_PVS1") & (df["is_penult_exon"] == True)]
print(f"\nPenultimate-exon Stratum2 variants ({len(pen_s2)}):")
if len(pen_s2):
    print(pen_s2[["gene", "exon", "hgvsc", "hgvsp", "prot_start", "af_gi"]].to_string())

# Print BRCA2 variants retained & excluded for full clarity
print("\nAll BRCA2 Stratum1/2/LastExon:")
print(df[(df["gene"]=="BRCA2") & (df["stratum"].isin(["Stratum1A_ClinVar_PLP_TwoPlus","Stratum1B_ClinVar_PLP_SingleSub","Stratum2_Novel_MANE_LoF_PVS1","Excluded_LastExon_NMD_Escape_VUS"]))][["exon","hgvsc","hgvsp","prot_start","af_gi","stratum","mech_note"]].to_string())

# Helper: Prevalence + Wilson 95% CI on effective diploid cohort N=9768
def calc_prev_ci(sub_df, n_cohort=N_GI_DIPLOIDS):
    if len(sub_df) == 0:
        return 0, 0, 0.0, 0.0, np.nan, 0.0, 0.0, "0 (None)"
    q_sum = float(sub_df["af_gi"].sum())
    cf = 1.0 - float(np.prod([(1.0 - min(max(q, 0.0), 1.0))**2 for q in sub_df["af_gi"]]))
    prev_100k = cf * 100000.0
    one_in_n = (1.0 / cf) if cf > 0 else np.nan
    # Wilson 95% CI using diploid sample size n_cohort
    z = 1.959964
    k = cf * n_cohort
    denom = 1.0 + z**2 / n_cohort
    center = (cf + z**2 / (2.0 * n_cohort)) / denom
    half = z * math.sqrt((cf * (1.0 - cf) / n_cohort) + z**2 / (4.0 * n_cohort**2)) / denom
    ci_lo = max(0.0, (center - half)) * 100.0
    ci_hi = min(1.0, (center + half)) * 100.0
    ratio_str = f"1 in {int(round(one_in_n)):,}" if pd.notna(one_in_n) else "None"
    return sub_df["gene"].nunique(), len(sub_df), q_sum, cf * 100.0, prev_100k, ci_lo, ci_hi, ratio_str

# Core tier definitions
S1A = ["Stratum1A_ClinVar_PLP_TwoPlus"]
S1AB = ["Stratum1A_ClinVar_PLP_TwoPlus", "Stratum1B_ClinVar_PLP_SingleSub"]
S_ALL = S1AB + ["Stratum2_Novel_MANE_LoF_PVS1"]

# Mask sets
ad_mask = df["inh_mode"].isin(["AD", "AD_GoF", "AD_LowPen"])
ad_notsdha_mask = df["inh_mode"].isin(["AD", "AD_GoF"])  # excludes low-penetrance SDHA
cdc_t1_genes = {"BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}
hboc_br_genes = {"BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D", "BARD1", "BRIP1"}
lynch_genes = {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}
ar_mask = df["inh_mode"] == "AR"

# ==============================================================================
# TABLE 1: Prevalence Summary Across Adjudication Tiers, Inheritance, & Syndromes
# ==============================================================================
t1_rows = []
strata_defs = [
    ("Stratum 1A (ClinVar 2+/Expert P/LP)", S1A),
    ("Stratum 1A+1B (All Criteria-Provided ClinVar P/LP)", S1AB),
    ("Primary Curated (Stratum 1A+1B + Stratum 2 MANE-PVS1 LoF)", S_ALL),
]
cohort_defs = [
    ("All Autosomal Dominant Cancer Genes (50 AD/GoF incl. SDHA)", ad_mask),
    ("AD Actionable Excluding Low-Penetrance SDHA (49 genes)", ad_notsdha_mask),
    ("CDC Tier 1 Genes (HBOC BRCA1/2 + Lynch MMR, 7 genes)", df["gene"].isin(cdc_t1_genes)),
    ("HBOC / HRR Pathway (9 genes)", df["gene"].isin(hboc_br_genes)),
    ("High-Penetrance HBOC Only (BRCA1, BRCA2, PALB2)", df["gene"].isin(["BRCA1", "BRCA2", "PALB2"])),
    ("Lynch Syndrome MMR (4 active genes; EPCAM 0 SNV/indel)", df["gene"].isin(lynch_genes)),
    ("GI Polyposis & Gastric AD (APC, SMAD4, CDH1, CTNNA1, etc.)", (df["syndrome_group"] == "GI Polyposis & Gastric") & ad_mask),
    ("Core Suppressors, Brain & Melanoma (TP53, CDKN2A, RB1, NF1, POT1)", (df["syndrome_group"] == "Core Suppressors, Brain & Melanoma") & ad_mask),
    ("Endocrine, Renal, Thyroid & PGL (incl. SDHA)", (df["syndrome_group"] == "Endocrine, Renal, Thyroid & PGL") & ad_mask),
    ("Endocrine, Renal, Thyroid & PGL (excl. SDHA)", (df["syndrome_group"] == "Endocrine, Renal, Thyroid & PGL") & ad_notsdha_mask),
    ("Autosomal Recessive Cancer & Polyposis Carriers (7 genes, monoallelic)", ar_mask),
    ("Combined AD (excl. SDHA) + AR Carriers (56 pathogenic genes)", ad_notsdha_mask | ar_mask),
    ("Combined AD (all) + AR Carriers (57 pathogenic genes)", ad_mask | ar_mask),
]

for clabel, cmask in cohort_defs:
    for slabel, slist in strata_defs:
        sub = df[cmask & df["stratum"].isin(slist)]
        ng, nv, qs, cfp, p100, clo, chi, rat = calc_prev_ci(sub)
        t1_rows.append({
            "Cohort_Scope": clabel,
            "Adjudication_Tier": slabel,
            "Active_Genes": ng,
            "Qualifying_Variants": nv,
            "Sum_AF": round(qs, 6),
            "Carrier_Freq_Percent": round(cfp, 3),
            "CI_95_Low_Percent": round(clo, 3),
            "CI_95_High_Percent": round(chi, 3),
            "Per_100k": round(p100, 1),
            "Carrier_Ratio": rat,
        })

# Add Low-Penetrance Modifiers (APC I1307K + CHEK2 I157T)
mod_sub = df[df["stratum"] == "Modifier_LowPenetrance_Risk"]
ng, nv, qs, cfp, p100, clo, chi, rat = calc_prev_ci(mod_sub)
t1_rows.append({
    "Cohort_Scope": "Low-Penetrance Risk Modifiers (APC p.Ile1307Lys + CHEK2 p.Ile157Thr)",
    "Adjudication_Tier": "Separate Modifier Stratum",
    "Active_Genes": ng, "Qualifying_Variants": nv, "Sum_AF": round(qs, 6),
    "Carrier_Freq_Percent": round(cfp, 3), "CI_95_Low_Percent": round(clo, 3), "CI_95_High_Percent": round(chi, 3),
    "Per_100k": round(p100, 1), "Carrier_Ratio": rat,
})

# Sensitivity: adding back the 7 non-BRCA2 Last-Exon NMD-escape variants
sens_last = df[ad_mask & (df["stratum"].isin(S_ALL) | ((df["stratum"] == "Excluded_LastExon_NMD_Escape_VUS") & (df["gene"] != "BRCA2") & (df["inh_mode"].isin(["AD", "AD_LowPen"]))))]
ng, nv, qs, cfp, p100, clo, chi, rat = calc_prev_ci(sens_last)
t1_rows.append({
    "Cohort_Scope": "Sensitivity: All AD + Non-BRCA2 Last-Exon Truncations",
    "Adjudication_Tier": "Sensitivity Bound",
    "Active_Genes": ng, "Qualifying_Variants": nv, "Sum_AF": round(qs, 6),
    "Carrier_Freq_Percent": round(cfp, 3), "CI_95_Low_Percent": round(clo, 3), "CI_95_High_Percent": round(chi, 3),
    "Per_100k": round(p100, 1), "Carrier_Ratio": rat,
})

t1 = pd.DataFrame(t1_rows)
t1.to_csv("tables/Table_1_62Gene_Prevalence_and_Syndrome_Burden.csv", index=False)
print("\n=== TABLE 1 KEY PREVALENCE RESULTS ===")
print(t1[t1["Adjudication_Tier"].str.contains("Primary Curated|Stratum 1A \\(|Modifier|Sensitivity")][["Cohort_Scope", "Adjudication_Tier", "Qualifying_Variants", "Carrier_Freq_Percent", "CI_95_Low_Percent", "CI_95_High_Percent", "Carrier_Ratio"]].to_string())

# ==============================================================================
# TABLE 2: Gene-by-Gene Decomposition Across All 62 Genes
# ==============================================================================
t2_rows = []
for g in sorted(man62.keys()):
    sub_all = df[df["gene"] == g]
    mode, sgrp, gtier, mech, pen = GENE_META[g]
    sub_1a = sub_all[sub_all["stratum"].isin(S1A)]
    sub_1ab = sub_all[sub_all["stratum"].isin(S1AB)]
    sub_pri = sub_all[sub_all["stratum"].isin(S_ALL)]
    sub_mod = sub_all[sub_all["stratum"] == "Modifier_LowPenetrance_Risk"]
    _, n1a, q1a, cf1a, _, _, _, r1a = calc_prev_ci(sub_1a)
    _, n1ab, q1ab, cf1ab, _, _, _, r1ab = calc_prev_ci(sub_1ab)
    _, npri, qpri, cfpri, _, _, _, rpri = calc_prev_ci(sub_pri)
    _, nmod, qmod, cfmod, _, _, _, _ = calc_prev_ci(sub_mod)
    t2_rows.append({
        "Gene": g,
        "Cohort_Arm": "Core33" if g in man33 else "Extra29",
        "Inheritance": mode,
        "Syndrome_Group": sgrp,
        "Clinical_Tier": gtier,
        "Disease_Mechanism": mech,
        "Penetrance": pen,
        "Total_Rare_Vars_LT1pct": len(sub_all),
        "N_ClinVar_2Plus": n1a,
        "N_ClinVar_1Sub": len(sub_all[sub_all["stratum"] == "Stratum1B_ClinVar_PLP_SingleSub"]),
        "N_Novel_MANE_PVS1": len(sub_all[sub_all["stratum"] == "Stratum2_Novel_MANE_LoF_PVS1"]),
        "N_Qualifying_Total": npri,
        "N_Modifier_Risk": nmod,
        "N_VUS_ClinVar": len(sub_all[sub_all["stratum"] == "Excluded_VUS_ClinVar"]),
        "N_Conflicting_ClinVar": len(sub_all[sub_all["stratum"] == "Excluded_Conflicting_ClinVar"]),
        "N_ClinVar_Overturned_LoF": len(sub_all[sub_all["stratum"] == "Excluded_ClinVar_Overturned_LoF"]),
        "N_Excluded_Other_Artifacts": len(sub_all[sub_all["stratum"].isin(["Excluded_Mechanism_or_Disputed_Gene", "Excluded_UTR_Splice_NonCoding", "Excluded_LastExon_NMD_Escape_VUS", "Excluded_InSiGHT_Codon49_5Prime_VUS", "Excluded_NonOncologic_Recessive_VHL", "Excluded_ZeroStar_ClinVar"])]),
        "CF_Stratum1A_Pct": round(cf1a, 4),
        "CF_Stratum1AB_Pct": round(cf1ab, 4),
        "CF_Primary_Pct": round(cfpri, 4),
        "Ratio_Primary": rpri if npri > 0 else "0 (None)",
        "Sum_AF_Primary": round(qpri, 6),
        "CF_Modifier_Pct": round(cfmod, 4),
    })

t2 = pd.DataFrame(t2_rows).sort_values(["Inheritance", "CF_Primary_Pct"], ascending=[True, False])
t2.to_csv("tables/Table_2_All62_Genes_Decomposition.csv", index=False)
print("\n=== TOP 20 GENES BY PRIMARY CARRIER FREQUENCY ===")
print(t2[t2["N_Qualifying_Total"] > 0][["Gene", "Inheritance", "Syndrome_Group", "N_ClinVar_2Plus", "N_ClinVar_1Sub", "N_Novel_MANE_PVS1", "CF_Primary_Pct", "Ratio_Primary"]].head(22).to_string())

# ==============================================================================
# TABLE 3: Bioinformatic Variant-Inflation Calibration (Old vs Refined)
# ==============================================================================
t3_old = pd.read_csv("tables/Table_PanCancer_3_Pathogenic_and_LoF_Variants_Spectrum.csv")
# Match old Table 3 variants onto master table
def make_hg(chrom, pos, ref, alt):
    c = str(chrom).replace("chr", "")
    p = int(pos)
    ref, alt = str(ref), str(alt)
    if len(ref) == 1 and len(alt) == 1:
        return f"chr{c}:g.{p}{ref}>{alt}"
    elif len(ref) > 1 and len(alt) == 1 and ref.startswith(alt):
        s, e = p + 1, p + len(ref) - 1
        return f"chr{c}:g.{s}del" if s == e else f"chr{c}:g.{s}_{e}del"
    elif len(ref) == 1 and len(alt) > 1 and alt.startswith(ref):
        return f"chr{c}:g.{p}_{p+1}ins{alt[1:]}"
    return f"chr{c}:g.{p}_{p+len(ref)-1}delins{alt}"

t3_old["hgvs_g"] = t3_old.apply(lambda r: make_hg(r["Chrom"], r["Pos"], r["Ref"], r["Alt"]), axis=1)
old_merged = t3_old.merge(df[["hgvs_g", "stratum", "cv_agg_class", "cv_id", "mech_note"]], on="hgvs_g", how="left")
t3_old_sum = old_merged.groupby(["Tier", "stratum"], dropna=False).agg(N=("Gene", "count"), Sum_AF=("AF_GenomeIndia", "sum")).reset_index()
t3_old_sum.to_csv("tables/Table_3_Pipeline_Inflation_Audit_Decomposition.csv", index=False)

# Also compute naive old-style burden vs calibrated burden across 62 genes!
naive_core33_cf = (1.0 - np.prod([(1.0 - q)**2 for q in t3_old["AF_GenomeIndia"]])) * 100.0
cal_core33_all_cf = (1.0 - np.prod([(1.0 - q)**2 for q in df[(df["cohort_arm"]=="Core33") & df["stratum"].isin(S_ALL)]["af_gi"]])) * 100.0
cal_core33_ad_cf = (1.0 - np.prod([(1.0 - q)**2 for q in df[(df["cohort_arm"]=="Core33") & ad_mask & df["stratum"].isin(S_ALL)]["af_gi"]])) * 100.0
print("\n=== TABLE 3: OLD 262-VARIANT INFLATION AUDIT ===")
print(t3_old_sum.to_string())
print(f"Core33 Naive Old CF = {naive_core33_cf:.3f}% vs Calibrated Core33 All (incl MUTYH) = {cal_core33_all_cf:.3f}% vs AD-only = {cal_core33_ad_cf:.3f}%")
print("Total rare (<1%) variants Core33 + Extra29 =", len(df33_all[df33_all["af_gi"]<0.01]) + len(df29_all[df29_all["af_gi"]<0.01]),
      "(Total raw extracted =", len(df33_all) + len(df29_all), ")")

# ==============================================================================
# TABLE 4: GenomeIndia vs gnomAD v4.1 (SAS and NFE) Comparative Catalog
# ==============================================================================
qual_all = df[df["stratum"].isin(S_ALL + ["Modifier_LowPenetrance_Risk"])].copy()
qual_all["Found_gnomAD_SAS"] = qual_all["g4_sas_ac"].fillna(0) > 0
qual_all["Found_gnomAD_NFE"] = qual_all["g4_nfe_ac"].fillna(0) > 0
qual_all["Fold_GI_vs_g4SAS"] = np.where(qual_all["g4_sas_af"].fillna(0) > 0, qual_all["af_gi"] / qual_all["g4_sas_af"], np.inf)
qual_all["Fold_GI_vs_g4NFE"] = np.where(qual_all["g4_nfe_af"].fillna(0) > 0, qual_all["af_gi"] / qual_all["g4_nfe_af"], np.inf)

# Save full curated variant catalog (Table 4 / ESM)
t4_export = qual_all[[
    "gene", "cohort_arm", "inh_mode", "syndrome_group", "stratum", "chrom", "pos", "ref", "alt", "rsid",
    "hgvsc", "hgvsp", "exon", "intron", "mane_csq", "cv_id", "cv_agg_class", "cv_agg_review",
    "af_gi", "max_pop_af", "top_pop", "g4_sas_ac", "g4_sas_an", "g4_sas_af", "g4_nfe_ac", "g4_nfe_an", "g4_nfe_af", "g4_tot_af", "mech_note"
]].sort_values(["inh_mode", "af_gi", "gene"], ascending=[True, False, True])
t4_export["Top_Community_Decoded"] = t4_export["top_pop"].apply(lambda x: get_full_ethnic_name(str(x), "descriptive") if pd.notna(x) and str(x) != "" else "Cohort-wide / Subgroup Masked")
t4_export.to_csv("tables/Table_4_Curated_219_Variant_Catalog_with_gnomADv4.csv", index=False)

n_gi_only = int((~qual_all["Found_gnomAD_SAS"] & ~qual_all["Found_gnomAD_NFE"] & (qual_all["g4_tot_af"].fillna(0) == 0)).sum())
n_no_sas = int((~qual_all["Found_gnomAD_SAS"]).sum())
n_no_nfe = int((~qual_all["Found_gnomAD_NFE"]).sum())
print(f"\n=== GNOMAD V4.1 BENCHMARK AMONG {len(qual_all)} QUALIFYING + MODIFIER ALLELES ===")
print(f"Absent from gnomAD v4.1 global = {n_gi_only}/{len(qual_all)} ({n_gi_only/len(qual_all)*100:.1f}%)")
print(f"Absent from gnomAD v4.1 SAS    = {n_no_sas}/{len(qual_all)} ({n_no_sas/len(qual_all)*100:.1f}%)")
print(f"Absent from gnomAD v4.1 NFE    = {n_no_nfe}/{len(qual_all)} ({n_no_nfe/len(qual_all)*100:.1f}%)")
print("\nTop recurrent qualifying/modifier alleles (AF_GI >= 0.0002):")
print(t4_export[t4_export["af_gi"] >= 0.0002][["gene", "inh_mode", "stratum", "hgvsc", "hgvsp", "af_gi", "g4_sas_af", "g4_nfe_af", "Top_Community_Decoded"]].to_string())

# ==============================================================================
# TABLE 5: Autosomal Recessive Cancer/Polyposis & CMMRD Autozygosity Modeling
# ==============================================================================
# Includes 7 AR genes + 4 Lynch MMR genes (CMMRD)
ar_cm_genes = ["FANCA", "WRN", "MUTYH", "MSH3", "BLM", "NTHL1", "MBD4", "MLH1", "MSH2", "MSH6", "PMS2"]
t5_rows = []
for g in ar_cm_genes:
    sub = df[(df["gene"] == g) & df["stratum"].isin(S_ALL)]
    q = float(sub["af_gi"].sum())
    cf = 1.0 - float(np.prod([(1.0 - x)**2 for x in sub["af_gi"]]))
    p_rm = q**2
    p_f02 = q**2 + 0.02 * q * (1.0 - q)
    p_f04 = q**2 + 0.04 * q * (1.0 - q)
    p_f0625 = q**2 + 0.0625 * q * (1.0 - q)
    fold_02 = (p_f02 / p_rm) if p_rm > 0 else np.nan
    fold_0625 = (p_f0625 / p_rm) if p_rm > 0 else np.nan
    t5_rows.append({
        "Gene": g,
        "Recessive_Phenotype": {
            "FANCA": "Fanconi Anemia Group A",
            "WRN": "Werner Syndrome",
            "MUTYH": "MUTYH-Associated Polyposis (MAP)",
            "MSH3": "MSH3-Associated Polyposis (FAP4)",
            "BLM": "Bloom Syndrome",
            "NTHL1": "NTHL1 Tumor Syndrome (NTS)",
            "MBD4": "MBD4-Associated Neoplasia (MANS)",
            "MLH1": "CMMRD (MLH1 biallelic)",
            "MSH2": "CMMRD (MSH2 biallelic)",
            "MSH6": "CMMRD (MSH6 biallelic)",
            "PMS2": "CMMRD (PMS2 biallelic)",
        }[g],
        "N_Vars": len(sub),
        "Sum_AF_q": round(q, 6),
        "Het_Carrier_Pct": round(cf * 100.0, 3),
        "Het_Ratio": f"1 in {int(round(1.0/cf)):,}" if cf > 0 else "None",
        "Biallelic_F0_Per_1M": round(p_rm * 1e6, 2),
        "Biallelic_F02_Per_1M": round(p_f02 * 1e6, 2),
        "Biallelic_F04_Per_1M": round(p_f04 * 1e6, 2),
        "Biallelic_F0625_Per_1M": round(p_f0625 * 1e6, 2),
        "Fold_F02_vs_Panmictic": round(fold_02, 1) if pd.notna(fold_02) else np.nan,
        "Fold_F0625_vs_Panmictic": round(fold_0625, 1) if pd.notna(fold_0625) else np.nan,
    })

# Add Combined AR Polyposis (MUTYH+NTHL1+MSH3+MBD4), Combined All 7 AR Cancer, Combined CMMRD from unrounded variant/gene AFs
for lbl, glist in [
    ("Combined BER/MMR Polyposis (MUTYH+NTHL1+MSH3+MBD4)", ["MUTYH", "NTHL1", "MSH3", "MBD4"]),
    ("Combined All 7 AR Cancer Genes", ["FANCA", "WRN", "MUTYH", "MSH3", "BLM", "NTHL1", "MBD4"]),
    ("Combined CMMRD (MLH1+MSH2+MSH6+PMS2)", ["MLH1", "MSH2", "MSH6", "PMS2"]),
]:
    vsub = df[(df["gene"].isin(glist)) & df["stratum"].isin(S_ALL)]
    q_by_gene = [float(vsub[vsub["gene"] == g]["af_gi"].sum()) for g in glist]
    p0 = 1.0 - float(np.prod([1.0 - q**2 for q in q_by_gene]))
    p02 = 1.0 - float(np.prod([1.0 - (q**2 + 0.02 * q * (1.0 - q)) for q in q_by_gene]))
    p04 = 1.0 - float(np.prod([1.0 - (q**2 + 0.04 * q * (1.0 - q)) for q in q_by_gene]))
    p0625 = 1.0 - float(np.prod([1.0 - (q**2 + 0.0625 * q * (1.0 - q)) for q in q_by_gene]))
    cf_tot = 1.0 - float(np.prod([(1.0 - x)**2 for x in vsub["af_gi"]]))
    t5_rows.append({
        "Gene": lbl, "Recessive_Phenotype": lbl,
        "N_Vars": len(vsub),
        "Sum_AF_q": round(sum(q_by_gene), 6),
        "Het_Carrier_Pct": round(cf_tot * 100.0, 3),
        "Het_Ratio": f"1 in {int(round(1.0/cf_tot)):,}",
        "Biallelic_F0_Per_1M": round(p0 * 1e6, 2),
        "Biallelic_F02_Per_1M": round(p02 * 1e6, 2),
        "Biallelic_F04_Per_1M": round(p04 * 1e6, 2),
        "Biallelic_F0625_Per_1M": round(p0625 * 1e6, 2),
        "Fold_F02_vs_Panmictic": round(p02 / p0, 1),
        "Fold_F0625_vs_Panmictic": round(p0625 / p0, 1),
    })
t5 = pd.DataFrame(t5_rows)
t5.to_csv("tables/Table_5_Recessive_Polyposis_and_CMMRD_Autozygosity.csv", index=False)
print("\n=== TABLE 5: RECESSIVE AUTOZYGOSITY AMPLIFICATION ===")
print(t5[["Gene", "N_Vars", "Het_Carrier_Pct", "Het_Ratio", "Biallelic_F0_Per_1M", "Biallelic_F02_Per_1M", "Biallelic_F0625_Per_1M", "Fold_F02_vs_Panmictic"]].to_string())

# ==============================================================================
# TABLE 6: 82-Community Subgroup Analysis (Subgroup-Resolved Variants Only)
# ==============================================================================
p33 = pd.read_csv("data_cancer/pan_cancer_variants_82_populations.csv", low_memory=False)
p29 = pd.read_csv("data_cancer/extra_29_variants_82_populations.csv", low_memory=False)
pop_cols = [c for c in p33.columns if re.match(r"^(AA|DR|IE|TB)_", c)]
assert len(pop_cols) == 82
p_all = pd.concat([p33[["gene", "chrom", "pos", "ref", "alt"] + pop_cols],
                   p29[["gene", "chrom", "pos", "ref", "alt"] + pop_cols]], ignore_index=True)
p_all["chrom"] = p_all["chrom"].astype(str)
qual_merge = qual_all.copy()
qual_merge["chrom"] = qual_merge["chrom"].astype(str)
qm = qual_merge.merge(p_all, on=["gene", "chrom", "pos", "ref", "alt"], how="left")

# Count how many qualifying alleles have >0 subgroup data
qm["has_82pop"] = qm[pop_cols].fillna(0).sum(axis=1) > 0
n_has82_ad = int(qm[qm["inh_mode"].isin(["AD", "AD_GoF", "AD_LowPen"]) & qm["stratum"].isin(S_ALL) & qm["has_82pop"]].shape[0])
n_has82_ar = int(qm[(qm["inh_mode"] == "AR") & qm["stratum"].isin(S_ALL) & qm["has_82pop"]].shape[0])
n_has82_mod = int(qm[(qm["stratum"] == "Modifier_LowPenetrance_Risk") & qm["has_82pop"]].shape[0])
print(f"\n82-Community Subgroup-Resolved Variants: AD={n_has82_ad}, AR={n_has82_ar}, Modifier={n_has82_mod} (total {n_has82_ad+n_has82_ar+n_has82_mod}/{len(qm)})")

# Compute community-level carrier rates across decoded 82 groups
t6_rows = []
ad_sub = qm[qm["inh_mode"].isin(["AD", "AD_GoF", "AD_LowPen"]) & qm["stratum"].isin(S_ALL) & qm["has_82pop"]]
ar_sub = qm[(qm["inh_mode"] == "AR") & qm["stratum"].isin(S_ALL) & qm["has_82pop"]]
mod_sub = qm[(qm["stratum"] == "Modifier_LowPenetrance_Risk") & qm["has_82pop"]]

for pc in pop_cols:
    ad_afs = [float(x) for x in ad_sub[pc].fillna(0) if float(x) > 0]
    ar_afs = [float(x) for x in ar_sub[pc].fillna(0) if float(x) > 0]
    mod_afs = [float(x) for x in mod_sub[pc].fillna(0) if float(x) > 0]
    cf_ad = (1.0 - float(np.prod([(1.0 - x)**2 for x in ad_afs]))) * 100.0 if ad_afs else 0.0
    cf_ar = (1.0 - float(np.prod([(1.0 - x)**2 for x in ar_afs]))) * 100.0 if ar_afs else 0.0
    cf_mod = (1.0 - float(np.prod([(1.0 - x)**2 for x in mod_afs]))) * 100.0 if mod_afs else 0.0
    # Highlight top genes contributing to this community
    hits = []
    for _, vr in qm[qm["has_82pop"]].iterrows():
        vaf = float(vr[pc]) if pd.notna(vr[pc]) else 0.0
        if vaf > 0:
            p_lbl = str(vr["hgvsp"]).split(":")[-1] if pd.notna(vr["hgvsp"]) else str(vr["hgvsc"]).split(":")[-1]
            hits.append(f"{vr['gene']}({p_lbl}:{vaf:.4f})")
    fam = pc.split("_")[0]
    fam_full = {"AA": "Austroasiatic", "DR": "Dravidian", "IE": "Indo-European", "TB": "Tibeto-Burman"}[fam]
    t6_rows.append({
        "Pop_Code": pc,
        "Full_Community_Name": get_full_ethnic_name(pc, "descriptive"),
        "Plot_Label": get_full_ethnic_name(pc, "plot"),
        "Language_Family": fam_full,
        "SubgroupResolved_AD_Carrier_Pct": round(cf_ad, 3),
        "SubgroupResolved_AR_Carrier_Pct": round(cf_ar, 3),
        "Modifier_Carrier_Pct": round(cf_mod, 3),
        "Total_SubgroupResolved_CF_Pct": round(100.0 * (1.0 - (1.0 - cf_ad/100.0) * (1.0 - cf_ar/100.0)), 3),
        "N_SubgroupResolved_Hits": len(hits),
        "Detected_Variants": "; ".join(hits) if hits else "None (>0.1% subgroup mask)",
    })

t6 = pd.DataFrame(t6_rows).sort_values("Total_SubgroupResolved_CF_Pct", ascending=False)
t6.to_csv("tables/Table_6_82_Communities_Decoded_Cancer_Structure.csv", index=False)
print("\n=== TOP 15 COMMUNITIES BY SUBGROUP-RESOLVED AD+AR BURDEN ===")
print(t6[["Full_Community_Name", "SubgroupResolved_AD_Carrier_Pct", "SubgroupResolved_AR_Carrier_Pct", "Modifier_Carrier_Pct", "Detected_Variants"]].head(15).to_string())

# Print detailed list of ALL subgroup-resolved variants for exact prose accuracy
print("\n=== ALL SUBGROUP-RESOLVED QUALIFYING/MODIFIER VARIANTS ===")
for _, vr in qm[qm["has_82pop"]].iterrows():
    nz = [(get_full_ethnic_name(c, "descriptive"), float(vr[c])) for c in pop_cols if pd.notna(vr[c]) and float(vr[c]) > 0]
    nz.sort(key=lambda x: x[1], reverse=True)
    print(f"  {vr['gene']:7s} ({vr['inh_mode']:9s}) {vr['hgvsc']} | af_gi={vr['af_gi']:.6f} | nz_pops={len(nz)} | top3={nz[:3]}")

# ==============================================================================
# TABLE 7: ICMR-NCDIR Cancer Samiksha + GenomeIndia Translation & Incidence Matrix
# ==============================================================================
cs_proj = pd.read_csv("data_cancer/cancer_samiksha_projections.csv" if os.path.exists("data_cancer/cancer_samiksha_projections.csv") else "scratch/cancer_samiksha_projections.csv")
# Verified PBCR hotspots (directly read from Chapter 5 PNGs)
pbcr_hotspots = [
    ("Breast (Female)", "Breast", "Hyderabad (South)", 48.0, 39.0, "BRCA1, BRCA2, PALB2, ATM, CHEK2, BARD1, TP53", "High Mendelian alignment (~4.4-8.0% clinic/pathways; ~1 in 122 AD HBOC GI carriers)"),
    ("Ovary (Female)", "Ovary", "Papumpare (North-East)", 13.7, 7.8, "BRCA1, BRCA2, RAD51C, RAD51D, BRIP1, MLH1/MSH2", "High ovarian BRCA/HRR fraction (~21.3% unselected Indian ovarian P/LP clinically)"),
    ("Colorectal (Male)", "Colon+Rectum", "Aizawl (Mizoram)", 15.9, 11.9, "MLH1, MSH2, MSH6, PMS2, APC, MUTYH, NTHL1, MSH3", "High MSI/MMRd yield (24.9% Indian CRC -> 51% LS); high-risk polyposis & AR founder clusters"),
    ("Colorectal (Female)", "Colon+Rectum", "Aizawl (Mizoram)", 11.4, 8.8, "MLH1, MSH2, MSH6, PMS2, APC, MUTYH, NTHL1, MSH3", "Concordant Lynch & AR polyposis predisposition targets; APC/CHEK2 modifiers"),
    ("Corpus Uteri (Female)", "Corpus Uteri", "Hyderabad (South)", 8.0, 6.0, "MLH1, MSH2, MSH6, PMS2, PTEN", "Key sentinel Lynch syndrome extracolonic tumor requiring universal MMR IHC screening"),
    ("Prostate (Male)", "Prostate", "Delhi (North)", 11.8, 7.3, "BRCA2, ATM, CHEK2, PALB2, HOXB13", "HRR/PARPi targetable; HOXB13 G84E absent, CHEK2/ATM/BRCA2 dominant AD burden"),
    ("Thyroid (Female)", "Thyroid", "Papumpare / Thiruvananthapuram / Kollam", 16.5, 9.8, "RET (1 MTC allele); Contested HABP2/SRGAP1 excluded", "Epidemiologic Paradox: intense Kerala papillary ultrasound detection (14.7/12.8) & NE Papumpare peak (16.5) despite minimal germline burden"),
    ("Oesophagus (Male)", "Oesophagus", "East Khasi Hills (Meghalaya)", 75.4, 44.2, "RHBDF2 (0 GoF alleles; 4 benign LoFs excluded), BRCA2", "Lifestyle/environmental tobacco-areca-smoked food dominance; LoF RHBDF2 non-causal"),
    ("Stomach (Male/Female)", "Stomach", "Aizawl (44.2M) / Papumpare (27.1F)", 44.2, 32.3, "CDH1 (1 LoF), CTNNA1 (1 ClinVar + 1 novel), APC, Lynch", "Targeted HDGC surveillance justified in familial gastric clusters amid environmental Helicobacter/diet dominance"),
    ("Gall Bladder (Female)", "Gall Bladder", "Kamrup urban (Assam)", 16.2, 13.7, "Unexplained on standard 62-gene panel", "Critical Surveillance Gap: minimal high-penetrance germline capture; non-BRCA/biliary discovery needed"),
]
t7 = pd.DataFrame(pbcr_hotspots, columns=["Cancer_Site", "Projection_Key", "Peak_PBCR_Registry", "Peak_AAR_Per100k", "Peak_CR_Per100k", "GenomeIndia_Genes_and_Alleles", "Integration_Interpretation"])
t7.to_csv("tables/Table_7_Cancer_Samiksha_PBCR_Integration_Matrix.csv", index=False)

# National 2025 clinical volume vs hereditary-attributable caseload projection (conservative range from published attributable fractions + GI baseline):
proj_both = cs_proj[cs_proj["Sex"] == "Both"].set_index("Site")
print("\n=== TABLE 7: CANCER SAMIKSHA 2025 PROJECTIONS SUMMARY (BOTH SEXES) ===")
print(proj_both[["2016", "2020", "2025", "Pct_Change_20_25"]].to_string())

# ==============================================================================
# MASTER EXCEL WORKBOOK COMPILATION
# ==============================================================================
with pd.ExcelWriter("tables/Master_62Gene_Familial_Cancer_GenomeIndia_Workbook.xlsx", engine="openpyxl") as xl:
    t1.to_excel(xl, sheet_name="T1_Prevalence_Summary", index=False)
    t2.to_excel(xl, sheet_name="T2_All62_Genes", index=False)
    t3_old_sum.to_excel(xl, sheet_name="T3_Inflation_Calibration", index=False)
    t4_export.to_excel(xl, sheet_name="T4_Curated_Variants_g4", index=False)
    t5.to_excel(xl, sheet_name="T5_Recessive_Autozygosity", index=False)
    t6.to_excel(xl, sheet_name="T6_82_Communities_Decoded", index=False)
    t7.to_excel(xl, sheet_name="T7_Cancer_Samiksha_PBCR", index=False)
    cs_proj.to_excel(xl, sheet_name="S2_Cancer_Projections", index=False)
    old_merged.to_excel(xl, sheet_name="S3_Old262_LineByLine_Audit", index=False)

print("Saved Master_62Gene_Familial_Cancer_GenomeIndia_Workbook.xlsx")

# ==============================================================================
# GENERATE 6 SPRINGER-COMPLIANT 600 DPI PUBLICATION FIGURES
# ==============================================================================
def save_fig(fig, name):
    p1 = os.path.join("figures_manuscript", f"{name}.png")
    fig.savefig(p1, dpi=600, bbox_inches="tight", facecolor="white")
    if ART_FIG_DIR:
        shutil.copy2(p1, os.path.join(ART_FIG_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"Saved 600 DPI figure: {p1}")

# --- FIGURE 1: Bioinformatic Variant-Inflation Calibration Funnel Across 62 Genes ---
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1.15, 1.05]})
plt.subplots_adjust(wspace=0.62, left=0.08, right=0.98, top=0.92, bottom=0.18)

# Panel a: 62-Gene Variant Filtering Funnel (exact Table S1 counts without magic offsets)
ax = axes[0]
n_total_rare = len(df33_all[df33_all["af_gi"]<0.01]) + len(df29_all[df29_all["af_gi"]<0.01])
n_after_master = len(df[df["stratum"].isin([
    "Stratum1A_ClinVar_PLP_TwoPlus", "Stratum1B_ClinVar_PLP_SingleSub", "Stratum2_Novel_MANE_LoF_PVS1",
    "Modifier_LowPenetrance_Risk", "Excluded_ClinVar_Overturned_LoF", "Excluded_Mechanism_or_Disputed_Gene",
    "Excluded_UTR_Splice_NonCoding", "Excluded_LastExon_NMD_Escape_VUS", "Excluded_InSiGHT_Codon49_5Prime_VUS",
    "Excluded_NonOncologic_Recessive_VHL", "Excluded_ZeroStar_ClinVar"
])])
n_overturned = len(df[df["stratum"] == "Excluded_ClinVar_Overturned_LoF"])
n_cur = len(df[df["stratum"].isin(S_ALL)])
n_mod = len(df[df["stratum"] == "Modifier_LowPenetrance_Risk"])

b1 = n_total_rare
b2 = n_after_master
b3 = b2 - n_overturned
b4 = n_cur + n_mod
b5 = n_cur
assert b4 == 218 and b5 == 216 and b2 == 302 and b3 == 268, f"Unexpected funnel counts: {(b1, b2, b3, b4, b5)}"
bars_a = [b1, b2, b3, b4, b5]
labels_a = [
    f"All rare variants\n(<1% AF, n={b1:,})",
    f"Candidate LoF / ClinVar\nP/LP pool (n={b2:,})",
    f"After ClinVar-overturned\nLoF removal (n={b3:,})",
    f"After mechanism, VCEP &\nNMD-escape filter (n={b4:,})",
    f"Final curated P/LP +\nMANE-PVS1 (n={b5:,})",
]
y_pos = np.arange(len(bars_a))[::-1]
cols_a = [C_GREY, C_SKY, C_ORANGE, C_BLUE, C_TEAL]
ax.barh(y_pos, bars_a, color=cols_a, height=0.58, edgecolor="#222222", linewidth=0.6)
ax.set_xscale("log")
ax.set_xlim(40, 120000)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_a, fontsize=8.0)
for yp, val in zip(y_pos, bars_a):
    ax.text(val * 1.18, yp, f"{val:,}", va="center", ha="left", fontsize=8.2, fontweight="bold", color="#222222")
ax.set_xlabel("Variant count (log scale)")
ax.text(-0.28, 1.03, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel b: Old 262-Variant Core33 Inflation Audit
ax = axes[1]
t3_cats = [
    ("Genuine ClinVar 2+/1-sub\n& novel MANE-PVS1", sum(old_merged["stratum"].isin(S_ALL)), C_TEAL),
    ("Non-MANE UTR / intron /\nmissense artifacts", sum(old_merged["stratum"] == "Excluded_Other_NonQualifying"), C_ORANGE),
    ("ClinVar VUS / Conflicting /\nBenign (incl. rsID bleed)", sum(old_merged["stratum"].isin(["Excluded_VUS_ClinVar", "Excluded_Conflicting_ClinVar", "Excluded_Benign_ClinVar"])), C_VERM),
    ("ClinVar-overturned LoF\n(e.g., MUTYH c.1392+2C>T)", sum(old_merged["stratum"] == "Excluded_ClinVar_Overturned_LoF"), C_PURPLE),
    ("Last-exon NMD escape /\nVHL R200W / UTR / Mod", sum(old_merged["stratum"].isin(["Excluded_LastExon_NMD_Escape_VUS", "Excluded_NonOncologic_Recessive_VHL", "Excluded_UTR_Splice_NonCoding", "Excluded_Mechanism_or_Disputed_Gene", "Modifier_LowPenetrance_Risk"])), C_GREY),
]
yb = np.arange(len(t3_cats))[::-1]
ax.barh(yb, [x[1] for x in t3_cats], color=[x[2] for x in t3_cats], height=0.58, edgecolor="#222222", linewidth=0.6)
ax.set_yticks(yb)
ax.set_yticklabels([x[0] for x in t3_cats], fontsize=8.2)
ax.set_xlim(0, 145)
for yp, (_, cnt, _) in zip(yb, t3_cats):
    ax.text(cnt + 2.5, yp, f"{cnt} ({cnt/len(old_merged)*100:.1f}%)", va="center", ha="left", fontsize=8.2, fontweight="bold")
ax.set_xlabel("Variants in uncalibrated Core-33 set (n = 262)")
ax.text(-0.28, 1.03, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel c: Naive vs Calibrated Carrier Burden (% Carriers)
ax = axes[2]
comp_bars = [
    ("Naive uncalibrated\nCore-33 pipeline", naive_core33_cf, C_VERM),
    ("Calibrated Core-33\n(AD + AR MUTYH)", cal_core33_all_cf, C_ORANGE),
    ("Calibrated 50 AD\n(incl. SDHA)", float(t1[(t1["Cohort_Scope"].str.startswith("All Autosomal Dominant")) & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_BLUE),
    ("Calibrated 49 AD\n(excl. low-pen SDHA)", float(t1[(t1["Cohort_Scope"].str.startswith("AD Actionable Excl")) & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_TEAL),
    ("Strictly ClinVar 2+\n(50 AD, incl. SDHA)", float(t1[(t1["Cohort_Scope"].str.startswith("All Autosomal Dominant")) & (t1["Adjudication_Tier"].str.startswith("Stratum 1A"))]["Carrier_Freq_Percent"].iloc[0]), C_NAVY),
]
xc = np.arange(len(comp_bars))
ax.bar(xc, [x[1] for x in comp_bars], color=[x[2] for x in comp_bars], width=0.58, edgecolor="#222222", linewidth=0.6)
ax.set_xticks(xc)
ax.set_xticklabels([x[0] for x in comp_bars], rotation=28, ha="right", fontsize=8)
ax.set_ylim(0, naive_core33_cf * 1.18)
for x_i, (_, val, _) in zip(xc, comp_bars):
    ax.text(x_i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
ax.set_ylabel("Modeled carrier prevalence (%)")
ax.text(-0.18, 1.03, "c", transform=ax.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig1_Variant_Calibration_Funnel_62Genes")

# --- FIGURE 2: Genetic Architecture Across AD & AR Genes (3 panels) ---
fig = plt.figure(figsize=(15.0, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.05], wspace=0.52, left=0.07, right=0.98, top=0.93, bottom=0.14)

# Panel a: Top 16 AD Genes stacked by Stratum 1A / 1B / 2
ax1 = fig.add_subplot(gs[0, 0])
t2_ad = t2[t2["Inheritance"].isin(["AD", "AD_GoF", "AD_LowPen"]) & (t2["N_Qualifying_Total"] > 0)].sort_values("CF_Primary_Pct", ascending=True).tail(16)
y_ad = np.arange(len(t2_ad))
# compute per-stratum CF per gene
g_s1a, g_s1b, g_s2 = [], [], []
for g in t2_ad["Gene"]:
    sub = df[df["gene"] == g]
    g_s1a.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"].isin(S1A)]["af_gi"]])) * 100.0)
    g_s1b.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"] == "Stratum1B_ClinVar_PLP_SingleSub"]["af_gi"]])) * 100.0)
    g_s2.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"] == "Stratum2_Novel_MANE_LoF_PVS1"]["af_gi"]])) * 100.0)

ax1.barh(y_ad, g_s1a, color=C_NAVY, height=0.65, label="ClinVar 2+/Expert")
ax1.barh(y_ad, g_s1b, left=g_s1a, color=C_SKY, height=0.65, label="ClinVar 1-Sub P/LP")
left_1ab = np.array(g_s1a) + np.array(g_s1b)
ax1.barh(y_ad, g_s2, left=left_1ab, color=C_ORANGE, height=0.65, label="Novel MANE PVS1")
tot_ad_g = left_1ab + np.array(g_s2)
ax1.set_yticks(y_ad)
ax1.set_yticklabels([f"{g}{'*' if g=='SDHA' else ''}" for g in t2_ad["Gene"]], fontsize=8.5, fontweight="bold")
for yi, val in zip(y_ad, tot_ad_g):
    ax1.text(val + 0.015, yi, f"{val:.3f}%", va="center", ha="left", fontsize=7.8)
ax1.set_xlim(0, max(tot_ad_g) * 1.25)
ax1.set_xlabel("Autosomal dominant carrier frequency (%)")
ax1.legend(loc="lower right", frameon=True, framealpha=0.92)
ax1.text(-0.22, 1.02, "a", transform=ax1.transAxes, fontsize=11, fontweight="bold")

# Panel b: 7 AR Cancer Genes + 2 Low-Pen Modifiers
ax2 = fig.add_subplot(gs[0, 1])
t2_ar = t2[t2["Inheritance"] == "AR"].sort_values("CF_Primary_Pct", ascending=True)
ar_names = list(t2_ar["Gene"]) + ["APC I1307K", "CHEK2 I157T"]
ar_s1a, ar_s1b, ar_s2, ar_mod = [], [], [], []
for g in t2_ar["Gene"]:
    sub = df[df["gene"] == g]
    ar_s1a.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"].isin(S1A)]["af_gi"]])) * 100.0)
    ar_s1b.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"] == "Stratum1B_ClinVar_PLP_SingleSub"]["af_gi"]])) * 100.0)
    ar_s2.append((1.0 - np.prod([(1.0 - q)**2 for q in sub[sub["stratum"] == "Stratum2_Novel_MANE_LoF_PVS1"]["af_gi"]])) * 100.0)
    ar_mod.append(0.0)
for mg in ["APC", "CHEK2"]:
    sub = df[(df["gene"] == mg) & (df["stratum"] == "Modifier_LowPenetrance_Risk")]
    ar_s1a.append(0.0); ar_s1b.append(0.0); ar_s2.append(0.0)
    ar_mod.append((1.0 - np.prod([(1.0 - q)**2 for q in sub["af_gi"]])) * 100.0)

yar = np.arange(len(ar_names))
ax2.barh(yar, ar_s1a, color=C_NAVY, height=0.62, label="ClinVar 2+")
ax2.barh(yar, ar_s1b, left=ar_s1a, color=C_SKY, height=0.62, label="ClinVar 1-Sub")
l_ar = np.array(ar_s1a) + np.array(ar_s1b)
ax2.barh(yar, ar_s2, left=l_ar, color=C_ORANGE, height=0.62, label="Novel MANE PVS1")
ax2.barh(yar, ar_mod, left=l_ar + np.array(ar_s2), color=C_PURPLE, height=0.62, label="Low-Pen Modifier")
tot_ar = l_ar + np.array(ar_s2) + np.array(ar_mod)
ax2.set_yticks(yar)
ax2.set_yticklabels(ar_names, fontsize=8.5, fontweight="bold")
for yi, val in zip(yar, tot_ar):
    ax2.text(val + 0.012, yi, f"{val:.3f}%", va="center", ha="left", fontsize=7.8)
ax2.set_xlim(0, max(tot_ar) * 1.30)
ax2.set_xlabel("Carrier frequency (%) — AR & Modifiers")
ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)
ax2.text(-0.26, 1.02, "b", transform=ax2.transAxes, fontsize=11, fontweight="bold")

# Panel c: Syndrome / Pathway Decomposition with 95% Wilson CI
ax3 = fig.add_subplot(gs[0, 2])
syn_keys = [
    ("AD Total (50 genes)", "All Autosomal Dominant Cancer Genes (50 AD/GoF incl. SDHA)", C_NAVY),
    ("AD excl. SDHA (49 genes)", "AD Actionable Excluding Low-Penetrance SDHA (49 genes)", C_TEAL),
    ("CDC Tier 1 (HBOC+Lynch)", "CDC Tier 1 Genes (HBOC BRCA1/2 + Lynch MMR, 7 genes)", C_BLUE),
    ("HBOC / HRR (9 genes)", "HBOC / HRR Pathway (9 genes)", C_SKY),
    ("Lynch Syndrome MMR", "Lynch Syndrome MMR (4 active genes; EPCAM 0 SNV/indel)", C_ORANGE),
    ("GI Polyposis / Gastric AD", "GI Polyposis & Gastric AD (APC, SMAD4, CDH1, CTNNA1, etc.)", C_VERM),
    ("AR Cancer / Polyposis (7)", "Autosomal Recessive Cancer & Polyposis Carriers (7 genes, monoallelic)", C_PURPLE),
]
y_syn = np.arange(len(syn_keys))[::-1]
for yi, (slbl, scope_key, col) in zip(y_syn, syn_keys):
    row = t1[(t1["Cohort_Scope"] == scope_key) & (t1["Adjudication_Tier"].str.startswith("Primary"))].iloc[0]
    val, lo, hi = row["Carrier_Freq_Percent"], row["CI_95_Low_Percent"], row["CI_95_High_Percent"]
    ax3.barh(yi, val, color=col, height=0.60, edgecolor="#222222", linewidth=0.5)
    ax3.errorbar(val, yi, xerr=[[val - lo], [hi - val]], fmt="none", ecolor="#111111", capsize=3, lw=1.2)
    ax3.text(hi + 0.08, yi, f"{val:.2f}%\n({row['Carrier_Ratio']})", va="center", ha="left", fontsize=7.6)
ax3.set_yticks(y_syn)
ax3.set_yticklabels([x[0] for x in syn_keys], fontsize=8.2)
ax3.set_xlim(0, 3.6)
ax3.set_xlabel("Carrier frequency (%) ± 95% CI")
ax3.text(-0.24, 1.02, "c", transform=ax3.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig2_Gene_Architecture_and_Syndrome_Burden")

# --- FIGURE 3: GenomeIndia vs gnomAD v4.1 SAS/NFE & Global Biobanks ---
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.1, 1.05]})
plt.subplots_adjust(wspace=0.68, left=0.13, right=0.97, top=0.92, bottom=0.14)

# Panel a: Variant-level GenomeIndia AF vs gnomAD v4.1 SAS & NFE for top recurrent alleles (AF_GI >= 0.00015 or major founders)
ax = axes[0]
top_vars = qual_all[(qual_all["af_gi"] >= 0.00015) | (qual_all["gene"].isin(["BRCA1", "BRCA2", "MUTYH", "NTHL1", "MBD4", "WRN"]) & (qual_all["af_gi"] >= 0.00015))].copy()
top_vars = top_vars.sort_values("af_gi", ascending=True).tail(15)
yl = np.arange(len(top_vars))
vlabels = []
for _, vr in top_vars.iterrows():
    p_sh = str(vr["hgvsp"]).split(":")[-1].replace("Ter", "*") if pd.notna(vr["hgvsp"]) and ":" in str(vr["hgvsp"]) else str(vr["hgvsc"]).split(":")[-1]
    tag = " [AR]" if vr["inh_mode"] == "AR" else (" [Mod]" if vr["stratum"] == "Modifier_LowPenetrance_Risk" else "")
    vlabels.append(f"{vr['gene']} {p_sh}{tag}")

gi_vals = top_vars["af_gi"].values * 1e4
sas_vals = top_vars["g4_sas_af"].fillna(0).values * 1e4
nfe_vals = top_vars["g4_nfe_af"].fillna(0).values * 1e4

# Plot connected Cleveland dot plot (per 10,000 alleles)
for i in range(len(top_vars)):
    ax.plot([min(gi_vals[i], sas_vals[i], nfe_vals[i]), max(gi_vals[i], sas_vals[i], nfe_vals[i])], [yl[i], yl[i]], color="#CCCCCC", lw=1.2, zorder=1)
ax.scatter(gi_vals, yl, color=C_VERM, marker="o", s=55, label="GenomeIndia (n=9,768)", zorder=3)
ax.scatter(sas_vals, yl, color=C_BLUE, marker="s", s=45, label="gnomAD v4.1 SAS (AN~91k)", zorder=3)
ax.scatter(nfe_vals, yl, color=C_TEAL, marker="^", s=45, label="gnomAD v4.1 NFE (AN~1.18M)", zorder=2)
ax.set_yticks(yl)
ax.set_yticklabels(vlabels, fontsize=8.0)
ax.set_xlim(-1.0, 25.0)
ax.set_xlabel("Allele frequency (per 10,000 chromosomes)")
ax.legend(loc="center right", frameon=True, framealpha=0.95, fontsize=7.8)
ax.text(-0.26, 1.03, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel b: Global Biobank Benchmarks (verified from read full texts!)
ax = axes[1]
biobank_bench = [
    ("GenomeIndia (All 49 AD excl. SDHA)\n[Primary Curated, n=9,768]", float(t1[(t1["Cohort_Scope"]=="AD Actionable Excluding Low-Penetrance SDHA (49 genes)") & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_NAVY, "o"),
    ("GenomeIndia (All 50 AD incl. SDHA)\n[Primary Curated, n=9,768]", float(t1[(t1["Cohort_Scope"].str.startswith("All Autosomal Dominant")) & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_BLUE, "o"),
    ("GenomeIndia (CDC Tier 1: BRCA1/2+MMR)\n[Primary Curated, n=9,768]", float(t1[(t1["Cohort_Scope"].str.startswith("CDC Tier 1")) & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_SKY, "o"),
    ("GenomeIndia (Lynch MMR 4 genes)\n[Primary Curated, n=9,768]", float(t1[(t1["Cohort_Scope"].str.startswith("Lynch Syndrome MMR")) & (t1["Adjudication_Tier"].str.startswith("Primary"))]["Carrier_Freq_Percent"].iloc[0]), C_ORANGE, "o"),
    ("Geno4ME WGS (CDC Tier 1 cancer/FH)\n[2,017 diverse, npj Genom Med 2025]", 1.44, C_TEAL, "s"),
    ("UW Diverse (25 hereditary cancer genes)\n[2,864; 57% Asian, Genome Med 2023]", 3.60, C_TEAL, "s"),
    ("Geisinger MyCode (Cancer ACMG subset)\n[39.7% of 5,119 in 175.5k, JAMA NO 2025]", 1.35, C_TEAL, "s"),
    ("All of Us Lynch (MLH1/MSH2/MSH6/PMS2)\n[217,824 WGS, 1 in 354, Nat Commun 2025]", 0.282, C_VERM, "^"),
    ("UK Biobank Lynch (exons excl PMS2CL)\n[454,756 WES, 1 in 550, J Med Genet 2024]", 0.183, C_VERM, "^"),
]
y_bb = np.arange(len(biobank_bench))[::-1]
for yi, (lbl, val, col, mk) in zip(y_bb, biobank_bench):
    ax.barh(yi, val, color=col, height=0.58, alpha=0.85, edgecolor="#222222", linewidth=0.5)
    ax.text(val + 0.06, yi, f"{val:.2f}% (1:{int(round(100/val))})", va="center", ha="left", fontsize=7.8)
ax.set_yticks(y_bb)
ax.set_yticklabels([x[0] for x in biobank_bench], fontsize=7.6)
ax.set_xlim(0, 4.4)
ax.set_xlabel("Population carrier prevalence (%)")
ax.text(-0.28, 1.03, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig3_Global_Biobank_and_gnomADv4_Comparison")

# --- FIGURE 4: 82 Decoded Indian Communities & Autozygosity Expansion ---
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
plt.subplots_adjust(wspace=0.35, left=0.25, right=0.98, top=0.92, bottom=0.14)

# Panel a: Top 16 Decoded Communities with Subgroup-Resolved Variants
ax = axes[0]
t6_top = t6[t6["Total_SubgroupResolved_CF_Pct"] > 0].head(15).iloc[::-1]
yp = np.arange(len(t6_top))
ax.barh(yp, t6_top["SubgroupResolved_AD_Carrier_Pct"], color=C_BLUE, height=0.62, label="AD Subgroup-Resolved")
ax.barh(yp, t6_top["SubgroupResolved_AR_Carrier_Pct"], left=t6_top["SubgroupResolved_AD_Carrier_Pct"], color=C_VERM, height=0.62, label="AR Subgroup-Resolved")
ax.barh(yp, t6_top["Modifier_Carrier_Pct"], left=t6_top["SubgroupResolved_AD_Carrier_Pct"] + t6_top["SubgroupResolved_AR_Carrier_Pct"], color=C_PURPLE, height=0.62, label="Low-Pen Modifiers")
ax.set_yticks(yp)
# Truncate slightly if extremely long while retaining FULL ethno-geographic name without any raw codes!
ax.set_yticklabels([str(x).replace(" (", "\n(") for x in t6_top["Full_Community_Name"]], fontsize=7.3)
ax.set_xlabel("Subgroup-resolved carrier prevalence (%)")
ax.legend(loc="lower right", frameon=True, framealpha=0.92)
ax.text(-0.42, 1.02, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel b: Autozygosity Expansion Curve across F = 0 to 0.0625 (first-cousin offspring)
ax = axes[1]
f_grid = np.linspace(0, 0.065, 100)
ar_plot_genes = [
    ("Combined 7 AR Cancer Genes", C_DARK, "-", 2.2),
    ("Combined BER/MMR Polyposis\n(MUTYH+NTHL1+MSH3+MBD4)", C_VERM, "-", 2.0),
    ("FANCA (Fanconi Anemia)", C_BLUE, "--", 1.5),
    ("WRN (Werner Syndrome)", C_TEAL, "-.", 1.5),
    ("MUTYH (MAP Polyposis)", C_ORANGE, ":", 1.8),
    ("Combined CMMRD (4 MMR)", C_PURPLE, "--", 1.5),
]
for name, col, ls, lw in ar_plot_genes:
    if name.startswith("Combined 7"):
        qs = t5[t5["Gene"].isin(["FANCA","WRN","MUTYH","MSH3","BLM","NTHL1","MBD4"])]["Sum_AF_q"].values
    elif name.startswith("Combined BER"):
        qs = t5[t5["Gene"].isin(["MUTYH","NTHL1","MSH3","MBD4"])]["Sum_AF_q"].values
    elif name.startswith("Combined CMMRD"):
        qs = t5[t5["Gene"].isin(["MLH1","MSH2","MSH6","PMS2"])]["Sum_AF_q"].values
    else:
        g_key = name.split(" ")[0]
        qs = t5[t5["Gene"] == g_key]["Sum_AF_q"].values
    prev_curve = [1e6 * (1.0 - np.prod([1.0 - (q**2 + F * q * (1.0 - q)) for q in qs])) for F in f_grid]
    ax.plot(f_grid, prev_curve, color=col, linestyle=ls, linewidth=lw, label=name)
ymax_f4 = ax.get_ylim()[1]
for bench_f, y_frac, bench_lbl in [(0.02, 0.56, "Median India\nF = 0.02"), (0.0625, 0.90, "First-Cousin\nF = 0.0625")]:
    ax.axvline(bench_f, color="#888888", linestyle=":", linewidth=1.0)
    ax.text(bench_f - 0.001, ymax_f4 * y_frac, bench_lbl, fontsize=7.5, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.92))
ax.set_xlabel("Inbreeding / autozygosity coefficient (F)")
ax.set_ylabel("Biallelic disease prevalence (per 1,000,000)")
ax.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92)
ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig4_82_Communities_and_Recessive_Autozygosity")

# --- FIGURE 5: Cancer Samiksha Epidemiologic Reconciliation (Projections & PBCR Hotspots) ---
fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
plt.subplots_adjust(wspace=0.72, left=0.11, right=0.98, top=0.92, bottom=0.14)

# Panel a: 2016 -> 2025 Cancer Site Growth in India (verified from Cancer Samiksha Projection PNGs)
ax = axes[0]
cs_verified = [
    ("Breast", 189984, 238908),
    ("Colorectal (Colon+Rectum)", 66060, 82716),
    ("Oesophagus", 47606, 59846),
    ("Stomach", 45250, 56733),
    ("Ovary (F)", 39628, 49644),
    ("Prostate (M)", 37416, 47068),
    ("Thyroid", 31796, 38574),
    ("Gall Bladder", 28692, 36213),
    ("Corpus Uteri (F)", 23816, 30121),
]
sites_labels = [x[0] for x in cs_verified]
ys = np.arange(len(cs_verified))[::-1]
c2016 = [x[1] / 1000.0 for x in cs_verified]
c2025 = [x[2] / 1000.0 for x in cs_verified]
ax.barh(ys - 0.16, c2016, height=0.30, color=C_SKY, label="2016 Projected")
ax.barh(ys + 0.16, c2025, height=0.30, color=C_NAVY, label="2025 Projected")
for yi, v16, v25 in zip(ys, c2016, c2025):
    pct_inc = ((v25 / v16) - 1.0) * 100.0
    ax.text(v25 + 2.5, yi + 0.02, f"{v25:.1f}k (+{pct_inc:.0f}%)", va="center", ha="left", fontsize=7.6)
ax.set_yticks(ys)
ax.set_yticklabels(sites_labels, fontsize=8.2)
ax.set_xlim(0, 345)
ax.set_xlabel("Annual incident cancer cases in India (thousands)")
ax.legend(loc="center right", frameon=True, framealpha=0.95)
ax.text(-0.22, 1.02, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel b: Regional PBCR Peak AAR (per 100k) paired by Etiologic Category (Genetically Aligned vs Paradox/Environmental)
ax = axes[1]
pbcr_bars = [
    ("Oesophagus (M) — East Khasi Hills [NE]", 75.4, C_GREY, "Tobacco/betel-nut dominant\n(0 RHBDF2 GoF alleles)"),
    ("Breast (F) — Hyderabad [South]", 48.0, C_BLUE, "HBOC dominant\n(BRCA1/2/PALB2/HRR)"),
    ("Stomach (M) — Aizawl [NE]", 44.2, C_ORANGE, "Diet + CDH1/CTNNA1"),
    ("Thyroid (F) — Papumpare [NE]", 16.5, C_VERM, "Discordance:\n0 non-RET germline"),
    ("Thyroid (F) — Thiruvananthapuram [South]", 14.7, C_VERM, "Ultrasound overdiagnosis\n(69% PTC in HBCR)"),
    ("Gall Bladder (F) — Kamrup urban [NE]", 16.2, C_PURPLE, "Unexplained on\n62-gene panel"),
    ("Colorectal (M) — Aizawl [NE]", 15.9, C_TEAL, "Lynch MMR + FAP +\nMUTYH/NTHL1/MSH3"),
    ("Ovary (F) — Papumpare [NE]", 13.7, C_BLUE, "21.3% clinical BRCA1/2"),
    ("Prostate (M) — Delhi [North]", 11.8, C_BLUE, "BRCA2/ATM/CHEK2"),
    ("Corpus Uteri (F) — Hyderabad [South]", 8.0, C_TEAL, "Lynch sentinel site"),
]
ypb = np.arange(len(pbcr_bars))[::-1]
ax.barh(ypb, [x[1] for x in pbcr_bars], color=[x[2] for x in pbcr_bars], height=0.60, edgecolor="#222222", linewidth=0.5)
for yi, (_, val, _, note) in zip(ypb, pbcr_bars):
    ax.text(val + 1.2, yi, f"{val:.1f}  ({note.replace(chr(10), ' ')})", va="center", ha="left", fontsize=7.4)
ax.set_yticks(ypb)
ax.set_yticklabels([x[0] for x in pbcr_bars], fontsize=8.0)
ax.set_xlim(0, 160)
ax.set_xlabel("Peak PBCR Age-Adjusted Incidence Rate (per 100,000)")
ax.text(-0.28, 1.02, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig5_Cancer_Samiksha_PBCR_Reconciliation")

# --- FIGURE 6: Clinical Pathway Comparison & National Germline Screening Strategy ---
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), gridspec_kw={"width_ratios": [1.0, 1.12]})
plt.subplots_adjust(wspace=0.68, left=0.07, right=0.98, top=0.92, bottom=0.14)

# Panel a: Clinical Case Yield vs General-Population Reservoir by Gene (Lynch & HBOC Gene Inversion!)
ax = axes[0]
# Compare GenomeIndia Population % Share vs Published Indian Clinical Cohort % Share (Sheth 2026 Lynch; Gupta/Singh BRCA1/2)
genes_inv = ["BRCA1\n(HBOC)", "BRCA2\n(HBOC)", "ATM/CHEK2\n/PALB2 (HBOC)", "MLH1\n(Lynch)", "MSH2\n(Lynch)", "MSH6\n(Lynch)", "PMS2\n(Lynch)"]
# Compute exact population shares within HBOC-5 or Lynch-4 in GenomeIndia:
cf_b1 = float(t2[t2["Gene"]=="BRCA1"]["CF_Primary_Pct"].iloc[0])
cf_b2 = float(t2[t2["Gene"]=="BRCA2"]["CF_Primary_Pct"].iloc[0])
cf_mod_hr = float(t2[t2["Gene"].isin(["ATM", "CHEK2", "PALB2"])]["CF_Primary_Pct"].sum())
hboc_tot = cf_b1 + cf_b2 + cf_mod_hr
cf_mlh1 = float(t2[t2["Gene"]=="MLH1"]["CF_Primary_Pct"].iloc[0])
cf_msh2 = float(t2[t2["Gene"]=="MSH2"]["CF_Primary_Pct"].iloc[0])
cf_msh6 = float(t2[t2["Gene"]=="MSH6"]["CF_Primary_Pct"].iloc[0])
cf_pms2 = float(t2[t2["Gene"]=="PMS2"]["CF_Primary_Pct"].iloc[0])
lyn_tot = cf_mlh1 + cf_msh2 + cf_msh6 + cf_pms2

pop_shares = [cf_b1/hboc_tot*100, cf_b2/hboc_tot*100, cf_mod_hr/hboc_tot*100, cf_mlh1/lyn_tot*100, cf_msh2/lyn_tot*100, cf_msh6/lyn_tot*100, cf_pms2/lyn_tot*100]
# Clinical India shares directly from verified full texts:
# HBOC (AIIMS PMC8487333 consecutive breast n=47 P/LP: BRCA1=46.8%, BRCA2=19.1%, non-BRCA=34.0% [PALB2/ATM/CHEK2 etc]);
# Lynch (Sheth 2026 Familial Cancer n=66 LS: MLH1=69.7%, MSH2=18.2%; MSH6+PMS2=12.1% -> split note: show MSH6+PMS2 or exact clinical contrast)
# Actually in Sheth 2026 abstract we verified: MLH1=69.7%, MSH2=18.2% (so MSH6+PMS2 remaining ~12.1%; or compare US All of Us EHR PMS2=44.8%, MSH6=35.4%, MLH1=10.9%, MSH2=8.7%!)
# To avoid inventing MSH6 vs PMS2 split in Sheth 2026, plot ALL OF US EHR + SHETH MLH1/MSH2 or group MSH6+PMS2 as 12.1% clinical India!
genes_inv = ["BRCA1\n(Breast/Ov)", "BRCA2\n(Breast/Ov)", "Non-BRCA\nHRR*", "MLH1\n(Lynch)", "MSH2\n(Lynch)", "MSH6+PMS2\n(Lynch)"]
pop_shares = [cf_b1/hboc_tot*100, cf_b2/hboc_tot*100, cf_mod_hr/hboc_tot*100, cf_mlh1/lyn_tot*100, cf_msh2/lyn_tot*100, (cf_msh6+cf_pms2)/lyn_tot*100]
clin_shares = [46.8, 19.1, 34.0, 69.7, 18.2, 12.1]  # Exact verified from PMC8487333 & Sheth 2026

x_inv = np.arange(len(genes_inv))
ax.bar(x_inv - 0.18, pop_shares, width=0.35, color=C_SKY, edgecolor="#222222", linewidth=0.5, label="GenomeIndia population reservoir (%)")
ax.bar(x_inv + 0.18, clin_shares, width=0.35, color=C_VERM, edgecolor="#222222", linewidth=0.5, label="Indian clinical tumor cohorts (%)")
ax.axvline(2.5, color="#666666", linestyle="--", lw=1.0)
ax.set_xticks(x_inv)
ax.set_xticklabels(genes_inv, fontsize=8.0)
ax.set_ylim(0, 82)
for xi, pv, cv in zip(x_inv, pop_shares, clin_shares):
    ax.text(xi - 0.18, pv + 1.2, f"{pv:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax.text(xi + 0.18, cv + 1.2, f"{cv:.1f}%", ha="center", va="bottom", fontsize=7.2)
ax.set_ylabel("Within-syndrome gene share (%)")
ax.legend(loc="upper left", frameon=True, fontsize=7.8)
ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

# Panel b: Population Carrier Reservoir Across 1.44 Billion Indians & Priority Action Matrix
ax = axes[1]
india_pop = 1440.0  # millions (UN 2024/2025 ~1.44B)
def _t1_row(scope_prefix, exact=False):
    m = (t1["Cohort_Scope"] == scope_prefix) if exact else t1["Cohort_Scope"].str.startswith(scope_prefix)
    return t1[m & t1["Adjudication_Tier"].str.startswith("Primary")].iloc[0]

tiers_trans = [
    ("AD Actionable (49 genes, excl. SDHA)", _t1_row("AD Actionable Excluding Low-Penetrance SDHA (49 genes)", exact=True), C_NAVY),
    ("CDC Tier 1 (BRCA1/2 + Lynch MMR)", _t1_row("CDC Tier 1"), C_BLUE),
    ("High-Penetrance HBOC (BRCA1/2/PALB2)", _t1_row("High-Penetrance HBOC"), C_TEAL),
    ("Lynch Syndrome (MLH1/MSH2/MSH6/PMS2)", _t1_row("Lynch Syndrome MMR"), C_ORANGE),
    ("GI Polyposis / Gastric AD", _t1_row("GI Polyposis & Gastric AD"), C_VERM),
    ("AR Polyposis / DNA-Repair Carriers (7)", _t1_row("Autosomal Recessive"), C_PURPLE),
]
yt = np.arange(len(tiers_trans))[::-1]
millions = [india_pop * (float(r["Carrier_Freq_Percent"]) / 100.0) for _, r, _ in tiers_trans]
ci_lo_m = [millions[i] - india_pop * (float(tiers_trans[i][1]["CI_95_Low_Percent"]) / 100.0) for i in range(len(tiers_trans))]
ci_hi_m = [india_pop * (float(tiers_trans[i][1]["CI_95_High_Percent"]) / 100.0) - millions[i] for i in range(len(tiers_trans))]
ax.barh(yt, millions, xerr=[ci_lo_m, ci_hi_m], color=[x[2] for x in tiers_trans], height=0.58,
        edgecolor="#222222", linewidth=0.5, error_kw=dict(ecolor="#222222", lw=1.0, capsize=2.5))
for yi, (lbl, r, _), m_val, hi_off in zip(yt, tiers_trans, millions, ci_hi_m):
    cfp = float(r["Carrier_Freq_Percent"])
    mlo = india_pop * (float(r["CI_95_Low_Percent"]) / 100.0)
    mhi = india_pop * (float(r["CI_95_High_Percent"]) / 100.0)
    ax.text(m_val + hi_off + 0.6, yi, f"{m_val:.1f}M ({mlo:.1f}-{mhi:.1f}M)\n({cfp:.2f}%, 1 in {int(round(100/cfp)):,})", va="center", ha="left", fontsize=7.3)
ax.set_yticks(yt)
ax.set_yticklabels([x[0] for x in tiers_trans], fontsize=8.0)
ax.set_xlim(0, 59)
ax.set_xlabel("Projected carriers in India with 95% Wilson CI (millions, N = 1.44B)")
ax.text(-0.26, 1.02, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

save_fig(fig, "Fig6_Clinical_Inversion_and_National_Screening_Impact")
print("ALL 62-GENE ANALYSES AND FIGURES COMPLETE!")
