from pathlib import Path

DATA_ROOT = Path('/workspace/data')
RESULTS_ROOT = Path('/workspace/results')
ANALYSIS_ROOT = RESULTS_ROOT / 'analysis'
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

TIMING_PATH = ANALYSIS_ROOT / 'time_to_produce.tsv'

CENSUS_VERSION = '2025-11-08'
SC_UNVEIL_VERSION = '2026-07-09_18-42-14_full'

EMB_NAME = 'embeddings'
IMPUTE_NAME = 'imputation'
MEASUREMENT_NAME = 'raw_measurement'

PBMC_50_MARKERS = [
    # ── Major lineage ──
    "CD3D",    # T cells
    "CD3E",    # T cells
    "CD79A",   # B cells
    "MS4A1",   # B cells (CD20)
    "LYZ",     # Myeloid
    "NKG7",    # NK / cytotoxic

    # ── Monocytes ──
    "CD14",    # CD14 Mono
    "S100A8",  # CD14 Mono
    "FCGR3A",  # CD16 Mono (+ NK_1/2)
    "CDKN1C",  # CD16 Mono

    # ── T cell: CD4 vs CD8 ──
    "CD4",
    "CD8A",
    "CD8B",    # strong in CD8 Naive

    # ── Naive / central memory gradient ──
    "CCR7",
    "LEF1",
    "SELL",
    "IL7R",    # naive T, ILC

    # ── Effector / memory gradient ──
    "S100A4",  # TEM / effector memory
    "GZMK",    # early effector (CD8 TEM_1/2, CD4 CTL)
    "GZMB",    # terminal effector
    "GZMH",    # CD8 TEM subset separation
    "GNLY",    # terminal cytotoxic
    "PRF1",    # cytotoxic

    # ── NK subsets ──
    "KLRD1",   # pan-NK
    "KLRF1",   # NK_2/3
    "NCAM1",   # NK_CD56bright (CD56)
    "FGFBP2",  # NK_1 (CD56dim cytotoxic)
    "XCL1",    # NK_CD56bright
    "KLRB1",   # NK, MAIT, ILC, CD4 CTL

    # ── Tregs ──
    "FOXP3",   # Treg Memory / Naive
    "IL2RA",   # Treg (CD25)
    "CTLA4",   # Treg Memory vs Naive split

    # ── Proliferating ──
    "MKI67",   # CD4/CD8/NK Proliferating

    # ── Unconventional T ──
    "SLC4A10", # MAIT
    "TRDC",    # gdT (all subsets)
    "TRGC2",   # gdT
    "TRDV2",   # gdT_1/2 vs gdT_3/4 (Vδ2 vs Vδ1)

    # ── DC subsets ──
    "FCER1A",  # cDC2_1, cDC2_2
    "CLEC9A",  # cDC1
    "CLEC4C",  # pDC
    "AXL",     # ASDC_mDC / ASDC_pDC

    # ── B cell fine subtypes ──
    "IGKC",    # kappa light chain
    "IGLC2",   # lambda light chain
    "IGHD",    # naive B
    "CD27",    # memory B

    # ── Plasma / Plasmablast ──
    "JCHAIN",  # Plasma + Plasmablast
    "MZB1",    # Plasma / Plasmablast

    # ── Rare / minor populations ──
    "PPBP",    # Platelet
    "HBB",     # Eryth
    "CD34",    # HSPC
]