# Pareto-fMRI Per-Segment Analysis Pipeline
## Documentation

---

## Overview

This pipeline analyzes CFG fMRI connectivity data at the segment level, comparing explore vs exploit conditions relative to 3 archetypes on a Pareto front.

**Archetypes:**
| Archetype | Computational goal (Amir et al.) |
|-----------|----------------------------------|
| Arch1 | Energetic efficiency |
| Arch2 | Goal-directed processing and cognitive control |
| Arch3 | Internal processing and memory |

---

## Directory Structure

```
Pareto-fMRI-main/
├── scripts/
│   ├── per_phase_analysis.py           # Main analysis (distributions, covariance, directions)
│   ├── covariance_paired_comparison.py # Paired statistical tests for covariance
│   ├── direction_toward_archetypes.py  # Direction analysis toward each archetype
│   ├── arch3_variance_behavior_correlation.py # Explore Var(Arch3) vs. CFG behavioral measures
│   ├── segment_count_check.py          # How the g vs. axis-movement result depends on segment count
│   ├── phase_position_check.py         # Average position and direction of change, explore vs. exploit
│   └── make_paper_figures.py           # Redraws paper Figures 1-3 + A1-A2 from the saved results
│
├── data/
│   ├── per_phase_data/                 # Per-segment matrices (17x17, GSR)
│   │   ├── sub-XXX_task-CFG_..._explore_seg01_correlation_matrix.csv
│   │   ├── sub-XXX_task-CFG_..._exploit_seg01_correlation_matrix.csv
│   │   └── ...
│   ├── pca_transformation.npz          # PCA parameters (from HCP)
│   ├── cfgmri_measures.csv             # CFG game behavioral measures (original explore/exploit algorithm)
│   ├── cfgmri_measures_mrisegmentation.csv # Same, "(sample)" cols use the MRI-adapted algorithm instead (unused so far)
│   └── corr_mats_combined_hemispheres/
│       └── combined_hemispheres/
│           └── CFG MRI Game QC.xlsx    # QC file
│
├── results/per_phase/                  # All outputs
│   ├── all_segments_pca.csv
│   ├── distribution_comparison.png     # Plot 1
│   ├── direction_comparison.png        # Plot 2
│   ├── covariance_comparison.png       # Plot 3
│   ├── covariance_paired_comparison.png # Plot 4
│   ├── direction_toward_archetypes.png # Plot 5
│   ├── arch3_variance_behavior_correlations.csv/.png # Explore Var(Arch3) vs. 28 behavioral measures (FDR-corrected)
│   ├── behavioral_measures_correlation_matrix.csv # Redundancy check among those 28 measures
│   ├── arch3_variance_vs_g.csv         # Explore Var(Arch3) vs. g
│   ├── segment_count_check.csv/.png    # g vs. axis movement, per segment and as total movement
│   └── phase_position_means.csv, phase_position_changes.csv # Average position and direction of change per phase
│
├── results/figures_for_paper/          # Paper Figures 1-3 + Appendix A1-A2 (made by make_paper_figures.py)
│
└── docs/                               # Write-up, plan, and paper materials
```

---

## Scripts

### 1. per_phase_analysis.py

**Purpose:** Main analysis script - generates 3 of the 5 final plots.

**Outputs:**
| File | Description |
|------|-------------|
| `distribution_comparison.png` | Scatter plot + PC1 & PC2 histograms |
| `direction_comparison.png` | Direction vectors (first→last) per condition |
| `covariance_comparison.png` | Variance/covariance bars + difference heatmap |
| `all_segments_pca.csv` | All segments with PC coordinates and distances |
| `covariance_matrices.csv` | Covariance matrix per subject per condition |
| `direction_vectors.csv` | Direction vectors per subject |

---

### 2. covariance_paired_comparison.py

**Purpose:** Paired statistical tests for covariance matrix elements.

**Outputs:**
| File | Description |
|------|-------------|
| `covariance_paired_comparison.png` | 6 paired plots with gray lines per subject |
| `covariance_paired_tests.csv` | Statistical test results (t-test, Wilcoxon) |

---

### 3. direction_toward_archetypes.py

**Purpose:** Analyze direction toward/away from each archetype.

**Outputs:**
| File | Description |
|------|-------------|
| `direction_toward_archetypes.png` | Paired comparison + correlation scatter |
| `direction_toward_archetypes.csv` | Delta distances per subject |

---

## Final Plots Summary

| # | Plot | What it shows | Key findings |
|---|------|---------------|--------------|
| 1 | distribution_comparison.png | Explore vs Exploit in PCA space | More variance in Explore |
| 2 | direction_comparison.png | First→last direction per condition | No consistent pattern |
| 3 | covariance_comparison.png | Variance/covariance of archetype distances | Var(Arch3) higher in Explore |
| 4 | covariance_paired_comparison.png | Paired tests per element | Var(Arch3) p=0.0395*, Cov(A2,A3) p=0.0505 |
| 5 | direction_toward_archetypes.png | Direction toward each archetype | Arch1 negative correlation p=0.044* |

---

## Archetype Order

All plots use order: **A1 → A2 → A3**

| Code variable | Display | Archetype |
|--------------|---------|-----------|
| cov_11 / Archetype_1 | Arch1 | Impulsive |
| cov_33 / Archetype_2 | Arch2 | Goal-directed |
| cov_22 / Archetype_3 | Arch3 | Memory |

---

## Quick Start

```bash
cd ~/Pareto-fMRI-main
# Requires: numpy, pandas, scipy, matplotlib, seaborn, openpyxl

# Step 1: Run main analysis
python scripts/per_phase_analysis.py

# Step 2: Run paired covariance tests
python scripts/covariance_paired_comparison.py

# Step 3: Run direction toward archetypes analysis
python scripts/direction_toward_archetypes.py
```

---

## Key Results

### Question א: Distribution differences
- **Finding:** Explore has higher variance, especially PC2 (p=0.037 within-subject)
- **Plot:** distribution_comparison.png

### Question ב: Covariance matrix differences
- **Finding:** Var(Arch3) significantly higher in Explore (p=0.0395)
- **Finding:** Cov(A2,A3) almost significant (p=0.0505) - coupled in Exploit, independent in Explore
- **Plots:** covariance_comparison.png, covariance_paired_comparison.png

### Question ג: Direction differences
- **Finding:** No consistent direction pattern
- **Finding:** Arch1 shows negative correlation between Explore and Exploit directions (p=0.044)
- **Plots:** direction_comparison.png, direction_toward_archetypes.png

---

## Notes

- All data uses GSR preprocessing to match HCP
- QC filtering: Keep=1, MRI artifacts column=0
- Archetype labels follow the original article naming
