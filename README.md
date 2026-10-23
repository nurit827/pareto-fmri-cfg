# Pareto-fMRI: explore and exploit dynamics in brain state space

Code for a course project (The Hebrew University of Jerusalem) by Nurit Tolkowsky.

Participants played the Creative Foraging Game (CFG; Hart et al., 2018) in the fMRI scanner. Each participant's search was divided into explore and exploit segments, and the functional connectivity of each segment (17 × 17 networks) was projected into a fixed two-dimensional brain state space. The space comes from a Pareto analysis of resting-state connectivity in the Human Connectome Project (Amir et al.). Its three archetypes are:

| Archetype | Computational goal |
|-----------|--------------------|
| Arch1 | Energetic efficiency |
| Arch2 | Goal-directed processing and cognitive control |
| Arch3 | Internal processing and memory |

For every segment, the pipeline computes the distance to each archetype and then compares explore and exploit segments.

## Main findings

1. Movement of the brain state is mostly a trade-off between Arch1 and Arch2. Changes in the distances to the two archetypes are strongly anti-correlated (r = −0.84 in explore, −0.95 in exploit).
2. The distance to Arch3 varies more during exploration than during exploitation (p = 0.04, uncorrected).
3. Participants with a lower g (a parameter of the CFG model of Hart et al., 2018) move more along the Arch1–Arch2 axis from one segment to the next (r = −0.31).

The figures are in [`results/figures_for_paper/`](results/figures_for_paper/).

## Data

**The data are not included in this repository.** That covers the per-segment connectivity matrices, the behavioral tables, the g values and the HCP-derived PCA basis. The scripts expect them under `data/`, in the layout described in [`docs/PIPELINE_DOCUMENTATION.md`](docs/PIPELINE_DOCUMENTATION.md).

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Running the pipeline

Run the scripts from the repository root, in this order. Some scripts open plot windows, so to run without a display, set `MPLBACKEND=Agg`.

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `scripts/save_pca_transformation.py` | Saves the fixed PCA basis and archetype coordinates (`data/pca_transformation.npz`) from the HCP analysis |
| 1 | `scripts/per_phase_analysis.py` | Projects each segment into the PC1–PC2 plane, computes the distances to the archetypes, and computes the per-participant variances and covariances |
| 2 | `scripts/covariance_paired_comparison.py` | Compares explore and exploit with paired tests (Results 2.2) |
| 3 | `scripts/direction_toward_archetypes.py` | Computes the first-to-last change in distance to each archetype |
| 4 | `scripts/arch1_arch2_axis_analysis.py` | Computes the Arch1–Arch2 trade-off from both changes and slopes (Results 2.1) |
| 5 | `scripts/switching_rate_regression.py` | Correlates movement along the axis with g (Results 2.3) |
| 6 | `scripts/arch3_variance_behavior_correlation.py` | Correlates explore Arch3 variance with g and behavioral measures |
| 7 | `scripts/segment_count_check.py` | Checks how the results depend on the number of segments (Limitations) |
| 8 | `scripts/phase_position_check.py` | Compares the average position and the direction of change in each phase (Discussion) |
| 9 | `scripts/make_paper_figures.py` | Redraws paper Figures 1–3 and A1–A2 from the saved results |

```bash
export MPLBACKEND=Agg
python scripts/per_phase_analysis.py
python scripts/covariance_paired_comparison.py
python scripts/direction_toward_archetypes.py
python scripts/arch1_arch2_axis_analysis.py
python scripts/switching_rate_regression.py
python scripts/arch3_variance_behavior_correlation.py
python scripts/segment_count_check.py
python scripts/phase_position_check.py
python scripts/make_paper_figures.py
```

The other scripts in `scripts/` are earlier, exploratory analyses that are not part of the paper.

## Repository layout

```
scripts/                     analysis code
results/figures_for_paper/   paper figures (Figures 1–3, A1–A2)
results/per_phase/           figures from the per-phase analyses
docs/PIPELINE_DOCUMENTATION.md  detailed description of each script, its inputs and outputs
```

## References

- Amir, D. H., Gavish, Y., Elias, U., Peters-Founshtein, G., Arzy, S., & Hart, Y. *Pareto optimality reveals the core computations of the human brain.* Unpublished manuscript.
- Hart, Y., Goldberg, H., Striem-Amit, E., Mayo, A. E., Noy, L., & Alon, U. (2018). Creative exploration as a scale-invariant search on a meaning landscape. *Nature Communications*, 9, 5411. https://doi.org/10.1038/s41467-018-07715-8
