"""
Phase Position Check
====================
Checks whether exploration and exploitation differ in the average position of the
brain state in the archetype space (each participant's average distance to each
archetype, compared across phases), and whether participants move toward or away
from each archetype over a phase (first-to-last change in distance).

Run from: ~/Pareto-fMRI-main/
"""
import pandas as pd
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
SEGMENTS_FILE = RESULTS_DIR / "all_segments_pca.csv"
DIRECTION_FILE = RESULTS_DIR / "direction_toward_archetypes.csv"

DIST_COLS = {'Arch1': 'dist_Archetype_1', 'Arch2': 'dist_Archetype_2', 'Arch3': 'dist_Archetype_3'}


def main():
    seg = pd.read_csv(SEGMENTS_FILE)
    means = seg.groupby(['subject_id', 'condition'])[list(DIST_COLS.values())].mean().unstack()

    rows = []
    print(f"A) Average distance to each archetype, explore vs exploit (paired, n = {len(means)}):")
    for arch, col in DIST_COLS.items():
        e, x = means[(col, 'explore')], means[(col, 'exploit')]
        t, p_t = stats.ttest_rel(e, x)
        p_w = stats.wilcoxon(e, x).pvalue
        rows.append({'archetype': arch, 'explore_mean': e.mean(), 'exploit_mean': x.mean(),
                     'diff': e.mean() - x.mean(), 'p_ttest': p_t, 'p_wilcoxon': p_w, 'n': len(e)})
        print(f"  {arch}: explore {e.mean():.2f} vs exploit {x.mean():.2f}, diff {e.mean() - x.mean():+.3f}, "
              f"t-test p = {p_t:.3f}, Wilcoxon p = {p_w:.3f}")
    out_means = RESULTS_DIR / "phase_position_means.csv"
    pd.DataFrame(rows).to_csv(out_means, index=False)

    direction = pd.read_csv(DIRECTION_FILE)
    rows = []
    print("\nB) Change in distance from the first to the last segment (negative = moved toward the archetype):")
    for condition in ['explore', 'exploit']:
        sub = direction[direction['condition'] == condition]
        for k, arch in enumerate(DIST_COLS, start=1):
            v = sub[f'delta_Archetype_{k}']
            t, p = stats.ttest_1samp(v, 0)
            rows.append({'condition': condition, 'archetype': arch, 'mean_change': v.mean(),
                         'p_vs_zero': p, 'n_toward': int((v < 0).sum()), 'n_away': int((v > 0).sum()), 'n': len(v)})
            print(f"  {condition} {arch}: mean {v.mean():+.2f}, p vs 0 = {p:.3f}; "
                  f"toward in {(v < 0).sum()}/{len(v)}, away in {(v > 0).sum()}/{len(v)}")
    out_changes = RESULTS_DIR / "phase_position_changes.csv"
    pd.DataFrame(rows).to_csv(out_changes, index=False)

    print(f"\n  Saved: {out_means}\n  Saved: {out_changes}")


if __name__ == "__main__":
    main()
