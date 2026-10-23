"""
Segment-Count Check for the g vs. Arch1-Arch2 Movement Result (R6)
===================================================================
The R6 movement measure is the slope of distance-to-Arch1 against segment
number, i.e. change per segment. This script checks how that measure depends
on the number of segments, and re-expresses it as total movement over the
session (|slope| x (n_segments - 1)) before relating it to g. It also relates the
variance of the distances to each archetype (per condition) to g and to the number
of segments.

Run from: ~/Pareto-fMRI-main/
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SLOPES_FILE = BASE_DIR / "results" / "per_phase" / "regression_slopes.csv"
G_FILE = BASE_DIR / "data" / "g_non_mri_segmentation.csv"
COV_FILE = BASE_DIR / "results" / "per_phase" / "covariance_matrices.csv"
RESULTS_DIR = BASE_DIR / "results" / "per_phase"


def normalize_id(raw):
    s = str(raw).strip()
    if s.startswith('sub-'):
        s = s[4:]
    return s.lstrip('0') or '0'


def main():
    slopes = pd.read_csv(SLOPES_FILE)
    slopes['key'] = slopes['subject_id'].apply(normalize_id)
    explore = slopes[slopes['condition'] == 'explore'].set_index('key')
    exploit = slopes[slopes['condition'] == 'exploit'].set_index('key')
    common = explore.index.intersection(exploit.index)

    g = pd.read_csv(G_FILE)
    g.columns = ['ID', 'g']
    g['key'] = g['ID'].apply(normalize_id)
    g = g.set_index('key')[['g']]

    d = pd.DataFrame(index=common)
    n_ex, n_xt = explore.loc[common, 'n_segments'], exploit.loc[common, 'n_segments']
    print(f"Segment counts differ between conditions in {(n_ex != n_xt).sum()} of {len(common)} participants")
    d['n_segments'] = (n_ex + n_xt) / 2
    d['abs_explore'] = explore.loc[common, 'slope_Archetype_1'].abs()
    d['abs_exploit'] = exploit.loc[common, 'slope_Archetype_1'].abs()
    d['mean_abs'] = (d['abs_explore'] + d['abs_exploit']) / 2
    d['drift_explore'] = d['abs_explore'] * (n_ex - 1)
    d['drift_exploit'] = d['abs_exploit'] * (n_xt - 1)
    d['mean_drift'] = (d['drift_explore'] + d['drift_exploit']) / 2
    d = d.join(g, how='inner')
    print(f"Participants with slopes in both conditions and a g value: n = {len(d)}")

    pairs = [
        ('per segment', 'mean_abs', 'g', 'mean |slope| (both conditions) vs g'),
        ('per segment', 'abs_explore', 'g', 'explore |slope| vs g'),
        ('per segment', 'mean_abs', 'n_segments', 'mean |slope| vs number of segments'),
        ('segment count', 'n_segments', 'g', 'number of segments vs g'),
        ('total movement', 'mean_drift', 'g', 'mean total movement (both conditions) vs g'),
        ('total movement', 'drift_explore', 'g', 'explore total movement vs g'),
    ]
    rows = []
    for family, x, y, label in pairs:
        for method, fn in [('pearson', stats.pearsonr), ('spearman', stats.spearmanr)]:
            r, p = fn(d[x], d[y])
            rows.append({'family': family, 'comparison': label, 'method': method,
                         'r': r, 'p_value': p, 'n': len(d)})
            print(f"  {label:45s} {method:8s} r={r:+.3f}  p={p:.4g}")

    cov = pd.read_csv(COV_FILE)
    cov['key'] = cov['subject_id'].apply(normalize_id)
    arch_cols = {'cov_11': 'Arch1', 'cov_33': 'Arch2', 'cov_22': 'Arch3'}
    for condition in ['explore', 'exploit']:
        v = cov[cov['condition'] == condition].set_index('key')[['n_segments'] + list(arch_cols)].join(g, how='inner')
        for col, arch in arch_cols.items():
            for other, name in [('g', 'g'), ('n_segments', 'number of segments')]:
                r, p = stats.spearmanr(v[col], v[other])
                label = f'{condition} Var({arch}) vs {name}'
                rows.append({'family': 'variance', 'comparison': label, 'method': 'spearman',
                             'r': r, 'p_value': p, 'n': len(v)})
                print(f"  {label:45s} spearman r={r:+.3f}  p={p:.4g}  n={len(v)}")

    out_csv = RESULTS_DIR / "segment_count_check.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, x, xlabel in [(axes[0], 'mean_abs', 'Mean |Arch1-Arch2 slope| per segment (both conditions)'),
                          (axes[1], 'mean_drift', 'Mean total movement over the session\n(|slope| x (n_segments - 1), both conditions)')]:
        r, p = stats.pearsonr(d[x], d['g'])
        ax.scatter(d[x], d['g'], alpha=0.7, s=25)
        fit = stats.linregress(d[x], d['g'])
        xs = [d[x].min(), d[x].max()]
        ax.plot(xs, [fit.intercept + fit.slope * v for v in xs], 'k--', alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('g')
        ax.set_title(f'r = {r:.3f}, p = {p:.4f}, n = {len(d)}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Arch1-Arch2 movement vs. g: per segment and as total movement', fontweight='bold')
    plt.tight_layout()
    out_png = RESULTS_DIR / "segment_count_check.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
