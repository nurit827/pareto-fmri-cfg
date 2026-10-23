"""
Arch3 Explore-Variance vs. Behavioral Measures Correlation
============================================================
Correlates each subject's explore-condition Var(Arch3) (cov_22) against all
behavioral measures extracted from the CFG game, to check whether the R2
finding (Var(Arch3) higher in Explore, p=0.0395) has behavioral signatures
beyond the explore/exploit contrast itself.

Run from: ~/Pareto-fMRI-main/
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COV_FILE = BASE_DIR / "results" / "per_phase" / "covariance_matrices.csv"
BEHAV_FILE = BASE_DIR / "data" / "cfgmri_measures.csv"
G_FILE = BASE_DIR / "data" / "g_non_mri_segmentation.csv"
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
RESULTS_DIR.mkdir(exist_ok=True)

NON_MEASURE_COLS = {'ID', 'Date/Time'}


def normalize_id(raw):
    s = str(raw).strip()
    if s.startswith('sub-'):
        s = s[4:]
    s = s.lstrip('0')
    return s if s else '0'


def main():
    print("\n" + "="*70)
    print("Arch3 Explore-Variance vs. Behavioral Measures Correlation")
    print("="*70)

    # Explore-condition Var(Arch3) per subject
    cov_df = pd.read_csv(COV_FILE)
    explore_cov = cov_df[cov_df['condition'] == 'explore'].copy()
    explore_cov['subject_key'] = explore_cov['subject_id'].apply(normalize_id)
    explore_cov = explore_cov.set_index('subject_key')['cov_22'].rename('var_arch3_explore').to_frame()
    print(f"\n  Subjects with explore Var(Arch3): {len(explore_cov)}")

    # Behavioral measures (original-algorithm file)
    behav_df = pd.read_csv(BEHAV_FILE)
    behav_df['subject_key'] = behav_df['ID'].apply(normalize_id)
    behav_df = behav_df.set_index('subject_key')

    measure_cols = [c for c in behav_df.columns if c not in NON_MEASURE_COLS and '(sample)' not in c]
    print(f"  Behavioral measures (main, non-sample): {len(measure_cols)}")

    # Merge
    merged = explore_cov.join(behav_df[measure_cols], how='inner')
    dropped = sorted(explore_cov.index.difference(behav_df.index), key=int)
    print(f"  Subjects with both: {len(merged)}")
    print(f"  Dropped (no behavioral match): {dropped}")

    # Correlate each measure against explore Var(Arch3)
    results = []
    for col in measure_cols:
        vals = pd.to_numeric(merged[col], errors='coerce')
        valid = vals.notna() & merged['var_arch3_explore'].notna()
        r, p = stats.spearmanr(merged.loc[valid, 'var_arch3_explore'], vals[valid])
        results.append({'measure': col, 'n': int(valid.sum()), 'spearman_r': r, 'p_value': p})

    results_df = pd.DataFrame(results)
    results_df['fdr_q'] = stats.false_discovery_control(results_df['p_value'], method='bh')
    results_df['significant_fdr05'] = results_df['fdr_q'] < 0.05
    results_df = results_df.sort_values('p_value').reset_index(drop=True)

    out_csv = RESULTS_DIR / "arch3_variance_behavior_correlations.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Redundancy check: correlation structure among the behavioral measures themselves
    behav_numeric = merged[measure_cols].apply(pd.to_numeric, errors='coerce')
    corr_matrix = behav_numeric.corr(method='spearman')
    out_corr = RESULTS_DIR / "behavioral_measures_correlation_matrix.csv"
    corr_matrix.to_csv(out_corr)
    print(f"  Saved: {out_corr}")

    # Separate, single pre-specified test: is this the same mercurial/thorough axis as g?
    # (g is Hart et al.'s model-fit switching-rate parameter, not one of the 28 game measures
    #  above, and this is one hypothesis-driven test, so no FDR correction is applied here.)
    g_df = pd.read_csv(G_FILE)
    g_df.columns = ['ID', 'g']
    g_df['subject_key'] = g_df['ID'].apply(normalize_id)
    g_df = g_df.set_index('subject_key')

    g_merged = explore_cov.join(g_df[['g']], how='inner')
    g_dropped = sorted(explore_cov.index.difference(g_df.index), key=int)
    g_r, g_p = stats.spearmanr(g_merged['var_arch3_explore'], g_merged['g'])

    g_result = pd.DataFrame([{'measure': 'g (switching rate, Hart et al. model param)',
                              'n': len(g_merged), 'spearman_r': g_r, 'p_value': g_p}])
    out_g = RESULTS_DIR / "arch3_variance_vs_g.csv"
    g_result.to_csv(out_g, index=False)
    print(f"\n  g test: n={len(g_merged)} (dropped: {g_dropped})")
    print(f"  Saved: {out_g}")

    # Plot: Spearman r per measure, sorted, FDR-significant ones marked
    plot_df = results_df.sort_values('spearman_r')
    colors = ['crimson' if sig else 'gray' for sig in plot_df['significant_fdr05']]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(plot_df['measure'], plot_df['spearman_r'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Spearman r (vs. explore Var(Arch3))')
    ax.set_title('Explore Var(Arch3) vs. Behavioral Measures\n(red = significant after FDR correction, q<0.05)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    out_png = RESULTS_DIR / "arch3_variance_behavior_correlations.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_png}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    sig = results_df[results_df['significant_fdr05']]
    if len(sig) > 0:
        print(f"\n  Significant after FDR correction (q<0.05):")
        for _, row in sig.iterrows():
            print(f"    {row['measure']}: r={row['spearman_r']:.3f}, p={row['p_value']:.4f}, q={row['fdr_q']:.4f}")
    else:
        print("\n  No measures survive FDR correction.")

    print(f"\n  Top 5 uncorrected (for reference):")
    for _, row in results_df.head(5).iterrows():
        print(f"    {row['measure']}: r={row['spearman_r']:.3f}, p={row['p_value']:.4f}, q={row['fdr_q']:.4f}")

    print(f"\n  g (switching rate, model param) vs. explore Var(Arch3): r={g_r:.3f}, p={g_p:.4f}, n={len(g_merged)}")


if __name__ == "__main__":
    main()
