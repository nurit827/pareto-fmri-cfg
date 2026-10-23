"""
Within-Subject Variance Analysis
================================
For each subject:
1. Calculate variance of PC1/PC2 across their explore segments
2. Calculate variance of PC1/PC2 across their exploit segments
3. Compare: paired t-test or Wilcoxon

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
INPUT_FILE = RESULTS_DIR / "all_segments_pca.csv"


def main():
    print("\n" + "=" * 70)
    print("Within-Subject Variance Analysis")
    print("=" * 70)

    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Total segments: {len(df)}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")

    # Calculate variance per subject per condition
    results = []

    for subject_id in df['subject_id'].unique():
        for condition in ['explore', 'exploit']:
            subset = df[(df['subject_id'] == subject_id) & (df['condition'] == condition)]

            if len(subset) < 2:  # Need at least 2 segments for variance
                continue

            results.append({
                'subject_id': subject_id,
                'condition': condition,
                'n_segments': len(subset),
                'var_PC1': subset['PC1'].var(),
                'var_PC2': subset['PC2'].var(),
                'std_PC1': subset['PC1'].std(),
                'std_PC2': subset['PC2'].std(),
            })

    var_df = pd.DataFrame(results)

    # Pivot to get paired data
    explore_var = var_df[var_df['condition'] == 'explore'].set_index('subject_id')
    exploit_var = var_df[var_df['condition'] == 'exploit'].set_index('subject_id')

    # Get subjects with both conditions
    common_subjects = explore_var.index.intersection(exploit_var.index)
    print(f"\n  Subjects with both conditions: {len(common_subjects)}")

    explore_var = explore_var.loc[common_subjects]
    exploit_var = exploit_var.loc[common_subjects]

    # ==========================================================================
    # Compare within-subject variance
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Within-Subject Variance (mean ± std across subjects)")
    print("-" * 50)

    print(f"\n  PC1 Variance:")
    print(f"    Explore: {explore_var['var_PC1'].mean():.4f} ± {explore_var['var_PC1'].std():.4f}")
    print(f"    Exploit: {exploit_var['var_PC1'].mean():.4f} ± {exploit_var['var_PC1'].std():.4f}")

    print(f"\n  PC2 Variance:")
    print(f"    Explore: {explore_var['var_PC2'].mean():.4f} ± {explore_var['var_PC2'].std():.4f}")
    print(f"    Exploit: {exploit_var['var_PC2'].mean():.4f} ± {exploit_var['var_PC2'].std():.4f}")

    # Paired tests
    print("\n" + "-" * 50)
    print("Paired Statistical Tests")
    print("-" * 50)

    # PC1
    t_pc1, p_pc1 = stats.ttest_rel(explore_var['var_PC1'], exploit_var['var_PC1'])
    w_pc1, pw_pc1 = stats.wilcoxon(explore_var['var_PC1'], exploit_var['var_PC1'])

    print(f"\n  PC1 Variance:")
    print(f"    Paired t-test: t={t_pc1:.3f}, p={p_pc1:.4f}")
    print(f"    Wilcoxon: W={w_pc1:.1f}, p={pw_pc1:.4f}")

    # PC2
    t_pc2, p_pc2 = stats.ttest_rel(explore_var['var_PC2'], exploit_var['var_PC2'])
    w_pc2, pw_pc2 = stats.wilcoxon(explore_var['var_PC2'], exploit_var['var_PC2'])

    print(f"\n  PC2 Variance:")
    print(f"    Paired t-test: t={t_pc2:.3f}, p={p_pc2:.4f}")
    print(f"    Wilcoxon: W={w_pc2:.1f}, p={pw_pc2:.4f}")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: PC1 variance paired comparison
    ax1 = axes[0]
    for i, subj in enumerate(common_subjects):
        ax1.plot([0, 1], [explore_var.loc[subj, 'var_PC1'], exploit_var.loc[subj, 'var_PC1']],
                 'o-', color='gray', alpha=0.3, markersize=4)
    ax1.boxplot([explore_var['var_PC1'], exploit_var['var_PC1']], positions=[0, 1])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Explore', 'Exploit'])
    ax1.set_ylabel('Variance')
    ax1.set_title(f'PC1 Within-Subject Variance\np={p_pc1:.4f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: PC2 variance paired comparison
    ax2 = axes[1]
    for i, subj in enumerate(common_subjects):
        ax2.plot([0, 1], [explore_var.loc[subj, 'var_PC2'], exploit_var.loc[subj, 'var_PC2']],
                 'o-', color='gray', alpha=0.3, markersize=4)
    ax2.boxplot([explore_var['var_PC2'], exploit_var['var_PC2']], positions=[0, 1])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Explore', 'Exploit'])
    ax2.set_ylabel('Variance')
    ax2.set_title(f'PC2 Within-Subject Variance\np={p_pc2:.4f}', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter of explore vs exploit variance
    ax3 = axes[2]
    ax3.scatter(explore_var['var_PC1'], exploit_var['var_PC1'], alpha=0.6, label='PC1', c='blue')
    ax3.scatter(explore_var['var_PC2'], exploit_var['var_PC2'], alpha=0.6, label='PC2', c='red')
    max_val = max(explore_var['var_PC1'].max(), exploit_var['var_PC1'].max(),
                  explore_var['var_PC2'].max(), exploit_var['var_PC2'].max())
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal')
    ax3.set_xlabel('Explore Variance')
    ax3.set_ylabel('Exploit Variance')
    ax3.set_title('Within-Subject Variance\n(points above line = Exploit > Explore)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save
    output_path = RESULTS_DIR / "within_subject_variance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    # Save data
    var_df.to_csv(RESULTS_DIR / "within_subject_variance.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'within_subject_variance.csv'}")

    plt.show()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    diff_pc1 = explore_var['var_PC1'].mean() - exploit_var['var_PC1'].mean()
    diff_pc2 = explore_var['var_PC2'].mean() - exploit_var['var_PC2'].mean()

    print(f"\n  PC1: Explore variance is {'HIGHER' if diff_pc1 > 0 else 'LOWER'} than Exploit by {abs(diff_pc1):.4f}")
    print(f"       Significant? {'YES' if p_pc1 < 0.05 else 'NO'} (p={p_pc1:.4f})")

    print(f"\n  PC2: Explore variance is {'HIGHER' if diff_pc2 > 0 else 'LOWER'} than Exploit by {abs(diff_pc2):.4f}")
    print(f"       Significant? {'YES' if p_pc2 < 0.05 else 'NO'} (p={p_pc2:.4f})")


if __name__ == "__main__":
    main()