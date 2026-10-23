"""
Covariance Matrix Paired Comparison
===================================
For each element of the covariance matrix:
- Show paired comparison (explore vs exploit per subject)
- Run statistical tests

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
INPUT_FILE = RESULTS_DIR / "covariance_matrices.csv"


def main():
    print("\n" + "="*70)
    print("Covariance Matrix Paired Comparison")
    print("="*70)
    
    # Load covariance data
    cov_df = pd.read_csv(INPUT_FILE)
    print(f"\n  Total rows: {len(cov_df)}")
    
    # Separate explore and exploit
    explore_cov = cov_df[cov_df['condition'] == 'explore'].set_index('subject_id')
    exploit_cov = cov_df[cov_df['condition'] == 'exploit'].set_index('subject_id')
    
    # Get common subjects
    common_subjects = explore_cov.index.intersection(exploit_cov.index)
    print(f"  Subjects with both conditions: {len(common_subjects)}")
    
    explore_cov = explore_cov.loc[common_subjects]
    exploit_cov = exploit_cov.loc[common_subjects]
    
    # Elements to compare - ordered A1, A2, A3
    elements = {
        'cov_11': 'Var(Arch1)',
        'cov_33': 'Var(Arch2)',
        'cov_22': 'Var(Arch3)',
        'cov_13': 'Cov(A1,A2)',
        'cov_12': 'Cov(A1,A3)',
        'cov_23': 'Cov(A2,A3)'
    }
    
    # ==========================================================================
    # Statistical Tests
    # ==========================================================================
    print("\n" + "-"*50)
    print("Paired Statistical Tests")
    print("-"*50)
    
    test_results = []
    
    for col, label in elements.items():
        explore_vals = explore_cov[col].values
        exploit_vals = exploit_cov[col].values
        
        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(explore_vals, exploit_vals)
        
        # Wilcoxon signed-rank test
        w_stat, p_wilcox = stats.wilcoxon(explore_vals, exploit_vals)
        
        mean_diff = explore_vals.mean() - exploit_vals.mean()
        
        test_results.append({
            'element': col,
            'label': label,
            'explore_mean': explore_vals.mean(),
            'exploit_mean': exploit_vals.mean(),
            'diff': mean_diff,
            't_stat': t_stat,
            'p_ttest': p_ttest,
            'w_stat': w_stat,
            'p_wilcox': p_wilcox,
            'significant': p_ttest < 0.05
        })
        
        sig_marker = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else ""
        print(f"\n  {label}:")
        print(f"    Explore: {explore_vals.mean():.4f} ± {explore_vals.std():.4f}")
        print(f"    Exploit: {exploit_vals.mean():.4f} ± {exploit_vals.std():.4f}")
        print(f"    Diff: {mean_diff:+.4f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_ttest:.4f} {sig_marker}")
        print(f"    Wilcoxon: W={w_stat:.1f}, p={p_wilcox:.4f}")
    
    results_df = pd.DataFrame(test_results)
    
    # ==========================================================================
    # Visualization - Variance elements (diagonal) - ordered A1, A2, A3
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    variance_elements = ['cov_11', 'cov_33', 'cov_22']  # A1, A2, A3
    variance_labels = ['Var(Arch1)', 'Var(Arch2)', 'Var(Arch3)']
    
    for idx, (col, label) in enumerate(zip(variance_elements, variance_labels)):
        ax = axes[0, idx]
        
        explore_vals = explore_cov[col].values
        exploit_vals = exploit_cov[col].values
        
        # Draw lines for each subject
        for i in range(len(common_subjects)):
            ax.plot([0, 1], [explore_vals[i], exploit_vals[i]], 
                    'o-', color='gray', alpha=0.3, markersize=4)
        
        # Boxplots
        bp = ax.boxplot([explore_vals, exploit_vals], positions=[0, 1], widths=0.3)
        
        # Color the boxes
        for patch, color in zip([bp['boxes'][0], bp['boxes'][1]], ['green', 'red']):
            patch.set_color(color)
        
        # Add mean values as text
        ax.text(0, ax.get_ylim()[1]*0.95, f'μ={explore_vals.mean():.2f}', ha='center', fontsize=10, color='green')
        ax.text(1, ax.get_ylim()[1]*0.95, f'μ={exploit_vals.mean():.2f}', ha='center', fontsize=10, color='red')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Explore', 'Exploit'])
        ax.set_ylabel('Variance')
        
        # Get p-value for title
        p_val = results_df[results_df['element'] == col]['p_ttest'].values[0]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        
        ax.set_title(f'{label}\np={p_val:.4f} {sig}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Visualization - Covariance elements (off-diagonal) - ordered A1-A2, A1-A3, A2-A3
    # ==========================================================================
    covariance_elements = ['cov_13', 'cov_12', 'cov_23']  # A1-A2, A1-A3, A2-A3
    covariance_labels = ['Cov(A1,A2)', 'Cov(A1,A3)', 'Cov(A2,A3)']
    
    for idx, (col, label) in enumerate(zip(covariance_elements, covariance_labels)):
        ax = axes[1, idx]
        
        explore_vals = explore_cov[col].values
        exploit_vals = exploit_cov[col].values
        
        # Draw lines for each subject
        for i in range(len(common_subjects)):
            ax.plot([0, 1], [explore_vals[i], exploit_vals[i]], 
                    'o-', color='gray', alpha=0.3, markersize=4)
        
        # Boxplots
        bp = ax.boxplot([explore_vals, exploit_vals], positions=[0, 1], widths=0.3)
        
        # Color the boxes
        for patch, color in zip([bp['boxes'][0], bp['boxes'][1]], ['green', 'red']):
            patch.set_color(color)
        
        # Add mean values as text
        ax.text(0, ax.get_ylim()[1]*0.95, f'μ={explore_vals.mean():.2f}', ha='center', fontsize=10, color='green')
        ax.text(1, ax.get_ylim()[1]*0.95, f'μ={exploit_vals.mean():.2f}', ha='center', fontsize=10, color='red')
        
        # Boxplots
        bp = ax.boxplot([explore_vals, exploit_vals], positions=[0, 1], widths=0.3)
        
        # Color the boxes
        for patch, color in zip([bp['boxes'][0], bp['boxes'][1]], ['green', 'red']):
            patch.set_color(color)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Explore', 'Exploit'])
        ax.set_ylabel('Covariance')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Get p-value for title
        p_val = results_df[results_df['element'] == col]['p_ttest'].values[0]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        
        ax.set_title(f'{label}\np={p_val:.4f} {sig}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Covariance Matrix Elements: Paired Comparison\n(each line = one subject)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = RESULTS_DIR / "covariance_paired_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    
    # Save test results
    results_df.to_csv(RESULTS_DIR / "covariance_paired_tests.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'covariance_paired_tests.csv'}")
    
    plt.show()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    sig_elements = results_df[results_df['significant']]
    
    if len(sig_elements) > 0:
        print(f"\n  Significant differences (p<0.05):")
        for _, row in sig_elements.iterrows():
            direction = "Explore > Exploit" if row['diff'] > 0 else "Explore < Exploit"
            print(f"    {row['label']}: {direction} (p={row['p_ttest']:.4f})")
    else:
        print("\n  No significant differences found.")
    
    print(f"\n  Non-significant elements:")
    nonsig = results_df[~results_df['significant']]
    for _, row in nonsig.iterrows():
        print(f"    {row['label']}: p={row['p_ttest']:.4f}")


if __name__ == "__main__":
    main()
