"""
Arch1-Arch2 Axis: Magnitude Analysis
=====================================
Test whether magnitude of movement along the Arch1-Arch2 axis
is related between conditions.

1. All subjects: correlate |explore_slope| with |exploit_slope|
2. Same-direction subjects: paired magnitude test
3. Opposite-direction subjects: paired magnitude test

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
SLOPES_FILE = RESULTS_DIR / "regression_slopes.csv"


def format_p(p_val):
    if p_val < 0.0001:
        return "p<0.0001"
    else:
        return f"p={p_val:.4f}"


def main():
    print("\n" + "=" * 70)
    print("Arch1-Arch2 Axis: Magnitude Analysis")
    print("=" * 70)

    slope_df = pd.read_csv(SLOPES_FILE)

    explore = slope_df[slope_df['condition'] == 'explore'].set_index('subject_id')
    exploit = slope_df[slope_df['condition'] == 'exploit'].set_index('subject_id')

    common = explore.index.intersection(exploit.index)
    explore = explore.loc[common]
    exploit = exploit.loc[common]

    explore_slope = explore['slope_Archetype_1'].values
    exploit_slope = exploit['slope_Archetype_1'].values

    explore_mag = np.abs(explore_slope)
    exploit_mag = np.abs(exploit_slope)

    same_mask = np.sign(explore_slope) == np.sign(exploit_slope)
    opposite_mask = ~same_mask

    n_total = len(common)
    n_same = same_mask.sum()
    n_opposite = opposite_mask.sum()

    print(f"\n  Total subjects: {n_total}")
    print(f"  Same direction: {n_same}")
    print(f"  Opposite direction: {n_opposite}")

    # =================================================================
    # 1. ALL SUBJECTS: magnitude correlation
    # =================================================================
    print("\n" + "=" * 70)
    print("1. ALL SUBJECTS: |explore_slope| vs |exploit_slope|")
    print("=" * 70)

    r_all, p_all = stats.pearsonr(explore_mag, exploit_mag)
    rho_all, p_rho_all = stats.spearmanr(explore_mag, exploit_mag)

    print(f"\n  Pearson:  r={r_all:.3f}, {format_p(p_all)} {'*' if p_all < 0.05 else ''}")
    print(f"  Spearman: rho={rho_all:.3f}, {format_p(p_rho_all)} {'*' if p_rho_all < 0.05 else ''}")

    t_all, p_t_all = stats.ttest_rel(explore_mag, exploit_mag)
    print(f"\n  Paired magnitude test (explore vs exploit):")
    print(f"    Explore |slope|: {explore_mag.mean():.4f} ± {explore_mag.std():.4f}")
    print(f"    Exploit |slope|: {exploit_mag.mean():.4f} ± {exploit_mag.std():.4f}")
    print(f"    Paired t-test: t={t_all:.3f}, {format_p(p_t_all)} {'*' if p_t_all < 0.05 else ''}")

    # =================================================================
    # 2. SAME-DIRECTION SUBJECTS: magnitude comparison
    # =================================================================
    print("\n" + "=" * 70)
    print(f"2. SAME-DIRECTION SUBJECTS (n={n_same})")
    print("=" * 70)

    explore_mag_same = explore_mag[same_mask]
    exploit_mag_same = exploit_mag[same_mask]

    r_same, p_same_corr = stats.pearsonr(explore_mag_same, exploit_mag_same)
    print(f"\n  Magnitude correlation: r={r_same:.3f}, {format_p(p_same_corr)} {'*' if p_same_corr < 0.05 else ''}")

    t_same, p_same = stats.ttest_rel(explore_mag_same, exploit_mag_same)
    w_same, pw_same = stats.wilcoxon(explore_mag_same, exploit_mag_same)

    print(f"\n  Explore |slope|: {explore_mag_same.mean():.4f} ± {explore_mag_same.std():.4f}")
    print(f"  Exploit |slope|: {exploit_mag_same.mean():.4f} ± {exploit_mag_same.std():.4f}")
    print(f"  Paired t-test: t={t_same:.3f}, {format_p(p_same)} {'*' if p_same < 0.05 else ''}")
    print(f"  Wilcoxon:      W={w_same:.1f}, {format_p(pw_same)} {'*' if pw_same < 0.05 else ''}")

    bigger_explore_same = (explore_mag_same > exploit_mag_same).sum()
    print(f"  Larger in explore: {bigger_explore_same}/{n_same}")
    print(f"  Larger in exploit: {n_same - bigger_explore_same}/{n_same}")

    # =================================================================
    # 3. OPPOSITE-DIRECTION SUBJECTS: magnitude comparison
    # =================================================================
    print("\n" + "=" * 70)
    print(f"3. OPPOSITE-DIRECTION SUBJECTS (n={n_opposite})")
    print("=" * 70)

    explore_mag_opp = explore_mag[opposite_mask]
    exploit_mag_opp = exploit_mag[opposite_mask]

    r_opp, p_opp_corr = stats.pearsonr(explore_mag_opp, exploit_mag_opp)
    print(f"\n  Magnitude correlation: r={r_opp:.3f}, {format_p(p_opp_corr)} {'*' if p_opp_corr < 0.05 else ''}")

    t_opp, p_opp = stats.ttest_rel(explore_mag_opp, exploit_mag_opp)
    w_opp, pw_opp = stats.wilcoxon(explore_mag_opp, exploit_mag_opp)

    print(f"\n  Explore |slope|: {explore_mag_opp.mean():.4f} ± {explore_mag_opp.std():.4f}")
    print(f"  Exploit |slope|: {exploit_mag_opp.mean():.4f} ± {exploit_mag_opp.std():.4f}")
    print(f"  Paired t-test: t={t_opp:.3f}, {format_p(p_opp)} {'*' if p_opp < 0.05 else ''}")
    print(f"  Wilcoxon:      W={w_opp:.1f}, {format_p(pw_opp)} {'*' if pw_opp < 0.05 else ''}")

    bigger_explore_opp = (explore_mag_opp > exploit_mag_opp).sum()
    print(f"  Larger in explore: {bigger_explore_opp}/{n_opposite}")
    print(f"  Larger in exploit: {n_opposite - bigger_explore_opp}/{n_opposite}")

    # =================================================================
    # VISUALIZATION
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Magnitude of Movement Along Arch1-Arch2 Axis',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: All subjects magnitude correlation ---
    ax1 = axes[0]
    ax1.scatter(explore_mag[same_mask], exploit_mag[same_mask],
                c='green', alpha=0.6, label=f'Same dir ({n_same})')
    ax1.scatter(explore_mag[opposite_mask], exploit_mag[opposite_mask],
                c='red', alpha=0.6, label=f'Opposite dir ({n_opposite})')

    # Fit line for all
    z = np.polyfit(explore_mag, exploit_mag, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(0, explore_mag.max(), 100)
    ax1.plot(x_line, p_line(x_line), 'k--', alpha=0.8)

    # Identity line
    max_val = max(explore_mag.max(), exploit_mag.max()) * 1.05
    ax1.plot([0, max_val], [0, max_val], ':', color='gray', alpha=0.5)

    ax1.set_xlabel('|Explore slope|')
    ax1.set_ylabel('|Exploit slope|')
    ax1.set_title(f'All Subjects (n={n_total})\nr={r_all:.3f}, {format_p(p_all)}',
                  fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)

    # --- Panel 2: Same-direction paired ---
    ax2 = axes[1]
    for i in range(n_same):
        ax2.plot([0, 1], [explore_mag_same[i], exploit_mag_same[i]],
                 'o-', color='green', alpha=0.3, markersize=4)

    bp2 = ax2.boxplot([explore_mag_same, exploit_mag_same], positions=[0, 1], widths=0.3,
                      patch_artist=True)
    bp2['boxes'][0].set_facecolor('steelblue')
    bp2['boxes'][0].set_alpha(0.5)
    bp2['boxes'][1].set_facecolor('indianred')
    bp2['boxes'][1].set_alpha(0.5)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Explore', 'Exploit'])
    ax2.set_ylabel('|slope|')
    ax2.set_title(f'Same Direction (n={n_same})\n{format_p(p_same)} (paired t-test)',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Opposite-direction paired ---
    ax3 = axes[2]
    for i in range(n_opposite):
        ax3.plot([0, 1], [explore_mag_opp[i], exploit_mag_opp[i]],
                 'o-', color='red', alpha=0.3, markersize=4)

    bp3 = ax3.boxplot([explore_mag_opp, exploit_mag_opp], positions=[0, 1], widths=0.3,
                      patch_artist=True)
    bp3['boxes'][0].set_facecolor('steelblue')
    bp3['boxes'][0].set_alpha(0.5)
    bp3['boxes'][1].set_facecolor('indianred')
    bp3['boxes'][1].set_alpha(0.5)

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Explore', 'Exploit'])
    ax3.set_ylabel('|slope|')
    ax3.set_title(f'Opposite Direction (n={n_opposite})\n{format_p(p_opp)} (paired t-test)',
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = RESULTS_DIR / "arch1_arch2_magnitude_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    plt.show()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  All subjects: magnitude correlation r={r_all:.3f}, {format_p(p_all)}")
    if p_all < 0.05:
        print(f"    → Subjects who move more in explore also move more in exploit")
    else:
        print(f"    → No relationship between magnitudes")

    print(f"\n  Same-direction (n={n_same}): magnitude {format_p(p_same)}")
    if p_same < 0.05:
        direction = "explore" if t_same > 0 else "exploit"
        print(f"    → Move more in {direction}")
    else:
        print(f"    → No magnitude difference")

    print(f"\n  Opposite-direction (n={n_opposite}): magnitude {format_p(p_opp)}")
    if p_opp < 0.05:
        direction = "explore" if t_opp > 0 else "exploit"
        print(f"    → Move more in {direction}")
    else:
        print(f"    → No magnitude difference")


if __name__ == "__main__":
    main()