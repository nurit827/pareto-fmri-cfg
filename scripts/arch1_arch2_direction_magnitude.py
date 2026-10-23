"""
Arch1-Arch2 Axis: Direction Consistency and Magnitude
=====================================================
For each subject, using the Arch1-Arch2 axis (captured by slope_Arch1):

Idea 1: Do subjects move in the same direction in explore and exploit?
Idea 2: Do subjects move more in one condition?

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
    """Format p-value: show '<0.0001' for very small values."""
    if p_val < 0.0001:
        return "p<0.0001"
    else:
        return f"p={p_val:.4f}"


def main():
    print("\n" + "=" * 70)
    print("Arch1-Arch2 Axis: Direction Consistency and Magnitude")
    print("=" * 70)

    slope_df = pd.read_csv(SLOPES_FILE)

    explore = slope_df[slope_df['condition'] == 'explore'].set_index('subject_id')
    exploit = slope_df[slope_df['condition'] == 'exploit'].set_index('subject_id')

    common = explore.index.intersection(exploit.index)
    explore = explore.loc[common]
    exploit = exploit.loc[common]

    print(f"\n  Subjects with both conditions: {len(common)}")

    explore_slope = explore['slope_Archetype_1'].values
    exploit_slope = exploit['slope_Archetype_1'].values

    # =================================================================
    # IDEA 1: Direction consistency
    # =================================================================
    print("\n" + "=" * 70)
    print("IDEA 1: Do subjects move in the same direction?")
    print("=" * 70)
    print("  (negative slope = toward Arch1, positive = toward Arch2)")

    # Correlation
    r, p_val = stats.pearsonr(explore_slope, exploit_slope)
    print(f"\n  Correlation between explore and exploit slopes:")
    print(f"    r={r:.3f}, {format_p(p_val)} {'*' if p_val < 0.05 else ''}")

    # Count same vs opposite direction
    same_direction = np.sum(np.sign(explore_slope) == np.sign(exploit_slope))
    opposite_direction = len(common) - same_direction
    pct_same = same_direction / len(common) * 100
    pct_opposite = opposite_direction / len(common) * 100

    print(f"\n  Same direction:     {same_direction}/{len(common)} ({pct_same:.1f}%)")
    print(f"  Opposite direction: {opposite_direction}/{len(common)} ({pct_opposite:.1f}%)")

    # Binomial test: is same/opposite significantly different from 50/50?
    binom_p = stats.binom_test(same_direction, len(common), 0.5)
    print(f"  Binomial test vs 50/50: {format_p(binom_p)} {'*' if binom_p < 0.05 else ''}")

    # Break down by direction
    toward_A1_both = np.sum((explore_slope < 0) & (exploit_slope < 0))
    toward_A2_both = np.sum((explore_slope > 0) & (exploit_slope > 0))
    A1_explore_A2_exploit = np.sum((explore_slope < 0) & (exploit_slope > 0))
    A2_explore_A1_exploit = np.sum((explore_slope > 0) & (exploit_slope < 0))

    print(f"\n  Breakdown:")
    print(f"    Toward Arch1 in both:                    {toward_A1_both}")
    print(f"    Toward Arch2 in both:                    {toward_A2_both}")
    print(f"    Toward Arch1 in explore, Arch2 in exploit: {A1_explore_A2_exploit}")
    print(f"    Toward Arch2 in explore, Arch1 in exploit: {A2_explore_A1_exploit}")

    # =================================================================
    # IDEA 2: Magnitude comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("IDEA 2: Do subjects move more in one condition?")
    print("=" * 70)

    explore_magnitude = np.abs(explore_slope)
    exploit_magnitude = np.abs(exploit_slope)

    print(f"\n  Explore magnitude: {explore_magnitude.mean():.4f} ± {explore_magnitude.std():.4f}")
    print(f"  Exploit magnitude: {exploit_magnitude.mean():.4f} ± {exploit_magnitude.std():.4f}")

    t_stat, p_paired = stats.ttest_rel(explore_magnitude, exploit_magnitude)
    w_stat, p_wilcox = stats.wilcoxon(explore_magnitude, exploit_magnitude)

    print(f"\n  Paired t-test: t={t_stat:.3f}, {format_p(p_paired)} {'*' if p_paired < 0.05 else ''}")
    print(f"  Wilcoxon:      W={w_stat:.1f}, {format_p(p_wilcox)} {'*' if p_wilcox < 0.05 else ''}")

    bigger_in_explore = np.sum(explore_magnitude > exploit_magnitude)
    bigger_in_exploit = np.sum(exploit_magnitude > explore_magnitude)
    print(f"\n  Subjects with larger movement in explore: {bigger_in_explore}/{len(common)}")
    print(f"  Subjects with larger movement in exploit: {bigger_in_exploit}/{len(common)}")

    # =================================================================
    # VISUALIZATION
    # =================================================================
    print("\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Movement Along Arch1-Arch2 Axis: Explore vs Exploit',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Direction correlation ---
    ax1 = axes[0]

    # Color by same/opposite direction
    colors = ['green' if np.sign(e) == np.sign(x) else 'red'
              for e, x in zip(explore_slope, exploit_slope)]

    ax1.scatter(explore_slope, exploit_slope, c=colors, alpha=0.6)

    # Fit line
    z = np.polyfit(explore_slope, exploit_slope, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(explore_slope.min(), explore_slope.max(), 100)
    ax1.plot(x_line, p_line(x_line), 'k--', alpha=0.8)

    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)

    # Quadrant labels
    lim = max(abs(explore_slope).max(), abs(exploit_slope).max()) * 1.1
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

    ax1.set_xlabel('Explore slope (Arch1-Arch2 axis)')
    ax1.set_ylabel('Exploit slope (Arch1-Arch2 axis)')
    ax1.set_title(f'Direction Consistency\nr={r:.3f}, {format_p(p_val)}\n'
                  f'Same direction: {same_direction}/{len(common)} ({pct_same:.0f}%)',
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                              label=f'Same direction ({same_direction})', markersize=8),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                              label=f'Opposite direction ({opposite_direction})', markersize=8)]
    ax1.legend(handles=legend_elements, loc='upper left')

    # --- Panel 2: Magnitude paired comparison ---
    ax2 = axes[1]

    for i in range(len(common)):
        ax2.plot([0, 1], [explore_magnitude[i], exploit_magnitude[i]],
                 'o-', color='gray', alpha=0.3, markersize=4)

    bp = ax2.boxplot([explore_magnitude, exploit_magnitude], positions=[0, 1], widths=0.3,
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('indianred')
    bp['boxes'][1].set_alpha(0.5)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Explore', 'Exploit'])
    ax2.set_ylabel('Movement magnitude (|slope|)')
    ax2.set_title(f'Movement Magnitude\n{format_p(p_paired)} (paired t-test)',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Direction breakdown ---
    ax3 = axes[2]

    categories = ['→Arch1\nboth', '→Arch2\nboth', '→Arch1 explore\n→Arch2 exploit',
                  '→Arch2 explore\n→Arch1 exploit']
    counts = [toward_A1_both, toward_A2_both, A1_explore_A2_exploit, A2_explore_A1_exploit]
    bar_colors = ['green', 'green', 'red', 'red']

    bars = ax3.bar(range(4), counts, color=bar_colors, alpha=0.7, edgecolor='black')

    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', fontsize=12, fontweight='bold')

    ax3.set_xticks(range(4))
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylabel('Number of subjects')
    ax3.set_title('Direction Breakdown', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = RESULTS_DIR / "arch1_arch2_direction_magnitude.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    plt.show()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if p_val < 0.05:
        if r > 0:
            print(f"\n  Direction: Subjects tend to move in the SAME direction in both conditions")
        else:
            print(f"\n  Direction: Subjects tend to move in OPPOSITE directions between conditions")
    else:
        print(f"\n  Direction: No significant relationship between conditions")

    if p_paired < 0.05:
        if t_stat > 0:
            print(f"  Magnitude: Subjects move MORE in explore")
        else:
            print(f"  Magnitude: Subjects move MORE in exploit")
    else:
        print(f"  Magnitude: No significant difference in movement magnitude")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()