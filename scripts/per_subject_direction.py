"""
Per-Subject Direction Consistency Analysis
===========================================
For each subject:
- Collect all their explore→exploit transitions
- Test if THEIR directions are clustered (Rayleigh test)
- Report how many subjects have consistent personal patterns

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
INPUT_FILE = RESULTS_DIR / "explore_to_exploit_transitions.csv"
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"


def rayleigh_test(angles):
    """
    Rayleigh test for non-uniformity of circular data.
    Returns R (mean resultant length) and p-value.
    """
    n = len(angles)
    if n < 2:
        return np.nan, np.nan

    # Mean resultant length
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C ** 2 + S ** 2)

    # Mean direction
    mean_angle = np.arctan2(S, C)

    # Rayleigh test statistic
    Z = n * R ** 2

    # P-value (approximation)
    p_value = np.exp(-Z)

    # Better approximation for small n
    if n < 50:
        p_value = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (
                    288 * n ** 2))

    return R, p_value, mean_angle


def main():
    print("\n" + "=" * 70)
    print("Per-Subject Direction Consistency Analysis")
    print("=" * 70)

    # Load transition data
    trans_df = pd.read_csv(INPUT_FILE)
    print(f"\n  Total transitions: {len(trans_df)}")
    print(f"  Unique subjects: {trans_df['subject_id'].nunique()}")

    # Load archetypes for plotting
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    archetypes_pca = pca_data['archetypes_pca']

    # Analyze each subject
    subject_results = []

    for subject_id in trans_df['subject_id'].unique():
        subj_data = trans_df[trans_df['subject_id'] == subject_id]
        n_transitions = len(subj_data)

        if n_transitions < 3:  # Need at least 3 for meaningful test
            continue

        angles = subj_data['angle_rad'].values
        R, p_value, mean_angle = rayleigh_test(angles)

        # Calculate mean dx, dy
        mean_dx = subj_data['dx'].mean()
        mean_dy = subj_data['dy'].mean()

        # Variance in directions
        angle_std = stats.circstd(angles)  # Circular standard deviation

        subject_results.append({
            'subject_id': subject_id,
            'n_transitions': n_transitions,
            'R': R,
            'p_value': p_value,
            'mean_angle_deg': np.degrees(mean_angle),
            'mean_dx': mean_dx,
            'mean_dy': mean_dy,
            'angle_circular_std': np.degrees(angle_std),
            'has_pattern': p_value < 0.05
        })

    results_df = pd.DataFrame(subject_results)

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Summary")
    print("-" * 50)

    n_subjects = len(results_df)
    n_with_pattern = results_df['has_pattern'].sum()
    pct_with_pattern = 100 * n_with_pattern / n_subjects

    print(f"\n  Subjects analyzed: {n_subjects}")
    print(f"  Subjects with consistent direction (p<0.05): {n_with_pattern} ({pct_with_pattern:.1f}%)")
    print(f"  Subjects with random directions: {n_subjects - n_with_pattern} ({100 - pct_with_pattern:.1f}%)")

    print(f"\n  Mean R across subjects: {results_df['R'].mean():.3f} ± {results_df['R'].std():.3f}")
    print(f"  (R close to 0 = random, R close to 1 = very consistent)")

    # ==========================================================================
    # Look at subjects WITH patterns
    # ==========================================================================
    if n_with_pattern > 0:
        print("\n" + "-" * 50)
        print(f"Subjects with consistent patterns (n={n_with_pattern})")
        print("-" * 50)

        pattern_subjects = results_df[results_df['has_pattern']]

        print(f"\n  Their mean directions:")
        for _, row in pattern_subjects.iterrows():
            direction = ""
            if row['mean_dx'] > 0.1:
                direction += "right "
            elif row['mean_dx'] < -0.1:
                direction += "left "
            if row['mean_dy'] > 0.1:
                direction += "up"
            elif row['mean_dy'] < -0.1:
                direction += "down"
            if direction == "":
                direction = "small movement"

            print(
                f"    {row['subject_id']}: {row['mean_angle_deg']:.0f}° (R={row['R']:.2f}, p={row['p_value']:.4f}) → {direction}")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Distribution of R values
    ax1 = axes[0, 0]
    ax1.hist(results_df['R'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax1.axvline(results_df['R'].mean(), color='red', linestyle='--', lw=2,
                label=f'Mean R = {results_df["R"].mean():.3f}')
    ax1.set_xlabel('R (direction consistency)', fontsize=12)
    ax1.set_ylabel('Number of subjects', fontsize=12)
    ax1.set_title('Distribution of Per-Subject Direction Consistency\n(R close to 1 = consistent pattern)',
                  fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: P-value distribution
    ax2 = axes[0, 1]
    ax2.hist(results_df['p_value'], bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax2.axvline(0.05, color='red', linestyle='--', lw=2, label='p=0.05 threshold')
    ax2.set_xlabel('P-value (Rayleigh test)', fontsize=12)
    ax2.set_ylabel('Number of subjects', fontsize=12)
    ax2.set_title(f'P-value Distribution\n({n_with_pattern} subjects with p<0.05)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean directions for subjects WITH patterns
    ax3 = axes[1, 0]

    # Draw triangle
    triangle = np.vstack([archetypes_pca, archetypes_pca[0]])
    ax3.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, zorder=5)

    # Draw archetypes
    archetype_colors = ['cyan', 'red', 'yellow']
    for i in range(3):
        ax3.scatter(archetypes_pca[i, 0], archetypes_pca[i, 1],
                    c=archetype_colors[i], s=150, marker='^',
                    edgecolors='black', linewidth=2, zorder=10)

    # Draw mean direction arrows for subjects with patterns
    if n_with_pattern > 0:
        pattern_subjects = results_df[results_df['has_pattern']]

        # Get start position (mean explore position) for each subject
        for _, row in pattern_subjects.iterrows():
            subj_trans = trans_df[trans_df['subject_id'] == row['subject_id']]
            start_x = subj_trans['explore_PC1'].mean()
            start_y = subj_trans['explore_PC2'].mean()

            # Scale arrow for visibility
            scale = 1.0
            ax3.annotate('',
                         xy=(start_x + row['mean_dx'] * scale, start_y + row['mean_dy'] * scale),
                         xytext=(start_x, start_y),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7))

    ax3.set_xlabel('PC1', fontsize=12)
    ax3.set_ylabel('PC2', fontsize=12)
    ax3.set_title(f'Mean Direction for Subjects WITH Patterns (n={n_with_pattern})\n(green arrows)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Plot 4: Polar plot of mean directions for subjects with patterns
    ax4 = plt.subplot(2, 2, 4, projection='polar')

    if n_with_pattern > 0:
        pattern_angles = np.radians(pattern_subjects['mean_angle_deg'].values)
        pattern_R = pattern_subjects['R'].values

        ax4.scatter(pattern_angles, pattern_R, c='green', s=100, alpha=0.7, label='Subjects with pattern')

    # Also show subjects without pattern (in gray)
    no_pattern = results_df[~results_df['has_pattern']]
    if len(no_pattern) > 0:
        no_pattern_angles = np.radians(no_pattern['mean_angle_deg'].values)
        no_pattern_R = no_pattern['R'].values
        ax4.scatter(no_pattern_angles, no_pattern_R, c='gray', s=30, alpha=0.3, label='No pattern')

    ax4.set_title('Mean Direction per Subject\n(distance from center = R consistency)', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    # Save
    output_path = RESULTS_DIR / "per_subject_direction_consistency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    # Save data
    results_df.to_csv(RESULTS_DIR / "per_subject_direction_consistency.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'per_subject_direction_consistency.csv'}")

    plt.show()

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if pct_with_pattern > 50:
        print(f"\n  MOST subjects ({pct_with_pattern:.0f}%) have consistent personal direction patterns")
        print("  → Individual patterns exist, but they differ between subjects")
    elif pct_with_pattern > 20:
        print(f"\n  SOME subjects ({pct_with_pattern:.0f}%) have consistent personal direction patterns")
        print("  → A subset of individuals show consistent behavior")
    else:
        print(f"\n  FEW subjects ({pct_with_pattern:.0f}%) have consistent personal direction patterns")
        print("  → Most individuals do NOT have a consistent explore→exploit direction")


if __name__ == "__main__":
    main()