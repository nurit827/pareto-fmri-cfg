"""
Analyze Distances to Archetypes
================================
This script:
1. Loads projected PCA data for explore and exploit conditions
2. Calculates Euclidean distance from each subject to each archetype
3. Compares distances between explore vs exploit groups per archetype
4. Generates visualizations

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"

# Output files from projection script
EXPLORE_FILE = RESULTS_DIR / "cfg_explore_pca.csv"
EXPLOIT_FILE = RESULTS_DIR / "cfg_exploit_pca.csv"

# =============================================================================
# ARCHETYPE LABELS (corrected to match article)
# =============================================================================
# Original positions: Arch1 (bottom-right), Arch2 (top-right), Arch3 (left)
# Article labels:     Arch1 (bottom-right), Arch3 (top-right), Arch2 (left)
# We swap 2↔3 for display purposes

ARCHETYPE_DISPLAY_NAMES = {
    0: "Archetype 1",  # Original Arch 1 stays as 1
    1: "Archetype 3",  # Original Arch 2 becomes 3 (top-right)
    2: "Archetype 2",  # Original Arch 3 becomes 2 (left)
}


def load_data():
    """Load projected PCA data and archetypes."""
    print("\n" + "=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    # Load archetypes
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    archetypes_pca = pca_data['archetypes_pca']

    print(f"  Archetypes loaded:")
    for i in range(3):
        print(f"    {ARCHETYPE_DISPLAY_NAMES[i]}: PC1={archetypes_pca[i, 0]:.3f}, PC2={archetypes_pca[i, 1]:.3f}")

    # Load explore data
    explore_df = pd.read_csv(EXPLORE_FILE)
    print(f"\n  Explore subjects: {len(explore_df)}")

    # Load exploit data
    exploit_df = pd.read_csv(EXPLOIT_FILE)
    print(f"  Exploit subjects: {len(exploit_df)}")

    return explore_df, exploit_df, archetypes_pca


def calculate_distances(df, archetypes_pca):
    """
    Calculate Euclidean distance from each subject to each archetype.
    Returns DataFrame with distance columns added.
    """
    df = df.copy()

    for i in range(3):
        arch_name = ARCHETYPE_DISPLAY_NAMES[i]
        col_name = f"dist_{arch_name.replace(' ', '_')}"

        # Euclidean distance
        df[col_name] = np.sqrt(
            (df['PC1'] - archetypes_pca[i, 0]) ** 2 +
            (df['PC2'] - archetypes_pca[i, 1]) ** 2
        )

    return df


def compare_groups(explore_df, exploit_df):
    """
    Compare distances between explore and exploit groups for each archetype.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Comparison (Explore vs Exploit)")
    print("=" * 70)

    results = []

    print(f"\n  {'Archetype':<15} {'Explore':<18} {'Exploit':<18} {'Difference':<12}")
    print("  " + "-" * 63)

    for i in range(3):
        arch_name = ARCHETYPE_DISPLAY_NAMES[i]
        col_name = f"dist_{arch_name.replace(' ', '_')}"

        explore_dist = explore_df[col_name]
        exploit_dist = exploit_df[col_name]

        explore_mean = explore_dist.mean()
        explore_std = explore_dist.std()
        exploit_mean = exploit_dist.mean()
        exploit_std = exploit_dist.std()
        difference = explore_mean - exploit_mean

        results.append({
            'Archetype': arch_name,
            'Explore_mean': explore_mean,
            'Explore_std': explore_std,
            'Exploit_mean': exploit_mean,
            'Exploit_std': exploit_std,
            'Difference': difference
        })

        print(
            f"  {arch_name:<15} {explore_mean:.3f} ± {explore_std:.3f}    {exploit_mean:.3f} ± {exploit_std:.3f}    {difference:+.3f}")

    return pd.DataFrame(results)


def create_visualizations(explore_df, exploit_df, archetypes_pca, output_dir):
    """Create box plots and violin plots comparing groups."""
    print("\n" + "=" * 70)
    print("STEP 3: Creating visualizations")
    print("=" * 70)

    # Prepare data for plotting
    dist_columns = [f"dist_{ARCHETYPE_DISPLAY_NAMES[i].replace(' ', '_')}" for i in range(3)]
    archetype_names = [ARCHETYPE_DISPLAY_NAMES[i] for i in range(3)]

    # =========================================================================
    # Figure 1: Box plots
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (col, arch_name) in enumerate(zip(dist_columns, archetype_names)):
        ax = axes1[idx]

        data_to_plot = [explore_df[col], exploit_df[col]]
        bp = ax.boxplot(data_to_plot, labels=['Explore', 'Exploit'], patch_artist=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('indianred')

        ax.set_ylabel('Distance to Archetype')
        ax.set_title(arch_name)
        ax.grid(True, alpha=0.3)

        # Add individual points
        for i, (data, color) in enumerate(zip(data_to_plot, ['steelblue', 'indianred'])):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=20, c=color, edgecolors='none')

    fig1.suptitle('Distance to Archetypes: Explore vs Exploit', fontsize=14, fontweight='bold')
    plt.tight_layout()

    boxplot_path = output_dir / "archetype_distances_boxplot.png"
    fig1.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {boxplot_path}")

    # =========================================================================
    # Figure 2: Violin plots
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (col, arch_name) in enumerate(zip(dist_columns, archetype_names)):
        ax = axes2[idx]

        data_to_plot = [explore_df[col].values, exploit_df[col].values]

        vp = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)

        # Color the violins
        colors = ['steelblue', 'indianred']
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Explore', 'Exploit'])
        ax.set_ylabel('Distance to Archetype')
        ax.set_title(arch_name)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('Distance to Archetypes: Explore vs Exploit (Violin Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    violin_path = output_dir / "archetype_distances_violin.png"
    fig2.savefig(violin_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {violin_path}")

    # =========================================================================
    # Figure 3: Bar plot with error bars
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    explore_means = [explore_df[col].mean() for col in dist_columns]
    explore_stds = [explore_df[col].std() for col in dist_columns]
    exploit_means = [exploit_df[col].mean() for col in dist_columns]
    exploit_stds = [exploit_df[col].std() for col in dist_columns]

    bars1 = ax3.bar(x - width / 2, explore_means, width, yerr=explore_stds,
                    label='Explore', color='steelblue', capsize=5, alpha=0.8)
    bars2 = ax3.bar(x + width / 2, exploit_means, width, yerr=exploit_stds,
                    label='Exploit', color='indianred', capsize=5, alpha=0.8)

    ax3.set_ylabel('Mean Distance to Archetype')
    ax3.set_title('Mean Distance to Archetypes by Condition')
    ax3.set_xticks(x)
    ax3.set_xticklabels(archetype_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    barplot_path = output_dir / "archetype_distances_barplot.png"
    fig3.savefig(barplot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {barplot_path}")

    # =========================================================================
    # Figure 4: PCA space with archetype distances visualized
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 8))

    # Plot data points
    ax4.scatter(explore_df['PC1'], explore_df['PC2'],
                c='steelblue', alpha=0.6, s=50, label=f'Explore (n={len(explore_df)})',
                edgecolors='darkblue', linewidth=0.5)
    ax4.scatter(exploit_df['PC1'], exploit_df['PC2'],
                c='indianred', alpha=0.6, s=50, label=f'Exploit (n={len(exploit_df)})',
                edgecolors='darkred', linewidth=0.5)

    # Plot triangle
    triangle = np.vstack([archetypes_pca, archetypes_pca[0]])
    ax4.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, zorder=5)

    # Plot archetypes
    archetype_colors = ['cyan', 'red', 'yellow']
    for i in range(3):
        ax4.scatter(archetypes_pca[i, 0], archetypes_pca[i, 1],
                    c=archetype_colors[i], s=300, marker='^',
                    edgecolors='black', linewidth=2, zorder=10)

        offset = [(0.2, -0.4), (0.2, 0.3), (-0.5, 0.1)]
        ax4.annotate(ARCHETYPE_DISPLAY_NAMES[i],
                     (archetypes_pca[i, 0], archetypes_pca[i, 1]),
                     xytext=(archetypes_pca[i, 0] + offset[i][0],
                             archetypes_pca[i, 1] + offset[i][1]),
                     fontsize=11, fontweight='bold')

    ax4.set_xlabel('PC1', fontsize=12)
    ax4.set_ylabel('PC2', fontsize=12)
    ax4.set_title('CFG Data in PCA Space with Archetypes', fontsize=14)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')

    pca_path = output_dir / "pca_space_with_archetypes.png"
    fig4.savefig(pca_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {pca_path}")

    plt.show()


def save_results(explore_df, exploit_df, stats_df, output_dir):
    """Save all results to files."""
    print("\n" + "=" * 70)
    print("STEP 4: Saving results")
    print("=" * 70)

    # Save distance data
    explore_path = output_dir / "cfg_explore_with_distances.csv"
    explore_df.to_csv(explore_path, index=False)
    print(f"  Saved: {explore_path}")

    exploit_path = output_dir / "cfg_exploit_with_distances.csv"
    exploit_df.to_csv(exploit_path, index=False)
    print(f"  Saved: {exploit_path}")

    # Save comparison summary
    stats_path = output_dir / "archetype_distance_comparison.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")


def main():
    print("\n" + "=" * 70)
    print("Archetype Distance Analysis")
    print("=" * 70)

    # Check files exist
    if not EXPLORE_FILE.exists():
        print(f"ERROR: Explore file not found: {EXPLORE_FILE}")
        print("Please run project_cfg_to_pca.py first!")
        return

    if not EXPLOIT_FILE.exists():
        print(f"ERROR: Exploit file not found: {EXPLOIT_FILE}")
        print("Please run project_cfg_to_pca.py first!")
        return

    # Step 1: Load data
    explore_df, exploit_df, archetypes_pca = load_data()

    # Step 2: Calculate distances
    explore_df = calculate_distances(explore_df, archetypes_pca)
    exploit_df = calculate_distances(exploit_df, archetypes_pca)

    # Step 3: Statistical comparison
    stats_df = compare_groups(explore_df, exploit_df)

    # Step 4: Create visualizations
    create_visualizations(explore_df, exploit_df, archetypes_pca, RESULTS_DIR)

    # Step 5: Save results
    save_results(explore_df, exploit_df, stats_df, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()