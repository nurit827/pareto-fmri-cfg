"""
Enrichment Analysis: Explore vs Exploit near Archetypes
========================================================
This script:
1. Pools explore + exploit data
2. For each archetype, takes the closest 10% and 20%
3. Tests if explore/exploit is enriched using hypergeometric test

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"

EXPLORE_FILE = RESULTS_DIR / "cfg_explore_pca.csv"
EXPLOIT_FILE = RESULTS_DIR / "cfg_exploit_pca.csv"

# Archetype labels (corrected to match article)
ARCHETYPE_DISPLAY_NAMES = {
    0: "Archetype 1",
    1: "Archetype 3",  # Swapped
    2: "Archetype 2",  # Swapped
}


def load_data():
    """Load PCA data and archetypes."""
    print("\n" + "="*70)
    print("STEP 1: Loading data")
    print("="*70)

    # Load archetypes
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    archetypes_pca = pca_data['archetypes_pca']

    # Load data
    explore_df = pd.read_csv(EXPLORE_FILE)
    exploit_df = pd.read_csv(EXPLOIT_FILE)

    print(f"  Explore subjects: {len(explore_df)}")
    print(f"  Exploit subjects: {len(exploit_df)}")

    return explore_df, exploit_df, archetypes_pca


def calculate_distances(df, archetypes_pca):
    """Calculate distance from each subject to each archetype."""
    df = df.copy()

    for i in range(3):
        arch_name = ARCHETYPE_DISPLAY_NAMES[i]
        col_name = f"dist_{arch_name.replace(' ', '_')}"

        df[col_name] = np.sqrt(
            (df['PC1'] - archetypes_pca[i, 0])**2 +
            (df['PC2'] - archetypes_pca[i, 1])**2
        )

    return df


def enrichment_analysis(explore_df, exploit_df, percentiles=[10, 20]):
    """
    Calculate enrichment of explore/exploit near each archetype.

    Method (from article):
    1. Pool all data
    2. For each archetype, take closest X%
    3. Count explore vs exploit in proximal bin
    4. Hypergeometric test for enrichment
    """
    print("\n" + "="*70)
    print("STEP 2: Enrichment Analysis")
    print("="*70)

    # Combine data
    explore_df = explore_df.copy()
    exploit_df = exploit_df.copy()
    explore_df['condition'] = 'explore'
    exploit_df['condition'] = 'exploit'

    combined_df = pd.concat([explore_df, exploit_df], ignore_index=True)

    N = len(combined_df)
    K_explore = len(explore_df)
    K_exploit = len(exploit_df)

    print(f"\n  Total: {N}")
    print(f"  Explore: {K_explore} ({100*K_explore/N:.1f}%)")
    print(f"  Exploit: {K_exploit} ({100*K_exploit/N:.1f}%)")

    results = []

    for pct in percentiles:
        print(f"\n  === Closest {pct}% ===")
        n = int(N * pct / 100)

        for i in range(3):
            arch_name = ARCHETYPE_DISPLAY_NAMES[i]
            dist_col = f"dist_{arch_name.replace(' ', '_')}"

            # Sort by distance, take closest n
            combined_sorted = combined_df.sort_values(dist_col)
            proximal = combined_sorted.head(n)

            # Count
            k_explore = (proximal['condition'] == 'explore').sum()
            k_exploit = (proximal['condition'] == 'exploit').sum()

            # Expected under null
            expected_explore = n * K_explore / N
            expected_exploit = n * K_exploit / N

            # Enrichment ratio
            enrich_explore = k_explore / expected_explore
            enrich_exploit = k_exploit / expected_exploit

            # Hypergeometric test
            # P(X >= k) for enrichment
            p_explore = hypergeom.sf(k_explore - 1, N, K_explore, n)
            p_exploit = hypergeom.sf(k_exploit - 1, N, K_exploit, n)

            results.append({
                'Archetype': arch_name,
                'Percentile': pct,
                'n_proximal': n,
                'Explore_observed': k_explore,
                'Explore_expected': round(expected_explore, 1),
                'Explore_enrichment': round(enrich_explore, 2),
                'Explore_p': round(p_explore, 4),
                'Exploit_observed': k_exploit,
                'Exploit_expected': round(expected_exploit, 1),
                'Exploit_enrichment': round(enrich_exploit, 2),
                'Exploit_p': round(p_exploit, 4),
            })

            # Print
            sig_e = "*" if p_explore < 0.05 else ""
            sig_x = "*" if p_exploit < 0.05 else ""

            print(f"\n  {arch_name}:")
            print(f"    Explore: {k_explore}/{n} (expected {expected_explore:.1f}), enrichment={enrich_explore:.2f}, p={p_explore:.4f} {sig_e}")
            print(f"    Exploit: {k_exploit}/{n} (expected {expected_exploit:.1f}), enrichment={enrich_exploit:.2f}, p={p_exploit:.4f} {sig_x}")

    return pd.DataFrame(results)


def create_visualization(results_df, output_dir):
    """Create enrichment bar plot similar to article figure."""

    print("\n" + "="*70)
    print("STEP 3: Creating visualization")
    print("="*70)

    # Use 10% results (or change to 20 if preferred)
    df_10 = results_df[results_df['Percentile'] == 10].copy()

    # Get archetype names in order 1, 2, 3
    arch_order = ['Archetype 1', 'Archetype 2', 'Archetype 3']
    df_10 = df_10.set_index('Archetype').loc[arch_order].reset_index()

    # Colors matching the article (cyan, red, yellow)
    colors = ['#40E0D0', '#DC143C', '#FFD700']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Explore enrichment
    ax1 = axes[0]
    x = np.arange(3)
    bars1 = ax1.bar(x, df_10['Explore_enrichment'], color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Expected (1.0)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['1', '2', '3'])
    ax1.set_xlabel('Archetype', fontsize=12)
    ax1.set_ylabel('Enrichment', fontsize=12)
    ax1.set_title('Explore Enrichment', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(df_10['Explore_enrichment'].max(), df_10['Exploit_enrichment'].max()) * 1.2)

    # Add significance stars
    for i, (_, row) in enumerate(df_10.iterrows()):
        if row['Explore_p'] < 0.05:
            ax1.annotate('*', (i, row['Explore_enrichment'] + 0.05), ha='center', fontsize=16)

    # Plot 2: Exploit enrichment
    ax2 = axes[1]
    bars2 = ax2.bar(x, df_10['Exploit_enrichment'], color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Expected (1.0)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['1', '2', '3'])
    ax2.set_xlabel('Archetype', fontsize=12)
    ax2.set_ylabel('Enrichment', fontsize=12)
    ax2.set_title('Exploit Enrichment', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(df_10['Explore_enrichment'].max(), df_10['Exploit_enrichment'].max()) * 1.2)

    # Add significance stars
    for i, (_, row) in enumerate(df_10.iterrows()):
        if row['Exploit_p'] < 0.05:
            ax2.annotate('*', (i, row['Exploit_enrichment'] + 0.05), ha='center', fontsize=16)

    plt.tight_layout()

    # Save
    output_path = output_dir / "enrichment_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.show()


def main():
    print("\n" + "="*70)
    print("Enrichment Analysis: Explore vs Exploit")
    print("="*70)

    # Check files
    if not EXPLORE_FILE.exists() or not EXPLOIT_FILE.exists():
        print("ERROR: Run project_cfg_to_pca.py first!")
        return

    # Load data
    explore_df, exploit_df, archetypes_pca = load_data()

    # Calculate distances
    explore_df = calculate_distances(explore_df, archetypes_pca)
    exploit_df = calculate_distances(exploit_df, archetypes_pca)

    # Enrichment analysis
    results_df = enrichment_analysis(explore_df, exploit_df, percentiles=[10, 20])

    # Create visualization
    create_visualization(results_df, RESULTS_DIR)

    # Save
    output_path = RESULTS_DIR / "enrichment_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()