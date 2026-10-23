"""
Arch1-Arch2 Axis Analysis
=========================
Test whether the negative correlations in direction toward Arch1 and Arch2
reflect movement along a single Arch1-Arch2 axis.

Two approaches:
  Approach 1 (Simple): Correlate delta_Arch1 with delta_Arch2 per condition
  Approach 2 (Regression): Fit linear trend across ALL segments, then correlate slopes

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
SEGMENTS_FILE = RESULTS_DIR / "all_segments_pca.csv"
DIRECTION_FILE = RESULTS_DIR / "direction_toward_archetypes.csv"

ARCHETYPE_NAMES = ['Archetype_1', 'Archetype_2', 'Archetype_3']
ARCHETYPE_LABELS = {'Archetype_1': 'Arch1', 'Archetype_2': 'Arch2', 'Archetype_3': 'Arch3'}


def format_p(p_val):
    """Format p-value: show '<0.0001' for very small values."""
    if p_val < 0.0001:
        return "p<0.0001"
    else:
        return f"p={p_val:.4f}"


# =============================================================================
# APPROACH 1: Simple delta (first → last)
# =============================================================================

def approach1_simple_delta():
    """Correlate delta_Arch1 with delta_Arch2 using first-to-last segment."""
    print("\n" + "="*70)
    print("APPROACH 1: Simple Delta (First → Last)")
    print("="*70)

    dir_df = pd.read_csv(DIRECTION_FILE)

    results = []

    for condition in ['explore', 'exploit']:
        cond_df = dir_df[dir_df['condition'] == condition]

        delta_A1 = cond_df['delta_Archetype_1'].values
        delta_A2 = cond_df['delta_Archetype_2'].values
        delta_A3 = cond_df['delta_Archetype_3'].values

        # Arch1 vs Arch2
        r_12, p_12 = stats.pearsonr(delta_A1, delta_A2)
        # Arch1 vs Arch3
        r_13, p_13 = stats.pearsonr(delta_A1, delta_A3)
        # Arch2 vs Arch3
        r_23, p_23 = stats.pearsonr(delta_A2, delta_A3)

        print(f"\n  {condition.upper()} (n={len(cond_df)}):")
        print(f"    delta_Arch1 vs delta_Arch2: r={r_12:.3f}, {format_p(p_12)} {'*' if p_12 < 0.05 else ''}")
        print(f"    delta_Arch1 vs delta_Arch3: r={r_13:.3f}, {format_p(p_13)} {'*' if p_13 < 0.05 else ''}")
        print(f"    delta_Arch2 vs delta_Arch3: r={r_23:.3f}, {format_p(p_23)} {'*' if p_23 < 0.05 else ''}")

        results.append({
            'condition': condition,
            'method': 'simple_delta',
            'r_A1_A2': r_12, 'p_A1_A2': p_12,
            'r_A1_A3': r_13, 'p_A1_A3': p_13,
            'r_A2_A3': r_23, 'p_A2_A3': p_23,
        })

    return results, dir_df


# =============================================================================
# APPROACH 2: Linear regression slope across all segments
# =============================================================================

def approach2_regression_slope():
    """Fit linear trend of distance vs segment number, then correlate slopes."""
    print("\n" + "="*70)
    print("APPROACH 2: Linear Regression Slope (All Segments)")
    print("="*70)

    df = pd.read_csv(SEGMENTS_FILE)
    dist_cols = [f'dist_{name}' for name in ARCHETYPE_NAMES]

    # Calculate slope for each subject x condition x archetype
    slope_data = []

    for subject_id in df['subject_id'].unique():
        for condition in ['explore', 'exploit']:
            subset = df[(df['subject_id'] == subject_id) & (df['condition'] == condition)]
            subset = subset.sort_values('segment')

            if len(subset) < 3:  # Need at least 3 points for meaningful regression
                continue

            row = {
                'subject_id': subject_id,
                'condition': condition,
                'n_segments': len(subset)
            }

            for dist_col, arch_name in zip(dist_cols, ARCHETYPE_NAMES):
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    subset['segment'].values, subset[dist_col].values
                )
                row[f'slope_{arch_name}'] = slope
                row[f'r2_{arch_name}'] = r_value**2

            slope_data.append(row)

    slope_df = pd.DataFrame(slope_data)

    print(f"\n  Subjects with >= 3 segments in both conditions:")

    results = []

    for condition in ['explore', 'exploit']:
        cond_df = slope_df[slope_df['condition'] == condition]

        slope_A1 = cond_df['slope_Archetype_1'].values
        slope_A2 = cond_df['slope_Archetype_2'].values
        slope_A3 = cond_df['slope_Archetype_3'].values

        # Arch1 vs Arch2
        r_12, p_12 = stats.pearsonr(slope_A1, slope_A2)
        # Arch1 vs Arch3
        r_13, p_13 = stats.pearsonr(slope_A1, slope_A3)
        # Arch2 vs Arch3
        r_23, p_23 = stats.pearsonr(slope_A2, slope_A3)

        print(f"\n  {condition.upper()} (n={len(cond_df)}):")
        print(f"    slope_Arch1 vs slope_Arch2: r={r_12:.3f}, {format_p(p_12)} {'*' if p_12 < 0.05 else ''}")
        print(f"    slope_Arch1 vs slope_Arch3: r={r_13:.3f}, {format_p(p_13)} {'*' if p_13 < 0.05 else ''}")
        print(f"    slope_Arch2 vs slope_Arch3: r={r_23:.3f}, {format_p(p_23)} {'*' if p_23 < 0.05 else ''}")

        results.append({
            'condition': condition,
            'method': 'regression_slope',
            'r_A1_A2': r_12, 'p_A1_A2': p_12,
            'r_A1_A3': r_13, 'p_A1_A3': p_13,
            'r_A2_A3': r_23, 'p_A2_A3': p_23,
        })

    return results, slope_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(dir_df, slope_df):
    """Create comparison plots for both approaches."""
    print("\n" + "="*70)
    print("Creating Visualization")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Correlation Between Movement Toward Arch1 and Arch2',
                 fontsize=14, fontweight='bold')

    # --- Top row: Approach 1 (Simple delta) ---
    for idx, condition in enumerate(['explore', 'exploit']):
        ax = axes[0, idx]
        cond_df = dir_df[dir_df['condition'] == condition]

        x = cond_df['delta_Archetype_1'].values
        y = cond_df['delta_Archetype_2'].values

        ax.scatter(x, y, alpha=0.6, c='steelblue' if condition == 'explore' else 'indianred')

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.8)

        r, p_val = stats.pearsonr(x, y)
        sig = "*" if p_val < 0.05 else "n.s."

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('delta_Arch1')
        ax.set_ylabel('delta_Arch2')
        ax.set_title(f'First-to-Last Change — {condition.upper()}\nr={r:.3f}, {format_p(p_val)} {sig}',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    # --- Bottom row: Approach 2 (Regression slope) ---
    for idx, condition in enumerate(['explore', 'exploit']):
        ax = axes[1, idx]
        cond_df = slope_df[slope_df['condition'] == condition]

        x = cond_df['slope_Archetype_1'].values
        y = cond_df['slope_Archetype_2'].values

        ax.scatter(x, y, alpha=0.6, c='steelblue' if condition == 'explore' else 'indianred')

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.8)

        r, p_val = stats.pearsonr(x, y)
        sig = "*" if p_val < 0.05 else "n.s."

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('slope_Arch1')
        ax.set_ylabel('slope_Arch2')
        ax.set_title(f'Linear Trend (All Segments) — {condition.upper()}\nr={r:.3f}, {format_p(p_val)} {sig}',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = RESULTS_DIR / "arch1_arch2_axis_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Arch1-Arch2 Axis Analysis")
    print("="*70)

    # Approach 1
    results1, dir_df = approach1_simple_delta()

    # Approach 2
    results2, slope_df = approach2_regression_slope()

    # Save combined results
    all_results = pd.DataFrame(results1 + results2)
    csv_path = RESULTS_DIR / "arch1_arch2_axis_results.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"\n  Saved results: {csv_path}")

    # Save slope data
    slope_path = RESULTS_DIR / "regression_slopes.csv"
    slope_df.to_csv(slope_path, index=False)
    print(f"  Saved slopes: {slope_path}")

    # Visualization
    create_visualization(dir_df, slope_df)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n  If Arch1 and Arch2 deltas/slopes are negatively correlated,")
    print("  it means subjects move along a single Arch1 <-> Arch2 axis.")
    print("  (Moving toward Impulsive = moving away from Goal-directed)")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()