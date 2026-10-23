"""
Direction Toward/Away from Archetypes
=====================================
For each subject:
- Calculate change in distance to each archetype (first → last)
- Compare explore vs exploit
- Is there a consistent pattern?

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

ARCHETYPE_NAMES = ['Archetype_1', 'Archetype_2', 'Archetype_3']  # Ordered A1, A2, A3
ARCHETYPE_LABELS = {
    'Archetype_1': 'Arch1',
    'Archetype_2': 'Arch2',
    'Archetype_3': 'Arch3'
}


def main():
    print("\n" + "="*70)
    print("Direction Toward/Away from Archetypes")
    print("="*70)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Total segments: {len(df)}")
    
    # Distance columns
    dist_cols = [f'dist_{name}' for name in ARCHETYPE_NAMES]
    
    # Calculate change in distance for each subject and condition
    results = []
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id]
        
        for condition in ['explore', 'exploit']:
            cond_data = subject_data[subject_data['condition'] == condition].sort_values('segment')
            
            if len(cond_data) < 2:
                continue
            
            first = cond_data.iloc[0]
            last = cond_data.iloc[-1]
            
            result = {
                'subject_id': subject_id,
                'condition': condition,
                'n_segments': len(cond_data)
            }
            
            # Change in distance for each archetype
            for dist_col, arch_name in zip(dist_cols, ARCHETYPE_NAMES):
                delta = last[dist_col] - first[dist_col]
                result[f'delta_{arch_name}'] = delta
                # Positive delta = moved AWAY from archetype
                # Negative delta = moved TOWARD archetype
            
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Separate explore and exploit
    explore = results_df[results_df['condition'] == 'explore'].set_index('subject_id')
    exploit = results_df[results_df['condition'] == 'exploit'].set_index('subject_id')
    
    common_subjects = explore.index.intersection(exploit.index)
    explore = explore.loc[common_subjects]
    exploit = exploit.loc[common_subjects]
    
    print(f"\n  Subjects with both conditions: {len(common_subjects)}")
    
    # ==========================================================================
    # Analysis 1: Group level - do subjects move toward/away from each archetype?
    # ==========================================================================
    print("\n" + "-"*50)
    print("Group Level: Mean change in distance (first → last)")
    print("-"*50)
    print("  (Negative = moved TOWARD archetype)")
    print("  (Positive = moved AWAY from archetype)")
    
    for arch_name in ARCHETYPE_NAMES:
        col = f'delta_{arch_name}'
        
        explore_mean = explore[col].mean()
        explore_std = explore[col].std()
        exploit_mean = exploit[col].mean()
        exploit_std = exploit[col].std()
        
        # t-test vs 0 (did they move significantly toward/away?)
        t_explore, p_explore = stats.ttest_1samp(explore[col], 0)
        t_exploit, p_exploit = stats.ttest_1samp(exploit[col], 0)
        
        print(f"\n  {ARCHETYPE_LABELS[arch_name]}:")
        print(f"    Explore: {explore_mean:+.3f} ± {explore_std:.3f} (t={t_explore:.2f}, p={p_explore:.4f})")
        print(f"    Exploit: {exploit_mean:+.3f} ± {exploit_std:.3f} (t={t_exploit:.2f}, p={p_exploit:.4f})")
    
    # ==========================================================================
    # Analysis 2: Paired comparison - explore vs exploit
    # ==========================================================================
    print("\n" + "-"*50)
    print("Paired Comparison: Explore vs Exploit direction")
    print("-"*50)
    
    paired_results = []
    
    for arch_name in ARCHETYPE_NAMES:
        col = f'delta_{arch_name}'
        
        t_stat, p_val = stats.ttest_rel(explore[col], exploit[col])
        diff = explore[col].mean() - exploit[col].mean()
        
        paired_results.append({
            'archetype': arch_name,
            'explore_mean': explore[col].mean(),
            'exploit_mean': exploit[col].mean(),
            'diff': diff,
            't_stat': t_stat,
            'p_value': p_val
        })
        
        sig = "*" if p_val < 0.05 else ""
        print(f"\n  {ARCHETYPE_LABELS[arch_name]}:")
        print(f"    Explore - Exploit = {diff:+.3f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    # ==========================================================================
    # Analysis 3: Correlation between explore and exploit direction
    # ==========================================================================
    print("\n" + "-"*50)
    print("Correlation: Does explore direction predict exploit direction?")
    print("-"*50)
    
    for arch_name in ARCHETYPE_NAMES:
        col = f'delta_{arch_name}'
        
        r, p = stats.pearsonr(explore[col], exploit[col])
        
        sig = "*" if p < 0.05 else ""
        print(f"\n  {ARCHETYPE_LABELS[arch_name]}:")
        print(f"    r = {r:.3f}, p = {p:.4f} {sig}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Paired comparison (like within_subject_variance)
    for idx, arch_name in enumerate(ARCHETYPE_NAMES):
        ax = axes[0, idx]
        col = f'delta_{arch_name}'
        
        explore_vals = explore[col].values
        exploit_vals = exploit[col].values
        
        # Draw lines for each subject
        for i in range(len(common_subjects)):
            ax.plot([0, 1], [explore_vals[i], exploit_vals[i]], 
                    'o-', color='gray', alpha=0.3, markersize=4)
        
        # Boxplots
        bp = ax.boxplot([explore_vals, exploit_vals], positions=[0, 1], widths=0.3)
        for patch, color in zip([bp['boxes'][0], bp['boxes'][1]], ['green', 'red']):
            patch.set_color(color)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add mean values as text
        ax.text(0, ax.get_ylim()[1]*0.9, f'μ={explore_vals.mean():.2f}', ha='center', fontsize=10, color='green')
        ax.text(1, ax.get_ylim()[1]*0.9, f'μ={exploit_vals.mean():.2f}', ha='center', fontsize=10, color='red')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Explore', 'Exploit'])
        ax.set_ylabel('Change in distance\n(- = toward, + = away)')
        
        p_val = paired_results[idx]['p_value']
        sig = "*" if p_val < 0.05 else "n.s."
        ax.set_title(f'{ARCHETYPE_LABELS[arch_name]}\np={p_val:.4f} {sig}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Correlation scatter
    for idx, arch_name in enumerate(ARCHETYPE_NAMES):
        ax = axes[1, idx]
        col = f'delta_{arch_name}'
        
        explore_vals = explore[col].values
        exploit_vals = exploit[col].values
        
        ax.scatter(explore_vals, exploit_vals, alpha=0.6, c='purple')
        
        # Regression line
        slope, intercept, r, p, _ = stats.linregress(explore_vals, exploit_vals)
        x_line = np.array([explore_vals.min(), explore_vals.max()])
        ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2)
        
        # Zero lines
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Explore Δdist')
        ax.set_ylabel(f'Exploit Δdist')
        ax.set_title(f'{ARCHETYPE_LABELS[arch_name]}\nr={r:.3f}, p={p:.4f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax.text(0.95, 0.95, 'Both away', transform=ax.transAxes, ha='right', va='top', fontsize=8, alpha=0.7)
        ax.text(0.05, 0.05, 'Both toward', transform=ax.transAxes, ha='left', va='bottom', fontsize=8, alpha=0.7)
        ax.text(0.95, 0.05, 'Explore away\nExploit toward', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, alpha=0.7)
        ax.text(0.05, 0.95, 'Explore toward\nExploit away', transform=ax.transAxes, ha='left', va='top', fontsize=8, alpha=0.7)
    
    plt.suptitle('Direction Toward/Away from Each Archetype\n(Negative = moved toward, Positive = moved away)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = RESULTS_DIR / "direction_toward_archetypes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    
    # Save data
    results_df.to_csv(RESULTS_DIR / "direction_toward_archetypes.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'direction_toward_archetypes.csv'}")
    
    plt.show()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n  Do subjects move toward/away from archetypes consistently?")
    for arch_name in ARCHETYPE_NAMES:
        col = f'delta_{arch_name}'
        t_explore, p_explore = stats.ttest_1samp(explore[col], 0)
        t_exploit, p_exploit = stats.ttest_1samp(exploit[col], 0)
        
        explore_dir = "TOWARD" if explore[col].mean() < 0 else "AWAY"
        exploit_dir = "TOWARD" if exploit[col].mean() < 0 else "AWAY"
        
        e_sig = "✓" if p_explore < 0.05 else ""
        x_sig = "✓" if p_exploit < 0.05 else ""
        
        print(f"\n  {ARCHETYPE_LABELS[arch_name]}:")
        print(f"    Explore: {explore_dir} (p={p_explore:.4f}) {e_sig}")
        print(f"    Exploit: {exploit_dir} (p={p_exploit:.4f}) {x_sig}")


if __name__ == "__main__":
    main()
