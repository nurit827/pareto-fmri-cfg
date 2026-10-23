"""
Direction Analysis in PCA Space (Per-Subject)
==============================================
For each subject, compute the direction toward each archetype
from THEIR starting position, then project their movement onto it.

1. Are subjects moving toward Arch1? (per-subject direction)
2. Do explore and exploit differ in direction?
3. Same for Arch2 and Arch3

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "per_phase"
DIRECTION_FILE = RESULTS_DIR / "direction_vectors.csv"
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"


def format_p(p_val):
    if p_val < 0.0001:
        return "p<0.0001"
    else:
        return f"p={p_val:.4f}"


def main():
    print("\n" + "="*70)
    print("Direction Analysis — Per-Subject Reference")
    print("="*70)

    # Load direction vectors
    dir_df = pd.read_csv(DIRECTION_FILE)

    # Load archetype positions
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    archetypes = pca_data['archetypes_pca']

    # Archetype positions (with swap correction)
    arch_positions = {
        'Arch1': archetypes[0],  # Impulsive - bottom right
        'Arch2': archetypes[2],  # Goal-directed - left (index 2 = display Arch2)
        'Arch3': archetypes[1],  # Memory - top right (index 1 = display Arch3)
    }

    print(f"\n  Archetype positions:")
    for name, pos in arch_positions.items():
        print(f"    {name}: PC1={pos[0]:.3f}, PC2={pos[1]:.3f}")

    # =================================================================
    # For each subject: project movement onto per-subject direction
    # =================================================================

    for arch_name, arch_pos in arch_positions.items():
        projections = []

        for _, row in dir_df.iterrows():
            # Vector from THIS subject's start to archetype
            to_arch = arch_pos - np.array([row['start_PC1'], row['start_PC2']])
            to_arch_norm = np.linalg.norm(to_arch)

            if to_arch_norm < 1e-10:
                projections.append(0)
                continue

            to_arch_unit = to_arch / to_arch_norm

            # Project subject's movement onto their personal "toward archetype" direction
            movement = np.array([row['dx'], row['dy']])
            proj = np.dot(movement, to_arch_unit)
            projections.append(proj)

        dir_df[f'proj_toward_{arch_name}'] = projections

    # Split by condition, match subjects
    explore = dir_df[dir_df['condition'] == 'explore'].set_index('subject_id')
    exploit = dir_df[dir_df['condition'] == 'exploit'].set_index('subject_id')
    common = explore.index.intersection(exploit.index)
    explore = explore.loc[common]
    exploit = exploit.loc[common]

    n = len(common)
    print(f"\n  Subjects with both conditions: {n}")

    # =================================================================
    # TEST: For each archetype — net movement and condition difference
    # =================================================================

    results = []

    for arch_name in ['Arch1', 'Arch2', 'Arch3']:
        col = f'proj_toward_{arch_name}'

        explore_vals = explore[col].values
        exploit_vals = exploit[col].values

        # One-sample t-test: is there net movement?
        t_exp, p_exp = stats.ttest_1samp(explore_vals, 0)
        t_ext, p_ext = stats.ttest_1samp(exploit_vals, 0)

        # Paired: do conditions differ?
        t_pair, p_pair = stats.ttest_rel(explore_vals, exploit_vals)

        results.append({
            'archetype': arch_name,
            'explore_mean': explore_vals.mean(),
            'explore_std': explore_vals.std(),
            'explore_p': p_exp,
            'exploit_mean': exploit_vals.mean(),
            'exploit_std': exploit_vals.std(),
            'exploit_p': p_ext,
            'paired_p': p_pair,
        })

        print(f"\n  --- {arch_name} ---")
        print(f"  (positive = toward {arch_name}, negative = away)")
        print(f"    Explore: {explore_vals.mean():+.4f} ± {explore_vals.std():.4f}, {format_p(p_exp)} {'*' if p_exp < 0.05 else ''}")
        print(f"    Exploit: {exploit_vals.mean():+.4f} ± {exploit_vals.std():.4f}, {format_p(p_ext)} {'*' if p_ext < 0.05 else ''}")
        print(f"    Paired:  diff={explore_vals.mean() - exploit_vals.mean():+.4f}, {format_p(p_pair)} {'*' if p_pair < 0.05 else ''}")

        # Count toward vs away
        exp_toward = (explore_vals > 0).sum()
        ext_toward = (exploit_vals > 0).sum()
        print(f"    Explore: {exp_toward}/{n} toward, {n-exp_toward}/{n} away")
        print(f"    Exploit: {ext_toward}/{n} toward, {n-ext_toward}/{n} away")

    # =================================================================
    # VISUALIZATION
    # =================================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-Subject Direction Toward Each Archetype\n(each subject's movement projected onto their personal direction to archetype)",
                 fontsize=13, fontweight='bold')

    arch_names = ['Arch1', 'Arch2', 'Arch3']

    # Top row: paired box plots
    for idx, arch_name in enumerate(arch_names):
        ax = axes[0, idx]
        col = f'proj_toward_{arch_name}'

        explore_vals = explore[col].values
        exploit_vals = exploit[col].values

        for i in range(n):
            ax.plot([0, 1], [explore_vals[i], exploit_vals[i]],
                     'o-', color='gray', alpha=0.2, markersize=3)

        bp = ax.boxplot([explore_vals, exploit_vals], positions=[0, 1], widths=0.3, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('indianred')
        bp['boxes'][1].set_alpha(0.5)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

        r = results[idx]
        exp_sig = '*' if r['explore_p'] < 0.05 else 'n.s.'
        ext_sig = '*' if r['exploit_p'] < 0.05 else 'n.s.'
        pair_sig = '*' if r['paired_p'] < 0.05 else 'n.s.'

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Explore', 'Exploit'])
        ax.set_ylabel(f'Projection (+ = toward {arch_name})')
        ax.set_title(f'{arch_name}\nExplore {format_p(r["explore_p"])} {exp_sig}, '
                     f'Exploit {format_p(r["exploit_p"])} {ext_sig}\n'
                     f'Paired {format_p(r["paired_p"])} {pair_sig}',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Bottom row: scatter explore vs exploit projection
    for idx, arch_name in enumerate(arch_names):
        ax = axes[1, idx]
        col = f'proj_toward_{arch_name}'

        explore_vals = explore[col].values
        exploit_vals = exploit[col].values

        ax.scatter(explore_vals, exploit_vals, alpha=0.6, c='purple')

        # Fit line
        z = np.polyfit(explore_vals, exploit_vals, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(explore_vals.min(), explore_vals.max(), 100)
        ax.plot(x_line, p_line(x_line), 'r--', alpha=0.8)

        r_corr, p_corr = stats.pearsonr(explore_vals, exploit_vals)

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(f'Explore projection toward {arch_name}')
        ax.set_ylabel(f'Exploit projection toward {arch_name}')
        ax.set_title(f'{arch_name}\nr={r_corr:.3f}, {format_p(p_corr)} {"*" if p_corr < 0.05 else "n.s."}',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = RESULTS_DIR / "pca_direction_per_subject.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    plt.show()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for r in results:
        arch = r['archetype']
        print(f"\n  {arch}:")
        if r['explore_p'] < 0.05 or r['exploit_p'] < 0.05:
            print(f"    Significant net movement detected!")
        else:
            print(f"    No net movement toward {arch} in either condition")
        if r['paired_p'] < 0.05:
            print(f"    Conditions DIFFER in direction toward {arch}")
        else:
            print(f"    No difference between conditions")


if __name__ == "__main__":
    main()