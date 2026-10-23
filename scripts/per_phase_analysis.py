"""
Per-Phase Analysis: Explore vs Exploit Dynamics
================================================
This script analyzes per-phase (segment) connectivity matrices to capture dynamics.

Three analyses:
(a) Distribution: Compare scatter of all explore vs exploit segments
(b) Covariance: 3x3 covariance matrix of archetype distances per subject
(c) Direction: Vector from first to last segment per subject

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import glob
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"
QC_FILE = BASE_DIR / "data" / "corr_mats_combined_hemispheres" / "combined_hemispheres" / "CFG MRI Game QC.xlsx"

# Input: per-segment data (GSR version, 17x17 combined hemispheres)
PER_PHASE_DATA_DIR = BASE_DIR / "data" / "per_phase_data"

# Output directory
OUTPUT_DIR = BASE_DIR / "results" / "per_phase"

# Network reordering: CFG -> HCP
CFG_TO_HCP_ORDER = [0, 1, 2, 3, 13, 14, 11, 12, 15, 16, 10, 8, 9, 7, 6, 4, 5]

# Archetype labels (corrected to match article)
ARCHETYPE_DISPLAY_NAMES = {
    0: "Archetype 1",
    1: "Archetype 3",  # Swapped
    2: "Archetype 2",  # Swapped
}

# Filename pattern:
# sub-063_task-CFG_acq-tr1090me5_run-01_space-fsLR_seg-4S456Parcels_stat-mean_timeseries_exploit_seg01_correlation_matrix.csv


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_qc_and_get_valid_subjects(qc_file):
    """
    Load QC file and return valid subject IDs.
    Criteria: Keep=1, second column (MRI artifacts)=0
    """
    print("  Loading QC file...")
    qc_df = pd.read_excel(qc_file)
    
    id_col = qc_df.columns[0]      # Subject ID
    keep_col = qc_df.columns[1]    # Keep
    artifact_col = qc_df.columns[2]  # MRI artifacts
    
    print(f"  Columns: {list(qc_df.columns[:5])}")
    
    valid_mask = (
        (qc_df[keep_col] == 1) & 
        (qc_df[artifact_col] == 0)
    )
    
    valid_subjects = qc_df.loc[valid_mask, id_col].tolist()
    print(f"  Valid subjects: {len(valid_subjects)}")
    return valid_subjects


def reorder_matrix_cfg_to_hcp(cfg_matrix):
    """Reorder 17x17 matrix from CFG to HCP order."""
    reorder_idx = np.array(CFG_TO_HCP_ORDER)
    return cfg_matrix[np.ix_(reorder_idx, reorder_idx)]


def extract_upper_triangle_features(matrix):
    """Extract 153 features from upper triangle (including diagonal)."""
    n = matrix.shape[0]
    features = []
    for i in range(n):
        for j in range(i, n):
            features.append(matrix[i, j])
    return np.array(features)


def load_pca_transformation():
    """Load PCA parameters."""
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    return {
        'mean': pca_data['mean'],
        'components': pca_data['components'],
        'archetypes_pca': pca_data['archetypes_pca']
    }


def project_to_pca(features, pca_params):
    """Project features to PCA space."""
    centered = features - pca_params['mean']
    return centered @ pca_params['components'].T


def calculate_distances(pc1, pc2, archetypes_pca):
    """Calculate distances to each archetype."""
    distances = {}
    for i in range(3):
        arch_name = ARCHETYPE_DISPLAY_NAMES[i]
        dist = np.sqrt((pc1 - archetypes_pca[i, 0])**2 + (pc2 - archetypes_pca[i, 1])**2)
        distances[f"dist_{arch_name.replace(' ', '_')}"] = dist
    return distances


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_phases(data_dir, valid_subjects, pca_params):
    """
    Load all per-segment matrices and project to PCA space.
    
    Filename format:
    sub-XXX_task-CFG_acq-tr1090me5_run-01_space-fsLR_seg-4S456Parcels_stat-mean_timeseries_explore_seg01_correlation_matrix.csv
    sub-XXX_task-CFG_acq-tr1090me5_run-01_space-fsLR_seg-4S456Parcels_stat-mean_timeseries_exploit_seg01_correlation_matrix.csv
    """
    print("\n" + "="*70)
    print("STEP 1: Loading and projecting all segments")
    print("="*70)
    
    all_data = []
    
    for condition in ['explore', 'exploit']:
        # Pattern to match: *_explore_seg*_correlation_matrix.csv or *_exploit_seg*_correlation_matrix.csv
        pattern = str(data_dir / f"*_task-CFG_*_{condition}_seg*_correlation_matrix.csv")
        files = glob.glob(pattern)
        
        print(f"\n  {condition}: found {len(files)} segment files")
        
        for filepath in files:
            filename = Path(filepath).name
            
            # Extract subject ID (e.g., "sub-063")
            subject_match = re.search(r'(sub-\d+)', filename)
            if not subject_match:
                continue
            subject_id = subject_match.group(1)
            
            # Check if subject is valid
            if subject_id not in valid_subjects:
                continue
            
            # Extract segment number (e.g., "seg01" -> 1)
            seg_match = re.search(r'_seg(\d+)_correlation', filename)
            if not seg_match:
                continue
            seg_num = int(seg_match.group(1))
            
            # Load matrix
            try:
                matrix = pd.read_csv(filepath, header=None).values
                if matrix.shape != (17, 17):
                    print(f"    WARNING: {filename} has shape {matrix.shape}, skipping")
                    continue
            except Exception as e:
                print(f"    ERROR loading {filename}: {e}")
                continue
            
            # Reorder and extract features
            matrix_hcp = reorder_matrix_cfg_to_hcp(matrix)
            features = extract_upper_triangle_features(matrix_hcp)
            
            # Project to PCA
            pca_coords = project_to_pca(features, pca_params)
            
            # Calculate distances
            distances = calculate_distances(
                pca_coords[0], pca_coords[1], 
                pca_params['archetypes_pca']
            )
            
            all_data.append({
                'subject_id': subject_id,
                'condition': condition,
                'segment': seg_num,
                'PC1': pca_coords[0],
                'PC2': pca_coords[1],
                **distances
            })
    
    df = pd.DataFrame(all_data)
    print(f"\n  Total segments loaded: {len(df)}")
    print(f"  Explore segments: {len(df[df['condition'] == 'explore'])}")
    print(f"  Exploit segments: {len(df[df['condition'] == 'exploit'])}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")
    
    return df


# =============================================================================
# ANALYSIS A: Distribution Comparison
# =============================================================================

def analysis_a_distribution(df, archetypes_pca, output_dir):
    """
    Compare distribution of explore vs exploit segments in PCA space.
    """
    print("\n" + "="*70)
    print("ANALYSIS A: Distribution Comparison")
    print("="*70)
    
    explore = df[df['condition'] == 'explore']
    exploit = df[df['condition'] == 'exploit']
    
    # Statistics
    print(f"\n  Explore: {len(explore)} segments from {explore['subject_id'].nunique()} subjects")
    print(f"  Exploit: {len(exploit)} segments from {exploit['subject_id'].nunique()} subjects")
    
    print(f"\n  PC1 - Explore: {explore['PC1'].mean():.3f} ± {explore['PC1'].std():.3f}")
    print(f"  PC1 - Exploit: {exploit['PC1'].mean():.3f} ± {exploit['PC1'].std():.3f}")
    print(f"  PC2 - Explore: {explore['PC2'].mean():.3f} ± {explore['PC2'].std():.3f}")
    print(f"  PC2 - Exploit: {exploit['PC2'].mean():.3f} ± {exploit['PC2'].std():.3f}")
    
    # Variance comparison
    print(f"\n  Variance PC1 - Explore: {explore['PC1'].var():.3f}")
    print(f"  Variance PC1 - Exploit: {exploit['PC1'].var():.3f}")
    print(f"  Variance PC2 - Explore: {explore['PC2'].var():.3f}")
    print(f"  Variance PC2 - Exploit: {exploit['PC2'].var():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(explore['PC1'], explore['PC2'], c='green', alpha=0.4, s=30, 
                label=f'Explore (n={len(explore)})', edgecolors='none')
    ax1.scatter(exploit['PC1'], exploit['PC2'], c='red', alpha=0.4, s=30,
                label=f'Exploit (n={len(exploit)})', edgecolors='none')
    
    # Triangle
    triangle = np.vstack([archetypes_pca, archetypes_pca[0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, zorder=5)
    
    # Archetypes
    archetype_colors = ['cyan', 'red', 'yellow']
    for i in range(3):
        ax1.scatter(archetypes_pca[i, 0], archetypes_pca[i, 1], 
                    c=archetype_colors[i], s=200, marker='^', 
                    edgecolors='black', linewidth=2, zorder=10)
        ax1.annotate(ARCHETYPE_DISPLAY_NAMES[i], 
                     (archetypes_pca[i, 0] + 0.2, archetypes_pca[i, 1]),
                     fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('All Segments in PCA Space', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: PC1 Distribution
    ax2 = axes[1]
    ax2.hist(explore['PC1'], bins=30, alpha=0.5, color='green', label='Explore', density=True)
    ax2.hist(exploit['PC1'], bins=30, alpha=0.5, color='red', label='Exploit', density=True)
    ax2.axvline(explore['PC1'].mean(), color='green', linestyle='--', lw=2)
    ax2.axvline(exploit['PC1'].mean(), color='red', linestyle='--', lw=2)
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'PC1 Distribution\nExplore: {explore["PC1"].mean():.2f}±{explore["PC1"].std():.2f}\nExploit: {exploit["PC1"].mean():.2f}±{exploit["PC1"].std():.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PC2 Distribution
    ax3 = axes[2]
    ax3.hist(explore['PC2'], bins=30, alpha=0.5, color='green', label='Explore', density=True)
    ax3.hist(exploit['PC2'], bins=30, alpha=0.5, color='red', label='Exploit', density=True)
    ax3.axvline(explore['PC2'].mean(), color='green', linestyle='--', lw=2)
    ax3.axvline(exploit['PC2'].mean(), color='red', linestyle='--', lw=2)
    ax3.set_xlabel('PC2', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title(f'PC2 Distribution\nExplore: {explore["PC2"].mean():.2f}±{explore["PC2"].std():.2f}\nExploit: {exploit["PC2"].mean():.2f}±{exploit["PC2"].std():.2f}', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "distribution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    
    plt.show()


# =============================================================================
# ANALYSIS B: Covariance Matrices
# =============================================================================

def analysis_b_covariance(df, output_dir):
    """
    Calculate 3x3 covariance matrix of archetype distances per subject per condition.
    Compare covariance patterns between explore and exploit.
    """
    print("\n" + "="*70)
    print("ANALYSIS B: Covariance Matrices")
    print("="*70)
    
    dist_cols = [f"dist_{ARCHETYPE_DISPLAY_NAMES[i].replace(' ', '_')}" for i in range(3)]
    
    cov_results = []
    
    for subject_id in df['subject_id'].unique():
        for condition in ['explore', 'exploit']:
            subset = df[(df['subject_id'] == subject_id) & (df['condition'] == condition)]
            
            if len(subset) < 3:  # Need at least 3 segments for covariance
                continue
            
            # Extract distances
            distances = subset[dist_cols].values  # (n_segments, 3)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(distances, rowvar=False)  # (3, 3)
            
            cov_results.append({
                'subject_id': subject_id,
                'condition': condition,
                'n_segments': len(subset),
                'cov_11': cov_matrix[0, 0],  # Var(dist_Arch1)
                'cov_22': cov_matrix[1, 1],  # Var(dist_Arch3)
                'cov_33': cov_matrix[2, 2],  # Var(dist_Arch2)
                'cov_12': cov_matrix[0, 1],  # Cov(Arch1, Arch3)
                'cov_13': cov_matrix[0, 2],  # Cov(Arch1, Arch2)
                'cov_23': cov_matrix[1, 2],  # Cov(Arch3, Arch2)
            })
    
    cov_df = pd.DataFrame(cov_results)
    
    print(f"\n  Subjects with enough segments: {cov_df['subject_id'].nunique()}")
    print(f"  Explore covariances: {len(cov_df[cov_df['condition'] == 'explore'])}")
    print(f"  Exploit covariances: {len(cov_df[cov_df['condition'] == 'exploit'])}")
    
    # Compare means
    explore_cov = cov_df[cov_df['condition'] == 'explore']
    exploit_cov = cov_df[cov_df['condition'] == 'exploit']
    
    print("\n  Mean covariance values:")
    print(f"  {'Element':<15} {'Explore':<15} {'Exploit':<15} {'Diff':<10}")
    print("  " + "-"*55)
    
    for col in ['cov_11', 'cov_22', 'cov_33', 'cov_12', 'cov_13', 'cov_23']:
        e_mean = explore_cov[col].mean()
        x_mean = exploit_cov[col].mean()
        diff = e_mean - x_mean
        print(f"  {col:<15} {e_mean:<15.4f} {x_mean:<15.4f} {diff:+.4f}")
    
    # Number of subjects for STE calculation
    n_subjects = len(explore_cov)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot diagonal elements (variances) - reorder to A1, A2, A3
    ax1 = axes[0]
    var_cols = ['cov_11', 'cov_33', 'cov_22']  # A1, A2, A3
    var_labels = ['Var(Arch1)', 'Var(Arch2)', 'Var(Arch3)']
    x = np.arange(3)
    width = 0.35

    explore_vars = [explore_cov[col].mean() for col in var_cols]
    exploit_vars = [exploit_cov[col].mean() for col in var_cols]
    explore_stds = [explore_cov[col].std() / np.sqrt(n_subjects) for col in var_cols]
    exploit_stds = [exploit_cov[col].std() / np.sqrt(n_subjects) for col in var_cols]

    bars1 = ax1.bar(x - width/2, explore_vars, width, yerr=explore_stds, label='Explore',
            color='steelblue', capsize=5, alpha=0.8)
    bars2 = ax1.bar(x + width/2, exploit_vars, width, yerr=exploit_stds, label='Exploit',
            color='indianred', capsize=5, alpha=0.8)

    # Add exact values on bars
    for bar, val in zip(bars1, explore_vars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, exploit_vars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(var_labels)
    ax1.set_ylabel('Variance')
    ax1.set_title('Variance of Archetype Distances', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot off-diagonal elements (covariances) - reorder to A1-A2, A1-A3, A2-A3
    ax2 = axes[1]
    cov_cols = ['cov_13', 'cov_12', 'cov_23']  # A1-A2, A1-A3, A2-A3
    cov_labels = ['Cov(A1,A2)', 'Cov(A1,A3)', 'Cov(A2,A3)']

    explore_covs = [explore_cov[col].mean() for col in cov_cols]
    exploit_covs = [exploit_cov[col].mean() for col in cov_cols]
    explore_cov_stds = [explore_cov[col].std() / np.sqrt(n_subjects) for col in cov_cols]
    exploit_cov_stds = [exploit_cov[col].std() / np.sqrt(n_subjects) for col in cov_cols]

    bars3 = ax2.bar(x - width/2, explore_covs, width, yerr=explore_cov_stds, label='Explore',
            color='steelblue', capsize=5, alpha=0.8)
    bars4 = ax2.bar(x + width/2, exploit_covs, width, yerr=exploit_cov_stds, label='Exploit',
            color='indianred', capsize=5, alpha=0.8)

    # Add exact values on bars
    for bar, val in zip(bars3, explore_covs):
        offset = 0.02 if val >= 0 else -0.06
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, f'{val:.2f}',
                ha='center', va=va, fontsize=9)
    for bar, val in zip(bars4, exploit_covs):
        offset = 0.02 if val >= 0 else -0.06
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, f'{val:.2f}',
                ha='center', va=va, fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cov_labels)
    ax2.set_ylabel('Covariance')
    ax2.set_title('Covariance Between Archetype Distances', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot mean covariance matrices as heatmaps
    ax3 = axes[2]

    # Calculate mean covariance matrix for explore (in original order: A1, A3, A2)
    mean_cov_explore_orig = np.array([
        [explore_cov['cov_11'].mean(), explore_cov['cov_12'].mean(), explore_cov['cov_13'].mean()],
        [explore_cov['cov_12'].mean(), explore_cov['cov_22'].mean(), explore_cov['cov_23'].mean()],
        [explore_cov['cov_13'].mean(), explore_cov['cov_23'].mean(), explore_cov['cov_33'].mean()]
    ])

    mean_cov_exploit_orig = np.array([
        [exploit_cov['cov_11'].mean(), exploit_cov['cov_12'].mean(), exploit_cov['cov_13'].mean()],
        [exploit_cov['cov_12'].mean(), exploit_cov['cov_22'].mean(), exploit_cov['cov_23'].mean()],
        [exploit_cov['cov_13'].mean(), exploit_cov['cov_23'].mean(), exploit_cov['cov_33'].mean()]
    ])

    # Reorder to A1, A2, A3 (swap rows/cols 1 and 2)
    reorder = [0, 2, 1]
    mean_cov_explore = mean_cov_explore_orig[np.ix_(reorder, reorder)]
    mean_cov_exploit = mean_cov_exploit_orig[np.ix_(reorder, reorder)]

    # Difference
    diff_cov = mean_cov_explore - mean_cov_exploit

    im = ax3.imshow(diff_cov, cmap='RdBu_r', vmin=-np.abs(diff_cov).max(), vmax=np.abs(diff_cov).max())
    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(['A1', 'A2', 'A3'])
    ax3.set_yticklabels(['A1', 'A2', 'A3'])
    ax3.set_title('Explore - Exploit\n(Covariance Difference)', fontweight='bold')
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # Add values
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{diff_cov[i, j]:.3f}', ha='center', va='center', fontsize=10)

    plt.suptitle(f'Error bars = STE (n={n_subjects})', fontsize=10, y=0.02, color='gray')
    plt.tight_layout()

    output_path = output_dir / "covariance_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    # Save data
    cov_path = output_dir / "covariance_matrices.csv"
    cov_df.to_csv(cov_path, index=False)
    print(f"  Saved: {cov_path}")

    plt.show()

    return cov_df


# =============================================================================
# ANALYSIS C: Direction Vectors
# =============================================================================

def analysis_c_direction(df, archetypes_pca, output_dir):
    """
    Calculate direction vector from first to last segment per subject.
    Compare directions between explore and exploit.
    """
    print("\n" + "="*70)
    print("ANALYSIS C: Direction Vectors (First → Last Segment)")
    print("="*70)

    direction_results = []

    for subject_id in df['subject_id'].unique():
        for condition in ['explore', 'exploit']:
            subset = df[(df['subject_id'] == subject_id) & (df['condition'] == condition)]
            subset = subset.sort_values('segment')

            if len(subset) < 2:  # Need at least 2 segments
                continue

            # First and last segment
            first = subset.iloc[0]
            last = subset.iloc[-1]

            # Direction vector
            dx = last['PC1'] - first['PC1']
            dy = last['PC2'] - first['PC2']

            # Magnitude and angle
            magnitude = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)  # in radians

            direction_results.append({
                'subject_id': subject_id,
                'condition': condition,
                'n_segments': len(subset),
                'start_PC1': first['PC1'],
                'start_PC2': first['PC2'],
                'end_PC1': last['PC1'],
                'end_PC2': last['PC2'],
                'dx': dx,
                'dy': dy,
                'magnitude': magnitude,
                'angle_rad': angle,
                'angle_deg': np.degrees(angle)
            })

    dir_df = pd.DataFrame(direction_results)

    print(f"\n  Subjects with direction vectors: {dir_df['subject_id'].nunique()}")

    explore_dir = dir_df[dir_df['condition'] == 'explore']
    exploit_dir = dir_df[dir_df['condition'] == 'exploit']

    print(f"\n  Explore vectors: {len(explore_dir)}")
    print(f"  Exploit vectors: {len(exploit_dir)}")

    print(f"\n  Mean magnitude - Explore: {explore_dir['magnitude'].mean():.3f} ± {explore_dir['magnitude'].std():.3f}")
    print(f"  Mean magnitude - Exploit: {exploit_dir['magnitude'].mean():.3f} ± {exploit_dir['magnitude'].std():.3f}")

    print(f"\n  Mean angle - Explore: {explore_dir['angle_deg'].mean():.1f}° ± {explore_dir['angle_deg'].std():.1f}°")
    print(f"  Mean angle - Exploit: {exploit_dir['angle_deg'].mean():.1f}° ± {exploit_dir['angle_deg'].std():.1f}°")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Vectors in PCA space
    ax1 = axes[0]

    # Plot triangle
    triangle = np.vstack([archetypes_pca, archetypes_pca[0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, zorder=5)

    # Plot archetypes
    archetype_colors = ['cyan', 'red', 'yellow']
    for i in range(3):
        ax1.scatter(archetypes_pca[i, 0], archetypes_pca[i, 1],
                    c=archetype_colors[i], s=200, marker='^',
                    edgecolors='black', linewidth=2, zorder=10)

    # Plot vectors
    for _, row in explore_dir.iterrows():
        ax1.arrow(row['start_PC1'], row['start_PC2'], row['dx'], row['dy'],
                  head_width=0.1, head_length=0.05, fc='steelblue', ec='steelblue', alpha=0.5)

    for _, row in exploit_dir.iterrows():
        ax1.arrow(row['start_PC1'], row['start_PC2'], row['dx'], row['dy'],
                  head_width=0.1, head_length=0.05, fc='indianred', ec='indianred', alpha=0.5)

    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('Direction Vectors (First → Last Segment)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Legend
    ax1.arrow([], [], [], [], head_width=0.1, fc='steelblue', ec='steelblue', label='Explore')
    ax1.arrow([], [], [], [], head_width=0.1, fc='indianred', ec='indianred', label='Exploit')
    ax1.legend(['Triangle', 'Explore', 'Exploit'], loc='upper left')

    # Plot 2: Magnitude comparison
    ax2 = axes[1]
    data_to_plot = [explore_dir['magnitude'], exploit_dir['magnitude']]
    bp = ax2.boxplot(data_to_plot, labels=['Explore', 'Exploit'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('indianred')
    ax2.set_ylabel('Magnitude (distance traveled)')
    ax2.set_title('Movement Magnitude', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Angle distribution (polar histogram)
    ax3 = axes[2]
    ax3 = plt.subplot(133, projection='polar')

    bins = np.linspace(-np.pi, np.pi, 25)
    ax3.hist(explore_dir['angle_rad'], bins=bins, alpha=0.5, color='steelblue', label='Explore')
    ax3.hist(exploit_dir['angle_rad'], bins=bins, alpha=0.5, color='indianred', label='Exploit')
    ax3.set_title('Direction Angle Distribution', fontweight='bold', pad=20)
    ax3.legend(loc='upper right')

    plt.tight_layout()

    output_path = output_dir / "direction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")

    # Save data
    dir_path = output_dir / "direction_vectors.csv"
    dir_df.to_csv(dir_path, index=False)
    print(f"  Saved: {dir_path}")

    plt.show()

    return dir_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Per-Phase Analysis: Explore vs Exploit Dynamics")
    print("="*70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {OUTPUT_DIR}")

    # Check input directory exists
    if not PER_PHASE_DATA_DIR.exists():
        print(f"\n  ERROR: Per-segment data directory not found: {PER_PHASE_DATA_DIR}")
        print("  Please create this directory and add per-segment connectivity matrices.")
        print("  Expected filename format: sub-XXX_task-CFG_..._explore_seg01_correlation_matrix.csv")
        return

    # Load PCA transformation
    print("\n  Loading PCA transformation...")
    pca_params = load_pca_transformation()
    archetypes_pca = pca_params['archetypes_pca']

    # Get valid subjects
    valid_subjects = load_qc_and_get_valid_subjects(QC_FILE)

    # Load all phases
    df = load_all_phases(PER_PHASE_DATA_DIR, valid_subjects, pca_params)

    if len(df) == 0:
        print("\n  ERROR: No phase data loaded!")
        return

    # Save all segments data
    all_segments_path = OUTPUT_DIR / "all_segments_pca.csv"
    df.to_csv(all_segments_path, index=False)
    print(f"\n  Saved: {all_segments_path}")

    # Analysis A: Distribution
    analysis_a_distribution(df, archetypes_pca, OUTPUT_DIR)

    # Analysis B: Covariance
    analysis_b_covariance(df, OUTPUT_DIR)

    # Analysis C: Direction
    analysis_c_direction(df, archetypes_pca, OUTPUT_DIR)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()