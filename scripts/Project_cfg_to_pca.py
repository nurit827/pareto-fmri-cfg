"""
Project CFG fMRI Data onto Existing PCA Space
==============================================
This script:
1. Reads QC file and filters valid subjects
2. Loads explore/exploit correlation matrices for valid subjects
3. Reorders networks from CFG order to HCP order
4. Extracts 153 features (upper triangle) in correct order
5. Applies PCA transformation
6. Visualizes results on the Pareto triangle

Run from: ~/Pareto-fMRI-main/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION - Update these paths if needed
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CFG_DATA_DIR = BASE_DIR / "data" / "corr_mats_combined_hemispheres" / "combined_hemispheres"
QC_FILE = CFG_DATA_DIR / "CFG MRI Game QC.xlsx"
PCA_FILE = BASE_DIR / "data" / "pca_transformation.npz"
OUTPUT_DIR = BASE_DIR / "results"

# =============================================================================
# NETWORK MAPPING: CFG order -> HCP order
# =============================================================================

# CFG network order (from JSON):
# 0: VisCent, 1: VisPeri, 2: SomMotA, 3: SomMotB, 4: DefaultA, 5: DefaultB,
# 6: DefaultC, 7: TempPar, 8: ContA, 9: ContB, 10: ContC, 11: SalVentAttnA,
# 12: SalVentAttnB, 13: DorsAttnA, 14: DorsAttnB, 15: LimbicA, 16: LimbicB

# HCP network order (from features.csv):
# 0: Visual Central (Visual A), 1: Visual Peripheral (Visual B),
# 2: Somatomotor A, 3: Somatomotor B, 4: Dorsal Attention A, 5: Dorsal Attention B,
# 6: Salience/Ventral Attention A, 7: Salience/Ventral Attention B,
# 8: Limbic A, 9: Limbic B, 10: Control C, 11: Control A, 12: Control B,
# 13: Temporal Parietal, 14: Default C, 15: Default A, 16: Default B

# Mapping: For each HCP position, which CFG index to use
CFG_TO_HCP_ORDER = [
    0,  # HCP 0 (Visual Central) <- CFG 0 (VisCent)
    1,  # HCP 1 (Visual Peripheral) <- CFG 1 (VisPeri)
    2,  # HCP 2 (Somatomotor A) <- CFG 2 (SomMotA)
    3,  # HCP 3 (Somatomotor B) <- CFG 3 (SomMotB)
    13,  # HCP 4 (Dorsal Attention A) <- CFG 13 (DorsAttnA)
    14,  # HCP 5 (Dorsal Attention B) <- CFG 14 (DorsAttnB)
    11,  # HCP 6 (Salience/Ventral Attention A) <- CFG 11 (SalVentAttnA)
    12,  # HCP 7 (Salience/Ventral Attention B) <- CFG 12 (SalVentAttnB)
    15,  # HCP 8 (Limbic A) <- CFG 15 (LimbicA)
    16,  # HCP 9 (Limbic B) <- CFG 16 (LimbicB)
    10,  # HCP 10 (Control C) <- CFG 10 (ContC)
    8,  # HCP 11 (Control A) <- CFG 8 (ContA)
    9,  # HCP 12 (Control B) <- CFG 9 (ContB)
    7,  # HCP 13 (Temporal Parietal) <- CFG 7 (TempPar)
    6,  # HCP 14 (Default C) <- CFG 6 (DefaultC)
    4,  # HCP 15 (Default A) <- CFG 4 (DefaultA)
    5,  # HCP 16 (Default B) <- CFG 5 (DefaultB)
]


def load_qc_and_get_valid_subjects(qc_file):
    """
    Load QC Excel file and return list of valid subject IDs.
    Criteria: Keep=1, Scanned_during_gradient_artifacts=0, third column=1
    """
    print("\n" + "=" * 70)
    print("STEP 1: Loading QC file and filtering subjects")
    print("=" * 70)

    qc_df = pd.read_excel(qc_file)
    print(f"  Total subjects in QC file: {len(qc_df)}")
    print(f"  Columns: {list(qc_df.columns)}")

    # Get column names (adjust if needed based on actual column names)
    id_col = qc_df.columns[0]  # Should be 'ID' or similar
    keep_col = qc_df.columns[1]  # 'Keep'
    artifact_col = qc_df.columns[2]  # 'Scanned_during_gradient_artifacts'
    usable_col = qc_df.columns[3]  # Third criterion column

    print(f"\n  Using columns:")
    print(f"    ID column: '{id_col}'")
    print(f"    Keep column: '{keep_col}'")
    print(f"    Artifact column: '{artifact_col}'")
    print(f"    Usable column: '{usable_col}'")

    # Apply filters
    valid_mask = (
            (qc_df[keep_col] == 1) &
            (qc_df[artifact_col] == 0) &
            (qc_df[usable_col] == 1)
    )

    valid_subjects = qc_df.loc[valid_mask, id_col].tolist()
    print(f"\n  Valid subjects: {len(valid_subjects)}")
    print(f"  Subject IDs: {valid_subjects[:10]}{'...' if len(valid_subjects) > 10 else ''}")

    return valid_subjects


def reorder_matrix_cfg_to_hcp(cfg_matrix):
    """
    Reorder a 17x17 correlation matrix from CFG network order to HCP network order.
    """
    # Create reordering indices
    reorder_idx = np.array(CFG_TO_HCP_ORDER)

    # Reorder both rows and columns
    hcp_matrix = cfg_matrix[np.ix_(reorder_idx, reorder_idx)]

    return hcp_matrix


def extract_upper_triangle_features(matrix):
    """
    Extract upper triangle features (including diagonal) from 17x17 matrix.
    Returns 153 features in the same order as HCP features.csv

    The HCP features include diagonal elements (network self-correlations).
    Order: (0,0), (0,1), (0,2), ..., (0,16), (1,1), (1,2), ..., (16,16)
    """
    n = matrix.shape[0]
    features = []

    # Extract upper triangle including diagonal (row by row)
    for i in range(n):
        for j in range(i, n):
            features.append(matrix[i, j])

    return np.array(features)


def load_correlation_matrix(filepath):
    """Load a correlation matrix CSV file."""
    df = pd.read_csv(filepath, header=None)
    return df.values


def process_subject(subject_id, data_dir, condition):
    """
    Process a single subject's correlation matrix.
    Returns 153 features in HCP order, or None if file not found.
    """
    # Construct filename
    filename = f"{subject_id}_task-CFG_acq-tr1090me5_run-01_space-fsLR_seg-4S456Parcels_stat-mean_timeseries_{condition}_correlation_matrix.csv"
    filepath = data_dir / filename

    if not filepath.exists():
        return None

    # Load CFG matrix (17x17)
    cfg_matrix = load_correlation_matrix(filepath)

    if cfg_matrix.shape != (17, 17):
        print(f"    WARNING: {subject_id} {condition} has shape {cfg_matrix.shape}, expected (17, 17)")
        return None

    # Reorder to HCP order
    hcp_matrix = reorder_matrix_cfg_to_hcp(cfg_matrix)

    # Extract 153 features
    features = extract_upper_triangle_features(hcp_matrix)

    return features


def load_and_process_all_subjects(valid_subjects, data_dir):
    """
    Load and process all valid subjects for both conditions.
    Returns dict with 'explore' and 'exploit' arrays.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Loading and processing correlation matrices")
    print("=" * 70)

    results = {
        'explore': {'subject_ids': [], 'features': []},
        'exploit': {'subject_ids': [], 'features': []}
    }

    for condition in ['explore', 'exploit']:
        print(f"\n  Processing {condition} condition...")
        success_count = 0

        for subject_id in valid_subjects:
            features = process_subject(subject_id, data_dir, condition)

            if features is not None:
                results[condition]['subject_ids'].append(subject_id)
                results[condition]['features'].append(features)
                success_count += 1

        # Convert to numpy array
        if results[condition]['features']:
            results[condition]['features'] = np.array(results[condition]['features'])

        print(f"    Successfully loaded: {success_count}/{len(valid_subjects)} subjects")

    return results


def apply_pca_transformation(features, pca_file):
    """
    Apply saved PCA transformation to new features.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Applying PCA transformation")
    print("=" * 70)

    # Load PCA parameters
    pca_data = np.load(pca_file, allow_pickle=True)
    mean = pca_data['mean']
    components = pca_data['components']
    archetypes_pca = pca_data['archetypes_pca']

    print(f"  PCA mean shape: {mean.shape}")
    print(f"  PCA components shape: {components.shape}")
    print(f"  Input features shape: {features.shape}")

    # Center the data
    features_centered = features - mean

    # Project to PC space
    features_pca = features_centered @ components.T

    print(f"  Output PCA shape: {features_pca.shape}")

    return features_pca, archetypes_pca


def create_visualization(explore_pca, exploit_pca, archetypes_pca, output_dir):
    """
    Create visualization of CFG data projected onto PCA space with Pareto triangle.

    NOTE: Archetype labels are swapped to match the article:
    - Original Arch 2 (top right) -> displayed as Arch 3
    - Original Arch 3 (left) -> displayed as Arch 2
    """
    print("\n" + "=" * 70)
    print("STEP 4: Creating visualization")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot CFG data points
    if explore_pca is not None and len(explore_pca) > 0:
        ax.scatter(explore_pca[:, 0], explore_pca[:, 1],
                   c='blue', alpha=0.6, s=50, label=f'Explore (n={len(explore_pca)})',
                   edgecolors='darkblue', linewidth=0.5)

    if exploit_pca is not None and len(exploit_pca) > 0:
        ax.scatter(exploit_pca[:, 0], exploit_pca[:, 1],
                   c='red', alpha=0.6, s=50, label=f'Exploit (n={len(exploit_pca)})',
                   edgecolors='darkred', linewidth=0.5)

    # Plot triangle (Pareto front)
    # Connect archetypes to form triangle
    triangle = np.vstack([archetypes_pca, archetypes_pca[0]])  # Close the triangle
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, zorder=5)

    # Plot archetypes with CORRECTED labels
    # Original: Arch1 (cyan, bottom-right), Arch2 (red, top-right), Arch3 (yellow, left)
    # Article: Arch1 (bottom-right), Arch2 (left), Arch3 (top-right)
    # So we swap labels 2 and 3

    archetype_colors = ['cyan', 'red', 'yellow']
    archetype_labels_corrected = ['Archetype 1', 'Archetype 3', 'Archetype 2']  # Swapped 2↔3

    for i in range(3):
        ax.scatter(archetypes_pca[i, 0], archetypes_pca[i, 1],
                   c=archetype_colors[i], s=300, marker='^',
                   edgecolors='black', linewidth=2, zorder=10)

        # Add label with offset
        offset = [(0.2, -0.4), (0.2, 0.3), (-0.5, 0.1)]  # Offsets for each archetype
        ax.annotate(archetype_labels_corrected[i],
                    (archetypes_pca[i, 0], archetypes_pca[i, 1]),
                    xytext=(archetypes_pca[i, 0] + offset[i][0],
                            archetypes_pca[i, 1] + offset[i][1]),
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('CFG fMRI Data Projected onto HCP PCA Space\n(Pareto Front Triangle)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio for accurate triangle representation
    ax.set_aspect('equal', adjustable='box')

    # Save figure
    output_path = output_dir / "cfg_projected_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure to: {output_path}")

    plt.show()

    return fig


def save_results(explore_pca, exploit_pca, explore_ids, exploit_ids, output_dir):
    """Save projected data to CSV files."""
    print("\n" + "=" * 70)
    print("STEP 5: Saving results")
    print("=" * 70)

    if explore_pca is not None and len(explore_pca) > 0:
        explore_df = pd.DataFrame({
            'subject_id': explore_ids,
            'PC1': explore_pca[:, 0],
            'PC2': explore_pca[:, 1],
            'condition': 'explore'
        })
        explore_path = output_dir / "cfg_explore_pca.csv"
        explore_df.to_csv(explore_path, index=False)
        print(f"  Saved explore data to: {explore_path}")

    if exploit_pca is not None and len(exploit_pca) > 0:
        exploit_df = pd.DataFrame({
            'subject_id': exploit_ids,
            'PC1': exploit_pca[:, 0],
            'PC2': exploit_pca[:, 1],
            'condition': 'exploit'
        })
        exploit_path = output_dir / "cfg_exploit_pca.csv"
        exploit_df.to_csv(exploit_path, index=False)
        print(f"  Saved exploit data to: {exploit_path}")

    # Combined file
    if explore_pca is not None and exploit_pca is not None:
        combined_df = pd.concat([explore_df, exploit_df], ignore_index=True)
        combined_path = output_dir / "cfg_combined_pca.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"  Saved combined data to: {combined_path}")


def main():
    print("\n" + "=" * 70)
    print("CFG to HCP PCA Projection Pipeline")
    print("=" * 70)

    # Check files exist
    if not QC_FILE.exists():
        print(f"ERROR: QC file not found: {QC_FILE}")
        return

    if not PCA_FILE.exists():
        print(f"ERROR: PCA transformation file not found: {PCA_FILE}")
        return

    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Get valid subjects
    valid_subjects = load_qc_and_get_valid_subjects(QC_FILE)

    if not valid_subjects:
        print("ERROR: No valid subjects found!")
        return

    # Step 2: Load and process correlation matrices
    results = load_and_process_all_subjects(valid_subjects, CFG_DATA_DIR)

    # Step 3: Apply PCA transformation
    explore_pca = None
    exploit_pca = None

    if len(results['explore']['features']) > 0:
        explore_pca, archetypes_pca = apply_pca_transformation(
            results['explore']['features'], PCA_FILE
        )

    if len(results['exploit']['features']) > 0:
        exploit_pca, archetypes_pca = apply_pca_transformation(
            results['exploit']['features'], PCA_FILE
        )

    # Step 4: Create visualization
    create_visualization(explore_pca, exploit_pca, archetypes_pca, OUTPUT_DIR)

    # Step 5: Save results
    save_results(
        explore_pca, exploit_pca,
        results['explore']['subject_ids'],
        results['exploit']['subject_ids'],
        OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)

    # Print summary statistics
    print("\nSummary:")
    if explore_pca is not None:
        print(f"  Explore: {len(explore_pca)} subjects")
        print(f"    PC1 range: [{explore_pca[:, 0].min():.2f}, {explore_pca[:, 0].max():.2f}]")
        print(f"    PC2 range: [{explore_pca[:, 1].min():.2f}, {explore_pca[:, 1].max():.2f}]")

    if exploit_pca is not None:
        print(f"  Exploit: {len(exploit_pca)} subjects")
        print(f"    PC1 range: [{exploit_pca[:, 0].min():.2f}, {exploit_pca[:, 0].max():.2f}]")
        print(f"    PC2 range: [{exploit_pca[:, 1].min():.2f}, {exploit_pca[:, 1].max():.2f}]")


if __name__ == "__main__":
    main()