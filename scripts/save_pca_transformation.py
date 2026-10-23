import numpy as np
import pandas as pd

print("="*70)
print("STAGE 2: Saving PCA Transformation from Original Analysis")
print("="*70)

# Load original features to compute the mean
print("\n1. Loading original features from data/features.csv...")
original_features = pd.read_csv('data/features.csv')
feature_columns = original_features.columns[1:]  # Skip 'id' column
X_original = original_features[feature_columns].values
print(f"   ✓ Original data shape: {X_original.shape}")
print(f"   ✓ Number of subjects: {X_original.shape[0]}")
print(f"   ✓ Number of features: {X_original.shape[1]}")

# Compute the mean that was used for centering
print("\n2. Computing mean of original data...")
X_mean = np.mean(X_original, axis=0)
print(f"   ✓ Mean shape: {X_mean.shape}")

# Load PCA components from results
print("\n3. Loading PCA components from results/results.npz...")
results = np.load('results/results.npz', allow_pickle=True)
pca_components = results['pca_components_']  # Shape: (2, 153)
print(f"   ✓ PCA components shape: {pca_components.shape}")
print(f"   ✓ Number of PCs: {pca_components.shape[0]}")

# Verify by transforming the original data
print("\n4. Verifying transformation...")
X_centered = X_original - X_mean
X_pca_reconstructed = X_centered @ pca_components.T
data_pca_stored = results['data_pca_']
max_diff = np.max(np.abs(X_pca_reconstructed - data_pca_stored))
print(f"   ✓ Max difference between reconstructed and stored: {max_diff:.2e}")

if max_diff < 1e-10:
    print("   ✓ Verification PASSED!")
else:
    print("   ⚠ WARNING: Large difference detected")

# Get archetypes for reference
archetypes_pca = results['archetypes_']  # Shape: (3, 2)
archetypes_original = results['archetypes_original_']

print("\n5. Archetype positions in PCA space:")
for i in range(3):
    print(f"   Archetype {i+1}: PC1={archetypes_pca[i, 0]:.3f}, PC2={archetypes_pca[i, 1]:.3f}")
# Save the transformation parameters
print("\n6. Saving transformation to 'data/pca_transformation.npz'...")
np.savez('data/pca_transformation.npz',
         mean=X_mean,
         components=pca_components,
         feature_names=np.array(feature_columns.tolist()),
         archetypes_pca=archetypes_pca,
         archetypes_original=archetypes_original)

print("   ✓ Saved successfully!")

print("\n" + "="*70)
print("STAGE 2 COMPLETE")
print("="*70)
print("\nYou now have 'data/pca_transformation.npz' containing:")
print("  - mean: for centering new data")
print("  - components: for projecting to PC space")
print("  - feature_names: order of the 153 features")
print("  - archetypes_pca: archetype positions")