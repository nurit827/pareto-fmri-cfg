import numpy as np
import pandas as pd

print("="*70)
print("STAGE 3: Examining Feature Order")
print("="*70)

# Load the saved transformation
data = np.load('data/pca_transformation.npz', allow_pickle=True)
feature_names = data['feature_names']

print(f"\nTotal features: {len(feature_names)}")
print("\nFirst 10 features:")
for i, name in enumerate(feature_names[:10]):
    print(f"  {i+1}. {name}")

print("\n...")
print("\nLast 10 features:")
for i, name in enumerate(feature_names[-10:], start=len(feature_names)-9):
    print(f"  {i}. {name}")

# Save to a text file for reference
with open('data/feature_order.txt', 'w') as f:
    f.write("Feature Order (153 features)\n")
    f.write("="*70 + "\n\n")
    for i, name in enumerate(feature_names, start=1):
        f.write(f"{i}. {name}\n")

print("\n✓ Full feature list saved to 'data/feature_order.txt'")

print("\n" + "="*70)
print("STAGE 3 COMPLETE")
print("="*70)