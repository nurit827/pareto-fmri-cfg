import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "results" / "per_phase" / "direction_toward_archetypes.csv")

# Separate explore and exploit
explore = df[df['condition'] == 'explore'].set_index('subject_id')
exploit = df[df['condition'] == 'exploit'].set_index('subject_id')

common = explore.index.intersection(exploit.index)
explore = explore.loc[common]
exploit = exploit.loc[common]

# Check Arch1 quadrants
explore_delta = explore['delta_Archetype_1'].values
exploit_delta = exploit['delta_Archetype_1'].values

# Count quadrants
q1 = ((explore_delta < 0) & (exploit_delta > 0)).sum()  # Explore toward, Exploit away
q2 = ((explore_delta > 0) & (exploit_delta < 0)).sum()  # Explore away, Exploit toward
q3 = ((explore_delta < 0) & (exploit_delta < 0)).sum()  # Both toward
q4 = ((explore_delta > 0) & (exploit_delta > 0)).sum()  # Both away

print("Arch1 (Impulsive) - Quadrant Analysis:")
print(f"  Explore TOWARD, Exploit AWAY: {q1} ({100*q1/len(common):.1f}%)")
print(f"  Explore AWAY, Exploit TOWARD: {q2} ({100*q2/len(common):.1f}%)")
print(f"  Both TOWARD: {q3} ({100*q3/len(common):.1f}%)")
print(f"  Both AWAY: {q4} ({100*q4/len(common):.1f}%)")
print(f"\n  'Opposite' quadrants (Q1+Q2): {q1+q2} ({100*(q1+q2)/len(common):.1f}%)")
print(f"  'Same' quadrants (Q3+Q4): {q3+q4} ({100*(q3+q4)/len(common):.1f}%)")