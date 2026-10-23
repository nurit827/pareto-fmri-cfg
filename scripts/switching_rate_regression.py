"""
Switching Rate (g) vs Arch1-Arch2 Axis: Regression Scatter Plots
All subjects only
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
SWITCH_FILE = BASE_DIR / "data" / "g_non_mri_segmentation.csv"
SLOPES_FILE = BASE_DIR / "results" / "per_phase" / "regression_slopes.csv"
OUTPUT_DIR  = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

def format_p(p):
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"

def pearson_label(r, p):
    sig = "*" if p < 0.05 else "n.s."
    return f"r={r:.3f}, {format_p(p)} {sig}"

sw = pd.read_csv(SWITCH_FILE)
sw.columns = ['subject_id', 'g']
sw['subject_id'] = sw['subject_id'].astype(str).str.strip()

slopes = pd.read_csv(SLOPES_FILE)
slopes['subject_id'] = slopes['subject_id'].astype(str).str.strip()

explore = slopes[slopes['condition'] == 'explore'].set_index('subject_id')
exploit = slopes[slopes['condition'] == 'exploit'].set_index('subject_id')
common  = explore.index.intersection(exploit.index)

axis_df = pd.DataFrame(index=common)
axis_df['slope_explore']  = explore.loc[common, 'slope_Archetype_1']
axis_df['slope_exploit']  = exploit.loc[common, 'slope_Archetype_1']
axis_df['abs_explore']    = axis_df['slope_explore'].abs()
axis_df['abs_exploit']    = axis_df['slope_exploit'].abs()
axis_df['mean_magnitude'] = (axis_df['abs_explore'] + axis_df['abs_exploit']) / 2
axis_df['flipped']        = np.sign(axis_df['slope_explore']) != np.sign(axis_df['slope_exploit'])

merged = axis_df.join(sw.set_index('subject_id')[['g']], how='inner')
print(f"Subjects: {len(merged)}")

same_mask = ~merged['flipped']
flip_mask  =  merged['flipped']

scatter_cfg = [
    ("slope_explore", "Explore slope\n(Arch1-Arch2 axis)"),
    ("slope_exploit", "Exploit slope\n(Arch1-Arch2 axis)"),
    ("mean_magnitude","Mean magnitude\n(both conditions)"),
    ("abs_explore",   "Explore |slope|"),
    ("abs_exploit",   "Exploit |slope|"),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle(f"Switching Rate (g) vs Arch1-Arch2 Axis Movement (n={len(merged)})", fontsize=13, fontweight='bold')

for i, (xcol, xlabel) in enumerate(scatter_cfg):
    ax = axes[i]
    x, y = merged[xcol], merged['g']
    ax.scatter(x[same_mask], y[same_mask], color='#4CAF50', alpha=0.7,
               edgecolors='white', linewidth=0.4, s=55, label='Same')
    ax.scatter(x[flip_mask], y[flip_mask], color='#F44336', alpha=0.7,
               edgecolors='white', linewidth=0.4, s=55, label='Flipped')
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), 'k--', alpha=0.7, linewidth=1.5)
    r, p = stats.pearsonr(x, y)
    ax.set_title(pearson_label(r, p), fontsize=9, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("g", fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
out = OUTPUT_DIR / "switching_rate_regression.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()