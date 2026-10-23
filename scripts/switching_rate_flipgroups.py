"""
Switching Rate (g) vs Arch1-Arch2 Flip Groups: Boxplots
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
axis_df['flipped']        = np.sign(axis_df['slope_explore']) != np.sign(axis_df['slope_exploit'])

axis_df['flip_type'] = 'same'
axis_df.loc[axis_df['flipped'] & (axis_df['slope_explore'] > 0) & (axis_df['slope_exploit'] < 0), 'flip_type'] = 'Arch1_to_Arch2'
axis_df.loc[axis_df['flipped'] & (axis_df['slope_explore'] < 0) & (axis_df['slope_exploit'] > 0), 'flip_type'] = 'Arch2_to_Arch1'

merged = axis_df.join(sw.set_index('subject_id')[['g']], how='inner')

same_g = merged.loc[~merged['flipped'], 'g']
flip_g = merged.loc[merged['flipped'],  'g']
a1a2_g = merged.loc[merged['flip_type'] == 'Arch1_to_Arch2', 'g']
a2a1_g = merged.loc[merged['flip_type'] == 'Arch2_to_Arch1', 'g']

t1, p1 = stats.ttest_ind(flip_g, same_g)
t2, p2 = stats.ttest_ind(a1a2_g, a2a1_g)
f,  pf = stats.f_oneway(same_g, a1a2_g, a2a1_g)

print(f"Subjects: {len(merged)}")
print(f"Flip vs Same: t={t1:.3f}, {format_p(p1)} {'*' if p1<0.05 else 'n.s.'}")
print(f"A1→A2 vs A2→A1: t={t2:.3f}, {format_p(p2)} {'*' if p2<0.05 else 'n.s.'}")
print(f"ANOVA: F={f:.3f}, {format_p(pf)} {'*' if pf<0.05 else 'n.s.'}")

ylim = (merged['g'].min() - 1, merged['g'].max() + 1)

def draw_panel(ax, groups, labels, colors, title):
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (data, xpos) in enumerate(zip(groups, range(1, len(groups)+1))):
        ax.scatter(np.random.normal(xpos, 0.04, len(data)), data,
                   color=colors[i], alpha=0.5, s=25, zorder=5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel("g", fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ylim)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Switching Rate (g) by Flip Group (n={len(merged)})", fontsize=13, fontweight='bold')

draw_panel(axes[0],
    [same_g.values, flip_g.values],
    [f'Same\n(n={len(same_g)})', f'Flipped\n(n={len(flip_g)})'],
    ['#4CAF50', '#F44336'],
    f"Flip vs Same\n{format_p(p1)} {'*' if p1<0.05 else 'n.s.'}"
)

draw_panel(axes[1],
    [same_g.values, a1a2_g.values, a2a1_g.values],
    [f'Same\n(n={len(same_g)})', f'Arch1→Arch2\n(n={len(a1a2_g)})', f'Arch2→Arch1\n(n={len(a2a1_g)})'],
    ['#4CAF50', '#2196F3', '#FF5722'],
    f"ANOVA: {format_p(pf)} {'*' if pf<0.05 else 'n.s.'}  |  A1→A2 vs A2→A1: {format_p(p2)} {'*' if p2<0.05 else 'n.s.'}"
)

plt.tight_layout()
out = OUTPUT_DIR / "switching_rate_flipgroups.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()