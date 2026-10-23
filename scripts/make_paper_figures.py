"""
Paper Figures
=============
Redraws the three figures of the paper from the saved results (no analysis is redone;
the original multi-panel figures are left untouched).

  Figure 1: change in distance to Arch1 vs. Arch2 (first to last segment), explore and exploit
  Figure 2: variance of the distance to each archetype, explore vs. exploit
  Figure 3: g vs. Arch1-Arch2 movement per segment, both phases (A) and explore only (B)
  Figure A1 (Appendix): g vs. Arch1-Arch2 total movement over the session
  Figure A2 (Appendix): slope of the distance to Arch1 vs. Arch2, explore and exploit

Run from: ~/Pareto-fMRI-main/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RES = BASE_DIR / "results" / "per_phase"
OUT = BASE_DIR / "results" / "figures_for_paper"
OUT.mkdir(parents=True, exist_ok=True)

EXPLORE, EXPLOIT, NEUTRAL, DARK = '#3B6FA0', '#C5504B', '#5B6C7A', '#333333'

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})


def minus(text):
    return text.replace('-', '−')


def fmt_p(p):
    return 'p < 0.0001' if p < 0.0001 else (f'p = {p:.4f}' if p < 0.05 else f'p = {p:.2f}')


def panel_letter(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')


def normalize_id(raw):
    s = str(raw).strip()
    if s.startswith('sub-'):
        s = s[4:]
    return s.lstrip('0') or '0'


def figure1():
    d = pd.read_csv(RES / "direction_toward_archetypes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, cond, color, letter in zip(axes, ['explore', 'exploit'], [EXPLORE, EXPLOIT], 'AB'):
        s = d[d['condition'] == cond]
        x, y = s['delta_Archetype_1'], s['delta_Archetype_2']
        r, p = stats.pearsonr(x, y)
        fit = stats.linregress(x, y)
        xs = np.array([x.min(), x.max()])
        ax.axhline(0, color='#c8c8c8', lw=0.8, zorder=0)
        ax.axvline(0, color='#c8c8c8', lw=0.8, zorder=0)
        ax.scatter(x, y, s=36, color=color, alpha=0.78, edgecolor='white', linewidth=0.6, zorder=2)
        ax.plot(xs, fit.intercept + fit.slope * xs, color=DARK, ls='--', lw=1.4, zorder=3)
        ax.set_xlabel('Change in distance to Arch1\n(last − first segment)')
        ax.set_ylabel('Change in distance to Arch2')
        ax.set_title(cond.capitalize(), fontweight='bold', color=color)
        ax.text(0.97, 0.96, minus(f'r = {r:.2f}\n{fmt_p(p)}\nn = {len(s)}'), transform=ax.transAxes,
                ha='right', va='top', fontsize=10.5)
        panel_letter(ax, letter)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_arch1_arch2_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure2():
    cov = pd.read_csv(RES / "covariance_matrices.csv")
    tests = pd.read_csv(RES / "covariance_paired_tests.csv").set_index('element')
    ex = cov[cov['condition'] == 'explore'].set_index('subject_id')
    xt = cov[cov['condition'] == 'exploit'].set_index('subject_id')
    common = ex.index.intersection(xt.index)
    archs = [('Arch1', 'cov_11'), ('Arch2', 'cov_33'), ('Arch3', 'cov_22')]
    n = len(common)
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ymax = 0
    for i, (name, col) in enumerate(archs):
        for off, df, color in [(-w / 2, ex, EXPLORE), (w / 2, xt, EXPLOIT)]:
            v = df.loc[common, col]
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
            ax.bar(i + off, m, width=w, color=color, alpha=0.9, yerr=se,
                   error_kw={'ecolor': DARK, 'capsize': 4, 'lw': 1.2})
            ymax = max(ymax, m + se)
        top = max(ex.loc[common, col].mean() + ex.loc[common, col].std(ddof=1) / np.sqrt(n),
                  xt.loc[common, col].mean() + xt.loc[common, col].std(ddof=1) / np.sqrt(n))
        p = tests.loc[col, 'p_ttest']
        ax.text(i, top + 0.03, fmt_p(p) + (' *' if p < 0.05 else ''), ha='center', va='bottom', fontsize=10,
                fontweight='bold' if p < 0.05 else 'normal')
    ax.set_xticks(range(3))
    ax.set_xticklabels([a for a, _ in archs])
    ax.set_ylabel('Variance of the distance to the archetype\n(across segments)')
    ax.set_ylim(0, ymax + 0.16)
    ax.bar(0, 0, color=EXPLORE, label='Explore')
    ax.bar(0, 0, color=EXPLOIT, label='Exploit')
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    ax.text(0.99, -0.16, f'Mean ± s.e.m., n = {n}; paired t-test', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#666666')
    fig.tight_layout()
    fig.savefig(OUT / "figure2_variance_by_phase.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def movement_vs_g():
    slopes = pd.read_csv(RES / "regression_slopes.csv")
    slopes['key'] = slopes['subject_id'].apply(normalize_id)
    ex = slopes[slopes['condition'] == 'explore'].set_index('key')
    xt = slopes[slopes['condition'] == 'exploit'].set_index('key')
    common = ex.index.intersection(xt.index)
    g = pd.read_csv(BASE_DIR / "data" / "g_non_mri_segmentation.csv")
    g.columns = ['ID', 'g']
    g['key'] = g['ID'].apply(normalize_id)
    g = g.set_index('key')[['g']]
    abs_ex = ex.loc[common, 'slope_Archetype_1'].abs()
    abs_xt = xt.loc[common, 'slope_Archetype_1'].abs()
    return pd.DataFrame({'per_segment': (abs_ex + abs_xt) / 2,
                         'explore': abs_ex,
                         'total': (abs_ex * (ex.loc[common, 'n_segments'] - 1) +
                                   abs_xt * (xt.loc[common, 'n_segments'] - 1)) / 2}).join(g, how='inner')


def scatter_vs_g(axes, d, specs):
    for ax, (col, xlabel, letter, color) in zip(axes, specs):
        r, p = stats.pearsonr(d[col], d['g'])
        fit = stats.linregress(d[col], d['g'])
        xs = np.array([d[col].min(), d[col].max()])
        ax.axhline(0, color='#c8c8c8', lw=0.8, zorder=0)
        ax.scatter(d[col], d['g'], s=36, color=color, alpha=0.75, edgecolor='white', linewidth=0.6, zorder=2)
        ax.plot(xs, fit.intercept + fit.slope * xs, color=DARK, ls='--', lw=1.4, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('g (rate of meaning accumulation)')
        ax.set_title(minus(f'r = {r:+.2f},  {fmt_p(p)},  n = {len(d)}'), fontsize=11)
        if letter:
            panel_letter(ax, letter)


def figure3():
    d = movement_vs_g()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    scatter_vs_g(axes, d, [
        ('per_segment', 'Mean |slope| of the distance to Arch1\nper segment (both phases)', 'A', NEUTRAL),
        ('explore', '|slope| of the distance to Arch1\nper segment (explore only)', 'B', EXPLORE)])
    fig.tight_layout()
    fig.savefig(OUT / "figure3_g_vs_movement.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure_a1():
    d = movement_vs_g()
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    scatter_vs_g([ax], d, [
        ('total', 'Mean total movement over the session\n(|slope| × (segments − 1), both phases)', None, NEUTRAL)])
    fig.tight_layout()
    fig.savefig(OUT / "figureA1_g_vs_total_movement.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure_a2():
    d = pd.read_csv(RES / "regression_slopes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, cond, color, letter in zip(axes, ['explore', 'exploit'], [EXPLORE, EXPLOIT], 'AB'):
        s = d[d['condition'] == cond]
        x, y = s['slope_Archetype_1'], s['slope_Archetype_2']
        r, p = stats.pearsonr(x, y)
        fit = stats.linregress(x, y)
        xs = np.array([x.min(), x.max()])
        ax.axhline(0, color='#c8c8c8', lw=0.8, zorder=0)
        ax.axvline(0, color='#c8c8c8', lw=0.8, zorder=0)
        ax.scatter(x, y, s=36, color=color, alpha=0.78, edgecolor='white', linewidth=0.6, zorder=2)
        ax.plot(xs, fit.intercept + fit.slope * xs, color=DARK, ls='--', lw=1.4, zorder=3)
        ax.set_xlabel('Slope of the distance to Arch1\n(change per segment)')
        ax.set_ylabel('Slope of the distance to Arch2')
        ax.set_title(cond.capitalize(), fontweight='bold', color=color)
        ax.text(0.97, 0.96, minus(f'r = {r:.2f}\n{fmt_p(p)}\nn = {len(s)}'), transform=ax.transAxes,
                ha='right', va='top', fontsize=10.5)
        panel_letter(ax, letter)
    fig.tight_layout()
    fig.savefig(OUT / "figureA2_arch1_arch2_slopes.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure_a1()
    figure_a2()
    print("Saved figures to", OUT)
