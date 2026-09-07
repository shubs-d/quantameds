"""
Analysis Script for:
Orbscan IIz Multimodal Corneal Topography Dataset
DOI: 10.17632/78wt2nc387.2

This script reproduces all descriptive statistics, figures, and correlation
analyses reported in the associated Data in Brief article.

Requirements:
    pip install -r requirements.txt

Usage:
    python analysis_script.py

Input files (must be in the same directory as this script):
    - metadata_with_QC.csv       : Original metadata with QC_status flag
    - metadata_numeric.csv       : Numeric analysis-ready values

Output files:
    - table3_statistics.csv      : Descriptive statistics (Table 3)
    - Figure_1_Distributions.png : Distribution plots (Fig. 1)
    - Figure_2_MaxK_Thinnest.png : Scatter plot (Fig. 2)
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df_qc  = pd.read_csv('metadata_with_QC.csv', encoding='utf-8-sig')
df_num = pd.read_csv('metadata_numeric.csv', encoding='utf-8-sig')

# Quality-controlled subset (PASS only)
qc_pass = df_num[df_num['QC_status'] == 'PASS'].copy().reset_index(drop=True)
N = len(qc_pass)
print(f"Total records  : {len(df_num)}")
print(f"QC PASS (PASS) : {N}")
print(f"QC FAIL (FAIL) : {len(df_num) - N}")

# ─────────────────────────────────────────────
# 2. ANATOMICAL PLAUSIBILITY FILTER
# ─────────────────────────────────────────────
plausibility = {
    'SimK_Astig_D':         (0, 15),
    'MaxK_D':               (35, 68),
    'MinK_D':               (35, 68),
    'Zone3mm_Irreg_D':      (0, 15),
    'Zone3mm_MeanPwr_D':    (35, 65),
    'Zone3mm_AstigPwr_D':   (0, 15),
    'Zone3mm_SteepAxis_deg':(0, 180),
    'Zone3mm_FlatAxis_deg': (0, 180),
    'Zone5mm_Irreg_D':      (0, 15),
    'Zone5mm_MeanPwr_D':    (35, 65),
    'Zone5mm_AstigPwr_D':   (0, 15),
    'Zone5mm_SteepAxis_deg':(0, 180),
    'Zone5mm_FlatAxis_deg': (0, 180),
    'WhiteToWhite_mm':      (9.5, 13.5),
    'PupilDiameter_mm':     (1.5, 8.5),
    'Thinnest_um':          (200, 700),
    'ACD_Ep_mm':            (2.0, 6.0),
    'Kappa_deg':            (0, 18),
    'KappaIntercept_x':     (-3, 3),
    'KappaIntercept_y':     (-3, 3),
}

qc_clean = qc_pass.copy()
for col, (lo, hi) in plausibility.items():
    if col in qc_clean.columns:
        qc_clean[col] = qc_clean[col].where(
            (qc_clean[col] >= lo) & (qc_clean[col] <= hi)
        )

# ─────────────────────────────────────────────
# 3. MISSING VALUES ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("MISSING VALUES — full released dataset (N = 3,000)")
print("="*70)
raw_cols = [c for c in df_qc.columns if c not in ['filename', 'QC_status']]
miss = df_qc[raw_cols].replace('', pd.NA).isnull().sum().sort_values(ascending=False)
for col, n in miss[miss > 0].items():
    print(f"  {col:<25}: {n:>4} missing ({n/len(df_qc)*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. DESCRIPTIVE STATISTICS (Table 3)
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 3: Descriptive statistics — QC-PASS subset")
print("="*70)

# Column name → paper label mapping
stat_cols = [
    ('SimK_Astig_D',          'SimK_Astig (D, absolute)'),
    ('MaxK_D',                'MaxK (D)'),
    ('MinK_D',                'MinK (D)'),
    ('Zone3mm_Irreg_D',       'Zone3mm_Irreg (D)'),
    ('Zone3mm_MeanPwr_D',     'Zone3mm_MeanPwr (D)'),
    ('Zone3mm_AstigPwr_D',    'Zone3mm_AstigPwr (D)'),
    ('Zone3mm_SteepAxis_deg', 'Zone3mm_SteepAxis (deg)'),
    ('Zone3mm_FlatAxis_deg',  'Zone3mm_FlatAxis (deg)'),
    ('Zone5mm_Irreg_D',       'Zone5mm_Irreg (D)'),
    ('Zone5mm_MeanPwr_D',     'Zone5mm_MeanPwr (D)'),
    ('Zone5mm_AstigPwr_D',    'Zone5mm_AstigPwr (D)'),
    ('Zone5mm_SteepAxis_deg', 'Zone5mm_SteepAxis (deg)'),
    ('Zone5mm_FlatAxis_deg',  'Zone5mm_FlatAxis (deg)'),
    ('WhiteToWhite_mm',       'WhiteToWhite (mm)'),
    ('PupilDiameter_mm',      'PupilDiameter (mm)'),
    ('Thinnest_um',           'Thinnest (µm)'),
    ('ACD_Ep_mm',             'ACD_Ep (mm)'),
    ('Kappa_deg',             'Kappa (°)'),
    ('KappaIntercept_x',      'KappaIntercept_x (mm)'),
]

results = []
for col, label in stat_cols:
    if col not in qc_clean.columns:
        continue
    s = qc_clean[col].dropna()
    if len(s) < 10:
        continue
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    iqr = q3 - q1
    n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    skew = s.skew()
    kurt = s.kurtosis()
    sw_stat, sw_p = scipy_stats.shapiro(
        s.sample(min(len(s), 5000), random_state=42)
    )
    sw_str = '<0.0001' if sw_p < 0.0001 else f'{sw_p:.4f}'
    results.append({
        'Parameter':    label,
        'N':            int(len(s)),
        'Mean':         round(s.mean(), 2),
        'SD':           round(s.std(), 2),
        'Median':       round(s.median(), 2),
        'Min':          round(s.min(), 2),
        'Max':          round(s.max(), 2),
        'Q1':           round(q1, 2),
        'Q3':           round(q3, 2),
        'IQR':          round(iqr, 2),
        'Skewness':     round(skew, 3),
        'Kurtosis':     round(kurt, 3),
        'Outliers_%':   round(n_out / len(s) * 100, 1),
        'SW_p':         sw_str,
    })
    print(f"  {label:<30}: N={len(s):4d}  Mean={s.mean():.2f}  "
          f"SD={s.std():.2f}  Min={s.min():.2f}  Max={s.max():.2f}")

results_df = pd.DataFrame(results)
results_df.to_csv('table3_statistics.csv', index=False)
print(f"\nTable 3 saved → table3_statistics.csv")

# ─────────────────────────────────────────────
# 5. PEARSON CORRELATIONS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("PEARSON CORRELATIONS — QC-PASS subset")
print("="*70)
corr_pairs = [
    ('MaxK_D',        'Thinnest_um',      'MaxK (D)',             'Thinnest (µm)'),
    ('MaxK_D',        'MinK_D',           'MaxK (D)',             'MinK (D)'),
    ('MaxK_D',        'Zone3mm_Irreg_D',  'MaxK (D)',             'Zone3mm Irreg (D)'),
    ('Thinnest_um',   'ACD_Ep_mm',        'Thinnest (µm)',        'ACD_Ep (mm)'),
    ('WhiteToWhite_mm','PupilDiameter_mm','White-to-White (mm)', 'Pupil Diameter (mm)'),
]
for c1, c2, l1, l2 in corr_pairs:
    pair = qc_clean[[c1, c2]].dropna()
    if len(pair) > 10:
        r, p = scipy_stats.pearsonr(pair[c1], pair[c2])
        p_str = '<0.0001' if p < 0.0001 else f'{p:.4f}'
        print(f"  {l1} vs {l2}: r = {r:.3f}, p = {p_str}, N = {len(pair):,}")

# ─────────────────────────────────────────────
# 6. FIGURE 1 — DISTRIBUTIONS
# ─────────────────────────────────────────────
print("\nGenerating Fig. 1 ...")

params_fig1 = [
    ('MaxK_D',          'Maximum Corneal Power — MaxK (D)',    'steelblue',  'MaxK (D)'),
    ('Thinnest_um',     'Thinnest Pachymetry (µm)',            'darkorange', 'Thinnest (µm)'),
    ('Zone3mm_Irreg_D', '3mm Zone Irregularity Index (D)',     'seagreen',   'Zone3mm Irregularity (D)'),
    ('WhiteToWhite_mm', 'White-to-White Diameter (mm)',        'purple',     'White-to-White (mm)'),
    ('ACD_Ep_mm',       'Anterior Chamber Depth — ACD (mm)',   'crimson',    'ACD Epithelium (mm)'),
    ('Kappa_deg',       'Angle Kappa (°)',                     'teal',       'Angle Kappa (°)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    f'Fig. 1. Frequency distributions of six selected clinical parameters\n'
    f'across the quality-controlled subset (N = {N:,})',
    fontsize=13, fontweight='bold', y=1.01
)

for ax, (col, xlabel, color, title) in zip(axes.flat, params_fig1):
    data = qc_clean[col].dropna()
    n    = len(data)
    mean = data.mean()

    ax.hist(data, bins=40, color=color, alpha=0.78,
            edgecolor='white', linewidth=0.4)
    ax.axvline(mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean:.2f}')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{title}\n(n = {n:,})', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('Figure_1_Distributions.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved → Figure_1_Distributions.png")

# ─────────────────────────────────────────────
# 7. FIGURE 2 — SCATTER PLOT
# ─────────────────────────────────────────────
print("Generating Fig. 2 ...")

pair2 = qc_clean[['MaxK_D', 'Thinnest_um']].dropna()
r2, p2 = scipy_stats.pearsonr(pair2['MaxK_D'], pair2['Thinnest_um'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(pair2['MaxK_D'], pair2['Thinnest_um'],
            alpha=0.22, s=8, c='steelblue', edgecolors='none')
ax2.axvline(47,  color='#C0392B', linestyle='--', linewidth=1.5,
            label='MaxK = 47 D (reference threshold)', alpha=0.85)
ax2.axhline(470, color='#E67E22', linestyle='--', linewidth=1.5,
            label='Thinnest = 470 µm (reference threshold)', alpha=0.85)
ax2.set_xlabel('Maximum Corneal Power — MaxK (D)', fontsize=12)
ax2.set_ylabel('Thinnest Pachymetry (µm)', fontsize=12)
ax2.set_title(
    f'Fig. 2. Maximum Corneal Power (MaxK) versus Thinnest Pachymetry\n'
    f'(r = {r2:.3f}, p < 0.0001, N = {len(pair2):,})',
    fontsize=12, fontweight='bold'
)
ax2.legend(fontsize=10, framealpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('Figure_2_MaxK_Thinnest.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved → Figure_2_MaxK_Thinnest.png  (r = {r2:.3f}, N = {len(pair2):,})")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("Output files:")
print("  table3_statistics.csv")
print("  Figure_1_Distributions.png")
print("  Figure_2_MaxK_Thinnest.png")
print("="*70)
