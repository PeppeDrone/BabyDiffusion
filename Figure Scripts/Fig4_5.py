import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Load your data
ours = pd.read_excel('evaluation_metrics_large.xlsx')
thibaud = pd.read_excel('evaluation_metrics_large_thibaud.xlsx')

# Define metrics
metrics = ['ssim', 'psnr', 'lpips']

# --- Filter your best configuration ---
ours_best = ours[
    (ours['batch_size'] == 2) &
    (ours['grad_accum'] == 4) &
    (ours['resolution'] == 768) &
    (ours['guidance'] == 6) &
    (ours['cond_scale'] == 1)
].reset_index(drop=True)

# --- Filter Thibaud's best configuration ---
thibaud_best = thibaud[
    (thibaud['guidance'] == 6) &
    (thibaud['cond_scale'] == 1)
].reset_index(drop=True)

# Align lengths for paired analysis
min_len = min(len(ours_best), len(thibaud_best))
ours_best = ours_best.iloc[:min_len]
thibaud_best = thibaud_best.iloc[:min_len]

# --- Paired Plot + t-test ---
fig, axes = plt.subplots(3,1, figsize=(6,6), sharex=True)

for i, metric in enumerate(metrics):
    # Perform paired t-test
    t_stat, p_val = ttest_rel(ours_best[metric], thibaud_best[metric])
    significance = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4e}"

    # Create paired line plot
    paired_df = pd.DataFrame({
        'Newborn': ours_best[metric],
        'Adults': thibaud_best[metric]
    })

    sns.lineplot(data=paired_df, palette='Set2', ax=axes[i], markers=True)
    axes[i].set_title(f'{metric.upper()} (t = {t_stat:.4f}, {significance})', fontsize=12)
    axes[i].set_ylabel(metric.upper())
    axes[i].set_xlabel('Sample Index')
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
# Save the figure
fig.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')