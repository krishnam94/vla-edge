#!/usr/bin/env python3
"""Regenerate AUROC comparison figure without fabricated stall data."""
import matplotlib.pyplot as plt
import numpy as np

# Verified data from monitor_analysis.json
monitors = ['Reversal', 'ATV', 'Jerk', 'Coherence', 'Spectral', 'Velocity']
vqbet =    [0.93,       0.89,  0.88,  0.86,        0.75,       0.69]
diffusion = [0.79,       0.71,  0.43,  0.70,        0.34,       0.41]

x = np.arange(len(monitors))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - width/2, vqbet, width, label='VQ-BeT', color='#E07070', edgecolor='white')
bars2 = ax.bar(x + width/2, diffusion, width, label='Diffusion', color='#7EB5D6', edgecolor='white')

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random')
ax.set_ylabel('AUROC (failure prediction)', fontsize=12)
ax.set_title('Architecture-Matched Failure Prediction (PushT)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(monitors, fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0.25, 1.05)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fig_auroc_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_auroc_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig_auroc_comparison.pdf and .png")
