"""
Simple visualization of multi-quantum ensemble results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path("outputs/god_ai/multi_quantum_ensemble_20251104_171322.json")
with open(results_file, 'r') as f:
    data = json.load(f)

print("=" * 80)
print("🏆 MULTI-QUANTUM ENSEMBLE RESULTS")
print("=" * 80)

# Extract all test scores
phase_based_scores = []
adaptive_scores = []

for horizon in data['results']:
    for test in horizon['tests']:
        if test['strategy'] == 'phase_based':
            phase_based_scores.append(test['total_score'])
        elif test['strategy'] == 'adaptive':
            adaptive_scores.append(test['total_score'])

# Calculate totals
phase_total = sum(phase_based_scores)
adaptive_total = sum(adaptive_scores)

print(f"\n📊 TOTAL SCORES:")
print(f"   Phase-Based: {phase_total:,.0f}")
print(f"   Adaptive:    {adaptive_total:,.0f}")
print(f"   Winner:      {'Phase-Based' if phase_total > adaptive_total else 'Adaptive'}")
print(f"   Margin:      +{abs(phase_total - adaptive_total):,.0f} ({((phase_total/adaptive_total - 1)*100):+.1f}%)")

print(f"\n📈 AVERAGE PER RUN:")
print(f"   Phase-Based: {np.mean(phase_based_scores):,.0f}")
print(f"   Adaptive:    {np.mean(adaptive_scores):,.0f}")

print(f"\n🧬 SPECIALIST USAGE:")
for spec in data['ensemble_config']:
    name = spec['name']
    history = spec.get('performance_history', [])
    if history:
        avg = np.mean(history)
        print(f"   {name:25s}: {len(history):2d} uses, avg {avg:,.0f}")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Total score comparison
strategies = ['Phase-Based', 'Adaptive']
totals = [phase_total, adaptive_total]
colors = ['#2ecc71', '#3498db']

bars = ax1.bar(strategies, totals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Total Score', fontsize=12, fontweight='bold')
ax1.set_title('🏆 Total Score Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Individual run scores
x = np.arange(len(phase_based_scores))
width = 0.35

bars1 = ax2.bar(x - width/2, phase_based_scores, width, label='Phase-Based', 
               color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, adaptive_scores, width, label='Adaptive',
               color='#3498db', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Score per Run', fontsize=12, fontweight='bold')
ax2.set_xlabel('Run Number', fontsize=12, fontweight='bold')
ax2.set_title('📊 Individual Run Scores', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{i+1}' for i in x])
ax2.legend()
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 3. Specialist performance
specialists = []
specialist_scores = []
specialist_colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

for i, spec in enumerate(data['ensemble_config']):
    history = spec.get('performance_history', [])
    if history:
        specialists.append(spec['name'].replace('_', '\n'))
        specialist_scores.append(np.mean(history))

bars = ax3.barh(specialists, specialist_scores, color=specialist_colors[:len(specialists)],
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Average Score', fontsize=12, fontweight='bold')
ax3.set_title('🧬 Specialist Performance', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,}',
            ha='left', va='center', fontweight='bold', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 4. Cooperation rates
phase_coop = []
adaptive_coop = []

for horizon in data['results']:
    for test in horizon['tests']:
        if test['strategy'] == 'phase_based':
            phase_coop.append(test['cooperation_rate'])
        elif test['strategy'] == 'adaptive':
            adaptive_coop.append(test['cooperation_rate'])

x = np.arange(len(phase_coop))
bars1 = ax4.bar(x - width/2, phase_coop, width, label='Phase-Based',
               color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x + width/2, adaptive_coop, width, label='Adaptive',
               color='#3498db', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Cooperation Rate (%)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Run Number', fontsize=12, fontweight='bold')
ax4.set_title('🤝 Cooperation Rates', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{i+1}' for i in x])
ax4.legend()
ax4.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Multi-Quantum Ensemble Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
output_file = Path("outputs/god_ai/ensemble_analysis.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved: {output_file}")
print("=" * 80)

plt.show()
