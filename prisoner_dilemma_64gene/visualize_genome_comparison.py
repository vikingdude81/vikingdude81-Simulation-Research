"""
Visualize genome differences between 50-gen and 150-gen training
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load genomes
with open("outputs/god_ai/quantum_evolution_champion.json", 'r') as f:
    genome_50 = json.load(f)['champion']['genome']

with open("outputs/god_ai/quantum_evolution_150gen_20251104_140931.json", 'r') as f:
    genome_150 = json.load(f)['champion']['genome']

param_names = [
    'intervention\nthreshold',
    'welfare\ntarget %', 
    'welfare\namount',
    'stimulus\namount',
    'spawn\nthreshold',
    'emergency\nthreshold',
    'cooperation\nthreshold',
    'intervention\ncooldown'
]

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Side-by-side comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(param_names))
width = 0.35

bars1 = ax1.bar(x - width/2, genome_50, width, label='50-gen (Winner)', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, genome_150, width, label='150-gen (Failed)', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Parameter', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Genome Comparison: 50-gen vs 150-gen', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Log scale comparison (to handle large differences)
ax2 = plt.subplot(2, 2, 2)
bars1 = ax2.bar(x - width/2, genome_50, width, label='50-gen (Winner)', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x + width/2, genome_150, width, label='150-gen (Failed)', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Parameter', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('Log Scale Comparison (handles extreme differences)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
ax2.set_yscale('log')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Percentage change
ax3 = plt.subplot(2, 2, 3)
pct_changes = []
for i in range(len(genome_50)):
    if genome_50[i] == 0:
        pct_changes.append(0)
    else:
        pct = ((genome_150[i] - genome_50[i]) / genome_50[i]) * 100
        pct_changes.append(pct)

colors = ['#e74c3c' if p < 0 else '#2ecc71' for p in pct_changes]
bars = ax3.bar(x, pct_changes, color=colors, alpha=0.8)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Parameter', fontsize=12, fontweight='bold')
ax3.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
ax3.set_title('150-gen vs 50-gen: Percentage Change', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add text annotations for extreme changes
for i, (bar, pct) in enumerate(zip(bars, pct_changes)):
    if abs(pct) > 100:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct:+.0f}%', ha='center', va='bottom' if pct > 0 else 'top',
                fontsize=8, fontweight='bold')

# 4. Key insights text
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

insights_text = """
🔍 KEY INSIGHTS

50-gen Strategy (WINNER):
✅ Rare interventions (threshold=5.0)
✅ Microscopic welfare ($0.0001)
✅ Magic 2π stimulus ($6.28)
✅ Quick cooldown (10 gen)
✅ Philosophy: "Do little, do it perfectly"

150-gen Strategy (FAILED):
❌ Constant interventions (threshold=0.47)
❌ Massive welfare ($126.25)
❌ Lost the magic (stimulus=$1695)
❌ Slow cooldown (15 gen)
❌ Philosophy: "Intervene often, heavy-handed"

💡 THE LESSON:
Longer training ≠ Better performance
Constraints breed creativity (2π discovery!)
Over-intervention disrupts natural dynamics
"LESS IS MORE" wins!

🎯 MULTI-CONTROLLER HYPOTHESIS:
What if we need DIFFERENT controllers
for DIFFERENT lifecycle phases?
• Gen 0-50: Quantum 50-gen (specialist)
• Gen 51+: GPT-4 Neutral (adaptive)
"""

ax4.text(0.1, 0.9, insights_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/god_ai/genome_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: outputs/god_ai/genome_comparison.png")

# Create a second figure focusing on intervention behavior
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Intervention threshold comparison
strategies = ['50-gen\n(Winner)', '150-gen\n(Failed)']
thresholds = [genome_50[0], genome_150[0]]
colors_thresh = ['#2ecc71', '#e74c3c']

ax1.bar(strategies, thresholds, color=colors_thresh, alpha=0.8, edgecolor='black', linewidth=2)
ax1.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Typical Gini (~0.4)')
ax1.set_ylabel('Intervention Threshold', fontsize=12, fontweight='bold')
ax1.set_title('Intervention Trigger Comparison\n(Higher = Rarer interventions)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add annotations
ax1.text(0, genome_50[0], f'{genome_50[0]:.2f}\n(RARE)', ha='center', va='bottom', fontweight='bold')
ax1.text(1, genome_150[0], f'{genome_150[0]:.2f}\n(FREQUENT)', ha='center', va='bottom', fontweight='bold')

# Welfare amount comparison (log scale due to huge difference)
welfare_50 = genome_50[2]
welfare_150 = genome_150[2]

ax2.bar(strategies, [welfare_50, welfare_150], color=colors_thresh, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Welfare Amount ($)', fontsize=12, fontweight='bold')
ax2.set_title('Welfare Payment Comparison\n(Log scale - 150-gen gives 1.26M× MORE!)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

# Add annotations
ax2.text(0, welfare_50, f'${welfare_50:.6f}\n(SUBTLE)', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.text(1, welfare_150, f'${welfare_150:.2f}\n(MASSIVE)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/god_ai/intervention_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: outputs/god_ai/intervention_comparison.png")

print("\n" + "=" * 80)
print("📊 Genome visualization complete!")
print("=" * 80)
print("\nTwo charts created:")
print("  1. genome_comparison.png - Full parameter comparison + insights")
print("  2. intervention_comparison.png - Focus on intervention behavior")
print("\nKey finding: 50-gen learned 'less is more', 150-gen learned 'more is more'")
print("             Reality favors subtle guidance over heavy intervention!")
print()
