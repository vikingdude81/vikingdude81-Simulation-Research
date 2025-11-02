"""
🔄 MIXED ECONOMY GOVERNMENT SIMULATION - 200 Agents
===================================================

Runs simulation with Mixed Economy government:
- Adaptively switches between policies based on conditions
- Laissez-faire if cooperation > 60%
- Welfare if Gini > 0.5 (high inequality)
- Central banker if avg wealth < 15
- Authoritarian if cooperation < 30%
- Should optimize dynamically
"""

from ultimate_dashboard import run_ultimate_dashboard
from government_styles import GovernmentStyle

print("\n" + "="*80)
print("🔄 MIXED ECONOMY GOVERNMENT - Ultimate Dashboard")
print("="*80)
print("\nGovernment Policy:")
print("  - Adaptive switching based on conditions:")
print("    * High cooperation (>60%) → Laissez-faire")
print("    * High inequality (Gini >0.5) → Welfare")
print("    * Low wealth (<15) → Central banker")
print("    * Low cooperation (<30%) → Authoritarian")
print("  - Goal: Dynamically respond to conditions")
print("\nExpected Outcome:")
print("  - Best overall performance (65-75%)")
print("  - Most stable across metrics")
print("  - Demonstrates adaptive governance")
print("="*80)

run_ultimate_dashboard(
    initial_size=200,
    generations=300,
    government_style=GovernmentStyle.MIXED_ECONOMY,
    grid_size=(75, 75),
    use_gpu=False,
    update_every=3
)

print("\n✅ Mixed Economy simulation completed!")
