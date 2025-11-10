"""
🏦 CENTRAL BANKER GOVERNMENT SIMULATION - 200 Agents
====================================================

Runs simulation with Central Banker government:
- Issues stimulus (+10 wealth to everyone) when avg wealth < 15
- Prevents economic collapse
- Should stabilize economy and cooperation
"""

from ultimate_dashboard import run_ultimate_dashboard
from government_styles import GovernmentStyle

print("\n" + "="*80)
print("🏦 CENTRAL BANKER GOVERNMENT - Ultimate Dashboard")
print("="*80)
print("\nGovernment Policy:")
print("  - Stimulus: +10 wealth to all when avg < 15")
print("  - Goal: Prevent economic collapse, stabilize")
print("\nExpected Outcome:")
print("  - Moderate cooperation (55-65%)")
print("  - More stable population")
print("  - Prevents catastrophic wealth loss")
print("="*80)

run_ultimate_dashboard(
    initial_size=200,
    generations=300,
    government_style=GovernmentStyle.CENTRAL_BANKER,
    grid_size=(75, 75),
    use_gpu=False,
    update_every=3
)

print("\n✅ Central Banker simulation completed!")
