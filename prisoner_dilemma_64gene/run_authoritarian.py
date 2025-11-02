"""
⚔️ AUTHORITARIAN GOVERNMENT SIMULATION - 200 Agents
===================================================

Runs simulation with Authoritarian government:
- Removes defectors forcibly (sets wealth to -9999)
- Harsh enforcement of cooperation
- Should achieve highest cooperation but may reduce diversity
"""

from ultimate_dashboard import run_ultimate_dashboard
from government_styles import GovernmentStyle

print("\n" + "="*80)
print("⚔️  AUTHORITARIAN GOVERNMENT - Ultimate Dashboard")
print("="*80)
print("\nGovernment Policy:")
print("  - Enforcement: Remove all defectors (wealth → -9999)")
print("  - Goal: Force cooperation through punishment")
print("\nExpected Outcome:")
print("  - Highest cooperation (80-90%)")
print("  - Reduced genetic diversity")
print("  - Artificial selection pressure")
print("="*80)

run_ultimate_dashboard(
    initial_size=200,
    generations=300,
    government_style=GovernmentStyle.AUTHORITARIAN,
    grid_size=(75, 75),
    use_gpu=False,
    update_every=3
)

print("\n✅ Authoritarian simulation completed!")
