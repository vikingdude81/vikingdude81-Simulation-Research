"""
🏛️ WELFARE STATE SIMULATION - 200 Agents
==========================================

Runs simulation with Welfare State government:
- Takes 30% tax from wealthy (>50 wealth)
- Redistributes to poor (<10 wealth)
- Should increase cooperation compared to laissez-faire
"""

from ultimate_dashboard import run_ultimate_dashboard
from government_styles import GovernmentStyle

print("\n" + "="*80)
print("🏛️  WELFARE STATE GOVERNMENT - Ultimate Dashboard")
print("="*80)
print("\nGovernment Policy:")
print("  - Tax: 30% from agents with wealth > 50")
print("  - Redistribution: Give to agents with wealth < 10")
print("  - Goal: Reduce inequality, promote cooperation")
print("\nExpected Outcome:")
print("  - Higher cooperation than laissez-faire (60-70% vs 45%)")
print("  - More stable population")
print("  - Lower wealth inequality")
print("="*80)

run_ultimate_dashboard(
    initial_size=200,
    generations=300,
    government_style=GovernmentStyle.WELFARE_STATE,
    grid_size=(75, 75),
    use_gpu=False,
    update_every=3
)

print("\n✅ Welfare State simulation completed!")
