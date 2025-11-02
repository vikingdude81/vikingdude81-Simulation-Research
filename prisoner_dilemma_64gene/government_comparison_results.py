"""
📊 GOVERNMENT COMPARISON RESULTS
================================

Results from all 5 government style experiments (200 agents, 300 generations)

Run this after completing all simulations to see comparison.
"""

print("\n" + "="*80)
print("📊 GOVERNMENT COMPARISON - RESULTS SUMMARY")
print("="*80)
print("\nAll simulations: 200 agents, 300 generations, 75×75 grid")
print("\n" + "-"*80)

results = {
    "Laissez-Faire": {
        "cooperation": 45.1,
        "population": 2812,
        "description": "No intervention, pure market forces"
    },
    "Welfare State": {
        "cooperation": 57.1,
        "population": 2812,
        "description": "30% tax on rich, redistribute to poor"
    },
    "Authoritarian": {
        "cooperation": None,  # Fill after completion
        "population": None,
        "description": "Remove defectors forcibly"
    },
    "Central Banker": {
        "cooperation": None,  # Fill after completion
        "population": None,
        "description": "Stimulus when avg wealth < 15"
    },
    "Mixed Economy": {
        "cooperation": None,  # Fill after completion
        "population": None,
        "description": "Adaptive policy switching"
    }
}

print("\n📈 COOPERATION RATE COMPARISON:\n")
for gov, data in results.items():
    if data["cooperation"] is not None:
        bar = "█" * int(data["cooperation"] / 2)  # Scale for visual
        print(f"{gov:20s} {data['cooperation']:5.1f}% {bar}")
        print(f"{'':20s} {data['description']}")
        print()
    else:
        print(f"{gov:20s} PENDING...")
        print()

print("\n" + "-"*80)
print("\n🎯 PREDICTIONS:")
print("  1. Authoritarian: 80-90% (forced cooperation)")
print("  2. Mixed Economy: 65-75% (adaptive optimization)")
print("  3. Welfare State: 57.1% ✅ (confirmed)")
print("  4. Central Banker: 55-65% (economic stabilization)")
print("  5. Laissez-Faire: 45.1% ✅ (confirmed)")

print("\n💡 HYPOTHESIS:")
print("  Government intervention can increase cooperation by 12-45%")
print("  Redistribution and enforcement both effective")
print("  Adaptive policy may be optimal")

print("\n" + "="*80)
print("\n⏳ Waiting for remaining simulations to complete...")
print("   Update this script with final cooperation rates!")
print("\n" + "="*80)
