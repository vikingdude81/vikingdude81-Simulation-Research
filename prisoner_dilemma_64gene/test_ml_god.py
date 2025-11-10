"""
🧬 TEST QUANTUM ML-BASED GOD CONTROLLER

Run a quick simulation with the quantum ML god controller to verify integration.
"""

import sys
from pathlib import Path

# Run a short simulation with ML-based God
from prisoner_echo_god import run_god_echo_simulation

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧬 TESTING QUANTUM ML-BASED GOD CONTROLLER")
    print("="*70)
    
    # Run simulation with ML-based God
    result = run_god_echo_simulation(
        generations=50,  # Short test
        initial_size=300,
        god_mode="ML_BASED",  # Use quantum controller!
        update_frequency=10
    )
    
    print("\n📊 Simulation Results:")
    print(f"   Final population: {len(result.agents)}")
    avg_wealth = sum(a.resources for a in result.agents) / len(result.agents)
    print(f"   Avg wealth: {avg_wealth:.1f}")
    total_actions = sum(a.cooperations + a.defections for a in result.agents)
    total_coop = sum(a.cooperations for a in result.agents)
    coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
    print(f"   Cooperation rate: {coop_rate:.1%}")
    print(f"   God interventions: {result.god.total_interventions}")
    
    # Show quantum controller stats if available
    if hasattr(result.god, 'quantum_controller') and result.god.quantum_controller:
        stats = result.god.quantum_controller.get_statistics()
        print(f"\n🧬 Quantum Controller Stats:")
        print(f"   Total interventions: {stats['total_interventions']}")
        print(f"   Timesteps evolved: {stats['timesteps']}")
        print(f"   Current fitness: {stats['fitness']:.0f}")
        print(f"   Current traits:")
        print(f"      Creativity: {stats['current_traits']['creativity']:.2f}")
        print(f"      Coherence: {stats['current_traits']['coherence']:.2f}")
        print(f"      Longevity: {stats['current_traits']['longevity']:.2f}")
    
    print("\n✅ Test complete!")
