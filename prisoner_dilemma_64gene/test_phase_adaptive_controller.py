"""
ADVANCED Multi-Controller Hypothesis:
Implement ACTUAL controller switching mid-simulation.

Strategy:
- Phase 1 (Gen 0-50):   Use Quantum ML 50-gen (aggressive early game)
- Phase 2 (Gen 51-100): Switch to GPT-4 Neutral (adaptive mid-game)  
- Phase 3 (Gen 101+):   Keep GPT-4 Neutral (long-term stability)

This requires modifying the simulation to switch controllers dynamically.
"""

import json
from datetime import datetime
from prisoner_echo_god import GodEchoPopulation
import os

def run_phase_adaptive_simulation(generations=150, initial_size=300):
    """
    Run simulation with controller switching at phase boundaries
    """
    
    print("=" * 80)
    print("🎭 PHASE-ADAPTIVE MULTI-CONTROLLER SIMULATION")
    print("=" * 80)
    print()
    print("Controller Strategy:")
    print("  Phase 1 (Gen 0-50):   Quantum ML 50-gen (specialist)")
    print("  Phase 2 (Gen 51-100): GPT-4 Neutral (adaptive)")
    print("  Phase 3 (Gen 101+):   GPT-4 Neutral (stable)")
    print()
    print(f"Running {generations} generations with {initial_size} initial agents...")
    print()
    
    # Load 50-gen genome
    with open("outputs/god_ai/quantum_evolution_champion.json", 'r') as f:
        genome_50 = json.load(f)['champion']['genome']
    
    # Phase 1: Quantum ML (0-50 gen)
    print("🚀 PHASE 1: Quantum ML Controller (Gen 0-50)")
    print("-" * 80)
    
    pop_phase1 = GodEchoPopulation(
        size=initial_size,
        width=100,
        height=100,
        god_mode="QUANTUM_ML",
        quantum_genome=genome_50
    )
    
    for gen in range(51):  # 0-50
        pop_phase1.step()
        
        if gen % 10 == 0:
            stats = pop_phase1.get_stats()
            print(f"  Gen {gen:3d}: Pop={len(pop_phase1.agents):4d} | "
                  f"Coop={stats['cooperation_rate']:.1%} | "
                  f"Wealth=${stats['avg_wealth']:.0f} | "
                  f"Interventions={len(pop_phase1.god_controller.intervention_log)}")
    
    phase1_stats = pop_phase1.get_stats()
    phase1_interventions = len(pop_phase1.god_controller.intervention_log)
    
    print(f"\n✅ Phase 1 Complete:")
    print(f"   Population: {len(pop_phase1.agents)}")
    print(f"   Cooperation: {phase1_stats['cooperation_rate']:.1%}")
    print(f"   Avg Wealth: ${phase1_stats['avg_wealth']:.0f}")
    print(f"   Interventions: {phase1_interventions}")
    print()
    
    # Phase 2: GPT-4 Neutral (51-100 gen)
    if generations > 50:
        print("🔄 PHASE 2: GPT-4 Neutral Controller (Gen 51-100)")
        print("-" * 80)
        
        # Switch controller
        pop_phase1.god_controller.switch_mode("API_BASED", prompt_style="neutral")
        
        for gen in range(51, min(101, generations + 1)):
            pop_phase1.step()
            
            if gen % 10 == 0:
                stats = pop_phase1.get_stats()
                print(f"  Gen {gen:3d}: Pop={len(pop_phase1.agents):4d} | "
                      f"Coop={stats['cooperation_rate']:.1%} | "
                      f"Wealth=${stats['avg_wealth']:.0f} | "
                      f"Interventions={len(pop_phase1.god_controller.intervention_log)}")
        
        phase2_stats = pop_phase1.get_stats()
        phase2_interventions = len(pop_phase1.god_controller.intervention_log) - phase1_interventions
        
        print(f"\n✅ Phase 2 Complete:")
        print(f"   Population: {len(pop_phase1.agents)}")
        print(f"   Cooperation: {phase2_stats['cooperation_rate']:.1%}")
        print(f"   Avg Wealth: ${phase2_stats['avg_wealth']:.0f}")
        print(f"   Phase 2 Interventions: {phase2_interventions}")
        print()
    
    # Phase 3: Continue GPT-4 Neutral (101+ gen)
    if generations > 100:
        print("⚡ PHASE 3: GPT-4 Neutral Controller (Gen 101+)")
        print("-" * 80)
        
        for gen in range(101, generations + 1):
            pop_phase1.step()
            
            if gen % 10 == 0:
                stats = pop_phase1.get_stats()
                print(f"  Gen {gen:3d}: Pop={len(pop_phase1.agents):4d} | "
                      f"Coop={stats['cooperation_rate']:.1%} | "
                      f"Wealth=${stats['avg_wealth']:.0f} | "
                      f"Interventions={len(pop_phase1.god_controller.intervention_log)}")
        
        phase3_stats = pop_phase1.get_stats()
        phase3_interventions = len(pop_phase1.god_controller.intervention_log) - phase1_interventions - phase2_interventions
        
        print(f"\n✅ Phase 3 Complete:")
        print(f"   Population: {len(pop_phase1.agents)}")
        print(f"   Cooperation: {phase3_stats['cooperation_rate']:.1%}")
        print(f"   Avg Wealth: ${phase3_stats['avg_wealth']:.0f}")
        print(f"   Phase 3 Interventions: {phase3_interventions}")
        print()
    
    # Final results
    final_stats = pop_phase1.get_stats()
    total_interventions = len(pop_phase1.god_controller.intervention_log)
    
    print("=" * 80)
    print("🏁 FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"Final Generation: {generations}")
    print(f"Final Population: {len(pop_phase1.agents)}")
    print(f"Final Cooperation: {final_stats['cooperation_rate']:.1%}")
    print(f"Final Wealth: ${final_stats['avg_wealth']:.0f}")
    print(f"Final Gini: {final_stats['gini']:.3f}")
    print(f"Total Interventions: {total_interventions}")
    print()
    
    # Calculate score
    score = (
        final_stats['avg_wealth'] +
        final_stats['cooperation_rate'] * 10000 +
        (1 - final_stats['gini']) * 10000
    )
    
    print(f"📊 Final Score: {score:.0f}")
    print()
    
    # Save results
    result = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'strategy': 'phase_adaptive',
        'generations': generations,
        'phases': {
            'phase1': {
                'name': 'Quantum ML 50-gen',
                'generations': '0-50',
                'interventions': phase1_interventions,
                'final_cooperation': phase1_stats['cooperation_rate'],
                'final_wealth': phase1_stats['avg_wealth']
            },
            'phase2': {
                'name': 'GPT-4 Neutral',
                'generations': '51-100',
                'interventions': phase2_interventions if generations > 50 else 0,
                'final_cooperation': phase2_stats['cooperation_rate'] if generations > 50 else None,
                'final_wealth': phase2_stats['avg_wealth'] if generations > 50 else None
            },
            'phase3': {
                'name': 'GPT-4 Neutral',
                'generations': '101+',
                'interventions': phase3_interventions if generations > 100 else 0,
                'final_cooperation': phase3_stats['cooperation_rate'] if generations > 100 else None,
                'final_wealth': phase3_stats['avg_wealth'] if generations > 100 else None
            }
        },
        'final_stats': {
            'population': len(pop_phase1.agents),
            'cooperation': final_stats['cooperation_rate'],
            'wealth': final_stats['avg_wealth'],
            'gini': final_stats['gini'],
            'score': score,
            'total_interventions': total_interventions
        }
    }
    
    output_file = f"outputs/god_ai/phase_adaptive_simulation_{result['timestamp']}.json"
    os.makedirs("outputs/god_ai", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Results saved: {output_file}")
    print()
    
    return result

if __name__ == "__main__":
    # Run one test at each time horizon
    print("\n" + "=" * 80)
    print("🧪 TESTING PHASE-ADAPTIVE STRATEGY")
    print("=" * 80)
    print()
    print("This will test if dynamically switching controllers")
    print("outperforms using a single controller for the entire simulation.")
    print()
    
    test_lengths = [50, 100, 150]
    results = []
    
    for length in test_lengths:
        print(f"\n{'#' * 80}")
        print(f"Testing {length} Generation Scenario")
        print(f"{'#' * 80}\n")
        
        result = run_phase_adaptive_simulation(generations=length, initial_size=300)
        results.append(result)
    
    # Compare results
    print("\n" + "=" * 80)
    print("📊 PHASE-ADAPTIVE STRATEGY SUMMARY")
    print("=" * 80)
    print()
    
    for result in results:
        print(f"{result['generations']} generations:")
        print(f"  Score: {result['final_stats']['score']:.0f}")
        print(f"  Cooperation: {result['final_stats']['cooperation']:.1%}")
        print(f"  Wealth: ${result['final_stats']['wealth']:.0f}")
        print(f"  Interventions: {result['final_stats']['total_interventions']}")
        print()
    
    print("💡 Compare these scores with the single-controller results")
    print("   from the ultimate showdown to see if adaptive switching helps!")
    print()
