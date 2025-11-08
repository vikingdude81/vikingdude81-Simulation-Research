"""
Quick runner for wealth inequality analysis.
Tests 3 governments: Communist, Oligarchy, Laissez-Faire
"""

from wealth_inequality_tracker import WealthInequalityTracker
from enhanced_government_styles import EnhancedGovernmentStyle
from ultimate_echo_simulation import UltimateEchoSimulation

def main():
    print("\n" + "="*70)
    print("💰 WEALTH INEQUALITY & SUPER CITIZEN ANALYSIS")
    print("="*70)
    
    # Test 3 extreme cases
    governments = [
        EnhancedGovernmentStyle.COMMUNIST,      # Should have NO super citizens
        EnhancedGovernmentStyle.OLIGARCHY,      # Should have EXTREME super citizens
        EnhancedGovernmentStyle.LAISSEZ_FAIRE,  # Natural emergence
    ]
    
    for gov_style in governments:
        print(f"\n{'='*70}")
        print(f"Testing: {gov_style.value.upper()}")
        print(f"{'='*70}")
        
        # Create simulation
        sim = UltimateEchoSimulation(initial_size=200, grid_size=(50, 50))
        
        # Create wealth tracker
        tracker = WealthInequalityTracker(government_type=gov_style.value)
        
        # Run 300 generations
        for gen in range(300):
            sim.step()
            
            # Track wealth inequality
            if len(sim.agents) > 0:
                tracker.track_generation(sim.agents, gen)
            
            # Progress
            if (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1}/300: Pop={len(sim.agents)}, "
                      f"Top1%={tracker.top_1_percent_share[-1]:.1f}%, "
                      f"Gini={tracker.gini_history[-1]:.3f}")
        
        # Get summary
        summary = tracker.get_summary()
        print(f"\n📊 Summary:")
        print(f"  Top 1% owns: {summary['final_top_1_percent']:.1f}% of wealth")
        print(f"  Top 10% owns: {summary['final_top_10_percent']:.1f}% of wealth")
        print(f"  Gini coefficient: {summary['final_gini']:.3f}")
        print(f"  Super citizens emerged: {summary['super_citizens_emerged']}")
        
        # Save results
        filename = f"wealth_analysis_{gov_style.value}.json"
        tracker.save_results(filename)
        print(f"  ✅ Saved to: {filename}")

if __name__ == "__main__":
    main()
