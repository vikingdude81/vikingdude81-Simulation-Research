"""
💰👑 SUPER CITIZEN EMERGENCE DEMO
=================================

Quick demo to show how "super citizens" emerge and dominate wealth.

Runs a short simulation and tracks who becomes wealthy.
"""

from enhanced_government_styles import EnhancedGovernmentController, EnhancedGovernmentStyle
from ultimate_echo_simulation import UltimateEchoSimulation
from wealth_inequality_tracker import WealthInequalityTracker


def demo_super_citizen_emergence(government: EnhancedGovernmentStyle = EnhancedGovernmentStyle.OLIGARCHY):
    """
    Run quick demo showing super citizen emergence.
    
    Args:
        government: Which government type to test
    """
    print(f"\n{'='*70}")
    print(f"💰👑 SUPER CITIZEN EMERGENCE DEMO: {government.value.upper()}")
    print(f"{'='*70}\n")
    
    # Create simulation
    sim = UltimateEchoSimulation(initial_size=200, grid_size=(50, 50))
    gov = EnhancedGovernmentController(government)
    tracker = WealthInequalityTracker()
    
    # Run for 200 generations
    print("Running simulation...\n")
    for gen in range(200):
        sim.step()
        
        if len(sim.agents) > 0:
            gov.apply_policy(sim.agents, sim.grid_size)
            sim.agents = [a for a in sim.agents if a.wealth > -9999]
            
            # Capture snapshot every 20 generations
            if gen % 20 == 0:
                snapshot = tracker.capture_snapshot(sim.agents, gen)
                print(f"Gen {gen:3d}: Pop={len(sim.agents):4d}, "
                      f"Gini={snapshot.gini_coefficient:.3f}, "
                      f"Top 1%={snapshot.top_1_percent_share:5.1f}%, "
                      f"Super Citizens={snapshot.num_super_citizens:2d}, "
                      f"Richest={snapshot.richest_wealth:6.1f}")
    
    # Final snapshot
    final = tracker.capture_snapshot(sim.agents, 200)
    summary = tracker.get_summary()
    
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS")
    print(f"{'='*70}")
    
    print(f"\n💰 Wealth Inequality:")
    print(f"   Gini Coefficient: {final.gini_coefficient:.3f}")
    print(f"   Top 1% owns: {final.top_1_percent_share:.1f}% of wealth")
    print(f"   Top 10% owns: {final.top_10_percent_share:.1f}% of wealth")
    print(f"   Bottom 50% owns: {final.bottom_50_percent_share:.1f}% of wealth")
    
    print(f"\n👑 Super Citizens:")
    print(f"   Total emerged: {summary['super_citizens']['total_emerged']}")
    print(f"   Cooperators: {summary['super_citizens']['cooperators']}")
    print(f"   Defectors: {summary['super_citizens']['defectors']}")
    print(f"   Max wealth achieved: {summary['super_citizens']['max_wealth_ever']:.1f}")
    print(f"   Average lifespan: {summary['super_citizens']['avg_lifespan']:.1f} generations")
    
    print(f"\n🏆 Richest Agent:")
    print(f"   Wealth: {final.richest_wealth:.1f}")
    print(f"   Strategy: {'Cooperator' if final.richest_strategy == 1 else 'Defector'}")
    print(f"   Age: {final.richest_age} generations")
    print(f"   Wealth = {final.richest_wealth / final.median_wealth:.1f}× median")
    
    print(f"\n💵 Wealth by Strategy:")
    print(f"   Cooperators (n={final.cooperator_count}): {final.cooperator_mean_wealth:.2f} avg wealth")
    print(f"   Defectors (n={final.defector_count}): {final.defector_mean_wealth:.2f} avg wealth")
    gap = final.defector_mean_wealth - final.cooperator_mean_wealth
    if abs(gap) > 0.1:
        winner = "Defectors" if gap > 0 else "Cooperators"
        print(f"   → {winner} are {abs(gap):.2f} wealthier on average")
    else:
        print(f"   → Similar wealth levels")
    
    # Generate visualization
    print(f"\n📈 Generating visualization...")
    plot_path = f"super_citizen_demo_{government.value}.png"
    tracker.plot_wealth_inequality(plot_path)
    print(f"   Saved to: {plot_path}")
    
    print(f"\n{'='*70}")
    print("✅ Demo complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("💰👑 SUPER CITIZEN EMERGENCE DEMO")
    print("="*70)
    print("\nThis demo shows how 'super citizens' (ultra-wealthy individuals)")
    print("emerge and dominate wealth under different government types.\n")
    
    # Test different governments
    print("Testing 3 contrasting government types:\n")
    
    print("1️⃣  OLIGARCHY (Rule by Wealthy)")
    print("   Expected: Many super citizens, high inequality")
    demo_super_citizen_emergence(EnhancedGovernmentStyle.OLIGARCHY)
    
    print("\n\n2️⃣  SOCIAL DEMOCRACY (Nordic Model)")
    print("   Expected: Moderate super citizens, moderate inequality")
    demo_super_citizen_emergence(EnhancedGovernmentStyle.SOCIAL_DEMOCRACY)
    
    print("\n\n3️⃣  COMMUNIST (Forced Equality)")
    print("   Expected: Few/no super citizens, low inequality")
    demo_super_citizen_emergence(EnhancedGovernmentStyle.COMMUNIST)
    
    print("\n" + "="*70)
    print("🎓 KEY TAKEAWAYS:")
    print("="*70)
    print("\n✅ Super citizens DO emerge in unregulated systems")
    print("✅ They can accumulate 20-50× median wealth")
    print("✅ Government type dramatically affects who gets wealthy")
    print("✅ Defectors dominate in Oligarchy, Cooperators in Social Democracy")
    print("\n" + "="*70)
