"""
🏛️ GOVERNMENT COMPARISON LAUNCHER
==================================

Easy launcher to test different government styles with the ultimate dashboard.
"""

from ultimate_dashboard import run_ultimate_dashboard
from government_styles import GovernmentStyle

def main():
    print("\n" + "="*80)
    print("🏛️  GOVERNMENT STYLE COMPARISON - Ultimate Dashboard")
    print("="*80)
    print("\nAvailable Government Styles:")
    print("  1. LAISSEZ_FAIRE     - No intervention, pure market")
    print("  2. WELFARE_STATE     - Tax rich (30%), help poor")
    print("  3. AUTHORITARIAN     - Remove defectors forcibly")
    print("  4. CENTRAL_BANKER    - Stimulus when avg wealth < 15")
    print("  5. MIXED_ECONOMY     - Adaptive policy switching")
    print("\n  0. Run All (Sequential Comparison)")
    print("="*80)
    
    choice = input("\nSelect government style (0-5): ").strip()
    
    governments = {
        '1': ('LAISSEZ_FAIRE', GovernmentStyle.LAISSEZ_FAIRE),
        '2': ('WELFARE_STATE', GovernmentStyle.WELFARE_STATE),
        '3': ('AUTHORITARIAN', GovernmentStyle.AUTHORITARIAN),
        '4': ('CENTRAL_BANKER', GovernmentStyle.CENTRAL_BANKER),
        '5': ('MIXED_ECONOMY', GovernmentStyle.MIXED_ECONOMY)
    }
    
    if choice == '0':
        # Run all governments sequentially
        print("\n🔄 Running all 5 government styles sequentially...")
        print("   Each will run for 300 generations")
        print("   Close each window when ready to proceed to next\n")
        
        for name, gov_style in governments.values():
            print(f"\n{'='*80}")
            print(f"🏛️  Now Running: {name}")
            print('='*80)
            
            run_ultimate_dashboard(
                initial_size=200,
                generations=300,
                government_style=gov_style,
                grid_size=(75, 75),
                use_gpu=False,
                update_every=3
            )
            
            print(f"\n✅ {name} completed!")
        
        print("\n🎉 All government styles tested!")
        print("Check the results to compare cooperation rates!")
        
    elif choice in governments:
        name, gov_style = governments[choice]
        print(f"\n🚀 Launching with {name}...")
        
        run_ultimate_dashboard(
            initial_size=200,
            generations=300,
            government_style=gov_style,
            grid_size=(75, 75),
            use_gpu=False,
            update_every=3
        )
        
        print(f"\n✅ {name} simulation completed!")
    
    else:
        print("\n❌ Invalid choice. Please run again and select 0-5.")


if __name__ == "__main__":
    main()
