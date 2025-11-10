"""
Quick test with your OpenAI API key.

Usage:
1. Set your API key:
   $env:OPENAI_API_KEY='sk-...'
   
2. Run this script:
   python test_with_your_api_key.py

Or pass key directly when prompted.
"""

from llm_god_controller import LLMGodController
import os

def test_with_real_gpt4():
    print("\n" + "="*70)
    print("🤖 TESTING LLM GOD WITH YOUR OPENAI API KEY")
    print("="*70)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        print("\nOptions:")
        print("1. Set environment variable:")
        print("   $env:OPENAI_API_KEY='sk-...'")
        print("\n2. Or enter it now (will be used for this session only):")
        
        api_key = input("\nEnter API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("\n❌ No API key provided. Exiting...")
            return
    
    print(f"\n✅ Using API key: sk-...{api_key[-4:]}")
    print("   (showing last 4 characters only for security)")
    
    # Create controller
    try:
        controller = LLMGodController(
            provider="openai",
            model="gpt-4",
            api_key=api_key,
            intervention_cooldown=0
        )
        print("✅ Controller initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing controller: {e}")
        return
    
    # Test scenario 1: High inequality
    print("\n" + "="*70)
    print("📊 TEST 1: HIGH INEQUALITY SCENARIO")
    print("="*70)
    
    state1 = {
        'population': 850,
        'avg_wealth': 120,
        'total_wealth': 102000,
        'cooperation_rate': 0.52,
        'clustering': 0.68,
        'gini_coefficient': 0.72,
        'wealth_inequality': 15.5,
        'tribe_diversity': 0.55,
        'tribe_dominance': 0.65,
        'growth_rate': -0.02,
        'generation': 50
    }
    
    print(f"\n📈 State:")
    print(f"   Population: {state1['population']} (healthy)")
    print(f"   Avg Wealth: ${state1['avg_wealth']} (good)")
    print(f"   Cooperation: {state1['cooperation_rate']:.1%}")
    print(f"   Gini: {state1['gini_coefficient']} (VERY HIGH - inequality!)")
    print(f"   Wealth declining: {state1['growth_rate']:.1%}")
    
    print("\n⏳ Asking GPT-4 for decision... (may take 2-3 seconds)")
    
    try:
        decision = controller.decide_intervention(state1, 50)
        
        if decision:
            itype, reasoning, params = decision
            print(f"\n✅ GPT-4 DECISION:")
            print(f"\n🎯 Intervention Type: {itype}")
            print(f"\n💭 Reasoning:")
            # Remove the provider prefix for cleaner output
            clean_reasoning = reasoning.replace("🤖 LLM GOD DECISION (openai/gpt-4)\n   ", "")
            print(f"   {clean_reasoning}")
            print(f"\n⚙️  Parameters:")
            for key, value in params.items():
                print(f"   - {key}: {value}")
        else:
            print("\n🤔 GPT-4 decided no intervention needed")
            print("   (This is unusual for such high inequality)")
    
    except Exception as e:
        print(f"\n❌ Error calling GPT-4: {e}")
        print("\n   Common issues:")
        print("   - Invalid API key")
        print("   - No credits remaining")
        print("   - Network connectivity")
        print("   - Rate limit exceeded")
        return
    
    # Test scenario 2: Population crisis
    print("\n" + "="*70)
    print("📊 TEST 2: POPULATION CRISIS")
    print("="*70)
    
    state2 = {
        'population': 150,
        'avg_wealth': 45,
        'total_wealth': 6750,
        'cooperation_rate': 0.28,
        'clustering': 0.35,
        'gini_coefficient': 0.65,
        'wealth_inequality': 18.2,
        'tribe_diversity': 0.42,
        'tribe_dominance': 0.78,
        'growth_rate': -0.08,
        'generation': 75
    }
    
    print(f"\n🚨 CRITICAL STATE:")
    print(f"   Population: {state2['population']} (VERY LOW!)")
    print(f"   Avg Wealth: ${state2['avg_wealth']} (collapsing)")
    print(f"   Cooperation: {state2['cooperation_rate']:.1%} (very low)")
    print(f"   Declining rapidly: {state2['growth_rate']:.1%} per generation")
    
    print("\n⏳ Asking GPT-4 for crisis response...")
    
    try:
        decision = controller.decide_intervention(state2, 75)
        
        if decision:
            itype, reasoning, params = decision
            print(f"\n✅ GPT-4 DECISION:")
            print(f"\n🎯 Intervention Type: {itype}")
            print(f"\n💭 Reasoning:")
            clean_reasoning = reasoning.replace("🤖 LLM GOD DECISION (openai/gpt-4)\n   ", "")
            print(f"   {clean_reasoning}")
            print(f"\n⚙️  Parameters:")
            for key, value in params.items():
                print(f"   - {key}: {value}")
        else:
            print("\n❌ GPT-4 decided no intervention")
            print("   (This is very concerning for a crisis state!)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Show stats
    print("\n" + "="*70)
    print("📊 SESSION STATISTICS")
    print("="*70)
    
    stats = controller.get_statistics()
    print(f"\n   Total API calls: {stats['total_api_calls']}")
    print(f"   Failed calls: {stats['failed_calls']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Interventions made: {stats['total_interventions']}")
    
    if stats['interventions_by_type']:
        print(f"\n   Interventions by type:")
        for itype, count in stats['interventions_by_type'].items():
            print(f"      {itype}: {count}")
    
    # Cost estimate
    estimated_cost = stats['total_api_calls'] * 0.003  # Rough estimate
    print(f"\n   Estimated cost: ${estimated_cost:.3f}")
    print(f"   (Based on ~$0.003 per call for GPT-4)")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print("\n💡 Key Observations:")
    print("   - LLM provides detailed, contextual reasoning")
    print("   - Decisions take 2-3 seconds (vs instant for Quantum)")
    print("   - Natural language explanations are human-readable")
    print("   - Can draw on broader knowledge (economics, history)")
    print("\n📚 Compare with:")
    print("   - Quantum ML: Instant, free, proven performance")
    print("   - Rule-based: Instant, free, but rigid")
    print("\n🎯 Best use case for LLM:")
    print("   - Explaining decisions to stakeholders")
    print("   - Handling novel/unexpected scenarios")
    print("   - Research requiring interpretability")


if __name__ == "__main__":
    test_with_real_gpt4()
