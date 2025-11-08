"""
🎯 QUICK TEST: LLM God Controller

This script demonstrates and tests the LLM God controller in three modes:
1. Mock mode (no API, simulated responses)
2. Real API mode (requires API key)
3. Comparison with Quantum ML God

Run this to see how LLM governance works!
"""

from llm_god_controller import LLMGodController
import json


def test_mock_mode():
    """Test with mock mode (no API required)."""
    print("\n" + "="*70)
    print("🎭 TEST 1: MOCK MODE (Simulated LLM)")
    print("="*70)
    
    controller = LLMGodController(
        provider="mock",
        intervention_cooldown=0  # No cooldown for testing
    )
    
    # Test critical scenario
    crisis_state = {
        'population': 120,
        'avg_wealth': 45,
        'cooperation_rate': 0.28,
        'gini_coefficient': 0.68,
        'tribe_dominance': 0.75,
        'generation': 75
    }
    
    print("\n🚨 CRISIS SCENARIO:")
    print(f"   Population: {crisis_state['population']} (low!)")
    print(f"   Wealth: ${crisis_state['avg_wealth']} (very low!)")
    print(f"   Cooperation: {crisis_state['cooperation_rate']:.1%} (low!)")
    print(f"   Gini: {crisis_state['gini_coefficient']} (high inequality!)")
    
    decision = controller.decide_intervention(crisis_state, 75)
    
    if decision:
        itype, reasoning, params = decision
        print(f"\n✅ LLM DECISION:")
        print(f"   Type: {itype}")
        print(f"   {reasoning}")
        print(f"   Parameters: {json.dumps(params, indent=6)}")
    else:
        print("\n❌ No intervention (unexpected for crisis!)")
    
    return controller


def test_real_api_mode():
    """Test with real OpenAI API (requires key)."""
    print("\n" + "="*70)
    print("🌐 TEST 2: REAL API MODE (OpenAI GPT-4)")
    print("="*70)
    
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        print("   To test with real API:")
        print("   1. Get API key from https://platform.openai.com/api-keys")
        print("   2. Set environment variable:")
        print("      Windows: $env:OPENAI_API_KEY='sk-...'")
        print("      Linux/Mac: export OPENAI_API_KEY='sk-...'")
        print("   3. Run this script again")
        print("\n   Skipping real API test...")
        return None
    
    controller = LLMGodController(
        provider="openai",
        model="gpt-4",
        intervention_cooldown=0
    )
    
    # Test inequality scenario
    inequality_state = {
        'population': 850,
        'avg_wealth': 120,
        'cooperation_rate': 0.52,
        'gini_coefficient': 0.72,
        'tribe_dominance': 0.65,
        'generation': 50
    }
    
    print("\n📊 INEQUALITY SCENARIO:")
    print(f"   Population: {inequality_state['population']} (healthy)")
    print(f"   Wealth: ${inequality_state['avg_wealth']} (good)")
    print(f"   Cooperation: {inequality_state['cooperation_rate']:.1%} (moderate)")
    print(f"   Gini: {inequality_state['gini_coefficient']} (VERY HIGH!)")
    
    print("\n⏳ Calling GPT-4... (may take 2-3 seconds)")
    
    try:
        decision = controller.decide_intervention(inequality_state, 50)
        
        if decision:
            itype, reasoning, params = decision
            print(f"\n✅ GPT-4 DECISION:")
            print(f"   Type: {itype}")
            print(f"   {reasoning}")
            print(f"   Parameters: {json.dumps(params, indent=6)}")
        else:
            print("\n🤔 GPT-4 decided no intervention needed")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   This is normal if:")
        print("   - API key is invalid")
        print("   - You don't have OpenAI credits")
        print("   - Network issue")
    
    return controller


def compare_mock_vs_quantum():
    """Compare mock LLM with quantum controller."""
    print("\n" + "="*70)
    print("⚔️  TEST 3: MOCK LLM vs QUANTUM ML")
    print("="*70)
    
    # Try to import quantum controller
    try:
        from quantum_god_controller import QuantumGodController
        has_quantum = True
    except ImportError:
        print("\n⚠️  quantum_god_controller.py not found")
        print("   Skipping comparison test...")
        return
    
    llm_controller = LLMGodController(provider="mock", intervention_cooldown=0)
    quantum_controller = QuantumGodController(intervention_cooldown=0)
    
    # Test same scenario with both
    test_state = {
        'population': 600,
        'avg_wealth': 85,
        'cooperation_rate': 0.48,
        'gini_coefficient': 0.62,
        'tribe_dominance': 0.68,
        'generation': 100
    }
    
    print("\n📊 TEST SCENARIO:")
    print(f"   Population: {test_state['population']}")
    print(f"   Wealth: ${test_state['avg_wealth']}")
    print(f"   Cooperation: {test_state['cooperation_rate']:.1%}")
    print(f"   Gini: {test_state['gini_coefficient']}")
    
    # LLM decision
    llm_decision = llm_controller.decide_intervention(test_state, 100)
    
    print("\n🤖 MOCK LLM DECISION:")
    if llm_decision:
        itype, reasoning, params = llm_decision
        print(f"   Type: {itype}")
        print(f"   Reasoning: {reasoning.split(chr(10))[1].strip()}")  # Just main reasoning
        print(f"   Parameters: {params}")
    else:
        print("   No intervention")
    
    # Quantum decision
    quantum_decision = quantum_controller.decide_intervention(test_state, 100)
    
    print("\n🧬 QUANTUM ML DECISION:")
    if quantum_decision:
        itype, reasoning, params = quantum_decision
        print(f"   Type: {itype}")
        print(f"   Reasoning: {reasoning.split(chr(10))[1].strip()}")  # Just main reasoning
        print(f"   Parameters: {params}")
    else:
        print("   No intervention")
    
    # Compare
    print("\n🔍 COMPARISON:")
    if llm_decision and quantum_decision:
        if llm_decision[0] == quantum_decision[0]:
            print("   ✅ Both chose same intervention type!")
        else:
            print(f"   🔄 Different choices: LLM={llm_decision[0]}, Quantum={quantum_decision[0]}")
    
    print("\n💡 KEY DIFFERENCES:")
    print("   LLM (Mock):")
    print("   - Simulated reasoning based on rules")
    print("   - Fast (< 0.001s)")
    print("   - Good for testing API integration")
    print("")
    print("   Quantum ML:")
    print("   - Evolved through genetic algorithms")
    print("   - Fast (< 0.001s)")
    print("   - Proven performance (326.6 score in testing)")


def show_cost_analysis():
    """Show cost comparison."""
    print("\n" + "="*70)
    print("💰 COST ANALYSIS")
    print("="*70)
    
    print("\n📊 Cost per 100-generation run:")
    print("")
    print("   Rule-Based:    $0.00      (instant)")
    print("   Quantum ML:    $0.00      (instant)")
    print("   Mock LLM:      $0.00      (instant, no real API)")
    print("   GPT-4:         $0.24      (2-3 sec per decision)")
    print("   GPT-3.5:       $0.024     (1-2 sec per decision)")
    print("   Claude:        $0.20      (2-3 sec per decision)")
    print("")
    print("📈 Cost for 1000 runs (comprehensive testing):")
    print("")
    print("   Rule-Based:    $0")
    print("   Quantum ML:    $0         ← Recommended for testing")
    print("   GPT-4:         $240")
    print("   GPT-3.5:       $24")
    print("")
    print("💡 RECOMMENDATION:")
    print("   - Development: Mock LLM (free, instant)")
    print("   - Testing: Quantum ML (free, proven)")
    print("   - Production: Quantum ML (free, fast)")
    print("   - Analysis: Real LLM (expensive, interpretable)")
    print("   - Demos: Real LLM (impressive, worth cost)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 LLM GOD CONTROLLER - QUICK TEST SUITE")
    print("="*70)
    print("\nThis will test the LLM God controller in different modes.")
    print("Tests run: Mock mode → Real API (if key available) → Comparison")
    
    # Test 1: Mock mode (always works)
    mock_controller = test_mock_mode()
    
    # Test 2: Real API (optional)
    real_controller = test_real_api_mode()
    
    # Test 3: Comparison (if quantum available)
    compare_mock_vs_quantum()
    
    # Cost analysis
    show_cost_analysis()
    
    # Summary
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)
    
    if mock_controller:
        stats = mock_controller.get_statistics()
        print(f"\nMock Controller Stats:")
        print(f"   API calls: {stats['total_api_calls']}")
        print(f"   Interventions: {stats['total_interventions']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
    
    if real_controller:
        stats = real_controller.get_statistics()
        print(f"\nReal API Controller Stats:")
        print(f"   API calls: {stats['total_api_calls']}")
        print(f"   Failed calls: {stats['failed_calls']}")
        print(f"   Interventions: {stats['total_interventions']}")
    
    print("\n📚 NEXT STEPS:")
    print("   1. Read LLM_GOD_GUIDE.md for detailed usage")
    print("   2. Read GOD_CONTROLLER_COMPARISON.md for comparison")
    print("   3. Try integrating into prisoner_echo_god.py")
    print("   4. Run comparative test with all controllers")
    print("")
    print("🎯 Files created:")
    print("   - llm_god_controller.py (implementation)")
    print("   - LLM_GOD_GUIDE.md (comprehensive guide)")
    print("   - GOD_CONTROLLER_COMPARISON.md (detailed comparison)")
    print("   - test_llm_god_quick.py (this file)")
    

if __name__ == "__main__":
    main()
