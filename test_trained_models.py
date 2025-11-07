"""
Test Trained Models and Compare to Baseline
Test Enhanced ML Predictor and GA Conductor predictions
"""

import torch
import numpy as np
import json
from phase2_complete_implementation import EnhancedMLPredictor, GAConductor


def load_trained_models():
    """Load trained models from checkpoints"""
    print("\n" + "="*70)
    print("📂 LOADING TRAINED MODELS")
    print("="*70 + "\n")
    
    # Load Enhanced ML Predictor
    predictor = EnhancedMLPredictor(hidden_size=256)
    checkpoint = torch.load('outputs/enhanced_ml_predictor_best.pth')
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.eval()
    print(f"✅ Loaded Enhanced ML Predictor (best checkpoint)")
    
    # Load GA Conductor
    conductor = GAConductor(hidden_size=512)
    checkpoint = torch.load('outputs/ga_conductor_best.pth')
    conductor.load_state_dict(checkpoint['model_state_dict'])
    conductor.eval()
    print(f"✅ Loaded GA Conductor (best checkpoint)")
    
    return predictor, conductor


def test_predictor_scenarios(predictor):
    """Test Enhanced ML Predictor on different scenarios"""
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED ML PREDICTOR")
    print("="*70 + "\n")
    
    scenarios = [
        {
            'name': 'Early Gen, Low Diversity, High Crossover',
            'state': [
                0.5, 0.6, 0.4, 0.1, 0.1, 0.4,  # pop stats
                0.1, 0.0, 0.5, 0.0,  # progress
                0.8, 0.1, 0.5  # config: high crossover, low mutation
            ],
            'expected': 'Should INCREASE mutation (explore), DECREASE crossover (inverse)'
        },
        {
            'name': 'Mid Gen, High Diversity, Converging',
            'state': [
                0.8, 0.9, 0.7, 0.1, 0.8, 0.4,  # pop stats
                0.5, 0.0, 0.3, 0.0,  # progress
                0.3, 0.5, 0.5  # config: low crossover, moderate mutation
            ],
            'expected': 'Should DECREASE mutation (converge), INCREASE crossover'
        },
        {
            'name': 'Late Gen, Low Diversity, Stagnant',
            'state': [
                0.7, 0.8, 0.6, 0.1, 0.2, 0.4,  # pop stats
                0.9, 0.8, 0.0, 1.0,  # progress: late, stagnant!
                0.5, 0.2, 0.5  # config: moderate crossover, low mutation
            ],
            'expected': 'Should INCREASE mutation (break stagnation)'
        }
    ]
    
    predictor.eval()
    for scenario in scenarios:
        state = torch.FloatTensor([scenario['state']])
        
        with torch.no_grad():
            mut, cross = predictor(state)
        
        print(f"📋 {scenario['name']}")
        print(f"   Current config: Mutation={scenario['state'][11]:.2f}, Crossover={scenario['state'][10]:.2f}")
        print(f"   Predicted: Mutation={mut.item():.3f}, Crossover={cross.item():.3f}")
        print(f"   Expected: {scenario['expected']}")
        print()


def test_conductor_scenarios(conductor):
    """Test GA Conductor on different scenarios"""
    print("\n" + "="*70)
    print("🧪 TESTING GA CONDUCTOR")
    print("="*70 + "\n")
    
    scenarios = [
        {
            'name': 'Early Gen, Low Diversity, Exploring',
            'state': [
                # Population stats (10)
                0.5, 0.6, 0.4, 0.1, 0.1,  # fitness
                0.1, 0.0, 0.5, 0.0, 0.4,  # progress
                # Wealth percentiles (6)
                0.2, 0.3, 0.5, 0.7, 0.9, 0.5,
                # Age metrics (3)
                5.0, 20.0, 0.8,
                # Strategy diversity (2)
                10.0, 0.4,
                # Config (4)
                0.4, 0.7, 0.1, 0.5
            ],
            'expected': 'HIGH mutation, immigration, diversity injection'
        },
        {
            'name': 'Late Gen, High Diversity, Converging',
            'state': [
                # Population stats (10)
                0.9, 0.95, 0.85, 0.05, 0.8,  # high fitness, high diversity
                0.9, 0.0, 0.3, 0.0, 0.8,  # late gen, converging
                # Wealth percentiles (6)
                0.7, 0.8, 0.9, 0.95, 0.98, 0.2,  # good distribution
                # Age metrics (3)
                100.0, 250.0, 0.1,  # old population
                # Strategy diversity (2)
                25.0, 0.2,  # high diversity
                # Config (4)
                0.8, 0.3, 0.5, 0.7
            ],
            'expected': 'LOW mutation, elite preservation, minimal intervention'
        },
        {
            'name': 'Crisis: Low Fitness, High Inequality',
            'state': [
                # Population stats (10)
                0.2, 0.4, -0.2, 0.3, 0.15,  # low fitness, low diversity
                0.5, 0.8, -0.3, 1.0, 0.3,  # mid gen, declining, stagnant
                # Wealth percentiles (6)
                -0.2, 0.0, 0.2, 0.8, 0.95, 0.85,  # extreme inequality
                # Age metrics (3)
                30.0, 120.0, 0.3,
                # Strategy diversity (2)
                3.0, 0.9,  # monoculture!
                # Config (4)
                0.3, 0.8, 0.05, 0.4
            ],
            'expected': 'Extinction trigger, welfare, diversity injection, high mutation'
        }
    ]
    
    conductor.eval()
    for scenario in scenarios:
        state = torch.FloatTensor([scenario['state']])
        
        with torch.no_grad():
            outputs = conductor(state)
        
        print(f"📋 {scenario['name']}")
        print(f"   Expected: {scenario['expected']}\n")
        
        print(f"   Evolution:")
        print(f"      Mutation:   {outputs['mutation_rate'].item():.3f}")
        print(f"      Crossover:  {outputs['crossover_rate'].item():.3f}")
        print(f"      Selection:  {outputs['selection_pressure'].item():.3f}")
        
        print(f"\n   Population:")
        print(f"      Delta:      {outputs['population_delta'].item():+.2f}")
        print(f"      Immigration: {outputs['immigration_rate'].item():.3f}")
        print(f"      Culling:    {outputs['culling_rate'].item():.3f}")
        print(f"      Diversity:  {outputs['diversity_injection'].item():.3f}")
        
        print(f"\n   Crisis:")
        print(f"      Extinction: {outputs['extinction_trigger'].item():.3f}")
        print(f"      Elite Pres: {outputs['elite_preservation'].item():.3f}")
        print(f"      Restart:    {outputs['restart_signal'].item():.3f}")
        
        print(f"\n   Institutional:")
        print(f"      Welfare:    {outputs['welfare_amount'].item():.3f}")
        print(f"      Tax:        {outputs['tax_rate'].item():.3f}")
        
        print("\n" + "-"*70 + "\n")


def compare_to_baseline():
    """Compare trained models to baseline fixed parameters"""
    print("\n" + "="*70)
    print("📊 BASELINE vs TRAINED MODELS COMPARISON")
    print("="*70 + "\n")
    
    # Load test data
    data = np.load('outputs/baseline_training_data.npz')
    states_13 = data['states_13']
    states_25 = data['states_25']
    
    # Baseline configuration (fixed)
    baseline_mutation = 0.1
    baseline_crossover = 0.7
    
    print(f"Baseline (Fixed):")
    print(f"  Mutation Rate:   {baseline_mutation:.3f} (NEVER CHANGES)")
    print(f"  Crossover Rate:  {baseline_crossover:.3f} (NEVER CHANGES)")
    print()
    
    # Load trained models
    predictor, conductor = load_trained_models()
    
    # Test on sample states
    sample_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800]
    
    print(f"Trained Models (Adaptive):\n")
    print(f"{'Gen':<6} {'Diversity':<10} {'Baseline':<20} {'Predictor':<20} {'Conductor':<20}")
    print("-" * 80)
    
    predictor.eval()
    conductor.eval()
    
    for idx in sample_indices:
        state_13 = torch.FloatTensor([states_13[idx]])
        state_25 = torch.FloatTensor([states_25[idx]])
        
        diversity = states_13[idx][4]
        generation = int(states_13[idx][6] * 300)
        
        with torch.no_grad():
            pred_mut, pred_cross = predictor(state_13)
            cond_outputs = conductor(state_25)
            cond_mut = cond_outputs['mutation_rate'].item()
            cond_cross = cond_outputs['crossover_rate'].item()
        
        print(f"{generation:<6} {diversity:<10.3f} "
              f"M:{baseline_mutation:.2f} C:{baseline_crossover:.2f}    "
              f"M:{pred_mut.item():.2f} C:{pred_cross.item():.2f}    "
              f"M:{cond_mut:.2f} C:{cond_cross:.2f}")
    
    print("\n" + "="*70)
    print("✅ Key Insight:")
    print("   Baseline: FIXED parameters throughout training")
    print("   Trained Models: ADAPTIVE parameters based on population state")
    print("   Expected: Faster convergence + better final fitness with adaptation")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🚀 TESTING TRAINED MODELS")
    print("="*70)
    
    # Load models
    predictor, conductor = load_trained_models()
    
    # Test scenarios
    test_predictor_scenarios(predictor)
    test_conductor_scenarios(conductor)
    
    # Compare to baseline
    compare_to_baseline()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\nNext step: Retrain a specialist with conductor control")
    print("and measure actual performance improvement!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
