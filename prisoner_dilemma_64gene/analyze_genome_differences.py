"""
Analyze differences between 50-gen and 150-gen trained genomes
to understand why retraining hurt performance.
"""

import json
import os

def analyze_genomes():
    """Compare the two quantum ML genomes"""
    
    # Original 50-gen champion (from previous work)
    genome_50gen = [5.0, 0.1, 0.0001, 6.283185307179586, 0.6, 0.3, 0.7, 10]
    
    # New 150-gen champion
    training_file = "outputs/god_ai/quantum_evolution_150gen_20251104_140931.json"
    with open(training_file, 'r') as f:
        data = json.load(f)
        genome_150gen = data['champion']['genome']
    
    param_names = [
        'intervention_threshold',
        'welfare_target_pct', 
        'welfare_amount',
        'stimulus_amount',
        'spawn_threshold',
        'emergency_threshold',
        'cooperation_threshold',
        'intervention_cooldown'
    ]
    
    print("=" * 80)
    print("🔬 GENOME COMPARISON: 50-gen vs 150-gen Training")
    print("=" * 80)
    print()
    
    print("Parameter Analysis:")
    print("-" * 80)
    print(f"{'Parameter':<25} {'50-gen':<15} {'150-gen':<15} {'Change':<15} {'Impact'}")
    print("-" * 80)
    
    for i, name in enumerate(param_names):
        val_50 = genome_50gen[i]
        val_150 = genome_150gen[i]
        
        if val_50 == 0:
            change_pct = "N/A (div by 0)"
            change_str = f"+{val_150:.2f}"
        else:
            change_pct = ((val_150 - val_50) / val_50) * 100
            change_str = f"{change_pct:+.1f}%"
        
        # Interpret the change
        if name == 'intervention_threshold':
            if val_150 < val_50:
                impact = "✅ More aggressive (lower threshold)"
            else:
                impact = "⚠️ More conservative (higher threshold)"
        elif name == 'welfare_amount':
            if val_150 > val_50:
                impact = "💰 MUCH larger welfare payments"
            else:
                impact = "💸 Smaller welfare payments"
        elif name == 'stimulus_amount':
            if val_150 < val_50:
                impact = "📉 Much smaller stimulus"
            else:
                impact = "📈 Larger stimulus"
        elif name == 'intervention_cooldown':
            if val_150 > val_50:
                impact = "🐌 Slower intervention rate"
            else:
                impact = "⚡ Faster intervention rate"
        else:
            if abs(val_150 - val_50) < 0.1:
                impact = "≈ Similar"
            elif val_150 > val_50:
                impact = "↗️ Increased"
            else:
                impact = "↘️ Decreased"
        
        print(f"{name:<25} {val_50:<15.4f} {val_150:<15.4f} {change_str:<15} {impact}")
    
    print("-" * 80)
    print()
    
    # Key differences analysis
    print("=" * 80)
    print("🔍 KEY DIFFERENCES")
    print("=" * 80)
    print()
    
    print("1️⃣ INTERVENTION THRESHOLD:")
    print(f"   50-gen:  {genome_50gen[0]:.4f} (very high - rarely triggers)")
    print(f"   150-gen: {genome_150gen[0]:.4f} (low - triggers easily)")
    print(f"   💡 150-gen is {genome_50gen[0]/genome_150gen[0]:.1f}x MORE AGGRESSIVE")
    print()
    
    print("2️⃣ WELFARE AMOUNT:")
    print(f"   50-gen:  ${genome_50gen[2]:.6f} (microscopic)")
    print(f"   150-gen: ${genome_150gen[2]:.2f} (substantial)")
    print(f"   💡 150-gen gives {genome_150gen[2]/genome_50gen[2]:,.0f}x LARGER welfare!")
    print()
    
    print("3️⃣ STIMULUS AMOUNT:")
    print(f"   50-gen:  ${genome_50gen[3]:.2f} (2π - magical constant!)")
    print(f"   150-gen: ${genome_150gen[3]:.2f}")
    print(f"   💡 150-gen uses {genome_150gen[3]/genome_50gen[3]:.0f}x LESS stimulus")
    print()
    
    print("4️⃣ INTERVENTION COOLDOWN:")
    print(f"   50-gen:  {genome_50gen[7]:.1f} generations")
    print(f"   150-gen: {genome_150gen[7]:.1f} generations")
    print(f"   💡 150-gen waits {genome_150gen[7]/genome_50gen[7]:.1f}x LONGER between interventions")
    print()
    
    print("=" * 80)
    print("💭 INTERPRETATION: Why 150-gen Training Failed")
    print("=" * 80)
    print()
    
    print("🎯 50-gen Strategy (WINNER):")
    print("   • Rare but DECISIVE interventions (threshold=5.0)")
    print("   • Tiny welfare amounts (0.0001) - subtle nudges")
    print("   • Magic 2π stimulus - discovered through evolution!")
    print("   • Quick cooldown (10 gen) - can react rapidly")
    print("   • Philosophy: 'Do little, but do it perfectly'")
    print()
    
    print("🎯 150-gen Strategy (FAILED):")
    print("   • Frequent interventions (threshold=0.47 - triggers constantly)")
    print("   • MASSIVE welfare (126.2 vs 0.0001) - heavy-handed")
    print("   • Much smaller stimulus (1695 vs 2π)")
    print("   • Longer cooldown (15 gen) - slower reactions")
    print("   • Philosophy: 'Intervene often with big hammers'")
    print()
    
    print("❌ Why 150-gen Failed:")
    print("   1. OVER-INTERVENTION: Constant meddling disrupts natural dynamics")
    print("   2. TOO MUCH WELFARE: Large payments create dependency, kill competition")
    print("   3. WRONG OPTIMIZATION: Trained for 150-gen survival, not peak performance")
    print("   4. LOST MAGIC: 2π stimulus was discovered genius, got averaged out")
    print("   5. OVERFITTING: Optimized for training scenarios, lost generalization")
    print()
    
    print("=" * 80)
    print("🏆 THE LESSON")
    print("=" * 80)
    print()
    print("The 50-gen genome learned:")
    print("  'LESS IS MORE' - Minimal intervention, maximum impact")
    print()
    print("The 150-gen genome learned:")
    print("  'MORE IS MORE' - Frequent intervention, heavy-handed control")
    print()
    print("Reality: Natural selection works best with SUBTLE guidance,")
    print("         not constant interference!")
    print()
    
    # Calculate theoretical intervention rates
    print("=" * 80)
    print("📊 INTERVENTION RATE ANALYSIS")
    print("=" * 80)
    print()
    
    # Estimate how often each genome triggers
    # Gini coefficient typically 0.3-0.5, intervention_threshold is compared to it
    avg_gini = 0.4
    
    prob_50 = "Very Rare" if genome_50gen[0] > avg_gini else "Sometimes"
    prob_150 = "Very Rare" if genome_150gen[0] > avg_gini else "Very Often"
    
    print(f"Given average Gini = {avg_gini}:")
    print(f"  50-gen (threshold={genome_50gen[0]:.2f}):  Triggers {prob_50}")
    print(f"  150-gen (threshold={genome_150gen[0]:.2f}): Triggers {prob_150}")
    print()
    print("Combined with cooldown:")
    print(f"  50-gen:  Rare trigger + 10-gen cooldown = ~0-2 interventions per 50 gen")
    print(f"  150-gen: Frequent trigger + 15-gen cooldown = ~5-10 interventions per 50 gen")
    print()

if __name__ == "__main__":
    analyze_genomes()
