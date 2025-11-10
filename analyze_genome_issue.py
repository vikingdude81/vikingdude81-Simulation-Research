"""Analyze specialist genomes"""
import json

files = {
    'volatile': 'outputs/conductor_enhanced_volatile_20251107_004635.json',
    'trending': 'outputs/conductor_enhanced_trending_20251108_001047.json',
    'ranging': 'outputs/conductor_enhanced_ranging_20251108_024640.json'
}

print("="*70)
print("SPECIALIST GENOME ANALYSIS - Why Only 1 Trade?")
print("="*70)

for regime, filepath in files.items():
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    genome = data['best_agent']['genome']
    fitness = data['best_fitness']
    
    print(f"\n{regime.upper()} (fitness={fitness:.2f}):")
    print(f"  [0] Stop Loss:        {genome[0]:.4f}")
    print(f"  [1] Take Profit:      {genome[1]:.4f}")
    print(f"  [2] Position Size:    {genome[2]:.4f}")
    print(f"  [3] Entry Threshold:  {genome[3]:.4f}  ⚠️  (needs >this to enter)")
    print(f"  [4] Exit Threshold:   {genome[4]:.4f}  ⚠️  (exits if <this)")
    print(f"  [5] Max Hold Time:    {int(genome[5])} days  ⚠️⚠️  (0 = exit immediately!)")
    print(f"  [6] Volatility Scale: {genome[6]:.4f}")
    print(f"  [7] Trend Sensitivity:{genome[7]:.4f}")

print("\n" + "="*70)
print("PROBLEM IDENTIFIED!")
print("="*70)
print("\n❌ Max Hold Time = 0 for all specialists!")
print("   This means positions are exited the SAME DAY they're entered.")
print("   No wonder we only have 1 trade - they can't hold positions!")
print("\n💡 SOLUTION:")
print("   Max hold time should be 1-14 days (gene value * 14)")
print("   But gene[5] is being used directly as days instead of scaling!")
print("\n🔍 Need to check trading_specialist.py line ~93:")
print("   self.max_hold_time = int(self.genome[5])")
print("   Should be: self.max_hold_time = max(1, int(self.genome[5] * 14))")
