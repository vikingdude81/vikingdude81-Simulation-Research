"""Analyze specialist genomes to understand conservatism"""
import json
import glob

regimes = ['volatile', 'trending', 'ranging']

print("="*60)
print("SPECIALIST GENOME ANALYSIS")
print("="*60)

for regime in regimes:
    # Find most recent result file
    pattern = f'outputs/conductor_enhanced_{regime}_*.json'
    files = glob.glob(pattern)
    if not files:
        print(f"\n❌ No results found for {regime}")
        continue
    
    latest_file = max(files)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    genome = data['best_genome']
    fitness = data['best_fitness']
    
    print(f"\n{regime.upper()} (fitness={fitness:.2f}):")
    print(f"  Genome: {[f'{g:.4f}' for g in genome]}")
    print(f"  Entry Threshold:  {genome[3]:.4f}  (signal strength needed to enter)")
    print(f"  Exit Threshold:   {genome[4]:.4f}  (signal strength to exit)")
    print(f"  Max Hold Time:    {int(genome[5])} days")
    print(f"  Stop Loss:        {genome[0]:.4f}")
    print(f"  Take Profit:      {genome[1]:.4f}")
    print(f"  Position Size:    {genome[2]:.4f}")

print("\n" + "="*60)
print("CONSERVATISM ANALYSIS")
print("="*60)
print("\nHigh entry thresholds (>0.5) = very selective trading")
print("High exit thresholds (>0.5) = quick exits")
print("Short max hold times (<7 days) = frequent turnover")
print("\nFor more trades, we need to REDUCE entry thresholds")
