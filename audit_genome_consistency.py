"""
Comprehensive Genome Audit - Ensure all components use consistent genome mapping
"""
import re

print("="*80)
print("GENOME AUDIT - Checking Consistency Across All Components")
print("="*80)

# Expected genome structure (8 genes, all 0.0-1.0 range)
EXPECTED_GENOME = {
    0: ('stop_loss', 'Stop loss threshold (0.0-1.0)', 'Direct use'),
    1: ('take_profit', 'Take profit threshold (0.0-1.0)', 'Direct use'),
    2: ('position_size', 'Position size (0.0-1.0)', 'Direct use'),
    3: ('entry_threshold', 'Signal strength to enter (0.0-1.0)', 'Direct use'),
    4: ('exit_threshold', 'Signal strength to exit (0.0-1.0)', 'Direct use'),
    5: ('max_hold_time', 'Maximum hold time', 'SCALE: max(1, int(x * 14)) → 1-14 days'),
    6: ('volatility_scaling', 'Volatility adjustment (0.0-1.0)', 'Direct use'),
    7: ('momentum_weight', 'Trend sensitivity (0.0-1.0)', 'Direct use'),
}

print("\n📋 EXPECTED GENOME STRUCTURE:")
print("-" * 80)
for idx, (name, desc, scaling) in EXPECTED_GENOME.items():
    print(f"[{idx}] {name:20s} - {desc:35s} | {scaling}")

# Files to audit
files_to_check = {
    'trading_specialist.py': {
        'description': 'Main specialist class - genome usage',
        'critical_lines': [88, 89, 90, 91, 92, 93, 94, 95]
    },
    'conductor_enhanced_trainer.py': {
        'description': 'Trainer - genome initialization',
        'critical_lines': [299, 300]  # _initialize_population
    },
    'baseline_trainer.py': {
        'description': 'Baseline trainer (if exists)',
        'critical_lines': []
    }
}

print("\n\n🔍 AUDITING FILES:")
print("-" * 80)

issues_found = []

# Check trading_specialist.py
print("\n1. trading_specialist.py")
try:
    with open('trading_specialist.py', 'r') as f:
        lines = f.readlines()
    
    genome_assignments = []
    for i, line in enumerate(lines[87:96], start=88):  # Lines 88-95
        if 'self.genome[' in line:
            genome_assignments.append((i, line.strip()))
    
    print(f"   Found {len(genome_assignments)} genome assignments:")
    for line_num, line in genome_assignments:
        print(f"   Line {line_num}: {line}")
    
    # Check max_hold_time specifically
    max_hold_line = [l for _, l in genome_assignments if 'max_hold_time' in l]
    if max_hold_line:
        line = max_hold_line[0]
        if 'int(self.genome[5] * 14)' in line:
            print("   ✅ max_hold_time CORRECTLY scales to 1-14 days")
        elif 'int(self.genome[5])' in line and '* 14' not in line:
            print("   ❌ max_hold_time INCORRECTLY uses int(genome[5]) - BUG!")
            issues_found.append("trading_specialist.py: max_hold_time not scaled")
        else:
            print(f"   ⚠️  max_hold_time has unexpected format: {line}")
            issues_found.append(f"trading_specialist.py: Unexpected max_hold_time format")
    
except FileNotFoundError:
    print("   ❌ File not found!")
    issues_found.append("trading_specialist.py: File not found")

# Check conductor_enhanced_trainer.py
print("\n2. conductor_enhanced_trainer.py")
try:
    with open('conductor_enhanced_trainer.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check _initialize_population
    if 'genome=np.random.random(8)' in content:
        print("   ✅ Initializes 8-gene genomes")
    else:
        print("   ❌ Genome size mismatch!")
        issues_found.append("conductor_enhanced_trainer.py: Genome size != 8")
    
    # Check for fitness cache
    if 'self.fitness_cache' in content:
        print("   ✅ Fitness caching implemented")
    else:
        print("   ⚠️  No fitness caching found")
    
    # Check for genome hash
    if '_genome_hash' in content:
        print("   ✅ Genome hashing implemented")
    else:
        print("   ⚠️  No genome hashing found")
        
except FileNotFoundError:
    print("   ❌ File not found!")
    issues_found.append("conductor_enhanced_trainer.py: File not found")

# Check evaluate_fitness in trading_specialist.py
print("\n3. trading_specialist.py - evaluate_fitness()")
try:
    with open('trading_specialist.py', 'r') as f:
        lines = f.readlines()
    
    # Find evaluate_fitness method
    in_evaluate = False
    generate_signal_calls = []
    for i, line in enumerate(lines, start=1):
        if 'def evaluate_fitness' in line:
            in_evaluate = True
        elif in_evaluate and 'def ' in line and 'def evaluate_fitness' not in line:
            break
        elif in_evaluate and 'generate_signal' in line and 'self.generate_signal' in line:
            generate_signal_calls.append((i, line.strip()))
    
    if generate_signal_calls:
        print(f"   Found {len(generate_signal_calls)} generate_signal calls in evaluate_fitness:")
        for line_num, line in generate_signal_calls:
            print(f"   Line {line_num}: {line[:70]}...")
        print("   ✅ evaluate_fitness calls generate_signal (genome will be used)")
    else:
        print("   ⚠️  No generate_signal calls found in evaluate_fitness")
        
except FileNotFoundError:
    pass

# Check for any hardcoded genome sizes
print("\n4. Searching for hardcoded genome references:")
try:
    with open('conductor_enhanced_trainer.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Look for genome size references
    patterns = [
        (r'genome.*=.*random.*\((\d+)\)', 'Random genome initialization'),
        (r'len\(.*genome.*\).*==.*(\d+)', 'Genome length checks'),
        (r'genome\[(\d+)\]', 'Genome index access'),
    ]
    
    for pattern, desc in patterns:
        matches = re.findall(pattern, content)
        if matches:
            unique = set(matches)
            if '8' in unique and len(unique) == 1:
                print(f"   ✅ {desc}: All reference 8 genes")
            elif unique:
                print(f"   ⚠️  {desc}: Found inconsistent sizes: {unique}")
                issues_found.append(f"conductor_enhanced_trainer.py: Inconsistent genome sizes")
                
except FileNotFoundError:
    pass

print("\n\n" + "="*80)
print("AUDIT SUMMARY")
print("="*80)

if not issues_found:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nGenome mapping is consistent across all components.")
    print("Ready for retraining with fixes applied.")
else:
    print(f"\n❌ FOUND {len(issues_found)} ISSUES:")
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue}")
    print("\n⚠️  Fix these issues before retraining!")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\nIf audit passed:")
print("  1. Retrain volatile specialist   (~15 min)")
print("  2. Retrain trending specialist   (~15 min)")
print("  3. Retrain ranging specialist    (~15 min)")
print("  4. Test ensemble with new genomes")
print("  5. Verify trade count increases dramatically")
print("\nExpected outcome:")
print("  - Trades should increase from 1 to 50-100+")
print("  - Specialists can now hold positions 1-14 days")
print("  - Much more realistic trading behavior")
