"""
Quick status check for Quantum ML training progress
"""

import glob
import os
from datetime import datetime
import time

print("\n" + "="*80)
print("📊 QUANTUM ML TRAINING STATUS")
print("="*80)

# Check for training results
training_files = glob.glob("outputs/god_ai/quantum_evolution_150gen_*.json")

if training_files:
    latest = max(training_files, key=os.path.getctime)
    mtime = os.path.getctime(latest)
    age_minutes = (time.time() - mtime) / 60
    
    import json
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   File: {os.path.basename(latest)}")
    print(f"   Created: {age_minutes:.1f} minutes ago")
    print(f"   Champion Score: {data['champion']['score']:.1f}")
    print(f"   Training Time: {data['elapsed_time']/60:.1f} minutes")
    print(f"   Genome: {data['champion']['genome']}")
    
    print(f"\n✅ Ready for Ultimate Showdown!")
    print(f"   Run: python test_ultimate_showdown_final.py")
    
else:
    print(f"\n🟡 TRAINING IN PROGRESS...")
    print(f"   Looking for: quantum_evolution_150gen_*.json")
    print(f"   Check terminal for live updates")
    print(f"   Estimated completion: ~40 minutes from start")
    
    print(f"\n💡 What's happening:")
    print(f"   • 15 genomes per cycle")
    print(f"   • 25 evolution cycles total")
    print(f"   • Each genome tested on 150-gen simulation")
    print(f"   • Best genomes breed to create next generation")
    print(f"\n⏳ This takes time but will create a MUCH better controller!")

print("\n" + "="*80 + "\n")
