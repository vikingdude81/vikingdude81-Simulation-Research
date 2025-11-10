"""
Quick progress checker for conductor-enhanced training
"""

from pathlib import Path
import json
import time

def check_progress():
    """Check if training has produced any results yet"""
    
    output_dir = Path('outputs')
    conductor_files = list(output_dir.glob('conductor_enhanced_volatile_*.json'))
    
    if not conductor_files:
        print("⏳ Training in progress... No results file yet.")
        print("   (First generation evaluation takes ~1-2 minutes)")
        return
    
    # Get latest file
    latest_file = max(conductor_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    history = results['training_history']
    current_gen = len(history['generation'])
    total_gens = results['generations']
    
    if current_gen == 0:
        print("⏳ Training started but no generations complete yet...")
        return
    
    best_fitness = history['best_fitness'][-1]
    avg_fitness = history['avg_fitness'][-1]
    
    progress_pct = (current_gen / total_gens) * 100
    
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING PROGRESS")
    print(f"{'='*60}")
    print(f"Generation:    {current_gen:3d} / {total_gens} ({progress_pct:.1f}%)")
    print(f"Best Fitness:  {best_fitness:.2f}")
    print(f"Avg Fitness:   {avg_fitness:.2f}")
    print(f"File:          {latest_file.name}")
    print(f"{'='*60}")
    
    # Estimate time remaining
    if current_gen > 10:
        baseline_file = list(output_dir.glob('specialist_volatile_*.json'))[0]
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        b_fitness = baseline['best_agent']['fitness']
        improvement = ((best_fitness - b_fitness) / abs(b_fitness)) * 100
        
        print(f"\nVS BASELINE:")
        print(f"  Baseline fitness: {b_fitness:.2f}")
        print(f"  Current fitness:  {best_fitness:.2f}")
        print(f"  Improvement:      {improvement:+.1f}%")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    check_progress()
