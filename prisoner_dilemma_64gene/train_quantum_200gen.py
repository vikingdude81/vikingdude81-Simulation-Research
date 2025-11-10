"""
Retrain Quantum ML God Controller on 200-Generation Data

This script evolves a new champion genome optimized for LONG-TERM governance (200 generations).
The current champion was trained on 50-gen scenarios and fails at 100+.

Evolution process:
1. Create 1000 random genomes (intervention strategies)
2. Test each on 200-generation simulations
3. Score based on: final wealth + cooperation + survival + low inequality
4. Keep top 10%, mutate, repeat for 100 evolution cycles
5. Save champion genome for long-term God controller

Key differences from original training:
- 200 generations (vs 50) - learns late-game dynamics
- Larger population (500 vs 300) - more stable
- More evolution cycles (100 vs original) - better convergence
"""

import numpy as np
import random
import json
from datetime import datetime
from prisoner_echo_god import run_god_echo_simulation
from quantum_god_controller import QuantumGodController
import time

# Evolution parameters (optimized for speed while maintaining quality)
POPULATION_SIZE = 20  # Number of genomes per generation
EVOLUTION_CYCLES = 30  # Number of evolution generations
ELITE_SIZE = 4  # Top performers to keep
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.3

# Simulation parameters for training
TRAIN_GENERATIONS = 200  # KEY CHANGE: Train on 200-gen scenarios
TRAIN_POPULATION = 300  # Smaller but still representative
RUNS_PER_GENOME = 1  # Test each genome once (faster)

def create_random_genome():
    """Create a random genome (intervention strategy)."""
    # Genome structure: [intervention_threshold, welfare_target_pct, welfare_amount_scale, 
    #                    stimulus_threshold, spawn_threshold, emergency_threshold, ...]
    return np.array([
        random.uniform(0.3, 0.7),   # intervention_threshold (Gini trigger)
        random.uniform(0.05, 0.25), # welfare_target_pct
        random.uniform(50, 200),    # welfare_amount
        random.uniform(500, 2000),  # stimulus_amount
        random.uniform(0.7, 0.95),  # spawn_threshold (tribe dominance)
        random.uniform(50, 150),    # emergency_population_threshold
        random.uniform(0.4, 0.8),   # cooperation_threshold
        random.uniform(5, 20),      # intervention_cooldown
    ])

def mutate_genome(genome, strength=MUTATION_STRENGTH):
    """Mutate a genome."""
    mutated = genome.copy()
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] += random.gauss(0, strength) * mutated[i]
            # Keep values in reasonable bounds
            mutated[i] = max(0.01, mutated[i])
    return mutated

def crossover(parent1, parent2):
    """Create offspring from two parents."""
    child = np.zeros_like(parent1)
    for i in range(len(child)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child

def evaluate_genome(genome, genome_id, cycle):
    """
    Evaluate a genome by running simulations.
    
    Score = wealth/100 + cooperation*50 + (1-gini)*100 + survival*100
    Higher is better.
    """
    print(f"  Genome {genome_id}: ", end="", flush=True)
    
    scores = []
    for run in range(RUNS_PER_GENOME):
        try:
            # Create temporary controller with this genome
            controller = QuantumGodController(environment='standard')
            controller.champion_genome = genome.tolist()
            
            # Run simulation (suppress output)
            result = run_god_echo_simulation(
                generations=TRAIN_GENERATIONS,
                initial_size=TRAIN_POPULATION,
                god_mode="ML_BASED",
                update_frequency=999  # No output
            )
            
            # Calculate score
            if len(result.agents) > 0:
                final_pop = len(result.agents)
                avg_wealth = sum(a.resources for a in result.agents) / final_pop
                
                total_actions = sum(a.cooperations + a.defections for a in result.agents)
                total_coop = sum(a.cooperations for a in result.agents)
                coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
                
                # Calculate Gini
                resources = sorted([a.resources for a in result.agents])
                n = len(resources)
                cumsum = sum((i + 1) * r for i, r in enumerate(resources))
                gini = (2 * cumsum) / (n * sum(resources)) - (n + 1) / n
                
                survival_rate = final_pop / TRAIN_POPULATION
                
                score = (
                    avg_wealth / 100 +
                    coop_rate * 50 +
                    (1 - gini) * 100 +
                    survival_rate * 100
                )
                
                scores.append(score)
                print(f"Run{run+1}:{score:.1f} ", end="", flush=True)
            else:
                scores.append(0)
                print(f"Run{run+1}:EXTINCT ", end="", flush=True)
        
        except Exception as e:
            print(f"Run{run+1}:ERROR ", end="", flush=True)
            scores.append(0)
    
    avg_score = np.mean(scores) if scores else 0
    print(f"→ Avg:{avg_score:.1f}")
    
    return avg_score

def evolution_200gen():
    """
    Main evolution loop to find optimal genome for 200-generation scenarios.
    """
    print("\n" + "="*80)
    print("🧬 QUANTUM ML EVOLUTION - 200 GENERATION TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Population size: {POPULATION_SIZE} genomes")
    print(f"  Evolution cycles: {EVOLUTION_CYCLES}")
    print(f"  Training length: {TRAIN_GENERATIONS} generations")
    print(f"  Training population: {TRAIN_POPULATION} agents")
    print(f"  Runs per genome: {RUNS_PER_GENOME}")
    print(f"  Elite size: {ELITE_SIZE}")
    print(f"\n⏱️  Estimated time: ~{EVOLUTION_CYCLES * POPULATION_SIZE * RUNS_PER_GENOME * 10 / 60:.0f} minutes")
    print("="*80 + "\n")
    
    print("🚀 Starting evolution automatically...\n")
    start_time = time.time()
    
    # Initialize population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_score = 0
    evolution_history = []
    
    # Evolution loop
    for cycle in range(EVOLUTION_CYCLES):
        print(f"\n{'='*80}")
        print(f"🧬 EVOLUTION CYCLE {cycle + 1}/{EVOLUTION_CYCLES}")
        print(f"{'='*80}")
        
        # Evaluate all genomes
        scores = []
        for i, genome in enumerate(population):
            score = evaluate_genome(genome, i + 1, cycle)
            scores.append((score, genome))
        
        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Track best
        cycle_best_score = scores[0][0]
        cycle_best_genome = scores[0][1]
        
        if cycle_best_score > best_score:
            best_score = cycle_best_score
            best_genome = cycle_best_genome.copy()
            print(f"\n✨ NEW CHAMPION! Score: {best_score:.1f}")
        
        # Statistics
        avg_score = np.mean([s[0] for s in scores])
        print(f"\n📊 Cycle {cycle + 1} Stats:")
        print(f"   Best: {cycle_best_score:.1f}")
        print(f"   Average: {avg_score:.1f}")
        print(f"   Overall best: {best_score:.1f}")
        
        evolution_history.append({
            'cycle': cycle + 1,
            'best_score': cycle_best_score,
            'avg_score': avg_score,
            'overall_best': best_score
        })
        
        # Create next generation
        if cycle < EVOLUTION_CYCLES - 1:
            # Keep elite
            elite = [genome for _, genome in scores[:ELITE_SIZE]]
            
            # Fill rest with mutated/crossed offspring
            new_population = elite.copy()
            
            while len(new_population) < POPULATION_SIZE:
                # Select parents from elite
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutate
                child = mutate_genome(child)
                
                new_population.append(child)
            
            population = new_population
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("🏆 EVOLUTION COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"🎯 Champion score: {best_score:.1f}")
    print(f"🧬 Champion genome: {best_genome.tolist()}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'config': {
            'population_size': POPULATION_SIZE,
            'evolution_cycles': EVOLUTION_CYCLES,
            'train_generations': TRAIN_GENERATIONS,
            'train_population': TRAIN_POPULATION,
            'elite_size': ELITE_SIZE,
            'mutation_rate': MUTATION_RATE,
            'mutation_strength': MUTATION_STRENGTH
        },
        'champion': {
            'genome': best_genome.tolist(),
            'score': best_score
        },
        'evolution_history': evolution_history,
        'elapsed_time': elapsed
    }
    
    output_file = f"outputs/god_ai/quantum_evolution_200gen_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Update quantum controller with new champion
    print(f"\n📝 To use this champion, update quantum_god_controller.py:")
    print(f"   CHAMPION_GENOME_200GEN = {best_genome.tolist()}")
    
    print("\n✅ Training complete!\n")
    
    return best_genome, best_score

if __name__ == "__main__":
    champion, score = evolution_200gen()
