"""
FAST 200-GEN QUANTUM ML TRAINING

Optimized for speed while maintaining quality:
- 15 genomes (vs 50) - fewer but still diverse
- 25 evolution cycles (vs 50) - enough to converge
- 150 generations training (vs 200) - captures long-term dynamics
- 1 run per genome (vs 2) - faster evaluation
- Population 300 (vs 500) - representative sample

Estimated time: ~40 minutes (vs 2+ hours)
"""

import numpy as np
import random
import json
from datetime import datetime
from prisoner_echo_god import run_god_echo_simulation
from quantum_god_controller import QuantumGodController
import time

# OPTIMIZED parameters for speed
POPULATION_SIZE = 15  # Genomes per generation
EVOLUTION_CYCLES = 25  # Evolution generations
ELITE_SIZE = 3  # Top performers to keep
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.3

# Training parameters - LONG-TERM focused
TRAIN_GENERATIONS = 150  # Long enough to capture lifecycle dynamics
TRAIN_POPULATION = 300  # Good sample size
RUNS_PER_GENOME = 1  # Single run for speed

def create_random_genome():
    """Create a random genome (intervention strategy)."""
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
    Evaluate a genome by running simulation.
    
    Score focuses on LONG-TERM outcomes:
    - wealth/100: Economic success
    - cooperation*50: Social cohesion
    - (1-gini)*100: Equality
    - survival*100: Population health
    """
    print(f"  G{genome_id}: ", end="", flush=True)
    
    try:
        # Create temporary controller with this genome
        controller = QuantumGodController(environment='standard')
        controller.champion_genome = genome.tolist()
        
        # Run simulation (suppress output)
        result = run_god_echo_simulation(
            generations=TRAIN_GENERATIONS,
            initial_size=TRAIN_POPULATION,
            god_mode="ML_BASED",
            update_frequency=99999  # No output
        )
        
        # Calculate score
        if len(result.agents) > 0:
            final_pop = len(result.agents)
            avg_wealth = sum(a.resources for a in result.agents) / final_pop
            
            total_actions = sum(a.cooperations + a.defections for a in result.agents)
            total_coop = sum(a.cooperations for a in result.agents)
            coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
            
            # Gini coefficient
            resources = sorted([a.resources for a in result.agents])
            n = len(resources)
            cumsum = sum((i + 1) * r for i, r in enumerate(resources))
            gini = (2 * cumsum) / (n * sum(resources)) - (n + 1) / n
            
            survival_rate = final_pop / TRAIN_POPULATION
            
            # Score emphasizes cooperation and wealth at long horizons
            score = (
                avg_wealth / 100 +
                coop_rate * 50 +
                (1 - gini) * 100 +
                survival_rate * 100
            )
            
            print(f"${avg_wealth:.0f} {coop_rate:.1%} gini={gini:.2f} → {score:.1f}")
            return score
        else:
            print(f"EXTINCT → 0.0")
            return 0
    
    except Exception as e:
        print(f"ERROR: {e}")
        return 0

def fast_evolution_150gen():
    """Fast evolution optimized for 150-gen training."""
    
    print("\n" + "="*80)
    print("🚀 FAST QUANTUM ML EVOLUTION - 150 GENERATION TRAINING")
    print("="*80)
    print(f"\nOptimized Configuration:")
    print(f"  Population: {POPULATION_SIZE} genomes")
    print(f"  Cycles: {EVOLUTION_CYCLES}")
    print(f"  Training: {TRAIN_GENERATIONS} generations")
    print(f"  Agents: {TRAIN_POPULATION}")
    print(f"  Elite: {ELITE_SIZE}")
    print(f"\n⏱️  Estimated time: ~{EVOLUTION_CYCLES * POPULATION_SIZE * 10 / 60:.0f} minutes")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Initialize population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_score = 0
    evolution_history = []
    
    # Evolution loop
    for cycle in range(EVOLUTION_CYCLES):
        print(f"\n{'='*80}")
        print(f"🧬 CYCLE {cycle + 1}/{EVOLUTION_CYCLES}")
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
        print(f"\n📊 Cycle {cycle + 1}:")
        print(f"   Best: {cycle_best_score:.1f}")
        print(f"   Avg: {avg_score:.1f}")
        print(f"   Overall: {best_score:.1f}")
        
        evolution_history.append({
            'cycle': cycle + 1,
            'best': cycle_best_score,
            'avg': avg_score,
            'overall_best': best_score
        })
        
        # Create next generation
        if cycle < EVOLUTION_CYCLES - 1:
            # Keep elite
            elite = [genome for _, genome in scores[:ELITE_SIZE]]
            
            # Fill rest with offspring
            new_population = elite.copy()
            
            while len(new_population) < POPULATION_SIZE:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = crossover(parent1, parent2)
                child = mutate_genome(child)
                new_population.append(child)
            
            population = new_population
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("🏆 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Time: {elapsed/60:.1f} minutes")
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
            'elite_size': ELITE_SIZE
        },
        'champion': {
            'genome': best_genome.tolist(),
            'score': best_score
        },
        'evolution_history': evolution_history,
        'elapsed_time': elapsed
    }
    
    output_file = f"outputs/god_ai/quantum_evolution_150gen_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_file}")
    print(f"\n✅ Ready for testing!\n")
    
    return best_genome, best_score

if __name__ == "__main__":
    champion, score = fast_evolution_150gen()
