"""
Noisy 64-Gene Prisoner's Dilemma Evolution
==========================================

Tests robustness of evolved strategies when there are execution errors.
5% of moves are randomly flipped (C→D or D→C).

This tests:
- Does TFT-like cooperation survive noise?
- Do more forgiving strategies evolve?
- How does noise affect convergence?
"""

import numpy as np
import random
from prisoner_64gene import (
    AdvancedPrisonerAgent, 
    play_prisoner_dilemma,
    create_random_chromosome,
    create_tit_for_tat
)

class NoisyPrisonerEvolution:
    """Evolution with noisy execution (5% error rate)"""
    
    def __init__(self, population_size=50, mutation_rate=0.01, noise_rate=0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.noise_rate = noise_rate
        self.population = [
            AdvancedPrisonerAgent(i, create_random_chromosome()) 
            for i in range(population_size)
        ]
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.tft_similarity_history = []
        
    def noisy_play(self, agent1, agent2, rounds=30):
        """Play Prisoner's Dilemma with noise (execution errors)"""
        score1 = 0
        score2 = 0
        
        # Reset histories to default starting state
        agent1.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        agent2.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        
        for _ in range(rounds):
            # Get intended moves
            move1 = agent1.get_move()
            move2 = agent2.get_move()
            
            # Apply noise: 5% chance each move gets flipped
            if random.random() < self.noise_rate:
                move1 = 'D' if move1 == 'C' else 'C'
            if random.random() < self.noise_rate:
                move2 = 'D' if move2 == 'C' else 'C'
            
            # Score based on actual (noisy) moves
            if move1 == 'C' and move2 == 'C':
                score1 += 3
                score2 += 3
            elif move1 == 'D' and move2 == 'C':
                score1 += 5
                score2 += 0
            elif move1 == 'C' and move2 == 'D':
                score1 += 0
                score2 += 5
            else:  # Both defect
                score1 += 1
                score2 += 1
            
            # Update histories with actual (noisy) moves  
            agent1.update_history(move1, move2)
            agent2.update_history(move2, move1)
        
        return score1, score2
    
    def evaluate_fitness(self):
        """Evaluate all agents via noisy round-robin tournament"""
        for agent in self.population:
            agent.fitness = 0
        
        # Round-robin tournament with noise
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                score1, score2 = self.noisy_play(
                    self.population[i], 
                    self.population[j],
                    rounds=30
                )
                self.population[i].fitness += score1
                self.population[j].fitness += score2
    
    def select_parents(self):
        """Tournament selection"""
        tournament_size = 5
        selected = []
        
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected[0], selected[1]
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() < 0.7:
            point = random.randint(1, 63)
            child_genes = parent1.chromosome[:point] + parent2.chromosome[point:]
            child = AdvancedPrisonerAgent(
                agent_id=random.randint(1000, 9999),
                chromosome=child_genes
            )
            return child
        else:
            # No crossover, return copy of parent1
            child = AdvancedPrisonerAgent(
                agent_id=random.randint(1000, 9999),
                chromosome=parent1.chromosome
            )
            return child
    
    def mutate(self, agent):
        """Flip random bits with mutation_rate probability"""
        chrom_list = list(agent.chromosome)
        for i in range(64):
            if random.random() < self.mutation_rate:
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
        # Create new agent with mutated chromosome
        agent.chromosome = "".join(chrom_list)
    
    def evolve_generation(self):
        """Evolve one generation with elitism"""
        self.generation += 1
        
        # Evaluate fitness
        self.evaluate_fitness()
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best agent stats BEFORE population replacement
        best_agent = self.population[0]
        best_fitness = best_agent.fitness
        
        # Calculate TFT similarity manually
        tft = create_tit_for_tat()
        tft_sim = sum(1 for i in range(64) if best_agent.chromosome[i] == tft[i]) / 64 * 100
        
        self.best_fitness_history.append(best_fitness)
        avg_fitness = sum(a.fitness for a in self.population) / len(self.population)
        self.avg_fitness_history.append(avg_fitness)
        self.tft_similarity_history.append(tft_sim)
        
        # Elitism: keep top 5
        elite_size = 5
        new_population = self.population[:elite_size].copy()
        
        # Generate rest through evolution
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        
        return best_fitness, tft_sim
    
    def print_generation_summary(self, best_fitness, tft_sim):
        """Print generation statistics"""
        if self.generation % 20 == 0:
            print(f"  Gen {self.generation}: Fitness={int(best_fitness)}, TFT Similarity={tft_sim:.1f}%")

def run_noisy_experiment(generations=100):
    """Run complete noisy evolution experiment"""
    print("\n" + "="*70)
    print("🔬 NOISY 64-GENE PRISONER'S DILEMMA")
    print("="*70)
    print("\nExperiment: 5% execution error rate")
    print("Question: Does noise favor more forgiving strategies?")
    print("="*70)
    
    # Run evolution
    evolution = NoisyPrisonerEvolution(
        population_size=50,
        mutation_rate=0.01,
        noise_rate=0.05  # 5% error rate
    )
    
    print(f"\n🧬 Evolving {generations} generations with 5% noise...")
    print("-" * 70)
    
    for gen in range(generations):
        best_fitness, tft_sim = evolution.evolve_generation()
        evolution.print_generation_summary(best_fitness, tft_sim)
    
    # Final results
    best_agent = evolution.population[0]
    tft_chromosome = create_tit_for_tat()
    final_tft_sim = sum(1 for i in range(64) if best_agent.chromosome[i] == tft_chromosome[i]) / 64 * 100
    
    print("\n" + "="*70)
    print("✅ EVOLUTION COMPLETE")
    print("="*70)
    print(f"Final Generation: {evolution.generation}")
    print(f"Best Fitness: {int(best_agent.fitness)}")
    print(f"TFT Similarity: {final_tft_sim:.1f}%")
    
    # Compare with clean environment
    print("\n" + "="*70)
    print("📊 TESTING IN CLEAN ENVIRONMENT")
    print("="*70)
    print("Testing evolved strategy WITHOUT noise...")
    
    # Create TFT for comparison
    tft = AdvancedPrisonerAgent(agent_id=9999, chromosome=tft_chromosome)
    
    # Test both agents without noise
    clean_score_evolved, _ = play_prisoner_dilemma(best_agent, tft, rounds=100)
    clean_score_tft, _ = play_prisoner_dilemma(tft, tft, rounds=100)
    
    print(f"  Evolved strategy vs TFT (clean): {clean_score_evolved}")
    print(f"  TFT vs TFT (clean): {clean_score_tft}")
    
    # Test both agents WITH noise
    noisy_score_evolved, _ = evolution.noisy_play(best_agent, tft, rounds=100)
    noisy_score_tft, _ = evolution.noisy_play(tft, tft, rounds=100)
    
    print(f"\n  Evolved strategy vs TFT (noisy): {noisy_score_evolved}")
    print(f"  TFT vs TFT (noisy): {noisy_score_tft}")
    
    # Analysis
    print("\n" + "="*70)
    print("🔍 ANALYSIS")
    print("="*70)
    
    clean_advantage = clean_score_evolved - clean_score_tft
    noisy_advantage = noisy_score_evolved - noisy_score_tft
    
    print(f"Clean environment advantage: {clean_advantage:+.0f}")
    print(f"Noisy environment advantage: {noisy_advantage:+.0f}")
    
    if noisy_advantage > clean_advantage:
        print("\n✅ Evolved strategy is MORE robust to noise than pure TFT!")
    elif noisy_advantage < 0:
        print("\n⚠️  Evolved strategy performs worse than TFT in noise")
    else:
        print("\n➡️  Evolved strategy performs similarly to TFT in noise")
    
    # Show some key genes
    print("\n" + "="*70)
    print("🧬 EVOLVED STRATEGY SAMPLE")
    print("="*70)
    print("First 16 genes (CCC, CCD, CDC, CDD, DCC, DCD, DDC, DDD histories):")
    print(f"  {''.join(best_agent.chromosome[:16])}")
    print(f"\nPure TFT would be:")
    print(f"  {''.join(tft_chromosome[:16])}")
    print(f"\nDifferences: {sum(1 for i in range(16) if best_agent.chromosome[i] != tft_chromosome[i])}/16")
    
    return evolution, best_agent

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎲 ROBUSTNESS TEST: Noisy Prisoner's Dilemma")
    print("="*70)
    print("\nThis experiment tests whether TFT-like cooperation can survive")
    print("when there are execution errors (5% of moves get flipped).")
    print("\nKey questions:")
    print("  1. Do more forgiving strategies evolve?")
    print("  2. Is the evolved strategy robust to noise?")
    print("  3. Does noise prevent convergence?")
    print("="*70)
    
    evolution, best_agent = run_noisy_experiment(generations=100)
    
    print("\n" + "="*70)
    print("🎉 EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults show how evolution adapts strategies to noisy environments.")
    print("Check if TFT similarity changed compared to clean evolution (53%).")
    print("="*70)
