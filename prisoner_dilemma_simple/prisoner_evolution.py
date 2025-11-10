"""
🧬 Simple Prisoner's Dilemma Evolution (3-Gene Chromosome)

Based on John Holland's "Hidden Order" - Page 82-84
This implements the basic memory-based strategy evolution.

Chromosome Structure (3 genes):
    Gene 0: Action for FIRST move (C or D)
    Gene 1: Action if opponent's LAST move was 'C' (Cooperate)
    Gene 2: Action if opponent's LAST move was 'D' (Defect)

Example Strategies:
    "CDC" = Tit-for-Tat (Start C, copy opponent)
    "DDD" = Always Defect
    "CCC" = Always Cooperate
    "DDC" = Suspicious Tit-for-Tat (Start D, then copy)
"""

import random
import numpy as np
from typing import List, Tuple, Dict

# --- 1. GENETIC ALGORITHM FUNCTIONS ---

def create_random_chromosome() -> str:
    """Creates a random 3-gene strategy chromosome."""
    return "".join(random.choice(['C', 'D']) for _ in range(3))

def calculate_fitness(chromosome: str, opponent_chromosome: str, rounds: int = 50) -> float:
    """
    Calculates fitness by playing the Prisoner's Dilemma.
    Returns total score accumulated over all rounds.
    """
    # Payoff matrix: (My_Move, Opponent_Move) -> (My_Payoff, Opponent_Payoff)
    PAYOFFS = {
        ('C', 'C'): (3, 3),  # Both cooperate: Reward
        ('D', 'C'): (5, 0),  # I defect, they cooperate: Temptation
        ('C', 'D'): (0, 5),  # I cooperate, they defect: Sucker's Payoff
        ('D', 'D'): (1, 1),  # Both defect: Punishment
    }
    
    my_score = 0
    
    # Track last moves for BOTH players
    my_history = []
    opp_history = []
    
    for round_num in range(rounds):
        # Determine my move based on opponent's last move
        if len(opp_history) == 0:
            my_move = chromosome[0]  # Gene 0: First move
        elif opp_history[-1] == 'C':
            my_move = chromosome[1]  # Gene 1: Opponent cooperated last
        else:  # opp_history[-1] == 'D'
            my_move = chromosome[2]  # Gene 2: Opponent defected last
        
        # Determine opponent's move based on MY last move
        if len(my_history) == 0:
            opp_move = opponent_chromosome[0]  # First move
        elif my_history[-1] == 'C':
            opp_move = opponent_chromosome[1]  # I cooperated last
        else:  # my_history[-1] == 'D'
            opp_move = opponent_chromosome[2]  # I defected last
        
        # Get payoffs
        my_payoff, opp_payoff = PAYOFFS[(my_move, opp_move)]
        my_score += my_payoff
        
        # Update history
        my_history.append(my_move)
        opp_history.append(opp_move)
    
    return my_score

def crossover(parent1: str, parent2: str) -> str:
    """Performs single-point crossover."""
    point = random.randint(1, 2)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.1) -> str:
    """Mutates genes with given probability."""
    if random.random() < rate:
        gene_pos = random.randint(0, 2)
        new_gene = 'D' if chromosome[gene_pos] == 'C' else 'C'
        return chromosome[:gene_pos] + new_gene + chromosome[gene_pos+1:]
    return chromosome

# --- 2. AGENT CLASS ---

class PrisonerAgent:
    """An agent with a 3-gene strategy chromosome."""
    
    def __init__(self, agent_id: int, chromosome: str):
        self.id = agent_id
        self.chromosome = chromosome
        self.fitness: float = 0.0
        self.wins = 0
        self.losses = 0
        self.ties = 0
    
    def __repr__(self):
        return f"Agent{self.id}({self.chromosome}, fitness={self.fitness:.1f})"
    
    def reset_stats(self):
        """Reset fitness and game statistics."""
        self.fitness = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0

# --- 3. EVOLUTION SIMULATION ---

class PrisonerEvolution:
    """Manages the evolutionary simulation."""
    
    def __init__(self, 
                 population_size: int = 50,
                 elite_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 rounds_per_matchup: int = 50):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rounds_per_matchup = rounds_per_matchup
        
        # Initialize population
        self.population: List[PrisonerAgent] = []
        for i in range(population_size):
            chromosome = create_random_chromosome()
            self.population.append(PrisonerAgent(i, chromosome))
        
        # Evolution tracking
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.strategy_counts_history = []
    
    def evaluate_fitness(self):
        """
        Evaluate all agents by playing round-robin tournament.
        Each agent plays against every other agent.
        """
        for agent in self.population:
            agent.reset_stats()
        
        # Round-robin tournament
        for i, agent1 in enumerate(self.population):
            for j, agent2 in enumerate(self.population):
                if i >= j:  # Avoid duplicate matchups and self-play
                    continue
                
                # Play the game
                score1 = calculate_fitness(agent1.chromosome, agent2.chromosome, 
                                         self.rounds_per_matchup)
                score2 = calculate_fitness(agent2.chromosome, agent1.chromosome, 
                                         self.rounds_per_matchup)
                
                agent1.fitness += score1
                agent2.fitness += score2
                
                # Track wins/losses/ties
                if score1 > score2:
                    agent1.wins += 1
                    agent2.losses += 1
                elif score2 > score1:
                    agent2.wins += 1
                    agent1.losses += 1
                else:
                    agent1.ties += 1
                    agent2.ties += 1
    
    def select_parents(self) -> Tuple[List[PrisonerAgent], List[PrisonerAgent]]:
        """Tournament selection with elitism."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Keep elite
        elites = sorted_pop[:self.elite_size]
        
        return elites, sorted_pop
    
    def evolve_generation(self):
        """Run one generation of evolution."""
        # 1. Evaluate fitness
        self.evaluate_fitness()
        
        # 2. Track statistics BEFORE creating new generation
        fitnesses = [agent.fitness for agent in self.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_agent = max(self.population, key=lambda x: x.fitness)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Track strategy distribution
        strategy_counts = {}
        for agent in self.population:
            strategy_counts[agent.chromosome] = strategy_counts.get(agent.chromosome, 0) + 1
        self.strategy_counts_history.append(strategy_counts)
        
        # Store best agent info before population replacement
        self._current_best_chromosome = best_agent.chromosome
        self._current_best_fitness = best_fitness
        self._current_avg_fitness = avg_fitness
        self._current_strategy_counts = strategy_counts
        
        # 3. Selection
        elites, sorted_pop = self.select_parents()
        
        # 4. Create new generation
        new_population = []
        
        # Add elites directly
        for agent in elites:
            new_population.append(PrisonerAgent(len(new_population), agent.chromosome))
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Select parents from top half
            parent1 = random.choice(sorted_pop[:self.population_size // 2])
            parent2 = random.choice(sorted_pop[:self.population_size // 2])
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_chrom = crossover(parent1.chromosome, parent2.chromosome)
            else:
                child_chrom = parent1.chromosome
            
            # Mutation
            child_chrom = mutate(child_chrom, self.mutation_rate)
            
            new_population.append(PrisonerAgent(len(new_population), child_chrom))
        
        # Replace population
        self.population = new_population
        self.generation += 1
    
    def get_best_agent(self) -> PrisonerAgent:
        """Returns the current best agent."""
        return max(self.population, key=lambda x: x.fitness)
    
    def print_generation_summary(self, verbose: bool = True):
        """Print summary of current generation."""
        # Use stored values from BEFORE population replacement
        best_chrom = getattr(self, '_current_best_chromosome', 'N/A')
        best_fitness = getattr(self, '_current_best_fitness', 0)
        avg_fitness = getattr(self, '_current_avg_fitness', 0)
        strategy_counts = getattr(self, '_current_strategy_counts', {})
        
        # Sort by count
        top_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation {self.generation}")
            print(f"{'='*60}")
            print(f"Best Agent: {best_chrom} | Fitness: {best_fitness:.1f}")
            print(f"Average Fitness: {avg_fitness:.1f}")
            print(f"\nTop 5 Strategies:")
            for strategy, count in top_strategies:
                pct = (count / self.population_size) * 100
                print(f"  {strategy}: {count:2d} agents ({pct:5.1f}%)")
    
    def run(self, generations: int = 50, print_every: int = 10):
        """Run the evolution for specified generations."""
        print("\n🧬 Starting Prisoner's Dilemma Evolution")
        print(f"Population: {self.population_size} agents")
        print(f"Generations: {generations}")
        print(f"Elite size: {self.elite_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Rounds per matchup: {self.rounds_per_matchup}")
        
        for gen in range(generations):
            self.evolve_generation()
            
            if (gen + 1) % print_every == 0 or gen == 0:
                self.print_generation_summary()
        
        print(f"\n{'='*60}")
        print("🏁 Evolution Complete!")
        print(f"{'='*60}")
        self.print_generation_summary()
        
        # Print interpretation
        best_chrom = getattr(self, '_current_best_chromosome', 'DDD')
        print(f"\n📊 Strategy Interpretation:")
        interpret_strategy(best_chrom)

# --- 4. UTILITY FUNCTIONS ---

def interpret_strategy(chromosome: str):
    """Interprets what a strategy means in plain English."""
    print(f"\nChromosome: {chromosome}")
    print(f"  Gene 0 (First move): {'Cooperate' if chromosome[0] == 'C' else 'Defect'}")
    print(f"  Gene 1 (If opponent cooperated): {'Cooperate' if chromosome[1] == 'C' else 'Defect'}")
    print(f"  Gene 2 (If opponent defected): {'Cooperate' if chromosome[2] == 'C' else 'Defect'}")
    
    # Identify known strategies
    if chromosome == "CDC":
        print("\n✨ This is TIT-FOR-TAT!")
        print("   Start nice, then copy opponent's last move")
    elif chromosome == "DDD":
        print("\n💀 This is ALWAYS DEFECT!")
        print("   Never cooperate, always betray")
    elif chromosome == "CCC":
        print("\n😇 This is ALWAYS COOPERATE!")
        print("   Always be nice, even if betrayed")
    elif chromosome == "DDC":
        print("\n🤨 This is SUSPICIOUS TIT-FOR-TAT!")
        print("   Start mean, but forgive if opponent cooperates")
    elif chromosome == "CCD":
        print("\n😤 This is GRUDGER!")
        print("   Start nice, but never forgive a defection")

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    # Run the evolution
    sim = PrisonerEvolution(
        population_size=50,
        elite_size=5,
        mutation_rate=0.15,
        crossover_rate=0.7,
        rounds_per_matchup=50
    )
    
    sim.run(generations=50, print_every=10)
