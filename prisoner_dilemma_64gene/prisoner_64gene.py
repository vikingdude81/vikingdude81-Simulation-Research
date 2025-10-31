"""
🧬 Advanced Prisoner's Dilemma Evolution (64-Gene Chromosome)

Based on John Holland's "Hidden Order" - Page 82
This implements the full 3-move history lookup table.

Chromosome Structure (64 genes):
    Each gene corresponds to a unique 3-move history
    Index = (hist_t-3 * 16) + (hist_t-2 * 4) + (hist_t-1)
    
    Where each history value maps to:
        (C,C) = 0, (C,D) = 1, (D,C) = 2, (D,D) = 3

Example:
    History: [(C,C), (D,C), (C,D)]
    Index = (0 * 16) + (2 * 4) + (1) = 9
    Action = chromosome[9]

Famous Strategy:
    Tit-for-Tat = "CDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCD"
    (Alternating C/D based on opponent's last move)
"""

import random
import numpy as np
from typing import List, Tuple, Dict

# --- 1. GENETIC ALGORITHM FUNCTIONS ---

def create_random_chromosome() -> str:
    """Creates a random 64-gene strategy chromosome."""
    return "".join(random.choice(['C', 'D']) for _ in range(64))

def create_tit_for_tat() -> str:
    """
    Creates the famous Tit-for-Tat strategy.
    TFT only looks at opponent's LAST move in the history.
    If opponent cooperated last, play C (even indices).
    If opponent defected last, play D (odd indices).
    """
    return "CD" * 32  # Repeating CD pattern

def crossover(parent1: str, parent2: str) -> str:
    """Performs single-point crossover on 64-gene chromosomes."""
    point = random.randint(1, 63)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.01) -> str:
    """
    Mutates genes with given probability.
    Lower rate for 64-gene because more genes to mutate.
    """
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < rate:
            chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
    return "".join(chrom_list)

# --- 2. AGENT CLASS ---

class AdvancedPrisonerAgent:
    """An agent with a 64-gene strategy chromosome and 3-move memory."""
    
    # Map joint moves to values for history encoding
    MOVE_MAP = {
        ('C', 'C'): 0,
        ('C', 'D'): 1,
        ('D', 'C'): 2,
        ('D', 'D'): 3
    }
    
    def __init__(self, agent_id: int, chromosome: str):
        self.id = agent_id
        self.chromosome = chromosome
        self.fitness: float = 0.0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        
        # Initialize with default history of (C,C) three times
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    
    def __repr__(self):
        return f"Agent{self.id}(fitness={self.fitness:.1f}, {self.chromosome[:8]}...)"
    
    def reset_stats(self):
        """Reset fitness and game statistics."""
        self.fitness = 0.0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    
    def get_history_index(self) -> int:
        """
        Converts the 3-move joint history into an index from 0 to 63.
        This is the "lookup table" concept from page 82.
        
        Formula: (hist[0] * 16) + (hist[1] * 4) + hist[2]
        """
        val_0 = self.MOVE_MAP[self.history[0]]  # Oldest
        val_1 = self.MOVE_MAP[self.history[1]]
        val_2 = self.MOVE_MAP[self.history[2]]  # Most recent
        
        index = (val_0 * 16) + (val_1 * 4) + val_2
        return index
    
    def get_move(self) -> str:
        """
        Decides next move based on the chromosome.
        Looks up the gene corresponding to current history.
        """
        index = self.get_history_index()
        return self.chromosome[index]
    
    def update_history(self, my_move: str, opponent_move: str):
        """Updates the 3-move sliding window history."""
        self.history.append((my_move, opponent_move))
        self.history.pop(0)  # Remove oldest, keep most recent 3

# --- 3. GAME FUNCTIONS ---

def play_prisoner_dilemma(agent1: AdvancedPrisonerAgent, 
                         agent2: AdvancedPrisonerAgent, 
                         rounds: int = 100) -> Tuple[float, float]:
    """
    Play the Prisoner's Dilemma between two agents for N rounds.
    Returns (agent1_score, agent2_score).
    """
    PAYOFFS = {
        ('C', 'C'): (3, 3),  # Both cooperate: Reward
        ('D', 'C'): (5, 0),  # I defect, they cooperate: Temptation
        ('C', 'D'): (0, 5),  # I cooperate, they defect: Sucker's Payoff
        ('D', 'D'): (1, 1),  # Both defect: Punishment
    }
    
    score1 = 0.0
    score2 = 0.0
    
    # Reset agents' histories for this game
    agent1.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    agent2.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    
    for _ in range(rounds):
        # Get moves based on current histories
        move1 = agent1.get_move()
        move2 = agent2.get_move()
        
        # Calculate payoffs
        payoff1, payoff2 = PAYOFFS[(move1, move2)]
        score1 += payoff1
        score2 += payoff2
        
        # Update histories
        agent1.update_history(move1, move2)
        agent2.update_history(move2, move1)  # Note: reversed
    
    return score1, score2

# --- 4. EVOLUTION SIMULATION ---

class AdvancedPrisonerEvolution:
    """Manages the evolutionary simulation with 64-gene chromosomes."""
    
    def __init__(self, 
                 population_size: int = 50,
                 elite_size: int = 5,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7,
                 rounds_per_matchup: int = 100):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rounds_per_matchup = rounds_per_matchup
        
        # Initialize population
        self.population: List[AdvancedPrisonerAgent] = []
        for i in range(population_size):
            chromosome = create_random_chromosome()
            self.population.append(AdvancedPrisonerAgent(i, chromosome))
        
        # Evolution tracking
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome_history = []
    
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
                score1, score2 = play_prisoner_dilemma(agent1, agent2, 
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
        self.best_chromosome_history.append(best_agent.chromosome)
        
        # Store best agent info before population replacement
        self._current_best_agent = best_agent
        self._current_best_fitness = best_fitness
        self._current_avg_fitness = avg_fitness
        
        # 3. Selection
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.elite_size]
        
        # 4. Create new generation
        new_population = []
        
        # Add elites directly (deep copy)
        for elite in elites:
            new_population.append(AdvancedPrisonerAgent(len(new_population), 
                                                        elite.chromosome))
        
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
            
            new_population.append(AdvancedPrisonerAgent(len(new_population), 
                                                        child_chrom))
        
        # Replace population
        self.population = new_population
        self.generation += 1
    
    def get_best_agent(self) -> AdvancedPrisonerAgent:
        """Returns the current best agent."""
        return max(self.population, key=lambda x: x.fitness)
    
    def print_generation_summary(self, verbose: bool = True):
        """Print summary of current generation."""
        # Use stored values from BEFORE population replacement
        best = getattr(self, '_current_best_agent', None)
        best_fitness = getattr(self, '_current_best_fitness', 0)
        avg_fitness = getattr(self, '_current_avg_fitness', 0)
        
        if best is None:
            # Fallback for first generation
            best = self.get_best_agent()
            best_fitness = best.fitness
            avg_fitness = np.mean([a.fitness for a in self.population])
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation {self.generation}")
            print(f"{'='*60}")
            print(f"Best Agent: Agent{best.id}")
            print(f"  Fitness: {best_fitness:.1f}")
            print(f"  W/L/T: {best.wins}/{best.losses}/{best.ties}")
            print(f"  Chromosome: {best.chromosome[:20]}...{best.chromosome[-10:]}")
            print(f"Average Fitness: {avg_fitness:.1f}")
            
            # Check if it's Tit-for-Tat
            tft = create_tit_for_tat()
            if best.chromosome == tft:
                print("  🎯 This is perfect Tit-for-Tat!")
            
            # Calculate similarity to TFT
            similarity = sum(1 for a, b in zip(best.chromosome, tft) if a == b)
            similarity_pct = (similarity / 64) * 100
            print(f"  Similarity to TFT: {similarity_pct:.1f}%")
    
    def run(self, generations: int = 100, print_every: int = 20):
        """Run the evolution for specified generations."""
        print("\n🧬 Starting Advanced Prisoner's Dilemma Evolution (64-Gene)")
        print(f"Population: {self.population_size} agents")
        print(f"Generations: {generations}")
        print(f"Elite size: {self.elite_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Rounds per matchup: {self.rounds_per_matchup}")
        print(f"Total possible strategies: 2^64 = {2**64:,}")
        
        for gen in range(generations):
            self.evolve_generation()
            
            if (gen + 1) % print_every == 0 or gen == 0:
                self.print_generation_summary()
        
        print(f"\n{'='*60}")
        print("🏁 Evolution Complete!")
        print(f"{'='*60}")
        self.print_generation_summary()

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    # Run the evolution
    sim = AdvancedPrisonerEvolution(
        population_size=50,
        elite_size=5,
        mutation_rate=0.01,  # Lower for 64 genes
        crossover_rate=0.7,
        rounds_per_matchup=100
    )
    
    sim.run(generations=100, print_every=20)
    
    # Show final best strategy
    best = sim.get_best_agent()
    print(f"\n📊 Final Best Strategy Analysis:")
    print(f"Chromosome: {best.chromosome}")
    print(f"\nFirst 16 genes (what to do after (C,C), (C,C), [X,X]):")
    print(f"  {best.chromosome[:16]}")
    print(f"\nLast 16 genes (what to do after (D,D), (D,D), [X,X]):")
    print(f"  {best.chromosome[-16:]}")
