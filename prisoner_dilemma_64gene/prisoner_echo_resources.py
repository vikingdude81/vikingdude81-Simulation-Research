"""
🧬 ECHO Model: Evolution with Tags, Strategy, and Resource Exchange

Based on John Holland's "Hidden Order" - Pages 82-84 and Figure 3.3
This implements the full "Echo" model with:
1. 64-gene strategy lookup table (3-move memory)
2. 8-bit tag system for conditional interaction
3. Resource exchange and accumulation
4. Fitness based on resource wealth

Chromosome Structure (72 genes total):
    [TAG: 8 bits] + [STRATEGY: 64 bits]
    
    TAG (8 bits): Identity marker for selective interaction
        - Example: "10110011"
        - Agents match if Hamming distance ≤ 2
    
    STRATEGY (64 bits): Full 3-move history lookup table
        - Each gene = action for specific history state
        - Index = (hist_t-3 * 16) + (hist_t-2 * 4) + (hist_t-1)

Resource System:
    - Agents start with initial resource allocation
    - Cooperation generates surplus resources
    - Defection steals resources from partner
    - Fitness = accumulated resources
    - Death if resources drop to 0
    - Reproduction when resources exceed threshold
"""

import random
import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime

# --- CONFIGURATION ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2  # Max Hamming distance for tag matching

# Resource parameters
INITIAL_RESOURCES = 100
REPRODUCTION_THRESHOLD = 200
DEATH_THRESHOLD = 0
METABOLISM_COST = 1  # Cost per round just to survive

# --- 1. GENETIC ALGORITHM FUNCTIONS ---

def create_random_chromosome() -> str:
    """Creates a 72-gene chromosome (8-bit tag + 64-bit strategy)."""
    tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "".join(random.choice(['C', 'D']) for _ in range(STRATEGY_LENGTH))
    return tag + strategy

def create_tit_for_tat_with_tag(tag: str | None = None) -> str:
    """Creates Tit-for-Tat strategy with specified or random tag."""
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "CD" * 32  # Tit-for-Tat pattern
    return tag + strategy

def create_always_defect_with_tag(tag: str | None = None) -> str:
    """Creates Always Defect strategy with specified or random tag."""
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "D" * 64
    return tag + strategy

def crossover(parent1: str, parent2: str) -> str:
    """Performs single-point crossover on 72-gene chromosomes."""
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.01) -> str:
    """
    Mutates chromosome, respecting tag vs. strategy genes.
    Tag genes: 0 <-> 1
    Strategy genes: C <-> D
    """
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < rate:
            if i < TAG_LENGTH:
                # Mutate tag gene
                chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
            else:
                # Mutate strategy gene
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
    return "".join(chrom_list)

def hamming_distance(tag1: str, tag2: str) -> int:
    """Calculates Hamming distance between two tags."""
    return sum(c1 != c2 for c1, c2 in zip(tag1, tag2))

# --- 2. AGENT CLASS ---

class EchoAgent:
    """
    An agent with tag, strategy, and resources.
    Fitness is determined by resource accumulation.
    """
    
    MOVE_MAP = {
        ('C', 'C'): 0,
        ('C', 'D'): 1,
        ('D', 'C'): 2,
        ('D', 'D'): 3
    }
    
    # Resource exchange payoffs (replaces simple scores)
    RESOURCE_PAYOFFS = {
        ('C', 'C'): (20, 20),   # Cooperation generates surplus (Win-Win)
        ('D', 'C'): (30, -10),  # Defector steals from cooperator
        ('C', 'D'): (-10, 30),  # Cooperator loses to defector
        ('D', 'D'): (5, 5),     # Mutual defection generates little
    }
    
    def __init__(self, agent_id: int, chromosome: str, resources: float = INITIAL_RESOURCES):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy = chromosome[TAG_LENGTH:]
        self.resources = resources
        self.age = 0
        
        # Game history
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        
        # Statistics
        self.interactions = 0
        self.cooperations = 0
        self.defections = 0
        self.matches_found = 0
        self.matches_rejected = 0
        self.children_produced = 0
    
    def __repr__(self):
        return f"Agent{self.id}(R={self.resources:.1f}, tag={self.tag}, {self.strategy[:8]}...)"
    
    def reset_history(self):
        """Reset game history for new interaction."""
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    
    def get_history_index(self) -> int:
        """
        Converts 3-move joint history into index (0-63).
        Formula: (hist[0] * 16) + (hist[1] * 4) + hist[2]
        """
        val_0 = self.MOVE_MAP[self.history[0]]  # Oldest
        val_1 = self.MOVE_MAP[self.history[1]]
        val_2 = self.MOVE_MAP[self.history[2]]  # Most recent
        
        return (val_0 * 16) + (val_1 * 4) + val_2
    
    def get_move(self) -> str:
        """Decides next move based on chromosome strategy."""
        index = self.get_history_index()
        return self.strategy[index]
    
    def update_history(self, my_move: str, opponent_move: str):
        """Updates 3-move sliding window history."""
        self.history.append((my_move, opponent_move))
        self.history.pop(0)
    
    def can_match(self, other: 'EchoAgent') -> bool:
        """Check if this agent's tag matches another's."""
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def interact(self, other: 'EchoAgent', rounds: int = 5) -> Tuple[float, float]:
        """
        Play repeated Prisoner's Dilemma with resource exchange.
        Returns (my_resource_change, other_resource_change).
        """
        self.reset_history()
        other.reset_history()
        
        my_total = 0.0
        other_total = 0.0
        
        for _ in range(rounds):
            # Get moves
            my_move = self.get_move()
            other_move = other.get_move()
            
            # Get resource payoffs
            my_payoff, other_payoff = self.RESOURCE_PAYOFFS[(my_move, other_move)]
            my_total += my_payoff
            other_total += other_payoff
            
            # Update histories
            self.update_history(my_move, other_move)
            other.update_history(other_move, my_move)
            
            # Track statistics
            if my_move == 'C':
                self.cooperations += 1
            else:
                self.defections += 1
            
            if other_move == 'C':
                other.cooperations += 1
            else:
                other.defections += 1
        
        self.interactions += rounds
        other.interactions += rounds
        
        return my_total, other_total
    
    def metabolize(self):
        """Pay metabolism cost for survival."""
        self.resources -= METABOLISM_COST
        self.age += 1
    
    def is_alive(self) -> bool:
        """Check if agent has sufficient resources to survive."""
        return self.resources > DEATH_THRESHOLD
    
    def can_reproduce(self) -> bool:
        """Check if agent has sufficient resources to reproduce."""
        return self.resources >= REPRODUCTION_THRESHOLD
    
    def reproduce(self, mutation_rate: float = 0.01) -> 'EchoAgent':
        """
        Create offspring with mutation.
        Parent pays reproduction cost.
        """
        # Reproduction cost
        reproduction_cost = REPRODUCTION_THRESHOLD / 2
        self.resources -= reproduction_cost
        
        # Child gets starting resources
        child_resources = reproduction_cost
        
        # Mutate chromosome
        child_chromosome = mutate(self.chromosome, mutation_rate)
        
        # Create child
        child = EchoAgent(-1, child_chromosome, child_resources)  # ID assigned by population
        self.children_produced += 1
        
        return child

# --- 3. POPULATION MANAGEMENT ---

class EchoPopulation:
    """Manages the population of Echo agents with resources."""
    
    def __init__(self, 
                 initial_size: int = 50,
                 mutation_rate: float = 0.01,
                 rounds_per_interaction: int = 5):
        
        self.mutation_rate = mutation_rate
        self.rounds_per_interaction = rounds_per_interaction
        
        # Initialize population with random strategies
        self.agents: List[EchoAgent] = []
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            agent = EchoAgent(i, chromosome)
            self.agents.append(agent)
        
        # Add some known strategies
        # TFT agents with similar tags
        tft_tag = "11110000"
        for i in range(5):
            tft_agent = EchoAgent(len(self.agents), 
                                 create_tit_for_tat_with_tag(tft_tag))
            self.agents.append(tft_agent)
        
        self.generation = 0
        self.next_id = len(self.agents)
        
        # Statistics
        self.history = {
            'generation': [],
            'population_size': [],
            'avg_resources': [],
            'total_resources': [],
            'tag_diversity': [],
            'strategy_diversity': [],
            'cooperation_rate': [],
            'interaction_rate': []
        }
    
    def run_interactions(self):
        """Run random pairwise interactions among agents."""
        # Shuffle agents
        random.shuffle(self.agents)
        
        # Each agent gets multiple chances to interact
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            # Find potential partners
            potential_partners = [a for a in self.agents 
                                if a.id != agent.id and a.is_alive()]
            
            if not potential_partners:
                continue
            
            # Try to find matching partner
            matches = [p for p in potential_partners if agent.can_match(p)]
            
            if matches:
                # Interact with random matching partner
                partner = random.choice(matches)
                agent.matches_found += 1
                partner.matches_found += 1
                
                # Play game with resource exchange
                agent_gain, partner_gain = agent.interact(partner, 
                                                         self.rounds_per_interaction)
                
                agent.resources += agent_gain
                partner.resources += partner_gain
            else:
                # No matches found - loner payoff
                agent.matches_rejected += 1
                agent.resources += EchoAgent.RESOURCE_PAYOFFS[('D', 'D')][0]
    
    def metabolize_all(self):
        """All agents pay metabolism cost."""
        for agent in self.agents:
            agent.metabolize()
    
    def remove_dead(self) -> int:
        """Remove agents with insufficient resources."""
        initial_count = len(self.agents)
        self.agents = [a for a in self.agents if a.is_alive()]
        deaths = initial_count - len(self.agents)
        return deaths
    
    def handle_reproduction(self):
        """Agents with sufficient resources reproduce."""
        new_agents = []
        
        for agent in self.agents:
            if agent.can_reproduce():
                child = agent.reproduce(self.mutation_rate)
                child.id = self.next_id
                self.next_id += 1
                new_agents.append(child)
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def collect_statistics(self):
        """Collect population statistics."""
        if not self.agents:
            return
        
        resources = [a.resources for a in self.agents]
        tags = [a.tag for a in self.agents]
        strategies = [a.strategy for a in self.agents]
        
        total_interactions = sum(a.interactions for a in self.agents)
        total_cooperations = sum(a.cooperations for a in self.agents)
        
        self.history['generation'].append(self.generation)
        self.history['population_size'].append(len(self.agents))
        self.history['avg_resources'].append(np.mean(resources))
        self.history['total_resources'].append(np.sum(resources))
        self.history['tag_diversity'].append(len(set(tags)))
        self.history['strategy_diversity'].append(len(set(strategies)))
        self.history['cooperation_rate'].append(
            total_cooperations / total_interactions if total_interactions > 0 else 0
        )
        self.history['interaction_rate'].append(
            sum(a.matches_found for a in self.agents) / len(self.agents)
        )
    
    def step(self):
        """Run one generation of the Echo model."""
        # 1. Interactions (resource exchange)
        self.run_interactions()
        
        # 2. Metabolism (survival cost)
        self.metabolize_all()
        
        # 3. Death (remove agents with no resources)
        deaths = self.remove_dead()
        
        # 4. Reproduction (successful agents reproduce)
        births = self.handle_reproduction()
        
        # 5. Statistics
        self.collect_statistics()
        
        self.generation += 1
        
        return deaths, births
    
    def get_dominant_tags(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common tags in population."""
        tag_counts = {}
        for agent in self.agents:
            tag_counts[agent.tag] = tag_counts.get(agent.tag, 0) + 1
        
        return sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_dominant_strategies(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common strategies in population."""
        strategy_counts = {}
        for agent in self.agents:
            strategy_counts[agent.strategy] = strategy_counts.get(agent.strategy, 0) + 1
        
        return sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_richest_agents(self, top_n: int = 5) -> List[EchoAgent]:
        """Get agents with most resources."""
        return sorted(self.agents, key=lambda a: a.resources, reverse=True)[:top_n]
    
    def print_summary(self, verbose: bool = True):
        """Print generation summary."""
        if not verbose or not self.agents:
            return
        
        print(f"\n{'='*70}")
        print(f"Generation {self.generation} | Population: {len(self.agents)}")
        print(f"{'='*70}")
        
        resources = [a.resources for a in self.agents]
        print(f"Resources: Avg={np.mean(resources):.1f}, "
              f"Total={np.sum(resources):.0f}, "
              f"Min={np.min(resources):.1f}, "
              f"Max={np.max(resources):.1f}")
        
        # Dominant tags
        print(f"\nTop 3 Tags:")
        for tag, count in self.get_dominant_tags(3):
            pct = (count / len(self.agents)) * 100
            print(f"  {tag}: {count} agents ({pct:.1f}%)")
        
        # Dominant strategies
        print(f"\nTop 3 Strategies:")
        tft = "CD" * 32
        for strategy, count in self.get_dominant_strategies(3):
            pct = (count / len(self.agents)) * 100
            is_tft = " [TIT-FOR-TAT]" if strategy == tft else ""
            print(f"  {strategy[:16]}...: {count} agents ({pct:.1f}%){is_tft}")
        
        # Richest agents
        print(f"\nRichest Agents:")
        for i, agent in enumerate(self.get_richest_agents(3), 1):
            print(f"  {i}. Agent{agent.id}: R={agent.resources:.1f}, "
                  f"Age={agent.age}, Children={agent.children_produced}, "
                  f"Tag={agent.tag}")
        
        # Cooperation rate
        total_actions = sum(a.cooperations + a.defections for a in self.agents)
        total_coop = sum(a.cooperations for a in self.agents)
        coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
        print(f"\nCooperation Rate: {coop_rate:.1f}%")

# --- 4. SIMULATION RUNNER ---

def run_echo_simulation(generations: int = 200, 
                       initial_population: int = 50,
                       print_every: int = 25,
                       save_results: bool = True):
    """Run the full Echo simulation."""
    
    print("🧬 Starting ECHO Model Simulation")
    print("="*70)
    print(f"Initial Population: {initial_population}")
    print(f"Generations: {generations}")
    print(f"Chromosome: {TAG_LENGTH}-bit tag + {STRATEGY_LENGTH}-bit strategy")
    print(f"Match Threshold: Hamming distance ≤ {MATCH_THRESHOLD}")
    print(f"Resource System: Cooperation generates surplus, Defection steals")
    print("="*70)
    
    # Create population
    population = EchoPopulation(initial_size=initial_population)
    
    # Run simulation
    for gen in range(generations):
        deaths, births = population.step()
        
        if (gen + 1) % print_every == 0 or gen == 0:
            population.print_summary(verbose=True)
            print(f"Deaths: {deaths}, Births: {births}")
        
        # Check for extinction
        if len(population.agents) == 0:
            print("\n💀 EXTINCTION - Population died out!")
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏁 SIMULATION COMPLETE")
    print(f"{'='*70}")
    population.print_summary(verbose=True)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"echo_simulation_{timestamp}.json"
        
        results = {
            'metadata': {
                'generations': generations,
                'initial_population': initial_population,
                'final_population': len(population.agents),
                'tag_length': TAG_LENGTH,
                'strategy_length': STRATEGY_LENGTH,
                'match_threshold': MATCH_THRESHOLD
            },
            'history': population.history,
            'final_agents': [
                {
                    'id': a.id,
                    'tag': a.tag,
                    'strategy': a.strategy,
                    'resources': a.resources,
                    'age': a.age,
                    'children': a.children_produced
                }
                for a in population.agents[:20]  # Save top 20
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    return population

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    population = run_echo_simulation(
        generations=200,
        initial_population=50,
        print_every=25,
        save_results=True
    )
