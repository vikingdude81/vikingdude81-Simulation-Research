"""
🧬 GENETIC TRAITS - Extended Chromosome System
===============================================

This module extends the Echo model's genetic system beyond the simple
[TAG + STRATEGY] chromosome to include complex, evolvable traits.

Original Chromosome: [TAG (8 bits)] + [STRATEGY (1 bit)]

Extended Chromosome: [TAG (8 bits)] + [STRATEGY (1 bit)] + [METABOLISM (2 bits)] + 
                     [VISION (3 bits)] + [LIFESPAN (4 bits)] + [REPRODUCTION_COST (3 bits)]

Total: 21 bits per agent

New Traits:
1. METABOLISM: Cost of living per round (0-3 wealth)
2. VISION: Interaction range (1-8 squares)
3. LIFESPAN: Maximum age (1-15 generations)
4. REPRODUCTION_COST: Cost to reproduce (0-7 wealth)

Research Questions:
- Does metabolism create evolutionary pressure for cooperation?
- Do high-vision agents dominate low-vision agents?
- Do long-lived "dynasties" hoard resources?
- What's the optimal balance between traits?
"""

from typing import Tuple, List, Dict
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class GeneticTraits:
    """Decoded genetic traits from chromosome."""
    tag: int  # 8-bit tag for matching (0-255)
    strategy: int  # 1-bit: 0=Defect, 1=Cooperate
    metabolism: int  # 2-bit: Cost per round (0-3)
    vision: int  # 3-bit: Interaction range (1-8)
    lifespan: int  # 4-bit: Max age (1-15)
    reproduction_cost: int  # 3-bit: Cost to reproduce (0-7)


class ExtendedChromosome:
    """
    Extended genetic system with multiple evolvable traits.
    
    Chromosome structure (21 bits):
    [TAG:8] [STRATEGY:1] [METABOLISM:2] [VISION:3] [LIFESPAN:4] [REPRODUCTION_COST:3]
    
    Example: 11010011 1 10 101 1010 011
             ^^^^^^^^ ^ ^^ ^^^ ^^^^ ^^^
             TAG      S M  V   L    R
    """
    
    CHROMOSOME_LENGTH = 21
    
    # Gene positions (start_bit, num_bits)
    TAG_POS = (0, 8)
    STRATEGY_POS = (8, 1)
    METABOLISM_POS = (9, 2)
    VISION_POS = (11, 3)
    LIFESPAN_POS = (14, 4)
    REPRODUCTION_POS = (17, 3)
    
    # Vision cost per square (optional - can be 0)
    VISION_COST_PER_SQUARE = 0.5
    
    @staticmethod
    def create_random() -> np.ndarray:
        """Create a random 21-bit chromosome."""
        return np.random.randint(0, 2, ExtendedChromosome.CHROMOSOME_LENGTH)
    
    @staticmethod
    def decode(chromosome: np.ndarray) -> GeneticTraits:
        """
        Decode chromosome into individual traits.
        
        Args:
            chromosome: 21-bit numpy array
            
        Returns:
            GeneticTraits object with decoded values
        """
        # Extract each gene segment
        tag = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.TAG_POS)
        strategy = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.STRATEGY_POS)
        metabolism = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.METABOLISM_POS)
        vision_raw = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.VISION_POS)
        lifespan_raw = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.LIFESPAN_POS)
        reproduction = ExtendedChromosome._extract_gene(chromosome, *ExtendedChromosome.REPRODUCTION_POS)
        
        # Map vision from 0-7 to 1-8 (can't have 0 vision)
        vision = vision_raw + 1
        
        # Map lifespan from 0-15 to 1-16 (can't have 0 lifespan)
        lifespan = lifespan_raw + 1
        
        return GeneticTraits(
            tag=tag,
            strategy=strategy,
            metabolism=metabolism,
            vision=vision,
            lifespan=lifespan,
            reproduction_cost=reproduction
        )
    
    @staticmethod
    def _extract_gene(chromosome: np.ndarray, start: int, length: int) -> int:
        """Extract a gene segment and convert to integer."""
        gene_bits = chromosome[start:start+length]
        # Convert binary array to integer
        value = 0
        for i, bit in enumerate(gene_bits):
            value += bit * (2 ** (length - 1 - i))
        return value
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
        """
        Mutate chromosome with given probability per bit.
        
        Args:
            chromosome: 21-bit numpy array
            mutation_rate: Probability of flipping each bit
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover between two chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two child chromosomes
        """
        crossover_point = np.random.randint(1, ExtendedChromosome.CHROMOSOME_LENGTH)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2


class ExtendedEchoAgent:
    """
    Echo agent with extended genetic traits.
    
    This agent has:
    - Metabolism: Must pay cost each round to survive
    - Vision: Can interact with agents up to N squares away
    - Lifespan: Dies of old age after N generations
    - Reproduction cost: Must pay to create offspring
    """
    
    def __init__(self, chromosome: np.ndarray, initial_wealth: float = 10, pos: Tuple[int, int] = (0, 0)):
        self.chromosome = chromosome
        self.traits = ExtendedChromosome.decode(chromosome)
        self.wealth = initial_wealth
        self.pos = pos
        self.age = 0
        self.dead = False
        
        # Statistics
        self.interactions = 0
        self.children_born = 0
        self.total_payoffs = 0
    
    def get_tag(self) -> int:
        """Get agent's tag for matching."""
        return self.traits.tag
    
    def get_strategy(self) -> int:
        """Get agent's strategy (0=Defect, 1=Cooperate)."""
        return self.traits.strategy
    
    def get_vision_range(self) -> int:
        """Get agent's vision range."""
        return self.traits.vision
    
    def get_lifespan(self) -> int:
        """Get agent's maximum lifespan."""
        return self.traits.lifespan
    
    def pay_metabolism(self) -> bool:
        """
        Pay metabolism cost.
        
        Returns:
            True if agent survived, False if agent died (bankrupt)
        """
        cost = self.traits.metabolism
        
        # Optional: Add vision cost
        if ExtendedChromosome.VISION_COST_PER_SQUARE > 0:
            cost += self.traits.vision * ExtendedChromosome.VISION_COST_PER_SQUARE
        
        self.wealth -= cost
        
        if self.wealth <= 0:
            self.dead = True
            return False
        
        return True
    
    def age_one_generation(self) -> bool:
        """
        Age the agent by one generation.
        
        Returns:
            True if agent is still alive, False if died of old age
        """
        self.age += 1
        
        if self.age >= self.traits.lifespan:
            self.dead = True
            return False
        
        return True
    
    def can_reproduce(self) -> bool:
        """Check if agent has enough wealth to reproduce."""
        return self.wealth >= self.traits.reproduction_cost
    
    def reproduce(self, partner: 'ExtendedEchoAgent', mutation_rate: float = 0.01) -> 'ExtendedEchoAgent':
        """
        Reproduce with partner via genetic crossover.
        
        Args:
            partner: Other agent to reproduce with
            mutation_rate: Probability of mutation per bit
            
        Returns:
            New child agent
        """
        # Pay reproduction cost
        self.wealth -= self.traits.reproduction_cost / 2
        partner.wealth -= partner.traits.reproduction_cost / 2
        
        # Genetic crossover
        child_chromosome, _ = ExtendedChromosome.crossover(self.chromosome, partner.chromosome)
        
        # Mutation
        child_chromosome = ExtendedChromosome.mutate(child_chromosome, mutation_rate)
        
        # Create child near parent
        child_pos = self.pos  # Could add spatial variation
        child = ExtendedEchoAgent(child_chromosome, initial_wealth=5, pos=child_pos)
        
        # Update statistics
        self.children_born += 1
        partner.children_born += 1
        
        return child
    
    def find_neighbors(self, all_agents: List['ExtendedEchoAgent'], grid_size: Tuple[int, int], max_sample: int = 50) -> List['ExtendedEchoAgent']:
        """
        Find agents within vision range.
        
        OPTIMIZED: Uses random sampling for large populations (O(1) instead of O(N)).
        For small populations (<= max_sample), checks all agents.
        For large populations, randomly samples max_sample agents and checks those.
        
        Args:
            all_agents: List of all agents in simulation
            grid_size: (width, height) of grid
            max_sample: Maximum number of agents to check (default 50)
            
        Returns:
            List of agents within vision range
        """
        neighbors = []
        vision_range = self.traits.vision
        
        # Filter out self and dead agents
        candidates = [a for a in all_agents if a is not self and not a.dead]
        
        if len(candidates) == 0:
            return []
        
        # For large populations, randomly sample to avoid O(N²) complexity
        if len(candidates) > max_sample:
            candidates = random.sample(candidates, max_sample)
        
        # Check distance for sampled candidates
        for other in candidates:
            # Calculate Manhattan distance with wrapping
            dx = abs(self.pos[0] - other.pos[0])
            dy = abs(self.pos[1] - other.pos[1])
            
            # Handle wrapping
            if dx > grid_size[0] / 2:
                dx = grid_size[0] - dx
            if dy > grid_size[1] / 2:
                dy = grid_size[1] - dy
            
            manhattan_distance = dx + dy
            
            if manhattan_distance <= vision_range:
                neighbors.append(other)
        
        return neighbors
    
    def __repr__(self):
        return (f"Agent(tag={self.traits.tag}, strat={self.traits.strategy}, "
                f"meta={self.traits.metabolism}, vision={self.traits.vision}, "
                f"lifespan={self.traits.lifespan}, age={self.age}, wealth={self.wealth:.1f})")


def analyze_population_genetics(agents: List[ExtendedEchoAgent]) -> Dict:
    """
    Analyze genetic distribution across population.
    
    Returns:
        Dict with genetic statistics
    """
    if not agents:
        return {}
    
    strategies = [a.traits.strategy for a in agents if not a.dead]
    metabolisms = [a.traits.metabolism for a in agents if not a.dead]
    visions = [a.traits.vision for a in agents if not a.dead]
    lifespans = [a.traits.lifespan for a in agents if not a.dead]
    ages = [a.age for a in agents if not a.dead]
    
    return {
        'population': len([a for a in agents if not a.dead]),
        'cooperation_rate': np.mean(strategies) if strategies else 0,
        'avg_metabolism': np.mean(metabolisms) if metabolisms else 0,
        'avg_vision': np.mean(visions) if visions else 0,
        'avg_lifespan_gene': np.mean(lifespans) if lifespans else 0,
        'avg_age': np.mean(ages) if ages else 0,
        'oldest_agent': max(ages) if ages else 0,
        'metabolisms': {i: metabolisms.count(i) for i in range(4)},
        'visions': {i: visions.count(i) for i in range(1, 9)},
        'lifespans': {i: lifespans.count(i) for i in range(1, 17)}
    }


if __name__ == "__main__":
    print("🧬 GENETIC TRAITS MODULE")
    print("=" * 60)
    
    # Demo: Create and decode random chromosome
    print("\n📊 Random Agent Example:")
    chromosome = ExtendedChromosome.create_random()
    traits = ExtendedChromosome.decode(chromosome)
    
    print(f"Chromosome (21 bits): {chromosome}")
    print(f"\nDecoded Traits:")
    print(f"  TAG:          {traits.tag:3d} (8-bit, 0-255)")
    print(f"  STRATEGY:     {traits.strategy:3d} ({'Cooperate' if traits.strategy == 1 else 'Defect'})")
    print(f"  METABOLISM:   {traits.metabolism:3d} (cost per round)")
    print(f"  VISION:       {traits.vision:3d} (interaction range)")
    print(f"  LIFESPAN:     {traits.lifespan:3d} (max age)")
    print(f"  REPRODUCTION: {traits.reproduction_cost:3d} (cost to reproduce)")
    
    # Demo: Create agent and test methods
    print("\n🔬 Agent Simulation:")
    agent = ExtendedEchoAgent(chromosome, initial_wealth=20, pos=(25, 25))
    print(f"Initial: {agent}")
    
    # Simulate 5 rounds
    for round in range(1, 6):
        survived_metabolism = agent.pay_metabolism()
        survived_aging = agent.age_one_generation()
        
        if not survived_metabolism:
            print(f"Round {round}: DIED (bankruptcy)")
            break
        elif not survived_aging:
            print(f"Round {round}: DIED (old age)")
            break
        else:
            print(f"Round {round}: {agent}")
    
    print("\n✅ Module loaded successfully!")
    print("To use: Import ExtendedEchoAgent and ExtendedChromosome")
