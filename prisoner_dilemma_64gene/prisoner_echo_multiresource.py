"""
🧬 ECHO Model with MULTI-RESOURCE ECONOMY

Following Holland's "Hidden Order" Chapter 3:
- Multiple resource types (Food, Materials, Energy)
- Each agent consumes all resources (metabolism)
- Interactions exchange different resource combinations
- Reproduction requires ALL resources above threshold
- Different strategies can specialize in different resources
- Trade emerges: agents exchange surpluses
"""

import random
import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime
import time
from colorama import init, Fore, Style

init(autoreset=True)

# --- CONFIGURATION ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2

# Multi-resource parameters
RESOURCE_TYPES = ['food', 'materials', 'energy']
NUM_RESOURCES = len(RESOURCE_TYPES)

INITIAL_RESOURCES = {
    'food': 100,
    'materials': 100,
    'energy': 100
}

REPRODUCTION_THRESHOLDS = {
    'food': 200,
    'materials': 200,
    'energy': 200
}

DEATH_THRESHOLDS = {
    'food': 0,
    'materials': 0,
    'energy': 0
}

METABOLISM_COSTS = {
    'food': 2,      # Food consumed fastest
    'materials': 1,  # Materials used moderately
    'energy': 1.5    # Energy middle ground
}

MAX_POPULATION = 500

# Multi-resource payoffs: Different outcomes produce different resources!
# Format: (move1, move2) -> (agent_gains_dict, partner_gains_dict)
MULTI_RESOURCE_PAYOFFS = {
    # Both cooperate: Balanced gains in all resources (synergy!)
    ('C', 'C'): (
        {'food': 15, 'materials': 20, 'energy': 25},
        {'food': 15, 'materials': 20, 'energy': 25}
    ),
    # Defector exploits: Takes food & materials, gives nothing
    ('D', 'C'): (
        {'food': 30, 'materials': 30, 'energy': 10},
        {'food': -10, 'materials': -10, 'energy': 5}
    ),
    # Cooperator exploited: Loses food & materials
    ('C', 'D'): (
        {'food': -10, 'materials': -10, 'energy': 5},
        {'food': 30, 'materials': 30, 'energy': 10}
    ),
    # Both defect: Low gains, energy dominant
    ('D', 'D'): (
        {'food': 5, 'materials': 5, 'energy': 10},
        {'food': 5, 'materials': 5, 'energy': 10}
    ),
}

# --- GENETIC FUNCTIONS ---

def create_random_chromosome() -> str:
    tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "".join(random.choice(['C', 'D']) for _ in range(STRATEGY_LENGTH))
    return tag + strategy

def create_tit_for_tat_with_tag(tag: str | None = None) -> str:
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "CD" * 32
    return tag + strategy

def crossover(parent1: str, parent2: str) -> str:
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.01) -> str:
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < rate:
            if i < TAG_LENGTH:
                chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
            else:
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
    return "".join(chrom_list)

def hamming_distance(tag1: str, tag2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(tag1, tag2))

# --- MULTI-RESOURCE AGENT CLASS ---

class MultiResourceAgent:
    """Agent with multiple resource types."""
    
    MOVE_MAP = {
        ('C', 'C'): 0, ('C', 'D'): 1,
        ('D', 'C'): 2, ('D', 'D'): 3
    }
    
    def __init__(self, agent_id: int, chromosome: str, 
                 resources: Dict[str, float] = None):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy = chromosome[TAG_LENGTH:]
        
        # Multi-resource inventory
        if resources is None:
            self.resources = INITIAL_RESOURCES.copy()
        else:
            self.resources = resources.copy()
        
        self.age = 0
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        self.cooperations = 0
        self.defections = 0
        self.children_produced = 0
        
        # Track resource specialization
        self.resource_production = {r: 0.0 for r in RESOURCE_TYPES}
    
    def reset_history(self):
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
    
    def get_history_index(self) -> int:
        val_0 = self.MOVE_MAP[self.history[0]]
        val_1 = self.MOVE_MAP[self.history[1]]
        val_2 = self.MOVE_MAP[self.history[2]]
        return (val_0 * 16) + (val_1 * 4) + val_2
    
    def get_move(self) -> str:
        index = self.get_history_index()
        return self.strategy[index]
    
    def update_history(self, my_move: str, opponent_move: str):
        self.history.append((my_move, opponent_move))
        self.history.pop(0)
    
    def can_match(self, other: 'MultiResourceAgent') -> bool:
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def interact(self, other: 'MultiResourceAgent', rounds: int = 5):
        """Multi-resource interaction."""
        self.reset_history()
        other.reset_history()
        
        my_totals = {r: 0.0 for r in RESOURCE_TYPES}
        other_totals = {r: 0.0 for r in RESOURCE_TYPES}
        
        for _ in range(rounds):
            my_move = self.get_move()
            other_move = other.get_move()
            
            my_payoff, other_payoff = MULTI_RESOURCE_PAYOFFS[(my_move, other_move)]
            
            # Accumulate each resource type
            for resource_type in RESOURCE_TYPES:
                my_totals[resource_type] += my_payoff[resource_type]
                other_totals[resource_type] += other_payoff[resource_type]
            
            self.update_history(my_move, other_move)
            other.update_history(other_move, my_move)
            
            if my_move == 'C':
                self.cooperations += 1
            else:
                self.defections += 1
            
            if other_move == 'C':
                other.cooperations += 1
            else:
                other.defections += 1
        
        # Apply resource gains
        for resource_type in RESOURCE_TYPES:
            self.resources[resource_type] += my_totals[resource_type]
            other.resources[resource_type] += other_totals[resource_type]
            
            self.resource_production[resource_type] += my_totals[resource_type]
            other.resource_production[resource_type] += other_totals[resource_type]
    
    def metabolize(self):
        """Consume all resource types."""
        for resource_type in RESOURCE_TYPES:
            self.resources[resource_type] -= METABOLISM_COSTS[resource_type]
        self.age += 1
    
    def is_alive(self) -> bool:
        """Must have positive amounts of ALL resources to survive."""
        return all(self.resources[r] > DEATH_THRESHOLDS[r] for r in RESOURCE_TYPES)
    
    def can_reproduce(self) -> bool:
        """Must have ALL resources above threshold."""
        return all(self.resources[r] >= REPRODUCTION_THRESHOLDS[r] for r in RESOURCE_TYPES)
    
    def reproduce(self, mutation_rate: float = 0.01) -> 'MultiResourceAgent':
        """Reproduction costs resources from all types."""
        child_resources = {}
        
        for resource_type in RESOURCE_TYPES:
            cost = REPRODUCTION_THRESHOLDS[resource_type] / 2
            self.resources[resource_type] -= cost
            child_resources[resource_type] = cost
        
        child_chromosome = mutate(self.chromosome, mutation_rate)
        child = MultiResourceAgent(-1, child_chromosome, child_resources)
        child.age = 0
        self.children_produced += 1
        return child
    
    def get_total_wealth(self) -> float:
        """Sum of all resources."""
        return sum(self.resources.values())
    
    def get_limiting_resource(self) -> Tuple[str, float]:
        """Find which resource is most scarce."""
        limiting = min(RESOURCE_TYPES, key=lambda r: self.resources[r])
        return limiting, self.resources[limiting]

# --- MULTI-RESOURCE POPULATION ---

class MultiResourcePopulation:
    """Population with multi-resource economy."""
    
    def __init__(self, initial_size: int = 50, mutation_rate: float = 0.01,
                 rounds_per_interaction: int = 5):
        self.mutation_rate = mutation_rate
        self.rounds_per_interaction = rounds_per_interaction
        
        # Create agents
        self.agents: List[MultiResourceAgent] = []
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            agent = MultiResourceAgent(i, chromosome)
            self.agents.append(agent)
        
        # Add TFT agents
        tft_tag = "11110000"
        for i in range(5):
            tft_agent = MultiResourceAgent(len(self.agents), 
                                          create_tit_for_tat_with_tag(tft_tag))
            self.agents.append(tft_agent)
        
        self.generation = 0
        self.next_id = len(self.agents)
        self.start_time = time.time()
        
        # Multi-resource history
        self.history = {
            'population': [],
            'cooperation': [],
            'births': [],
            'deaths': [],
            'total_food': [],
            'total_materials': [],
            'total_energy': [],
            'avg_food': [],
            'avg_materials': [],
            'avg_energy': []
        }
    
    def run_interactions(self):
        """Run pairwise interactions."""
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            potential_partners = [a for a in self.agents 
                                if a.id != agent.id and a.is_alive()]
            
            if not potential_partners:
                continue
            
            matches = [p for p in potential_partners if agent.can_match(p)]
            
            if matches:
                partner = random.choice(matches)
                agent.interact(partner, self.rounds_per_interaction)
            else:
                # Loner score: Get defect-defect payoff
                for resource_type in RESOURCE_TYPES:
                    loner_payoff = MULTI_RESOURCE_PAYOFFS[('D', 'D')][0][resource_type]
                    agent.resources[resource_type] += loner_payoff
    
    def metabolize_all(self):
        for agent in self.agents:
            agent.metabolize()
    
    def remove_dead(self) -> int:
        initial_count = len(self.agents)
        self.agents = [a for a in self.agents if a.is_alive()]
        return initial_count - len(self.agents)
    
    def handle_reproduction(self):
        new_agents = []
        
        for agent in self.agents:
            if len(self.agents) + len(new_agents) >= MAX_POPULATION:
                break
            
            if agent.can_reproduce():
                child = agent.reproduce(self.mutation_rate)
                child.id = self.next_id
                self.next_id += 1
                new_agents.append(child)
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def step(self):
        """Run one generation."""
        self.run_interactions()
        self.metabolize_all()
        deaths = self.remove_dead()
        births = self.handle_reproduction()
        
        # Record multi-resource history
        if self.agents:
            total_actions = sum(a.cooperations + a.defections for a in self.agents)
            total_coop = sum(a.cooperations for a in self.agents)
            coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
            
            self.history['population'].append(len(self.agents))
            self.history['cooperation'].append(coop_rate)
            self.history['births'].append(births)
            self.history['deaths'].append(deaths)
            
            # Track each resource type
            for resource_type in RESOURCE_TYPES:
                total = sum(a.resources[resource_type] for a in self.agents)
                avg = total / len(self.agents)
                self.history[f'total_{resource_type}'].append(total)
                self.history[f'avg_{resource_type}'].append(avg)
        
        self.generation += 1
        return deaths, births
    
    def print_dashboard(self):
        """Print multi-resource dashboard."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}🧬 MULTI-RESOURCE ECHO MODEL (Gen {self.generation}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        
        if not self.agents:
            print(f"{Fore.RED}💀 EXTINCTION{Style.RESET_ALL}")
            return
        
        # Population stats
        print(f"{Fore.WHITE}Population: {Fore.GREEN}{len(self.agents)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Avg Age: {Fore.CYAN}{np.mean([a.age for a in self.agents]):.1f}{Style.RESET_ALL}\n")
        
        # Resource breakdown
        print(f"{Fore.YELLOW}RESOURCE INVENTORY:{Style.RESET_ALL}")
        for resource_type in RESOURCE_TYPES:
            total = sum(a.resources[resource_type] for a in self.agents)
            avg = total / len(self.agents)
            min_res = min(a.resources[resource_type] for a in self.agents)
            max_res = max(a.resources[resource_type] for a in self.agents)
            
            # Color based on resource
            if resource_type == 'food':
                color = Fore.GREEN
            elif resource_type == 'materials':
                color = Fore.YELLOW
            else:  # energy
                color = Fore.CYAN
            
            print(f"  {color}{resource_type.upper():<10}{Style.RESET_ALL} | ", end="")
            print(f"Total: {total:>10,.0f} | Avg: {avg:>8,.1f} | ", end="")
            print(f"Min: {min_res:>6,.1f} | Max: {max_res:>8,.1f}")
        
        # Cooperation
        total_actions = sum(a.cooperations + a.defections for a in self.agents)
        total_coop = sum(a.cooperations for a in self.agents)
        coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
        print(f"\n{Fore.WHITE}Cooperation Rate: {Fore.GREEN}{coop_rate:.1f}%{Style.RESET_ALL}")
        
        # Recent trends
        if len(self.history['births']) >= 5:
            recent_births = sum(self.history['births'][-5:])
            recent_deaths = sum(self.history['deaths'][-5:])
            print(f"{Fore.WHITE}Last 5 gens: {Fore.GREEN}+{recent_births} births{Style.RESET_ALL}, ", end="")
            print(f"{Fore.RED}-{recent_deaths} deaths{Style.RESET_ALL}")

# --- MAIN EXECUTION ---

def run_multi_resource_echo(generations: int = 200):
    """Run multi-resource Echo model."""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}🧬 MULTI-RESOURCE ECHO MODEL{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    print(f"{Fore.WHITE}Resource Types: {', '.join(RESOURCE_TYPES)}")
    print(f"Initial Population: 55 agents")
    print(f"Survival: Must have ALL resources > 0")
    print(f"Reproduction: Must have ALL resources >= 200{Style.RESET_ALL}\n")
    
    pop = MultiResourcePopulation()
    
    for gen in range(generations):
        pop.step()
        
        if (gen + 1) % 25 == 0:
            pop.print_dashboard()
    
    # Final summary
    elapsed = time.time() - pop.start_time
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"SIMULATION COMPLETE!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
    
    print(f"Time: {elapsed:.1f}s | Speed: {generations/elapsed:.2f} gen/s")
    print(f"Final Population: {len(pop.agents)}")
    
    # Analyze resource specialization
    if pop.agents:
        print(f"\n{Fore.YELLOW}RESOURCE ANALYSIS:{Style.RESET_ALL}")
        for resource_type in RESOURCE_TYPES:
            total = sum(a.resources[resource_type] for a in pop.agents)
            print(f"  {resource_type.capitalize()}: {total:,.0f} total")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multiresource_echo_{timestamp}.json"
    
    results = {
        'metadata': {
            'generations': generations,
            'initial_population': 55,
            'final_population': len(pop.agents),
            'resource_types': RESOURCE_TYPES,
            'elapsed_time': elapsed
        },
        'history': pop.history
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    run_multi_resource_echo(200)
