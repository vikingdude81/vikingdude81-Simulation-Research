"""
🧬 ECHO Model with SPATIAL STRUCTURE (Grid-Based Interactions)

Key additions:
- 2D grid topology (agents have positions)
- Local neighborhood interactions only (Moore neighborhood = 8 neighbors)
- Tag-based matching within local neighborhood
- Visual heatmap of population density
- Cluster formation analysis
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import time
import os
from colorama import init, Fore, Style

init(autoreset=True)

# --- CONFIGURATION ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2

# Spatial parameters
GRID_WIDTH = 30
GRID_HEIGHT = 30
NEIGHBORHOOD_RADIUS = 1  # Moore neighborhood (8 neighbors)

# Resource parameters
INITIAL_RESOURCES = 100
REPRODUCTION_THRESHOLD = 200
DEATH_THRESHOLD = 0
METABOLISM_COST = 1
MAX_POPULATION = 500

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

# --- SPATIAL AGENT CLASS ---

class SpatialEchoAgent:
    """Agent with position, tag, strategy, and resources."""
    
    MOVE_MAP = {
        ('C', 'C'): 0, ('C', 'D'): 1,
        ('D', 'C'): 2, ('D', 'D'): 3
    }
    
    RESOURCE_PAYOFFS = {
        ('C', 'C'): (20, 20),
        ('D', 'C'): (30, -10),
        ('C', 'D'): (-10, 30),
        ('D', 'D'): (5, 5),
    }
    
    def __init__(self, agent_id: int, chromosome: str, 
                 position: Tuple[int, int], resources: float = INITIAL_RESOURCES):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy = chromosome[TAG_LENGTH:]
        self.position = position  # (x, y) on grid
        self.resources = resources
        self.age = 0
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        self.cooperations = 0
        self.defections = 0
        self.children_produced = 0
    
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
    
    def can_match(self, other: 'SpatialEchoAgent') -> bool:
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def interact(self, other: 'SpatialEchoAgent', rounds: int = 5) -> Tuple[float, float]:
        self.reset_history()
        other.reset_history()
        
        my_total = 0.0
        other_total = 0.0
        
        for _ in range(rounds):
            my_move = self.get_move()
            other_move = other.get_move()
            
            my_payoff, other_payoff = self.RESOURCE_PAYOFFS[(my_move, other_move)]
            my_total += my_payoff
            other_total += other_payoff
            
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
        
        return my_total, other_total
    
    def metabolize(self):
        self.resources -= METABOLISM_COST
        self.age += 1
    
    def is_alive(self) -> bool:
        return self.resources > DEATH_THRESHOLD
    
    def can_reproduce(self) -> bool:
        return self.resources >= REPRODUCTION_THRESHOLD
    
    def reproduce(self, mutation_rate: float = 0.01) -> 'SpatialEchoAgent':
        reproduction_cost = REPRODUCTION_THRESHOLD / 2
        self.resources -= reproduction_cost
        child_resources = reproduction_cost
        child_chromosome = mutate(self.chromosome, mutation_rate)
        child = SpatialEchoAgent(-1, child_chromosome, self.position, child_resources)
        child.age = 0
        self.children_produced += 1
        return child

# --- SPATIAL POPULATION CLASS ---

class SpatialEchoPopulation:
    """Population with 2D grid spatial structure."""
    
    def __init__(self, initial_size: int = 50, mutation_rate: float = 0.01,
                 rounds_per_interaction: int = 5):
        self.mutation_rate = mutation_rate
        self.rounds_per_interaction = rounds_per_interaction
        
        # Initialize grid
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Create initial agents at random positions
        self.agents: List[SpatialEchoAgent] = []
        positions_used = set()
        
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            
            # Find empty position
            while True:
                x = random.randint(0, GRID_WIDTH - 1)
                y = random.randint(0, GRID_HEIGHT - 1)
                if (x, y) not in positions_used:
                    break
            
            agent = SpatialEchoAgent(i, chromosome, (x, y))
            self.agents.append(agent)
            self.grid[y][x] = agent
            positions_used.add((x, y))
        
        # Add TFT agents with same tag in one corner (cluster)
        tft_tag = "11110000"
        for i in range(5):
            # Place in top-left corner
            while True:
                x = random.randint(0, 5)
                y = random.randint(0, 5)
                if (x, y) not in positions_used:
                    break
            
            tft_agent = SpatialEchoAgent(len(self.agents), 
                                        create_tit_for_tat_with_tag(tft_tag),
                                        (x, y))
            self.agents.append(tft_agent)
            self.grid[y][x] = tft_agent
            positions_used.add((x, y))
        
        self.generation = 0
        self.next_id = len(self.agents)
        self.start_time = time.time()
        
        self.history = {
            'population': [],
            'resources': [],
            'cooperation': [],
            'births': [],
            'deaths': [],
            'clustering': []  # Track spatial clustering
        }
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[SpatialEchoAgent]:
        """Get Moore neighborhood (8 neighbors) that are alive."""
        x, y = position
        neighbors = []
        
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % GRID_WIDTH  # Toroidal wrap
                ny = (y + dy) % GRID_HEIGHT
                
                neighbor = self.grid[ny][nx]
                if neighbor is not None and neighbor.is_alive():
                    neighbors.append(neighbor)
        
        return neighbors
    
    def run_interactions(self):
        """Run interactions with spatial neighbors only."""
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            # Get local neighbors
            neighbors = self.get_neighbors(agent.position)
            
            if not neighbors:
                agent.resources += SpatialEchoAgent.RESOURCE_PAYOFFS[('D', 'D')][0]
                continue
            
            # Filter by tag match
            matches = [n for n in neighbors if agent.can_match(n)]
            
            if matches:
                partner = random.choice(matches)
                agent_gain, partner_gain = agent.interact(partner, self.rounds_per_interaction)
                agent.resources += agent_gain
                partner.resources += partner_gain
            else:
                agent.resources += SpatialEchoAgent.RESOURCE_PAYOFFS[('D', 'D')][0]
    
    def metabolize_all(self):
        for agent in self.agents:
            agent.metabolize()
    
    def remove_dead(self) -> int:
        initial_count = len(self.agents)
        dead_agents = [a for a in self.agents if not a.is_alive()]
        
        # Remove from grid
        for agent in dead_agents:
            x, y = agent.position
            self.grid[y][x] = None
        
        # Remove from list
        self.agents = [a for a in self.agents if a.is_alive()]
        return initial_count - len(self.agents)
    
    def find_empty_neighbor(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find an empty cell near the parent."""
        x, y = position
        attempts = []
        
        # Try Moore neighborhood first
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                if self.grid[ny][nx] is None:
                    attempts.append((nx, ny))
        
        if attempts:
            return random.choice(attempts)
        
        # If no nearby empty, try wider search
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                if self.grid[ny][nx] is None:
                    return (nx, ny)
        
        return None  # Grid full
    
    def handle_reproduction(self):
        new_agents = []
        
        for agent in self.agents:
            if len(self.agents) + len(new_agents) >= MAX_POPULATION:
                break
            
            if agent.can_reproduce():
                # Find empty spot near parent
                empty_pos = self.find_empty_neighbor(agent.position)
                if empty_pos is None:
                    continue  # No space, can't reproduce
                
                child = agent.reproduce(self.mutation_rate)
                child.id = self.next_id
                child.position = empty_pos
                self.next_id += 1
                new_agents.append(child)
                
                # Place on grid
                x, y = empty_pos
                self.grid[y][x] = child
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def calculate_clustering(self) -> float:
        """Calculate spatial clustering coefficient (0=random, 1=highly clustered)."""
        if len(self.agents) < 2:
            return 0.0
        
        same_tag_neighbors = 0
        total_neighbors = 0
        
        for agent in self.agents:
            neighbors = self.get_neighbors(agent.position)
            if not neighbors:
                continue
            
            for neighbor in neighbors:
                total_neighbors += 1
                if agent.tag == neighbor.tag:
                    same_tag_neighbors += 1
        
        if total_neighbors == 0:
            return 0.0
        
        return same_tag_neighbors / total_neighbors
    
    def step(self):
        """Run one generation."""
        self.run_interactions()
        self.metabolize_all()
        deaths = self.remove_dead()
        births = self.handle_reproduction()
        
        # Record history
        if self.agents:
            total_res = sum(a.resources for a in self.agents)
            total_actions = sum(a.cooperations + a.defections for a in self.agents)
            total_coop = sum(a.cooperations for a in self.agents)
            coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
            clustering = self.calculate_clustering()
            
            self.history['population'].append(len(self.agents))
            self.history['resources'].append(total_res)
            self.history['cooperation'].append(coop_rate)
            self.history['births'].append(births)
            self.history['deaths'].append(deaths)
            self.history['clustering'].append(clustering)
        
        self.generation += 1
        return deaths, births
    
    def print_grid_snapshot(self):
        """Print ASCII grid showing population density."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}SPATIAL GRID SNAPSHOT (Generation {self.generation}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        
        # Create density map
        for y in range(GRID_HEIGHT):
            row = ""
            for x in range(GRID_WIDTH):
                agent = self.grid[y][x]
                if agent is None:
                    row += Fore.WHITE + "·"
                elif agent.resources > 10000:
                    row += Fore.GREEN + "█"
                elif agent.resources > 1000:
                    row += Fore.YELLOW + "▓"
                else:
                    row += Fore.RED + "░"
            print(row + Style.RESET_ALL)
        
        print(f"\n{Fore.WHITE}Legend: {Fore.GREEN}█{Style.RESET_ALL}=Rich(>10K) ", end="")
        print(f"{Fore.YELLOW}▓{Style.RESET_ALL}=Medium(>1K) ", end="")
        print(f"{Fore.RED}░{Style.RESET_ALL}=Poor(<1K) ", end="")
        print(f"{Fore.WHITE}·{Style.RESET_ALL}=Empty\n")

# --- MAIN EXECUTION ---

def run_spatial_echo_model(generations: int = 200):
    """Run the spatial Echo model."""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}🧬 SPATIAL ECHO MODEL - Grid-Based Interactions{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    print(f"{Fore.WHITE}Grid Size: {GRID_WIDTH}×{GRID_HEIGHT} = {GRID_WIDTH * GRID_HEIGHT} cells")
    print(f"Initial Population: 55 agents")
    print(f"Neighborhood: Moore (8 neighbors)")
    print(f"Tag Matching: Hamming ≤ {MATCH_THRESHOLD}")
    print(f"Max Population: {MAX_POPULATION}{Style.RESET_ALL}\n")
    
    pop = SpatialEchoPopulation()
    
    for gen in range(generations):
        pop.step()
        
        # Print progress every 25 generations
        if (gen + 1) % 25 == 0:
            print(f"\n{Fore.YELLOW}Generation {gen + 1}/{generations}{Style.RESET_ALL}")
            print(f"  Population: {len(pop.agents)}")
            print(f"  Avg Resources: {np.mean([a.resources for a in pop.agents]):.1f}")
            print(f"  Cooperation: {pop.history['cooperation'][-1]*100:.1f}%")
            print(f"  Clustering: {pop.history['clustering'][-1]*100:.1f}%")
            
            if (gen + 1) % 50 == 0:
                pop.print_grid_snapshot()
    
    # Final summary
    elapsed = time.time() - pop.start_time
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"SIMULATION COMPLETE!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
    
    print(f"Time: {elapsed:.1f}s | Speed: {generations/elapsed:.2f} gen/s")
    print(f"Final Population: {len(pop.agents)}")
    print(f"Final Clustering: {pop.history['clustering'][-1]*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spatial_echo_{timestamp}.json"
    
    results = {
        'metadata': {
            'generations': generations,
            'initial_population': 55,
            'final_population': len(pop.agents),
            'grid_size': f"{GRID_WIDTH}×{GRID_HEIGHT}",
            'elapsed_time': elapsed
        },
        'history': pop.history
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    pop.print_grid_snapshot()

if __name__ == "__main__":
    run_spatial_echo_model(200)
