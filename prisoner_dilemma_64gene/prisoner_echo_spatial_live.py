"""
🧬 SPATIAL ECHO MODEL with LIVE VISUAL DASHBOARD

Combines:
- Live grid visualization (population density map)
- Real-time statistics
- Color-coded agents by wealth
- Cluster formation tracking
- Tag distribution
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
NEIGHBORHOOD_RADIUS = 1

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

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def color_tag(tag: str) -> str:
    """Color-code binary tag."""
    colored = ""
    for bit in tag:
        if bit == '0':
            colored += Fore.CYAN + bit
        else:
            colored += Fore.RED + bit
    return colored + Style.RESET_ALL

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
        self.position = position
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

# --- SPATIAL POPULATION WITH LIVE DASHBOARD ---

class SpatialEchoPopulation:
    """Population with 2D grid and live visualization."""
    
    def __init__(self, initial_size: int = 50, mutation_rate: float = 0.01,
                 rounds_per_interaction: int = 5):
        self.mutation_rate = mutation_rate
        self.rounds_per_interaction = rounds_per_interaction
        
        # Initialize grid
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Create initial agents
        self.agents: List[SpatialEchoAgent] = []
        positions_used = set()
        
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            
            while True:
                x = random.randint(0, GRID_WIDTH - 1)
                y = random.randint(0, GRID_HEIGHT - 1)
                if (x, y) not in positions_used:
                    break
            
            agent = SpatialEchoAgent(i, chromosome, (x, y))
            self.agents.append(agent)
            self.grid[y][x] = agent
            positions_used.add((x, y))
        
        # Add TFT cluster in top-left
        tft_tag = "11110000"
        for i in range(5):
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
            'clustering': []
        }
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[SpatialEchoAgent]:
        """Get Moore neighborhood (8 neighbors)."""
        x, y = position
        neighbors = []
        
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                
                neighbor = self.grid[ny][nx]
                if neighbor is not None and neighbor.is_alive():
                    neighbors.append(neighbor)
        
        return neighbors
    
    def run_interactions(self):
        """Run interactions with spatial neighbors."""
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            neighbors = self.get_neighbors(agent.position)
            
            if not neighbors:
                agent.resources += SpatialEchoAgent.RESOURCE_PAYOFFS[('D', 'D')][0]
                continue
            
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
        
        for agent in dead_agents:
            x, y = agent.position
            self.grid[y][x] = None
        
        self.agents = [a for a in self.agents if a.is_alive()]
        return initial_count - len(self.agents)
    
    def find_empty_neighbor(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find empty cell near parent."""
        x, y = position
        attempts = []
        
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
        
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                if self.grid[ny][nx] is None:
                    return (nx, ny)
        
        return None
    
    def handle_reproduction(self):
        new_agents = []
        
        for agent in self.agents:
            if len(self.agents) + len(new_agents) >= MAX_POPULATION:
                break
            
            if agent.can_reproduce():
                empty_pos = self.find_empty_neighbor(agent.position)
                if empty_pos is None:
                    continue
                
                child = agent.reproduce(self.mutation_rate)
                child.id = self.next_id
                child.position = empty_pos
                self.next_id += 1
                new_agents.append(child)
                
                x, y = empty_pos
                self.grid[y][x] = child
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def calculate_clustering(self) -> float:
        """Calculate spatial clustering coefficient."""
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
    
    def print_live_dashboard(self):
        """Print live spatial dashboard with grid visualization."""
        clear_screen()
        import sys
        sys.stdout.flush()
        
        elapsed = time.time() - self.start_time
        
        # Header
        print(f"{Fore.CYAN}{'='*90}")
        print(f"{Fore.YELLOW}🗺️  SPATIAL ECHO MODEL - LIVE GRID VISUALIZATION {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*90}{Style.RESET_ALL}\n")
        
        # Timing
        print(f"{Fore.WHITE}Generation: {Fore.YELLOW}{self.generation}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Elapsed: {Fore.GREEN}{elapsed:.1f}s{Style.RESET_ALL} | ", end="")
        speed = self.generation / elapsed if elapsed > 0 else 0
        print(f"{Fore.WHITE}Speed: {Fore.GREEN}{speed:.2f} gen/s{Style.RESET_ALL}\n")
        
        if not self.agents:
            print(f"{Fore.RED}💀 EXTINCTION{Style.RESET_ALL}")
            return
        
        # Population stats
        resources = [a.resources for a in self.agents]
        ages = [a.age for a in self.agents]
        
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}POPULATION STATS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Size: {Fore.GREEN}{len(self.agents)}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Avg Age: {Fore.CYAN}{np.mean(ages):.1f}{Style.RESET_ALL} (", end="")
        print(f"Min: {Fore.CYAN}{min(ages)}{Style.RESET_ALL}, Max: {Fore.CYAN}{max(ages)}{Style.RESET_ALL}) | ", end="")
        print(f"{Fore.WHITE}Clustering: {Fore.MAGENTA}{self.history['clustering'][-1]*100:.1f}%{Style.RESET_ALL}\n")
        
        # Resource distribution
        print(f"{Fore.WHITE}Resources: {Fore.GREEN}Avg={np.mean(resources):.1f}{Style.RESET_ALL}, ", end="")
        print(f"Min={Fore.RED}{min(resources):.0f}{Style.RESET_ALL}, ", end="")
        print(f"Max={Fore.GREEN}{max(resources):.0f}{Style.RESET_ALL}, ", end="")
        print(f"Total={Fore.CYAN}{sum(resources):.0f}{Style.RESET_ALL}")
        
        # Resource histogram
        very_rich = sum(1 for r in resources if r > 10000)
        rich = sum(1 for r in resources if 5000 < r <= 10000)
        medium = sum(1 for r in resources if 1000 < r <= 5000)
        poor = sum(1 for r in resources if 500 < r <= 1000)
        very_poor = sum(1 for r in resources if r <= 500)
        
        print(f"{Fore.WHITE}Wealth Distribution: ", end="")
        print(f"{Fore.GREEN}VeryRich={very_rich}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.GREEN}Rich={rich}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.YELLOW}Medium={medium}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.YELLOW}Poor={poor}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.RED}VeryPoor={very_poor}{Style.RESET_ALL}\n")
        
        # Cooperation
        total_actions = sum(a.cooperations + a.defections for a in self.agents)
        total_coop = sum(a.cooperations for a in self.agents)
        coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
        
        print(f"{Fore.WHITE}Cooperation: {Fore.GREEN}{coop_rate:.1f}%{Style.RESET_ALL} ", end="")
        print(f"(C={Fore.GREEN}{total_coop}{Style.RESET_ALL}, D={Fore.RED}{total_actions-total_coop}{Style.RESET_ALL}) ", end="")
        coop_bar = int(coop_rate / 5)
        print(f"[{Fore.GREEN}{'█' * coop_bar}{Fore.WHITE}{'░' * (20 - coop_bar)}{Style.RESET_ALL}]\n")
        
        # SPATIAL GRID VISUALIZATION
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}SPATIAL GRID ({GRID_WIDTH}×{GRID_HEIGHT}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        
        for y in range(GRID_HEIGHT):
            row = ""
            for x in range(GRID_WIDTH):
                agent = self.grid[y][x]
                if agent is None:
                    row += Fore.WHITE + "·"
                elif agent.resources > 10000:
                    row += Fore.GREEN + "█"  # Very rich
                elif agent.resources > 5000:
                    row += Fore.GREEN + "▓"  # Rich
                elif agent.resources > 1000:
                    row += Fore.YELLOW + "▓"  # Medium
                elif agent.resources > 500:
                    row += Fore.YELLOW + "░"  # Poor
                else:
                    row += Fore.RED + "░"  # Very poor
            print(row + Style.RESET_ALL)
        
        print(f"\n{Fore.WHITE}Legend: {Fore.GREEN}█{Style.RESET_ALL}=Very Rich(>10K) ", end="")
        print(f"{Fore.GREEN}▓{Style.RESET_ALL}=Rich(>5K) ", end="")
        print(f"{Fore.YELLOW}▓{Style.RESET_ALL}=Medium(>1K) ", end="")
        print(f"{Fore.YELLOW}░{Style.RESET_ALL}=Poor(>500) ", end="")
        print(f"{Fore.RED}░{Style.RESET_ALL}=Very Poor(<500) ", end="")
        print(f"{Fore.WHITE}·{Style.RESET_ALL}=Empty\n")
        
        # Top tags with detailed stats
        from collections import Counter
        tag_counts = Counter([a.tag for a in self.agents])
        top_tags = tag_counts.most_common(5)
        
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}TOP 5 DOMINANT TAGS (with avg resources & cooperation){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        
        for i, (tag, count) in enumerate(top_tags, 1):
            pct = count / len(self.agents) * 100
            bar_len = int(pct / 2)
            
            # Calculate avg resources and cooperation for this tag
            tag_agents = [a for a in self.agents if a.tag == tag]
            avg_res = np.mean([a.resources for a in tag_agents])
            tag_coop_actions = sum(a.cooperations + a.defections for a in tag_agents)
            tag_coop = sum(a.cooperations for a in tag_agents)
            tag_coop_rate = (tag_coop / tag_coop_actions * 100) if tag_coop_actions > 0 else 0
            
            print(f"{i}. {color_tag(tag)} | ", end="")
            print(f"Count: {Fore.CYAN}{count:>3}{Style.RESET_ALL} ({pct:>5.1f}%) | ", end="")
            print(f"Avg Res: {Fore.GREEN}{avg_res:>7.0f}{Style.RESET_ALL} | ", end="")
            print(f"Coop: {Fore.GREEN}{tag_coop_rate:>5.1f}%{Style.RESET_ALL}")
            print(f"   [{Fore.CYAN}{'█' * bar_len}{Style.RESET_ALL}]")
        
        # Recent activity and trends
        print(f"\n{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}GENERATION CHANGES{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        
        if len(self.history['births']) >= 1:
            last_births = self.history['births'][-1]
            last_deaths = self.history['deaths'][-1]
            
            print(f"{Fore.WHITE}THIS Generation: ", end="")
            print(f"{Fore.GREEN}Births={last_births}{Style.RESET_ALL}, ", end="")
            print(f"{Fore.RED}Deaths={last_deaths}{Style.RESET_ALL}, ", end="")
            print(f"{Fore.WHITE}Net={last_births - last_deaths:+d}{Style.RESET_ALL}")
        
        if len(self.history['births']) >= 5:
            recent_births = sum(self.history['births'][-5:])
            recent_deaths = sum(self.history['deaths'][-5:])
            
            print(f"{Fore.WHITE}LAST 5 Gens:     ", end="")
            print(f"{Fore.GREEN}Births={recent_births}{Style.RESET_ALL}, ", end="")
            print(f"{Fore.RED}Deaths={recent_deaths}{Style.RESET_ALL}, ", end="")
            print(f"{Fore.WHITE}Net={recent_births - recent_deaths:+d}{Style.RESET_ALL}")
        
        if len(self.history['population']) >= 10:
            pop_trend = self.history['population'][-1] - self.history['population'][-10]
            coop_trend = (self.history['cooperation'][-1] - self.history['cooperation'][-10]) * 100
            cluster_trend = (self.history['clustering'][-1] - self.history['clustering'][-10]) * 100
            
            print(f"{Fore.WHITE}10-Gen Trends:   ", end="")
            print(f"Pop={pop_trend:+d}, ", end="")
            print(f"Coop={coop_trend:+.1f}%, ", end="")
            print(f"Cluster={cluster_trend:+.1f}%{Style.RESET_ALL}")

# --- MAIN EXECUTION ---

def run_live_spatial_echo(generations: int = 200, update_every: int = 1):
    """Run spatial Echo with live grid visualization."""
    print(f"{Fore.CYAN}{'='*90}")
    print(f"{Fore.YELLOW}🧬 STARTING LIVE SPATIAL ECHO SIMULATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*90}{Style.RESET_ALL}\n")
    
    print(f"{Fore.WHITE}Configuration:")
    print(f"  Grid: {GRID_WIDTH}×{GRID_HEIGHT} = {GRID_WIDTH * GRID_HEIGHT} cells")
    print(f"  Initial Population: 55 agents")
    print(f"  Max Population: {MAX_POPULATION}")
    print(f"  Tag Matching: Hamming ≤ {MATCH_THRESHOLD}")
    print(f"  Update Frequency: Every {update_every} generation(s)")
    print(f"  Total Generations: {generations}{Style.RESET_ALL}\n")
    
    input(f"{Fore.YELLOW}Press ENTER to start...{Style.RESET_ALL}")
    
    pop = SpatialEchoPopulation()
    
    for gen in range(generations):
        pop.step()
        
        # Update display
        if (gen + 1) % update_every == 0 or gen == 0:
            pop.print_live_dashboard()
            
            # Slower viewing for detailed observation
            time.sleep(0.5)  # 500ms pause - much slower for careful observation
    
    # Final summary
    elapsed = time.time() - pop.start_time
    
    input(f"\n{Fore.YELLOW}Press ENTER to see final summary...{Style.RESET_ALL}")
    
    clear_screen()
    print(f"\n{Fore.GREEN}{'='*90}")
    print(f"🎉 SIMULATION COMPLETE!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*90}{Style.RESET_ALL}\n")
    
    print(f"Time: {elapsed:.1f}s | Speed: {generations/elapsed:.2f} gen/s")
    print(f"Final Population: {len(pop.agents)}")
    print(f"Final Cooperation: {pop.history['cooperation'][-1]*100:.1f}%")
    print(f"Final Clustering: {pop.history['clustering'][-1]*100:.1f}%")
    print(f"Total Births: {sum(pop.history['births'])}")
    print(f"Total Deaths: {sum(pop.history['deaths'])}")
    
    # Show final grid one more time
    print(f"\n{Fore.YELLOW}FINAL GRID STATE:{Style.RESET_ALL}\n")
    for y in range(GRID_HEIGHT):
        row = ""
        for x in range(GRID_WIDTH):
            agent = pop.grid[y][x]
            if agent is None:
                row += Fore.WHITE + "·"
            elif agent.resources > 10000:
                row += Fore.GREEN + "█"
            elif agent.resources > 5000:
                row += Fore.GREEN + "▓"
            elif agent.resources > 1000:
                row += Fore.YELLOW + "▓"
            elif agent.resources > 500:
                row += Fore.YELLOW + "░"
            else:
                row += Fore.RED + "░"
        print(row + Style.RESET_ALL)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spatial_echo_live_{timestamp}.json"
    
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
    
    print(f"\n{Fore.GREEN}Results saved to: {filename}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Run with live updates every generation
    run_live_spatial_echo(generations=200, update_every=1)
