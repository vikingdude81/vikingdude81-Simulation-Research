"""
🧬💰🌪️ ULTIMATE ECHO MODEL - Full Agent-Based Economy with External Shocks

Combines ALL features:
- 50×50 grid (2,500 cells) for large-scale dynamics
- 72-gene chromosome (8-bit tag + 64-bit strategy)
- Tag-based conditional interaction (Hamming ≤ 2)
- Wealth-driven fitness and evolution
- External shocks: Droughts, Disasters, Predators
- Live visualization with detailed statistics
- Resource accumulation economy
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import time
import os
from colorama import init, Fore, Style
from collections import Counter

init(autoreset=True)

# --- CONFIGURATION ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2

# Spatial parameters - BIGGER GRID
GRID_WIDTH = 50
GRID_HEIGHT = 50
NEIGHBORHOOD_RADIUS = 1

# Resource parameters
INITIAL_RESOURCES = 100
REPRODUCTION_THRESHOLD = 200
DEATH_THRESHOLD = 0
METABOLISM_COST = 1
MAX_POPULATION = 1000  # More agents on bigger grid

# External shock parameters
DROUGHT_CHANCE = 0.05        # 5% chance per generation
DROUGHT_PENALTY = 50         # Lose 50 resources
DISASTER_CHANCE = 0.02       # 2% chance per generation
DISASTER_RADIUS = 5          # Affects 11×11 area
DISASTER_KILL_CHANCE = 0.5   # 50% chance to die
PREDATOR_CHANCE = 0.03       # 3% chance per generation
PREDATOR_COUNT = 3           # Number of predators
PREDATOR_KILL_CHANCE = 0.3   # 30% chance to kill nearby agents

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
    colored = ""
    for bit in tag:
        if bit == '0':
            colored += Fore.CYAN + bit
        else:
            colored += Fore.RED + bit
    return colored + Style.RESET_ALL

# --- SPATIAL AGENT CLASS ---

class UltimateEchoAgent:
    """Agent with position, tag, strategy, resources, and survival history."""
    
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
    
    LONER_SCORE = 5
    
    def __init__(self, agent_id: int, chromosome: str, position: Tuple[int, int]):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy_genes = chromosome[TAG_LENGTH:]
        self.position = position
        self.resources = INITIAL_RESOURCES
        self.age = 0
        self.cooperations = 0
        self.defections = 0
        self.survived_droughts = 0
        self.survived_disasters = 0
        self.survived_predators = 0
        
    def can_match(self, other: 'UltimateEchoAgent') -> bool:
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def get_move(self, history: List[str]) -> str:
        if not history:
            return self.strategy_genes[0]
        
        opponent_history = tuple(history[-min(len(history), 6):])
        if len(opponent_history) < 6:
            opponent_history = ('C',) * (6 - len(opponent_history)) + opponent_history
        
        index = int(sum(
            self.MOVE_MAP.get((opponent_history[i], opponent_history[i+1]), 0) * (4 ** (2-i))
            for i in range(0, 6, 2)
        ))
        
        return self.strategy_genes[min(index, STRATEGY_LENGTH - 1)]
    
    def interact(self, other: 'UltimateEchoAgent', rounds: int = 5) -> Tuple[int, int]:
        my_history = []
        other_history = []
        
        my_total = 0
        other_total = 0
        
        for _ in range(rounds):
            my_move = self.get_move(other_history)
            other_move = other.get_move(my_history)
            
            my_gain, other_gain = self.RESOURCE_PAYOFFS[(my_move, other_move)]
            my_total += my_gain
            other_total += other_gain
            
            my_history.append(my_move)
            other_history.append(other_move)
            
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

# --- EXTERNAL SHOCK EVENTS ---

class ExternalShock:
    """Represents various external shocks to the population."""
    
    @staticmethod
    def apply_drought(agents: List[UltimateEchoAgent]) -> int:
        """Random drought affects all agents."""
        affected = 0
        for agent in agents:
            agent.resources -= DROUGHT_PENALTY
            agent.survived_droughts += 1
            affected += 1
        return affected
    
    @staticmethod
    def apply_disaster(agents: List[UltimateEchoAgent], grid: np.ndarray, 
                       center: Tuple[int, int]) -> Tuple[int, int]:
        """Localized disaster kills agents in radius."""
        cx, cy = center
        killed = 0
        affected = 0
        
        for agent in agents[:]:  # Copy list since we modify it
            ax, ay = agent.position
            distance = max(abs(ax - cx), abs(ay - cy))
            
            if distance <= DISASTER_RADIUS:
                affected += 1
                if random.random() < DISASTER_KILL_CHANCE:
                    agent.resources = -1000  # Instant death
                    killed += 1
                else:
                    agent.survived_disasters += 1
        
        return killed, affected
    
    @staticmethod
    def apply_predators(agents: List[UltimateEchoAgent], grid: np.ndarray, 
                       predator_positions: List[Tuple[int, int]]) -> int:
        """Predators hunt nearby agents."""
        killed = 0
        
        for pred_x, pred_y in predator_positions:
            # Check 3×3 area around predator
            for agent in agents[:]:
                ax, ay = agent.position
                distance = max(abs(ax - pred_x), abs(ay - pred_y))
                
                if distance <= 1:  # Adjacent to predator
                    if random.random() < PREDATOR_KILL_CHANCE:
                        agent.resources = -1000  # Eaten
                        killed += 1
                    else:
                        agent.survived_predators += 1
        
        return killed

# --- ULTIMATE POPULATION CLASS ---

class UltimateEchoPopulation:
    def __init__(self, initial_size: int = 100):
        self.agents: List[UltimateEchoAgent] = []
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), None, dtype=object)
        self.rounds_per_interaction = 5
        
        # Initialize population
        positions_used = set()
        
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            
            while True:
                x = random.randint(0, GRID_WIDTH - 1)
                y = random.randint(0, GRID_HEIGHT - 1)
                if (x, y) not in positions_used:
                    break
            
            agent = UltimateEchoAgent(i, chromosome, (x, y))
            self.agents.append(agent)
            self.grid[y][x] = agent
            positions_used.add((x, y))
        
        # Add TFT cluster
        tft_tag = "11110000"
        for i in range(10):
            while True:
                x = random.randint(0, 10)
                y = random.randint(0, 10)
                if (x, y) not in positions_used:
                    break
            
            tft_agent = UltimateEchoAgent(len(self.agents), 
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
            'clustering': [],
            'shocks': []
        }
        
        self.shock_log = []
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[UltimateEchoAgent]:
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
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if not agent.is_alive():
                continue
            
            neighbors = self.get_neighbors(agent.position)
            
            if not neighbors:
                agent.resources += UltimateEchoAgent.LONER_SCORE
                continue
            
            matches = [n for n in neighbors if agent.can_match(n)]
            
            if matches:
                partner = random.choice(matches)
                agent_gain, partner_gain = agent.interact(partner, self.rounds_per_interaction)
                agent.resources += agent_gain
                partner.resources += partner_gain
            else:
                agent.resources += UltimateEchoAgent.LONER_SCORE
    
    def apply_external_shocks(self) -> str:
        """Apply random external shocks and return description."""
        shocks = []
        
        # Drought
        if random.random() < DROUGHT_CHANCE:
            affected = ExternalShock.apply_drought(self.agents)
            shock_msg = f"🌵 DROUGHT! All {affected} agents lose {DROUGHT_PENALTY} resources"
            shocks.append(shock_msg)
            self.shock_log.append(('drought', self.generation, affected))
        
        # Disaster
        if random.random() < DISASTER_CHANCE:
            center = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            killed, affected = ExternalShock.apply_disaster(self.agents, self.grid, center)
            shock_msg = f"💥 DISASTER at ({center[0]},{center[1]})! {killed} killed, {affected} affected"
            shocks.append(shock_msg)
            self.shock_log.append(('disaster', self.generation, killed, affected, center))
        
        # Predators
        if random.random() < PREDATOR_CHANCE:
            predator_positions = [
                (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
                for _ in range(PREDATOR_COUNT)
            ]
            killed = ExternalShock.apply_predators(self.agents, self.grid, predator_positions)
            shock_msg = f"🦖 PREDATORS! {PREDATOR_COUNT} predators kill {killed} agents"
            shocks.append(shock_msg)
            self.shock_log.append(('predators', self.generation, killed, predator_positions))
        
        return " | ".join(shocks) if shocks else None
    
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
        x, y = position
        empty_spots = []
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                
                nx = (x + dx) % GRID_WIDTH
                ny = (y + dy) % GRID_HEIGHT
                
                if self.grid[ny][nx] is None:
                    empty_spots.append((nx, ny))
        
        return random.choice(empty_spots) if empty_spots else None
    
    def handle_reproduction(self) -> int:
        if len(self.agents) >= MAX_POPULATION:
            return 0
        
        parents = [a for a in self.agents if a.can_reproduce()]
        if not parents:
            return 0
        
        new_agents = []
        
        for _ in range(min(len(parents), MAX_POPULATION - len(self.agents))):
            parent = random.choice(parents)
            
            position = self.find_empty_neighbor(parent.position)
            if position is None:
                continue
            
            parent.resources -= REPRODUCTION_THRESHOLD // 2
            
            child_chromosome = mutate(parent.chromosome)
            child = UltimateEchoAgent(self.next_id, child_chromosome, position)
            self.next_id += 1
            
            x, y = position
            self.grid[y][x] = child
            new_agents.append(child)
        
        self.agents.extend(new_agents)
        return len(new_agents)
    
    def calculate_clustering(self) -> float:
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
        """Run one generation with external shocks."""
        self.run_interactions()
        
        # Apply external shocks BEFORE metabolism
        shock_msg = self.apply_external_shocks()
        
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
            self.history['shocks'].append(shock_msg)
        
        self.generation += 1
        return deaths, births, shock_msg
    
    def print_live_dashboard(self, shock_msg: Optional[str] = None):
        """Print live dashboard with external shock alerts."""
        clear_screen()
        import sys
        sys.stdout.flush()
        
        elapsed = time.time() - self.start_time
        
        # Header with SHOCK ALERT if active
        print(f"{Fore.CYAN}{'='*90}")
        if shock_msg:
            print(f"{Fore.RED}⚠️  {shock_msg} ⚠️{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*90}")
        print(f"{Fore.YELLOW}🗺️💰🌪️  ULTIMATE ECHO MODEL - Full Economy + External Shocks {Style.RESET_ALL}")
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
        
        # Wealth distribution
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
        
        # SPATIAL GRID VISUALIZATION (showing 40×40 sample of 50×50)
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}SPATIAL GRID (40×40 sample of {GRID_WIDTH}×{GRID_HEIGHT}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        
        # Show center 40×40 region
        start_y = (GRID_HEIGHT - 40) // 2
        start_x = (GRID_WIDTH - 40) // 2
        
        for y in range(start_y, start_y + 40):
            row = ""
            for x in range(start_x, start_x + 40):
                agent = self.grid[y][x]
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
        
        print(f"\n{Fore.WHITE}Legend: {Fore.GREEN}█{Style.RESET_ALL}=VeryRich(>10K) ", end="")
        print(f"{Fore.GREEN}▓{Style.RESET_ALL}=Rich(>5K) ", end="")
        print(f"{Fore.YELLOW}▓{Style.RESET_ALL}=Medium(>1K) ", end="")
        print(f"{Fore.YELLOW}░{Style.RESET_ALL}=Poor(>500) ", end="")
        print(f"{Fore.RED}░{Style.RESET_ALL}=VeryPoor(<500) ", end="")
        print(f"{Fore.WHITE}·{Style.RESET_ALL}=Empty\n")
        
        # Survival stats
        total_drought_survivors = sum(a.survived_droughts for a in self.agents)
        total_disaster_survivors = sum(a.survived_disasters for a in self.agents)
        total_predator_survivors = sum(a.survived_predators for a in self.agents)
        
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}SURVIVAL STATS (Total Survivals){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}🌵 Droughts: {Fore.CYAN}{total_drought_survivors}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}💥 Disasters: {Fore.CYAN}{total_disaster_survivors}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}🦖 Predators: {Fore.CYAN}{total_predator_survivors}{Style.RESET_ALL}\n")
        
        # Top tags with detailed stats
        tag_counts = Counter([a.tag for a in self.agents])
        top_tags = tag_counts.most_common(5)
        
        print(f"{Fore.CYAN}{'─'*90}")
        print(f"{Fore.YELLOW}TOP 5 DOMINANT TAGS (with avg resources & cooperation){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*90}{Style.RESET_ALL}")
        
        for i, (tag, count) in enumerate(top_tags, 1):
            pct = count / len(self.agents) * 100
            bar_len = int(pct / 2)
            
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

def run_ultimate_echo(generations: int = 300, update_every: int = 1):
    """Run ultimate Echo with all features."""
    print(f"{Fore.CYAN}{'='*90}")
    print(f"{Fore.YELLOW}🧬💰🌪️  STARTING ULTIMATE ECHO SIMULATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*90}{Style.RESET_ALL}\n")
    
    print(f"{Fore.WHITE}Configuration:")
    print(f"  Grid: {GRID_WIDTH}×{GRID_HEIGHT} = {GRID_WIDTH * GRID_HEIGHT} cells")
    print(f"  Initial Population: 110 agents")
    print(f"  Max Population: {MAX_POPULATION}")
    print(f"  Tag Matching: Hamming ≤ {MATCH_THRESHOLD}")
    print(f"  External Shocks: Droughts({DROUGHT_CHANCE*100:.0f}%), Disasters({DISASTER_CHANCE*100:.0f}%), Predators({PREDATOR_CHANCE*100:.0f}%)")
    print(f"  Update Frequency: Every {update_every} generation(s)")
    print(f"  Total Generations: {generations}{Style.RESET_ALL}\n")
    
    input(f"{Fore.YELLOW}Press ENTER to start the ultimate simulation...{Style.RESET_ALL}")
    
    pop = UltimateEchoPopulation()
    
    for gen in range(generations):
        deaths, births, shock_msg = pop.step()
        
        # Update display
        if (gen + 1) % update_every == 0 or gen == 0 or shock_msg:
            pop.print_live_dashboard(shock_msg)
            
            # Slower for viewing
            time.sleep(0.5)
    
    # Final summary
    elapsed = time.time() - pop.start_time
    
    input(f"\n{Fore.YELLOW}Press ENTER to see final summary...{Style.RESET_ALL}")
    
    clear_screen()
    print(f"\n{Fore.GREEN}{'='*90}")
    print(f"🎉 ULTIMATE SIMULATION COMPLETE!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*90}{Style.RESET_ALL}\n")
    
    print(f"Time: {elapsed:.1f}s | Speed: {generations/elapsed:.2f} gen/s")
    print(f"Final Population: {len(pop.agents)}")
    if pop.agents:
        print(f"Final Cooperation: {pop.history['cooperation'][-1]*100:.1f}%")
        print(f"Final Clustering: {pop.history['clustering'][-1]*100:.1f}%")
    print(f"Total Births: {sum(pop.history['births'])}")
    print(f"Total Deaths: {sum(pop.history['deaths'])}")
    
    # Shock summary
    print(f"\n{Fore.YELLOW}EXTERNAL SHOCKS SUMMARY:{Style.RESET_ALL}")
    drought_count = sum(1 for s in pop.shock_log if s[0] == 'drought')
    disaster_count = sum(1 for s in pop.shock_log if s[0] == 'disaster')
    predator_count = sum(1 for s in pop.shock_log if s[0] == 'predators')
    
    print(f"  🌵 Droughts: {drought_count}")
    print(f"  💥 Disasters: {disaster_count}")
    print(f"  🦖 Predator Attacks: {predator_count}")
    print(f"  Total Shocks: {len(pop.shock_log)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultimate_echo_{timestamp}.json"
    
    results = {
        'metadata': {
            'timestamp': timestamp,
            'generations': generations,
            'grid_size': (GRID_WIDTH, GRID_HEIGHT),
            'max_population': MAX_POPULATION,
            'external_shocks': True,
            'drought_chance': DROUGHT_CHANCE,
            'disaster_chance': DISASTER_CHANCE,
            'predator_chance': PREDATOR_CHANCE
        },
        'history': pop.history,
        'shock_log': pop.shock_log,
        'final_state': {
            'population': len(pop.agents),
            'cooperation': pop.history['cooperation'][-1] if pop.history['cooperation'] else 0,
            'clustering': pop.history['clustering'][-1] if pop.history['clustering'] else 0
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Fore.GREEN}Results saved to: {filename}{Style.RESET_ALL}")

if __name__ == "__main__":
    run_ultimate_echo(generations=300, update_every=1)
