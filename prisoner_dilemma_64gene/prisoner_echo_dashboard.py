"""
🧬 ECHO Model with Live Dashboard

Real-time visualization of:
- Population dynamics
- Tag distribution (color-coded)
- Strategy evolution (C/D visualization)
- Resource distribution
- Cooperation rate
- Generation progress with timing
"""

import random
import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime
import time
import os
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows color support
init(autoreset=True)

# --- CONFIGURATION ---
TAG_LENGTH = 8
STRATEGY_LENGTH = 64
CHROMOSOME_LENGTH = TAG_LENGTH + STRATEGY_LENGTH
MATCH_THRESHOLD = 2

# Resource parameters
INITIAL_RESOURCES = 100
REPRODUCTION_THRESHOLD = 200
DEATH_THRESHOLD = 0
METABOLISM_COST = 1
MAX_POPULATION = 500  # Cap to prevent slowdown

# --- COLOR HELPERS ---

def color_tag(tag: str) -> str:
    """Color-code binary tag (0=blue, 1=red)."""
    colored = ""
    for bit in tag:
        if bit == '0':
            colored += Fore.CYAN + bit
        else:
            colored += Fore.RED + bit
    return colored + Style.RESET_ALL

def color_strategy(strategy: str, max_len: int = 20) -> str:
    """Color-code strategy (C=green, D=red)."""
    colored = ""
    for i, gene in enumerate(strategy[:max_len]):
        if gene == 'C':
            colored += Fore.GREEN + gene
        else:
            colored += Fore.RED + gene
    if len(strategy) > max_len:
        colored += Fore.WHITE + "..."
    return colored + Style.RESET_ALL

def resource_bar(resources: float, max_res: float = 500) -> str:
    """Create a visual resource bar."""
    bar_length = 20
    filled = int((resources / max_res) * bar_length)
    filled = min(filled, bar_length)
    
    # Color based on resource level
    if resources < 50:
        color = Fore.RED
    elif resources < 150:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    
    bar = color + "█" * filled + Fore.WHITE + "░" * (bar_length - filled)
    return f"{bar} {resources:.0f}R"

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- GENETIC ALGORITHM FUNCTIONS ---

def create_random_chromosome() -> str:
    """Creates a 72-gene chromosome (8-bit tag + 64-bit strategy)."""
    tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "".join(random.choice(['C', 'D']) for _ in range(STRATEGY_LENGTH))
    return tag + strategy

def create_tit_for_tat_with_tag(tag: str | None = None) -> str:
    """Creates Tit-for-Tat strategy with specified or random tag."""
    if tag is None:
        tag = "".join(random.choice(['0', '1']) for _ in range(TAG_LENGTH))
    strategy = "CD" * 32
    return tag + strategy

def crossover(parent1: str, parent2: str) -> str:
    """Performs single-point crossover."""
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

def mutate(chromosome: str, rate: float = 0.01) -> str:
    """Mutates chromosome."""
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < rate:
            if i < TAG_LENGTH:
                chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
            else:
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
    return "".join(chrom_list)

def hamming_distance(tag1: str, tag2: str) -> int:
    """Calculates Hamming distance between two tags."""
    return sum(c1 != c2 for c1, c2 in zip(tag1, tag2))

# --- AGENT CLASS ---

class EchoAgent:
    """Agent with tag, strategy, and resources."""
    
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
    
    def __init__(self, agent_id: int, chromosome: str, resources: float = INITIAL_RESOURCES):
        self.id = agent_id
        self.chromosome = chromosome
        self.tag = chromosome[:TAG_LENGTH]
        self.strategy = chromosome[TAG_LENGTH:]
        self.resources = resources
        self.age = 0
        self.history = [('C', 'C'), ('C', 'C'), ('C', 'C')]
        self.interactions = 0
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
    
    def can_match(self, other: 'EchoAgent') -> bool:
        return hamming_distance(self.tag, other.tag) <= MATCH_THRESHOLD
    
    def interact(self, other: 'EchoAgent', rounds: int = 5) -> Tuple[float, float]:
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
        
        self.interactions += rounds
        other.interactions += rounds
        
        return my_total, other_total
    
    def metabolize(self):
        self.resources -= METABOLISM_COST
        self.age += 1
    
    def is_alive(self) -> bool:
        return self.resources > DEATH_THRESHOLD
    
    def can_reproduce(self) -> bool:
        return self.resources >= REPRODUCTION_THRESHOLD
    
    def reproduce(self, mutation_rate: float = 0.01) -> 'EchoAgent':
        reproduction_cost = REPRODUCTION_THRESHOLD / 2
        self.resources -= reproduction_cost
        child_resources = reproduction_cost
        child_chromosome = mutate(self.chromosome, mutation_rate)
        child = EchoAgent(-1, child_chromosome, child_resources)
        self.children_produced += 1
        return child

# --- POPULATION WITH DASHBOARD ---

class EchoPopulationDashboard:
    """Population manager with live dashboard."""
    
    def __init__(self, initial_size: int = 50, mutation_rate: float = 0.01):
        self.mutation_rate = mutation_rate
        self.rounds_per_interaction = 5
        
        # Initialize population
        self.agents: List[EchoAgent] = []
        for i in range(initial_size):
            chromosome = create_random_chromosome()
            agent = EchoAgent(i, chromosome)
            self.agents.append(agent)
        
        # Add TFT agents with same tag
        tft_tag = "11110000"
        for i in range(5):
            tft_agent = EchoAgent(len(self.agents), 
                                 create_tit_for_tat_with_tag(tft_tag))
            self.agents.append(tft_agent)
        
        self.generation = 0
        self.next_id = len(self.agents)
        self.start_time = time.time()
        
        # History
        self.history = {
            'population': [],
            'resources': [],
            'cooperation': [],
            'births': [],
            'deaths': []
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
                agent_gain, partner_gain = agent.interact(partner, self.rounds_per_interaction)
                agent.resources += agent_gain
                partner.resources += partner_gain
            else:
                agent.resources += EchoAgent.RESOURCE_PAYOFFS[('D', 'D')][0]
    
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
            # Stop reproduction if we hit population cap
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
        
        # Record history
        if self.agents:
            total_res = sum(a.resources for a in self.agents)
            total_actions = sum(a.cooperations + a.defections for a in self.agents)
            total_coop = sum(a.cooperations for a in self.agents)
            coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
            
            self.history['population'].append(len(self.agents))
            self.history['resources'].append(total_res)
            self.history['cooperation'].append(coop_rate)
            self.history['births'].append(births)
            self.history['deaths'].append(deaths)
        
        self.generation += 1
        return deaths, births
    
    def print_dashboard(self):
        """Print live dashboard."""
        clear_screen()
        import sys
        sys.stdout.flush()  # Force output
        
        elapsed = time.time() - self.start_time
        
        # Header
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}🧬 ECHO MODEL LIVE DASHBOARD {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        
        # Timing
        print(f"{Fore.WHITE}Generation: {Fore.YELLOW}{self.generation}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Elapsed: {Fore.GREEN}{elapsed:.1f}s{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Speed: {Fore.GREEN}{self.generation/elapsed if elapsed > 0 else 0:.2f} gen/s{Style.RESET_ALL}\n")
        
        if not self.agents:
            print(f"{Fore.RED}💀 EXTINCTION - Population died out!{Style.RESET_ALL}")
            return
        
        # Population Stats
        resources = [a.resources for a in self.agents]
        ages = [a.age for a in self.agents]
        
        print(f"{Fore.CYAN}{'─'*80}")
        print(f"{Fore.YELLOW}POPULATION STATS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Size: {Fore.GREEN}{len(self.agents)}{Style.RESET_ALL} agents | ", end="")
        print(f"{Fore.WHITE}Avg Age: {Fore.CYAN}{np.mean(ages):.1f}{Style.RESET_ALL} | ", end="")
        print(f"{Fore.WHITE}Max Age: {Fore.CYAN}{np.max(ages)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Resources: Avg={Fore.GREEN}{np.mean(resources):.1f}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.WHITE}Total={Fore.GREEN}{np.sum(resources):.0f}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.WHITE}Min={Fore.RED}{np.min(resources):.1f}{Style.RESET_ALL}, ", end="")
        print(f"{Fore.WHITE}Max={Fore.GREEN}{np.max(resources):.1f}{Style.RESET_ALL}")
        
        # Cooperation Rate
        total_actions = sum(a.cooperations + a.defections for a in self.agents)
        total_coop = sum(a.cooperations for a in self.agents)
        coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
        
        coop_bar = int(coop_rate / 5)  # 20 segments
        coop_color = Fore.GREEN if coop_rate > 50 else Fore.YELLOW if coop_rate > 30 else Fore.RED
        print(f"\n{Fore.WHITE}Cooperation Rate: {coop_color}{coop_rate:.1f}%{Style.RESET_ALL}")
        print(f"{coop_color}{'█' * coop_bar}{Fore.WHITE}{'░' * (20 - coop_bar)}{Style.RESET_ALL}")
        
        # Top Tags
        print(f"\n{Fore.CYAN}{'─'*80}")
        print(f"{Fore.YELLOW}TOP 5 TAGS (Identity){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
        
        tag_counts = {}
        for agent in self.agents:
            tag_counts[agent.tag] = tag_counts.get(agent.tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (tag, count) in enumerate(sorted_tags, 1):
            pct = (count / len(self.agents)) * 100
            bar = int(pct / 2)  # 50 segments
            print(f"{i}. {color_tag(tag)} ", end="")
            print(f"{Fore.GREEN}{'█' * bar}{Fore.WHITE}{'░' * (50 - bar)} ", end="")
            print(f"{Fore.YELLOW}{count:3d}{Style.RESET_ALL} ({pct:5.1f}%)")
        
        # Top Strategies
        print(f"\n{Fore.CYAN}{'─'*80}")
        print(f"{Fore.YELLOW}TOP 5 STRATEGIES (Behavior){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
        
        strategy_counts = {}
        for agent in self.agents:
            strategy_counts[agent.strategy] = strategy_counts.get(agent.strategy, 0) + 1
        
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        tft = "CD" * 32
        
        for i, (strategy, count) in enumerate(sorted_strategies, 1):
            pct = (count / len(self.agents)) * 100
            bar = int(pct / 2)
            is_tft = " [TFT]" if strategy == tft else ""
            print(f"{i}. {color_strategy(strategy, 16)} ", end="")
            print(f"{Fore.GREEN}{'█' * bar}{Fore.WHITE}{'░' * (50 - bar)} ", end="")
            print(f"{Fore.YELLOW}{count:3d}{Style.RESET_ALL} ({pct:5.1f}%){Fore.CYAN}{is_tft}{Style.RESET_ALL}")
        
        # Richest Agents
        print(f"\n{Fore.CYAN}{'─'*80}")
        print(f"{Fore.YELLOW}TOP 5 RICHEST AGENTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
        
        richest = sorted(self.agents, key=lambda a: a.resources, reverse=True)[:5]
        for i, agent in enumerate(richest, 1):
            print(f"{i}. {Fore.WHITE}Agent{agent.id:4d}{Style.RESET_ALL} | ", end="")
            print(f"{resource_bar(agent.resources, 500)} | ", end="")
            print(f"{Fore.WHITE}Age:{Fore.CYAN}{agent.age:3d}{Style.RESET_ALL} | ", end="")
            print(f"{Fore.WHITE}Kids:{Fore.GREEN}{agent.children_produced:2d}{Style.RESET_ALL}")
            print(f"   Tag: {color_tag(agent.tag)} | Strategy: {color_strategy(agent.strategy, 12)}")
        
        # Recent History (last 10 generations)
        if len(self.history['population']) >= 2:
            print(f"\n{Fore.CYAN}{'─'*80}")
            print(f"{Fore.YELLOW}RECENT TRENDS (Last 10 Gen){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
            
            recent = min(10, len(self.history['population']))
            pop_change = self.history['population'][-1] - self.history['population'][-recent]
            births_recent = sum(self.history['births'][-recent:])
            deaths_recent = sum(self.history['deaths'][-recent:])
            
            pop_color = Fore.GREEN if pop_change > 0 else Fore.RED if pop_change < 0 else Fore.YELLOW
            print(f"{Fore.WHITE}Population Change: {pop_color}{pop_change:+d}{Style.RESET_ALL} | ", end="")
            print(f"{Fore.GREEN}Births: {births_recent}{Style.RESET_ALL} | ", end="")
            print(f"{Fore.RED}Deaths: {deaths_recent}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Press Ctrl+C to stop...{Style.RESET_ALL}")

# --- MAIN SIMULATION ---

def run_echo_with_dashboard(generations: int = 200, 
                            initial_population: int = 50,
                            update_interval: int = 1):
    """Run Echo simulation with live dashboard."""
    
    print(f"{Fore.YELLOW}🧬 Starting ECHO Model with Live Dashboard...{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Initializing population...{Style.RESET_ALL}")
    time.sleep(1)
    
    population = EchoPopulationDashboard(initial_size=initial_population)
    
    try:
        for gen in range(generations):
            deaths, births = population.step()
            
            # Update dashboard at interval
            if gen % update_interval == 0 or gen == 0:
                population.print_dashboard()
                time.sleep(0.1)  # Brief pause for readability
            
            # Check for extinction
            if len(population.agents) == 0:
                population.print_dashboard()
                print(f"\n{Fore.RED}💀 Simulation ended - Population extinct at generation {gen}{Style.RESET_ALL}")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⏸️  Simulation paused by user{Style.RESET_ALL}")
    
    # Final summary
    print(f"\n\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.YELLOW}🏁 SIMULATION COMPLETE{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
    
    elapsed = time.time() - population.start_time
    print(f"{Fore.WHITE}Total Generations: {Fore.YELLOW}{population.generation}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Total Time: {Fore.GREEN}{elapsed:.1f}s{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Avg Speed: {Fore.GREEN}{population.generation/elapsed:.2f} gen/s{Style.RESET_ALL}")
    
    if population.agents:
        print(f"{Fore.WHITE}Final Population: {Fore.GREEN}{len(population.agents)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Survival Rate: {Fore.GREEN}{len(population.agents)/initial_population*100:.1f}%{Style.RESET_ALL}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"echo_dashboard_{timestamp}.json"
    
    results = {
        'metadata': {
            'generations': population.generation,
            'initial_population': initial_population,
            'final_population': len(population.agents),
            'elapsed_time': elapsed
        },
        'history': population.history
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Fore.CYAN}💾 Results saved to: {filename}{Style.RESET_ALL}")
    
    return population

if __name__ == "__main__":
    population = run_echo_with_dashboard(
        generations=200,
        initial_population=50,
        update_interval=1  # Update every generation
    )
