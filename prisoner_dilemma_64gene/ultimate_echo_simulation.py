"""
🌍 ULTIMATE ECHO SIMULATION - Government Styles + Genetic Traits
=================================================================

This simulation combines:
1. Top-Down Control: Government policy styles (government_styles.py)
2. Bottom-Up Evolution: Extended genetic traits (genetic_traits.py)

This creates a truly rich, complex adaptive system that can test
sophisticated economic and social theories.

Research Questions:
- How do different government styles affect genetic evolution?
- Which traits evolve under laissez-faire vs. welfare state?
- Do authoritarian governments suppress genetic diversity?
- What's the interaction between top-down policy and bottom-up adaptation?
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from colorama import Fore, Style, init

# Import our new modules
from government_styles import GovernmentController, GovernmentStyle, PolicyAction
from genetic_traits import ExtendedEchoAgent, ExtendedChromosome, analyze_population_genetics

# Initialize colorama for colored output
init(autoreset=True)


@dataclass
class ExternalShock:
    """External events that affect the simulation."""
    name: str
    probability: float
    
    def apply(self, agents: List[ExtendedEchoAgent]) -> int:
        """Apply shock to population. Returns number affected."""
        if random.random() < self.probability:
            if self.name == "DROUGHT":
                # Reduce wealth for random 30% of agents
                affected = random.sample(agents, int(len(agents) * 0.3))
                for agent in affected:
                    agent.wealth *= 0.5
                return len(affected)
            elif self.name == "DISASTER":
                # Kill random 10% of agents
                affected = random.sample(agents, int(len(agents) * 0.1))
                for agent in affected:
                    agent.dead = True
                return len(affected)
            elif self.name == "BOOM":
                # Increase wealth for everyone
                for agent in agents:
                    agent.wealth *= 1.2
                return len(agents)
        return 0


class UltimateEchoSimulation:
    """
    Complete Echo simulation with government policy and genetic traits.
    """
    
    def __init__(
        self,
        initial_size: int = 100,
        grid_size: Tuple[int, int] = (50, 50),
        government_style: GovernmentStyle = GovernmentStyle.LAISSEZ_FAIRE,
        mutation_rate: float = 0.01
    ):
        self.grid_size = grid_size
        self.government = GovernmentController(government_style)
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.max_population_ever = initial_size
        
        # Initialize population
        self.agents = self._create_initial_population(initial_size)
        
        # External shocks
        self.shocks = [
            ExternalShock("DROUGHT", 0.05),
            ExternalShock("DISASTER", 0.02),
            ExternalShock("BOOM", 0.03)
        ]
        
        # History tracking
        self.history = {
            'population': [],
            'cooperation': [],
            'avg_wealth': [],
            'avg_metabolism': [],
            'avg_vision': [],
            'avg_age': [],
            'policy_actions': [],
            'shocks': []
        }
    
    def _create_initial_population(self, size: int) -> List[ExtendedEchoAgent]:
        """Create initial random population."""
        agents = []
        for _ in range(size):
            chromosome = ExtendedChromosome.create_random()
            pos = (random.randint(0, self.grid_size[0]-1), 
                   random.randint(0, self.grid_size[1]-1))
            agent = ExtendedEchoAgent(chromosome, initial_wealth=10, pos=pos)
            agents.append(agent)
        return agents
    
    def step(self):
        """Execute one generation of the simulation."""
        self.generation += 1
        
        # Remove dead agents
        self.agents = [a for a in self.agents if not a.dead]
        
        if len(self.agents) == 0:
            return False  # Extinction
        
        # 1. METABOLISM PHASE: Agents pay cost of living
        for agent in self.agents:
            agent.pay_metabolism()
        
        # Remove agents who went bankrupt
        self.agents = [a for a in self.agents if not a.dead]
        
        # 2. INTERACTION PHASE: Agents play prisoner's dilemma
        self._interaction_phase()
        
        # 3. EXTERNAL SHOCKS: Random events
        for shock in self.shocks:
            affected = shock.apply(self.agents)
            if affected > 0:
                self.history['shocks'].append({
                    'generation': self.generation,
                    'type': shock.name,
                    'affected': affected
                })
        
        # Remove dead agents (from shocks)
        self.agents = [a for a in self.agents if not a.dead]
        
        # 4. GOVERNMENT POLICY: Top-down intervention
        policy_action = self.government.apply_policy(self.agents, self.grid_size)
        if policy_action:
            self.history['policy_actions'].append(policy_action)
            
            # Handle authoritarian removals (marked with wealth=-9999)
            self.agents = [a for a in self.agents if a.wealth > -9999]
        
        # 5. AGING PHASE: Agents age and die of old age
        for agent in self.agents:
            agent.age_one_generation()
        
        # Remove agents who died of old age
        self.agents = [a for a in self.agents if not a.dead]
        
        # 6. REPRODUCTION PHASE: Agents reproduce if they have enough wealth
        self._reproduction_phase()
        
        # Update max population
        self.max_population_ever = max(self.max_population_ever, len(self.agents))
        
        # Record history
        self._record_history()
        
        return True  # Simulation continues
    
    def _interaction_phase(self):
        """Agents interact with neighbors based on vision range."""
        # Shuffle agents for random interaction order
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if agent.dead:
                continue
            
            # Find neighbors within vision range
            neighbors = agent.find_neighbors(self.agents, self.grid_size)
            
            if not neighbors:
                continue
            
            # Choose best match based on tag similarity
            best_partner = self._find_best_partner(agent, neighbors)
            
            if best_partner is None:
                continue
            
            # Play prisoner's dilemma
            payoff_self, payoff_other = self._play_prisoners_dilemma(
                agent.get_strategy(),
                best_partner.get_strategy()
            )
            
            agent.wealth += payoff_self
            best_partner.wealth += payoff_other
            agent.interactions += 1
            best_partner.interactions += 1
    
    def _find_best_partner(self, agent: ExtendedEchoAgent, candidates: List[ExtendedEchoAgent]) -> Optional[ExtendedEchoAgent]:
        """Find partner with most similar tag."""
        if not candidates:
            return None
        
        # Calculate tag similarity (Hamming distance)
        similarities = []
        for candidate in candidates:
            # XOR tags and count matching bits
            xor = agent.get_tag() ^ candidate.get_tag()
            matching_bits = 8 - bin(xor).count('1')  # 8-bit tag
            similarities.append(matching_bits)
        
        # Return candidate with highest similarity
        best_idx = np.argmax(similarities)
        return candidates[best_idx]
    
    def _play_prisoners_dilemma(self, strategy1: int, strategy2: int) -> Tuple[float, float]:
        """
        Play prisoner's dilemma.
        
        Payoff matrix:
                    Cooperate   Defect
        Cooperate   (3, 3)      (0, 5)
        Defect      (5, 0)      (1, 1)
        """
        if strategy1 == 1 and strategy2 == 1:
            return 3, 3  # Both cooperate
        elif strategy1 == 1 and strategy2 == 0:
            return 0, 5  # Cooperator exploited
        elif strategy1 == 0 and strategy2 == 1:
            return 5, 0  # Defector exploits
        else:
            return 1, 1  # Both defect
    
    def _reproduction_phase(self):
        """Agents reproduce if they have enough wealth."""
        # Population cap to prevent explosion (grid capacity)
        max_population = self.grid_size[0] * self.grid_size[1] // 2  # 50% of grid cells
        
        if len(self.agents) >= max_population:
            return  # No reproduction if at capacity
        
        new_agents = []
        
        # Shuffle agents for random reproduction order
        random.shuffle(self.agents)
        
        for agent in self.agents:
            # Stop if we hit population cap
            if len(self.agents) + len(new_agents) >= max_population:
                break
            
            if not agent.can_reproduce():
                continue
            
            # Find partner within vision range
            neighbors = agent.find_neighbors(self.agents, self.grid_size)
            partners = [n for n in neighbors if n.can_reproduce() and n is not agent]
            
            if not partners:
                continue
            
            # Reproduce with random partner
            partner = random.choice(partners)
            child = agent.reproduce(partner, self.mutation_rate)
            new_agents.append(child)
        
        # Add new agents to population
        self.agents.extend(new_agents)
    
    def _record_history(self):
        """Record current generation statistics."""
        genetics = analyze_population_genetics(self.agents)
        
        self.history['population'].append(len(self.agents))
        self.history['cooperation'].append(genetics.get('cooperation_rate', 0))
        self.history['avg_wealth'].append(np.mean([a.wealth for a in self.agents]) if self.agents else 0)
        self.history['avg_metabolism'].append(genetics.get('avg_metabolism', 0))
        self.history['avg_vision'].append(genetics.get('avg_vision', 0))
        self.history['avg_age'].append(genetics.get('avg_age', 0))
    
    def print_dashboard(self):
        """Print live dashboard to console."""
        genetics = analyze_population_genetics(self.agents)
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}🌍 ULTIMATE ECHO SIMULATION (Gov: {self.government.style.value.upper()}) 🌍")
        print(f"{Fore.CYAN}{'='*80}")
        
        print(f"\n{Fore.GREEN}Generation: {self.generation}")
        print(f"{Fore.WHITE}Population: {len(self.agents)} (Max: {self.max_population_ever})")
        
        # Genetic traits
        print(f"\n{Fore.YELLOW}🧬 GENETIC TRAITS:")
        print(f"{Fore.WHITE}Cooperation: {genetics.get('cooperation_rate', 0)*100:.1f}%")
        print(f"Avg Metabolism: {genetics.get('avg_metabolism', 0):.2f} (cost per round)")
        print(f"Avg Vision: {genetics.get('avg_vision', 0):.2f} (interaction range)")
        print(f"Avg Age: {genetics.get('avg_age', 0):.1f} / {genetics.get('avg_lifespan_gene', 0):.1f} (current/max)")
        print(f"Oldest Agent: {genetics.get('oldest_agent', 0)} generations")
        
        # Economics
        if self.agents:
            wealths = [a.wealth for a in self.agents]
            print(f"\n{Fore.YELLOW}💰 ECONOMICS:")
            print(f"{Fore.WHITE}Total Wealth: {sum(wealths):,.0f}")
            print(f"Avg Wealth: {np.mean(wealths):.1f}")
            print(f"Wealth Range: {min(wealths):.1f} - {max(wealths):.1f}")
        
        # Government
        gov_summary = self.government.get_summary()
        print(f"\n{Fore.YELLOW}🏛️ GOVERNMENT ({self.government.style.value.upper()}):")
        print(f"{Fore.WHITE}Total Policy Actions: {gov_summary['total_actions']}")
        if gov_summary.get('action_breakdown'):
            for action_type, count in gov_summary['action_breakdown'].items():
                print(f"  {action_type}: {count}")
        
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def save_results(self, filename: Optional[str] = None):
        """Save simulation results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/ultimate_echo/ultimate_echo_{self.government.style.value}_{timestamp}.json"
        
        results = {
            'government_style': self.government.style.value,
            'generations': self.generation,
            'final_population': len(self.agents),
            'max_population': self.max_population_ever,
            'mutation_rate': self.mutation_rate,
            'grid_size': self.grid_size,
            'government_summary': self.government.get_summary(),
            'final_genetics': analyze_population_genetics(self.agents),
            'history': self.history
        }
        
        # Create output directory
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{Fore.GREEN}✅ Results saved to: {filename}")


def run_ultimate_echo(
    generations: int = 500,
    initial_size: int = 100,
    government_style: GovernmentStyle = GovernmentStyle.LAISSEZ_FAIRE,
    update_frequency: int = 10
):
    """
    Run the Ultimate Echo simulation.
    
    Args:
        generations: Number of generations to simulate
        initial_size: Starting population size
        government_style: Government policy style to use
        update_frequency: Print dashboard every N generations
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}🌍 ULTIMATE ECHO SIMULATION - Starting... 🌍")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.WHITE}Government Style: {government_style.value}")
    print(f"Generations: {generations}")
    print(f"Initial Population: {initial_size}")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    # Create simulation
    sim = UltimateEchoSimulation(
        initial_size=initial_size,
        government_style=government_style
    )
    
    # Run simulation
    import time
    start_time = time.time()
    
    for gen in range(generations):
        survived = sim.step()
        
        if not survived:
            print(f"\n{Fore.RED}💀 EXTINCTION at generation {sim.generation}!")
            break
        
        if (gen + 1) % update_frequency == 0:
            sim.print_dashboard()
    
    elapsed = time.time() - start_time
    
    # Final dashboard
    sim.print_dashboard()
    
    # Final summary
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.YELLOW}🏁 SIMULATION COMPLETE 🏁")
    print(f"{Fore.GREEN}{'='*80}")
    print(f"{Fore.WHITE}Total Time: {elapsed:.1f}s")
    print(f"Speed: {generations/elapsed:.2f} gen/s")
    print(f"Final Population: {len(sim.agents)}")
    print(f"Survival Rate: {len(sim.agents)/sim.max_population_ever*100:.1f}%")
    
    # Save results
    sim.save_results()
    
    return sim


if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}🌍 ULTIMATE ECHO SIMULATION 🌍")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"\n{Fore.WHITE}Available Government Styles:")
    for i, style in enumerate(GovernmentStyle, 1):
        print(f"  {i}. {style.value}")
    
    print(f"\n{Fore.WHITE}Quick Start:")
    print(f"  1. Laissez-Faire (no intervention):")
    print(f"     {Fore.GREEN}run_ultimate_echo(government_style=GovernmentStyle.LAISSEZ_FAIRE)")
    print(f"  2. Welfare State:")
    print(f"     {Fore.GREEN}run_ultimate_echo(government_style=GovernmentStyle.WELFARE_STATE)")
    print(f"  3. Authoritarian:")
    print(f"     {Fore.GREEN}run_ultimate_echo(government_style=GovernmentStyle.AUTHORITARIAN)")
    
    print(f"\n{Fore.YELLOW}Running default simulation (Laissez-Faire)...")
    run_ultimate_echo(generations=200, government_style=GovernmentStyle.LAISSEZ_FAIRE)
