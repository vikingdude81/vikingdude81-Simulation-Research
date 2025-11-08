"""
🚀 GPU-ACCELERATED ULTIMATE ECHO SIMULATION
===========================================

This runs the Ultimate Echo simulation with GPU acceleration for:
- Batch distance calculations (neighbor finding)
- Batch genetic operations (crossover, mutation)

Best for: Large populations (>500 agents) and long simulations
"""

from ultimate_echo_simulation import UltimateEchoSimulation, run_ultimate_echo
from government_styles import GovernmentStyle
from gpu_acceleration import get_gpu_accelerator, check_gpu_available
import numpy as np
import random
from colorama import Fore

# Check GPU before starting
gpu_info = check_gpu_available()
print(f"\n{Fore.CYAN}{'='*80}")
print(f"{Fore.YELLOW}🚀 GPU-ACCELERATED ULTIMATE ECHO SIMULATION 🚀")
print(f"{Fore.CYAN}{'='*80}")
print(f"{Fore.WHITE}GPU Available: {gpu_info['available']}")
if gpu_info['available']:
    print(f"Device: {gpu_info.get('device_name', 'Unknown')}")
    print(f"Backend: {gpu_info.get('backend', 'Unknown')}")
print(f"{Fore.CYAN}{'='*80}\n")


class GPUEchoSimulation(UltimateEchoSimulation):
    """
    GPU-accelerated version of Ultimate Echo Simulation.
    
    Uses GPU for:
    1. Batch neighbor finding (distance calculations)
    2. Batch genetic operations (crossover, mutation)
    """
    
    def __init__(self, *args, use_gpu: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = get_gpu_accelerator(use_gpu=use_gpu)
        self.use_gpu_neighbors = use_gpu and gpu_info['available']
        
        if self.use_gpu_neighbors:
            print(f"{Fore.GREEN}✅ GPU acceleration enabled for simulation")
        else:
            print(f"{Fore.YELLOW}⚠️ Running on CPU (GPU not available or disabled)")
    
    def _interaction_phase_gpu(self):
        """
        GPU-accelerated interaction phase.
        
        Uses batch distance calculations instead of per-agent calculations.
        """
        if not self.use_gpu_neighbors or len(self.agents) < 100:
            # Fall back to CPU for small populations
            return super()._interaction_phase()
        
        # Build position array
        positions = np.array([a.pos for a in self.agents])
        
        # Shuffle agents for random interaction order
        random.shuffle(self.agents)
        
        for agent in self.agents:
            if agent.dead:
                continue
            
            # GPU-accelerated distance calculation
            distances = self.gpu.manhattan_distances_batch(
                positions, 
                agent.pos, 
                self.grid_size
            )
            
            # Find neighbors within vision range
            neighbor_indices = np.where(distances <= agent.traits.vision)[0]
            
            # Filter out self and dead agents
            neighbors = []
            for idx in neighbor_indices:
                other = self.agents[idx]
                if other is not agent and not other.dead:
                    neighbors.append(other)
            
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
    
    def step(self):
        """Override step to use GPU-accelerated interaction phase."""
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
        
        # 2. INTERACTION PHASE: Use GPU-accelerated version
        if self.use_gpu_neighbors:
            self._interaction_phase_gpu()
        else:
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


def run_gpu_echo(
    generations: int = 500,
    initial_size: int = 100,
    government_style: GovernmentStyle = GovernmentStyle.LAISSEZ_FAIRE,
    update_frequency: int = 10,
    use_gpu: bool = True
):
    """
    Run GPU-accelerated Ultimate Echo simulation.
    
    Args:
        generations: Number of generations to simulate
        initial_size: Starting population size
        government_style: Government policy style to use
        update_frequency: Print dashboard every N generations
        use_gpu: Enable GPU acceleration (default True)
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}🚀 GPU-ACCELERATED ULTIMATE ECHO - Starting... 🚀")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.WHITE}Government Style: {government_style.value}")
    print(f"Generations: {generations}")
    print(f"Initial Population: {initial_size}")
    print(f"GPU Enabled: {use_gpu}")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    # Create GPU-accelerated simulation
    sim = GPUEchoSimulation(
        initial_size=initial_size,
        government_style=government_style,
        use_gpu=use_gpu
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
            
            # Show GPU stats
            if sim.use_gpu_neighbors:
                gpu_stats = sim.gpu.get_stats()
                if 'memory_allocated' in gpu_stats:
                    print(f"{Fore.CYAN}GPU Memory: {gpu_stats['memory_allocated']:.1f} MB allocated, "
                          f"{gpu_stats['memory_reserved']:.1f} MB reserved")
    
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
    
    # GPU stats
    if sim.use_gpu_neighbors:
        gpu_stats = sim.gpu.get_stats()
        print(f"\n{Fore.CYAN}🚀 GPU Stats:")
        for key, value in gpu_stats.items():
            print(f"{Fore.WHITE}  {key}: {value}")
    
    # Save results
    sim.save_results()
    
    return sim


if __name__ == "__main__":
    # Run GPU-accelerated simulation
    run_gpu_echo(
        generations=200,
        initial_size=100,
        government_style=GovernmentStyle.LAISSEZ_FAIRE,
        update_frequency=20,
        use_gpu=True
    )
