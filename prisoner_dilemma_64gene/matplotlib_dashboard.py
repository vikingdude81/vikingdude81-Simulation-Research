"""
🎨 MATPLOTLIB VISUAL DASHBOARD - Smooth Real-Time Visualization
================================================================

Better than terminal-based rich library:
- Smooth animations
- No flickering or choppiness
- Persistent window
- Beautiful charts and graphs
- Interactive controls

Shows:
- World map heatmap (agent density + cooperation)
- Population over time
- Cooperation rate over time
- Wealth distribution
- Genetic trait evolution
- Event timeline
- Live statistics
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import time

# Enable interactive mode
plt.ion()


class MatplotlibDashboard:
    """
    Real-time matplotlib dashboard for Echo simulation.
    
    Features:
    - 2x3 subplot layout
    - Live updating without flickering
    - Smooth animations
    - Professional appearance
    """
    
    def __init__(self, grid_size: Tuple[int, int], total_generations: int):
        self.grid_size = grid_size
        self.total_generations = total_generations
        
        # Data storage
        self.history_length = 200
        self.generations = deque(maxlen=self.history_length)
        self.population_history = deque(maxlen=self.history_length)
        self.cooperation_history = deque(maxlen=self.history_length)
        self.wealth_history = deque(maxlen=self.history_length)
        self.metabolism_history = deque(maxlen=self.history_length)
        self.vision_history = deque(maxlen=self.history_length)
        
        # Events
        self.events = []
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🌍 ULTIMATE ECHO SIMULATION - Live Dashboard', 
                         fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Top row: World map (large) + Stats
        self.ax_world = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        
        # Middle row: Genetics
        self.ax_genetics = self.fig.add_subplot(gs[1, 2])
        
        # Bottom row: Time series
        self.ax_population = self.fig.add_subplot(gs[2, 0])
        self.ax_cooperation = self.fig.add_subplot(gs[2, 1])
        self.ax_wealth = self.fig.add_subplot(gs[2, 2])
        
        # Initialize plots
        self._init_world_map()
        self._init_stats()
        self._init_genetics()
        self._init_time_series()
        
        # Show window
        plt.show(block=False)
        plt.pause(0.1)
        
        self.start_time = time.time()
    
    def _init_world_map(self):
        """Initialize world map heatmap."""
        self.ax_world.set_title('🗺️ World Map (Agent Density)', fontweight='bold', fontsize=12)
        self.ax_world.set_xlabel('X Position')
        self.ax_world.set_ylabel('Y Position')
        
        # Create empty heatmap
        self.world_data = np.zeros((40, 40))
        self.world_im = self.ax_world.imshow(
            self.world_data,
            cmap='RdYlGn',
            interpolation='nearest',
            vmin=-1,
            vmax=1,
            aspect='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(self.world_im, ax=self.ax_world, fraction=0.046, pad=0.04)
        cbar.set_label('Cooperation (Green) vs Defection (Red)', rotation=270, labelpad=20)
    
    def _init_stats(self):
        """Initialize statistics panel."""
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(
            0.1, 0.5, '', 
            fontsize=10, 
            family='monospace',
            verticalalignment='center'
        )
    
    def _init_genetics(self):
        """Initialize genetics panel."""
        self.ax_genetics.axis('off')
        self.genetics_text = self.ax_genetics.text(
            0.1, 0.5, '', 
            fontsize=10, 
            family='monospace',
            verticalalignment='center'
        )
    
    def _init_time_series(self):
        """Initialize time series plots."""
        # Population
        self.ax_population.set_title('👥 Population Over Time', fontsize=10, fontweight='bold')
        self.ax_population.set_xlabel('Generation')
        self.ax_population.set_ylabel('Population')
        self.ax_population.grid(True, alpha=0.3)
        self.line_population, = self.ax_population.plot([], [], 'b-', linewidth=2)
        
        # Cooperation
        self.ax_cooperation.set_title('🤝 Cooperation Rate', fontsize=10, fontweight='bold')
        self.ax_cooperation.set_xlabel('Generation')
        self.ax_cooperation.set_ylabel('Cooperation %')
        self.ax_cooperation.set_ylim(0, 100)
        self.ax_cooperation.grid(True, alpha=0.3)
        self.line_cooperation, = self.ax_cooperation.plot([], [], 'g-', linewidth=2)
        self.ax_cooperation.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
        
        # Wealth
        self.ax_wealth.set_title('💰 Average Wealth', fontsize=10, fontweight='bold')
        self.ax_wealth.set_xlabel('Generation')
        self.ax_wealth.set_ylabel('Wealth')
        self.ax_wealth.grid(True, alpha=0.3)
        self.line_wealth, = self.ax_wealth.plot([], [], 'gold', linewidth=2)
    
    def update(self, generation: int, agents: List, sim_data: Dict, government_style: str = "laissez_faire"):
        """Update all dashboard elements."""
        # Add to history
        self.generations.append(generation)
        self.population_history.append(sim_data.get('population', 0))
        self.cooperation_history.append(sim_data.get('cooperation', 0) * 100)
        self.wealth_history.append(sim_data.get('avg_wealth', 0))
        self.metabolism_history.append(sim_data.get('avg_metabolism', 0))
        self.vision_history.append(sim_data.get('avg_vision', 0))
        
        # Update world map
        self._update_world_map(agents)
        
        # Update stats
        self._update_stats(generation, sim_data, government_style)
        
        # Update genetics
        self._update_genetics(sim_data)
        
        # Update time series
        self._update_time_series()
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _update_world_map(self, agents: List):
        """Update world map heatmap."""
        map_size = (40, 40)
        
        # Create density and cooperation grids
        density_grid = np.zeros(map_size)
        coop_grid = np.zeros(map_size)
        
        # Scale factors
        scale_x = self.grid_size[0] / map_size[1]
        scale_y = self.grid_size[1] / map_size[0]
        
        for agent in agents:
            if hasattr(agent, 'dead') and agent.dead:
                continue
            
            # Map to display grid
            x = min(int(agent.pos[0] / scale_x), map_size[1] - 1)
            y = min(int(agent.pos[1] / scale_y), map_size[0] - 1)
            
            density_grid[y, x] += 1
            if agent.get_strategy() == 1:  # Cooperator
                coop_grid[y, x] += 1
        
        # Calculate cooperation rate per cell
        world_data = np.zeros(map_size)
        for y in range(map_size[0]):
            for x in range(map_size[1]):
                if density_grid[y, x] > 0:
                    coop_rate = coop_grid[y, x] / density_grid[y, x]
                    # Map to -1 (all defectors) to +1 (all cooperators)
                    world_data[y, x] = coop_rate * 2 - 1
                else:
                    world_data[y, x] = 0  # Empty
        
        # Update image data
        self.world_im.set_data(world_data)
    
    def _update_stats(self, generation: int, sim_data: Dict, government_style: str):
        """Update statistics text."""
        elapsed = time.time() - self.start_time
        speed = generation / elapsed if elapsed > 0 else 0
        progress = (generation / self.total_generations) * 100
        
        stats = f"""📊 STATISTICS
━━━━━━━━━━━━━━━━━━
Government: {government_style.upper()}
Generation: {generation}/{self.total_generations}
Progress: {progress:.1f}%

👥 Population: {sim_data.get('population', 0):,}
   Max: {sim_data.get('max_population', 0):,}

🤝 Cooperation: {sim_data.get('cooperation', 0)*100:.1f}%

💰 Wealth:
   Avg: {sim_data.get('avg_wealth', 0):.1f}
   Total: {sim_data.get('total_wealth', 0):,.0f}

⏳ Age:
   Avg: {sim_data.get('avg_age', 0):.1f} gen
   Oldest: {sim_data.get('oldest_agent', 0)} gen

⚡ Performance:
   Speed: {speed:.2f} gen/s
   Time: {elapsed:.1f}s
"""
        
        self.stats_text.set_text(stats)
    
    def _update_genetics(self, sim_data: Dict):
        """Update genetics text."""
        genetics = f"""🧬 GENETICS
━━━━━━━━━━━━━━━━━━
🔥 Metabolism: {sim_data.get('avg_metabolism', 0):.2f}
   (cost per round)

👁️ Vision: {sim_data.get('avg_vision', 0):.2f}
   (interaction range)

💗 Lifespan: {sim_data.get('avg_lifespan', 0):.1f}
   (max generations)

🏘️ Clustering: {sim_data.get('clustering', 0)*100:.1f}%
"""
        
        self.genetics_text.set_text(genetics)
    
    def _update_time_series(self):
        """Update time series plots."""
        if len(self.generations) < 2:
            return
        
        gens = list(self.generations)
        
        # Population
        self.line_population.set_data(gens, list(self.population_history))
        self.ax_population.relim()
        self.ax_population.autoscale_view()
        
        # Cooperation
        self.line_cooperation.set_data(gens, list(self.cooperation_history))
        self.ax_cooperation.set_xlim(min(gens), max(gens))
        
        # Wealth
        self.line_wealth.set_data(gens, list(self.wealth_history))
        self.ax_wealth.relim()
        self.ax_wealth.autoscale_view()
    
    def add_event(self, event_text: str, event_type: str = "info"):
        """Add event to log (shown in console)."""
        self.events.append((time.time(), event_text, event_type))
        print(f"[Gen {len(self.generations)}] {event_text}")
    
    def close(self):
        """Close the dashboard."""
        plt.close(self.fig)


def run_with_matplotlib_dashboard(
    initial_size: int = 100,
    generations: int = 500,
    government_style = None,
    grid_size: Tuple[int, int] = (50, 50),
    update_every: int = 5
):
    """
    Run simulation with matplotlib dashboard.
    
    Args:
        initial_size: Starting population
        generations: Number of generations
        government_style: GovernmentStyle enum
        grid_size: World grid size
        update_every: Update display every N generations
    """
    from ultimate_echo_simulation import UltimateEchoSimulation
    from government_styles import GovernmentStyle
    from genetic_traits import analyze_population_genetics
    
    if government_style is None:
        government_style = GovernmentStyle.LAISSEZ_FAIRE
    
    print("\n🎨 Starting simulation with matplotlib dashboard...")
    print(f"   Population: {initial_size}")
    print(f"   Generations: {generations}")
    print(f"   Government: {government_style.value}")
    print("\n✨ Dashboard window opening...\n")
    
    # Create simulation
    sim = UltimateEchoSimulation(
        initial_size=initial_size,
        grid_size=grid_size,
        government_style=government_style
    )
    
    # Create dashboard
    dashboard = MatplotlibDashboard(grid_size, generations)
    
    try:
        for gen in range(generations):
            # Step simulation
            survived = sim.step()
            
            if not survived:
                dashboard.add_event("💀 EXTINCTION!", "error")
                print("\n💀 Population went extinct!")
                break
            
            # Update dashboard
            if gen % update_every == 0 or gen == generations - 1:
                genetics = analyze_population_genetics(sim.agents)
                
                sim_data = {
                    'population': len(sim.agents),
                    'max_population': sim.max_population_ever,
                    'cooperation': genetics.get('cooperation_rate', 0),
                    'avg_wealth': np.mean([a.wealth for a in sim.agents]) if sim.agents else 0,
                    'total_wealth': sum(a.wealth for a in sim.agents),
                    'avg_age': genetics.get('avg_age', 0),
                    'oldest_agent': genetics.get('oldest_agent', 0),
                    'avg_metabolism': genetics.get('avg_metabolism', 0),
                    'avg_vision': genetics.get('avg_vision', 0),
                    'avg_lifespan': genetics.get('avg_lifespan_gene', 0),
                    'clustering': 0.5  # Placeholder
                }
                
                dashboard.update(gen + 1, sim.agents, sim_data, government_style.value)
            
            # Log events
            if sim.history['policy_actions'] and sim.history['policy_actions'][-1].generation == gen + 1:
                action = sim.history['policy_actions'][-1]
                dashboard.add_event(f"🏛️ {action.action_type}: {action.affected_agents} agents", "policy")
            
            if sim.history['shocks'] and sim.history['shocks'][-1]['generation'] == gen + 1:
                shock = sim.history['shocks'][-1]
                dashboard.add_event(f"⚠️ {shock['type']}: {shock['affected']} affected", "shock")
        
        # Keep window open for viewing
        print("\n✅ Simulation complete! Close the window to exit.")
        plt.show(block=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
    
    finally:
        dashboard.close()
    
    return sim


if __name__ == "__main__":
    from government_styles import GovernmentStyle
    
    print("=" * 60)
    print("🎨 MATPLOTLIB DASHBOARD - Smooth Visualization")
    print("=" * 60)
    
    run_with_matplotlib_dashboard(
        initial_size=100,
        generations=300,
        government_style=GovernmentStyle.LAISSEZ_FAIRE,
        update_every=3
    )
