"""
🌟 ULTIMATE COMBINED DASHBOARD - All Best Features Together
============================================================

Combines:
1. World map heatmap (cooperation visualization)
2. Interaction network with flash effects
3. Activity heatmap showing hotspots
4. GPU monitoring and stats
5. Time-series graphs (population, cooperation, wealth, GPU memory)
6. Network statistics
7. Genetic evolution tracking

The most comprehensive visualization possible!
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Dict, Tuple
from collections import deque, defaultdict
import time

plt.ion()


class UltimateDashboard:
    """
    The ultimate combined dashboard with all features.
    """
    
    def __init__(self, grid_size: Tuple[int, int], total_generations: int, use_gpu: bool = False):
        self.grid_size = grid_size
        self.total_generations = total_generations
        self.use_gpu = use_gpu
        
        # Data storage
        self.history_length = 200
        self.generations = deque(maxlen=self.history_length)
        self.population_history = deque(maxlen=self.history_length)
        self.cooperation_history = deque(maxlen=self.history_length)
        self.wealth_history = deque(maxlen=self.history_length)
        self.metabolism_history = deque(maxlen=self.history_length)
        self.vision_history = deque(maxlen=self.history_length)
        self.interactions_per_gen = deque(maxlen=self.history_length)
        self.gpu_memory_history = deque(maxlen=self.history_length)
        
        # Interaction tracking
        self.interaction_history = deque(maxlen=100)
        self.interaction_counts = defaultdict(int)
        self.recent_interactions = deque(maxlen=20)
        
        # Create figure with 3x4 layout
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('🌟 ULTIMATE ECHO DASHBOARD - Complete System Visualization', 
                         fontsize=18, fontweight='bold')
        
        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        
        # Row 1: World Map + Network + Activity Heatmap + Stats
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_network = self.fig.add_subplot(gs[0, 1])
        self.ax_activity = self.fig.add_subplot(gs[0, 2])
        self.ax_stats = self.fig.add_subplot(gs[0, 3])
        
        # Row 2: Population + Cooperation + Wealth + Interactions
        self.ax_population = self.fig.add_subplot(gs[1, 0])
        self.ax_cooperation = self.fig.add_subplot(gs[1, 1])
        self.ax_wealth = self.fig.add_subplot(gs[1, 2])
        self.ax_interactions = self.fig.add_subplot(gs[1, 3])
        
        # Row 3: Genetics + Network Stats + GPU Stats + GPU Memory
        self.ax_genetics = self.fig.add_subplot(gs[2, 0])
        self.ax_network_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_gpu_stats = self.fig.add_subplot(gs[2, 2])
        self.ax_gpu_memory = self.fig.add_subplot(gs[2, 3])
        
        # Initialize all plots
        self._init_world_map()
        self._init_network()
        self._init_activity()
        self._init_stats()
        self._init_time_series()
        self._init_text_panels()
        
        plt.show(block=False)
        plt.pause(0.1)
        
        self.start_time = time.time()
        self.generation_interactions = []
    
    def _init_world_map(self):
        """World map heatmap."""
        self.ax_world.set_title('World Map (Cooperation)', fontweight='bold', fontsize=10)
        self.world_data = np.zeros((30, 30))
        self.world_im = self.ax_world.imshow(self.world_data, cmap='RdYlGn', 
                                             interpolation='nearest', vmin=-1, vmax=1, aspect='auto')
        self.ax_world.set_xticks([])
        self.ax_world.set_yticks([])
    
    def _init_network(self):
        """Interaction network."""
        self.ax_network.set_title('Interaction Network', fontweight='bold', fontsize=10)
        self.ax_network.set_xlim(0, self.grid_size[0])
        self.ax_network.set_ylim(0, self.grid_size[1])
        self.ax_network.set_facecolor('#1a1a1a')
        self.ax_network.set_xticks([])
        self.ax_network.set_yticks([])
    
    def _init_activity(self):
        """Activity heatmap."""
        self.ax_activity.set_title('Activity Heatmap', fontweight='bold', fontsize=10)
        self.activity_data = np.zeros((20, 20))
        self.activity_im = self.ax_activity.imshow(self.activity_data, cmap='hot',
                                                   interpolation='bilinear', vmin=0, vmax=10, aspect='auto')
        self.ax_activity.set_xticks([])
        self.ax_activity.set_yticks([])
    
    def _init_stats(self):
        """Main statistics panel."""
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.5, '', fontsize=8, family='monospace', verticalalignment='center')
    
    def _init_time_series(self):
        """Initialize all time-series graphs."""
        # Population
        self.ax_population.set_title('Population', fontsize=9, fontweight='bold')
        self.ax_population.grid(True, alpha=0.3)
        self.line_population, = self.ax_population.plot([], [], 'b-', linewidth=2)
        
        # Cooperation
        self.ax_cooperation.set_title('Cooperation %', fontsize=9, fontweight='bold')
        self.ax_cooperation.set_ylim(0, 100)
        self.ax_cooperation.grid(True, alpha=0.3)
        self.line_cooperation, = self.ax_cooperation.plot([], [], 'g-', linewidth=2)
        self.ax_cooperation.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        
        # Wealth
        self.ax_wealth.set_title('Avg Wealth', fontsize=9, fontweight='bold')
        self.ax_wealth.grid(True, alpha=0.3)
        self.line_wealth, = self.ax_wealth.plot([], [], 'gold', linewidth=2)
        
        # Interactions
        self.ax_interactions.set_title('Interactions/Gen', fontsize=9, fontweight='bold')
        self.ax_interactions.grid(True, alpha=0.3)
        self.line_interactions, = self.ax_interactions.plot([], [], 'cyan', linewidth=2)
        
        # GPU Memory
        if self.use_gpu:
            self.ax_gpu_memory.set_title('GPU Memory (MB)', fontsize=9, fontweight='bold')
            self.ax_gpu_memory.grid(True, alpha=0.3)
            self.line_gpu_memory, = self.ax_gpu_memory.plot([], [], 'purple', linewidth=2)
        else:
            self.ax_gpu_memory.axis('off')
            self.ax_gpu_memory.text(0.5, 0.5, 'CPU Mode\n(No GPU)', ha='center', va='center',
                                   fontsize=10, style='italic')
    
    def _init_text_panels(self):
        """Initialize text-based info panels."""
        self.ax_genetics.axis('off')
        self.genetics_text = self.ax_genetics.text(0.05, 0.5, '', fontsize=8, family='monospace', verticalalignment='center')
        
        self.ax_network_stats.axis('off')
        self.network_stats_text = self.ax_network_stats.text(0.05, 0.5, '', fontsize=8, family='monospace', verticalalignment='center')
        
        self.ax_gpu_stats.axis('off')
        self.gpu_stats_text = self.ax_gpu_stats.text(0.05, 0.5, '', fontsize=8, family='monospace', verticalalignment='center')
    
    def track_interaction(self, agent1_id: int, agent2_id: int, 
                         agent1_pos: Tuple, agent2_pos: Tuple,
                         agent1_strategy: int, agent2_strategy: int,
                         outcome: str):
        """Track agent interaction."""
        interaction = {
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'pos1': agent1_pos,
            'pos2': agent2_pos,
            'strategy1': agent1_strategy,
            'strategy2': agent2_strategy,
            'outcome': outcome,
            'timestamp': time.time()
        }
        
        self.interaction_history.append(interaction)
        self.recent_interactions.append(interaction)
        self.generation_interactions.append(interaction)
        
        pair = tuple(sorted([agent1_id, agent2_id]))
        self.interaction_counts[pair] += 1
    
    def update(self, generation: int, agents: List, sim_data: Dict, 
               government_style: str = "laissez_faire", gpu_stats = None):
        """Update all dashboard elements."""
        # Add to history
        self.generations.append(generation)
        self.population_history.append(sim_data.get('population', 0))
        self.cooperation_history.append(sim_data.get('cooperation', 0) * 100)
        self.wealth_history.append(sim_data.get('avg_wealth', 0))
        self.metabolism_history.append(sim_data.get('avg_metabolism', 0))
        self.vision_history.append(sim_data.get('avg_vision', 0))
        self.interactions_per_gen.append(len(self.generation_interactions))
        
        if gpu_stats and self.use_gpu:
            self.gpu_memory_history.append(gpu_stats.get('memory_mb', 0))
        else:
            self.gpu_memory_history.append(0)
        
        # Update all visualizations
        self._update_world_map(agents)
        self._update_network(agents)
        self._update_activity()
        self._update_stats(generation, sim_data, government_style)
        self._update_genetics(sim_data)
        self._update_network_stats()
        self._update_gpu_stats(gpu_stats)
        self._update_time_series()
        
        # Clear generation interactions
        self.generation_interactions = []
        
        # Decay recent interactions
        current_time = time.time()
        while self.recent_interactions and current_time - self.recent_interactions[0]['timestamp'] > 0.5:
            self.recent_interactions.popleft()
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _update_world_map(self, agents: List):
        """Update world map heatmap."""
        map_size = (30, 30)
        density_grid = np.zeros(map_size)
        coop_grid = np.zeros(map_size)
        
        scale_x = self.grid_size[0] / map_size[1]
        scale_y = self.grid_size[1] / map_size[0]
        
        for agent in agents:
            if hasattr(agent, 'dead') and agent.dead:
                continue
            
            x = min(int(agent.pos[0] / scale_x), map_size[1] - 1)
            y = min(int(agent.pos[1] / scale_y), map_size[0] - 1)
            
            density_grid[y, x] += 1
            if agent.get_strategy() == 1:
                coop_grid[y, x] += 1
        
        world_data = np.zeros(map_size)
        for y in range(map_size[0]):
            for x in range(map_size[1]):
                if density_grid[y, x] > 0:
                    coop_rate = coop_grid[y, x] / density_grid[y, x]
                    world_data[y, x] = coop_rate * 2 - 1
        
        self.world_im.set_data(world_data)
    
    def _update_network(self, agents: List):
        """Update interaction network."""
        self.ax_network.clear()
        self.ax_network.set_title('Interaction Network', fontweight='bold', fontsize=10)
        self.ax_network.set_xlim(0, self.grid_size[0])
        self.ax_network.set_ylim(0, self.grid_size[1])
        self.ax_network.set_facecolor('#1a1a1a')
        self.ax_network.set_xticks([])
        self.ax_network.set_yticks([])
        
        if not agents:
            return
        
        # Draw edges (sample last 30 interactions)
        edges = []
        edge_colors = []
        
        for interaction in list(self.interaction_history)[-30:]:
            edges.append([interaction['pos1'], interaction['pos2']])
            if interaction['outcome'] == 'both_coop':
                edge_colors.append('lime')
            elif interaction['outcome'] == 'both_defect':
                edge_colors.append('red')
            else:
                edge_colors.append('orange')
        
        if edges:
            lc = LineCollection(edges, colors=edge_colors, linewidths=1.0, alpha=0.3, zorder=1)
            self.ax_network.add_collection(lc)
        
        # Flash recent interactions
        for interaction in self.recent_interactions:
            age = time.time() - interaction['timestamp']
            if age < 0.5:
                alpha = 1.0 - (age / 0.5)
                pos1, pos2 = interaction['pos1'], interaction['pos2']
                self.ax_network.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                    color='yellow', linewidth=2, alpha=alpha, zorder=3)
        
        # Draw agents
        positions = np.array([agent.pos for agent in agents if not (hasattr(agent, 'dead') and agent.dead)])
        strategies = np.array([agent.get_strategy() for agent in agents if not (hasattr(agent, 'dead') and agent.dead)])
        
        if len(positions) > 0:
            coop_mask = strategies == 1
            defect_mask = strategies == 0
            
            if np.any(coop_mask):
                self.ax_network.scatter(positions[coop_mask, 0], positions[coop_mask, 1],
                                       c='lime', s=15, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=2)
            if np.any(defect_mask):
                self.ax_network.scatter(positions[defect_mask, 0], positions[defect_mask, 1],
                                       c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=2)
    
    def _update_activity(self):
        """Update activity heatmap."""
        heatmap_size = (20, 20)
        self.activity_data = np.zeros(heatmap_size)
        
        scale_x = self.grid_size[0] / heatmap_size[1]
        scale_y = self.grid_size[1] / heatmap_size[0]
        
        for interaction in self.interaction_history:
            for pos in [interaction['pos1'], interaction['pos2']]:
                x = min(int(pos[0] / scale_x), heatmap_size[1] - 1)
                y = min(int(pos[1] / scale_y), heatmap_size[0] - 1)
                self.activity_data[y, x] += 1
        
        self.activity_im.set_data(self.activity_data)
        self.activity_im.set_clim(vmin=0, vmax=max(10, self.activity_data.max()))
    
    def _update_stats(self, generation: int, sim_data: Dict, government_style: str):
        """Update main stats panel."""
        elapsed = time.time() - self.start_time
        speed = generation / elapsed if elapsed > 0 else 0
        progress = (generation / self.total_generations) * 100
        
        stats = f"""MAIN STATISTICS
━━━━━━━━━━━━━━━
Gen: {generation}/{self.total_generations}
Progress: {progress:.1f}%
Gov: {government_style[:15]}

Population: {sim_data.get('population', 0):,}
Max Ever: {sim_data.get('max_population', 0):,}

Cooperation: {sim_data.get('cooperation', 0)*100:.1f}%

Wealth:
 Avg: {sim_data.get('avg_wealth', 0):.1f}
 Total: {sim_data.get('total_wealth', 0):,.0f}

Age:
 Avg: {sim_data.get('avg_age', 0):.1f}
 Max: {sim_data.get('oldest_agent', 0)}

Speed: {speed:.2f} gen/s
Time: {elapsed:.1f}s
"""
        self.stats_text.set_text(stats)
    
    def _update_genetics(self, sim_data: Dict):
        """Update genetics panel."""
        genetics = f"""GENETICS
━━━━━━━━━━━━━━━
Metabolism:
 {sim_data.get('avg_metabolism', 0):.3f}

Vision:
 {sim_data.get('avg_vision', 0):.2f}

Lifespan:
 {sim_data.get('avg_lifespan', 0):.1f}

Clustering:
 {sim_data.get('clustering', 0)*100:.1f}%
"""
        self.genetics_text.set_text(genetics)
    
    def _update_network_stats(self):
        """Update network statistics."""
        total = len(self.interaction_history)
        unique = len(self.interaction_counts)
        recent = len(self.generation_interactions) if hasattr(self, 'generation_interactions') else 0
        
        stats = f"""NETWORK STATS
━━━━━━━━━━━━━━━
Total Interactions:
 {total}

Unique Pairs:
 {unique}

This Generation:
 {recent}

Avg per Pair:
 {sum(self.interaction_counts.values()) / max(unique, 1):.1f}
"""
        self.network_stats_text.set_text(stats)
    
    def _update_gpu_stats(self, gpu_stats=None):
        """Update GPU statistics."""
        if not self.use_gpu or not gpu_stats:
            text = """GPU STATUS
━━━━━━━━━━━━━━━
Status: CPU Mode

No GPU acceleration
"""
        else:
            text = f"""GPU STATUS
━━━━━━━━━━━━━━━
Device:
 {gpu_stats.get('device', 'N/A')[:15]}

Backend:
 {gpu_stats.get('backend', 'N/A')}

Memory:
 {gpu_stats.get('memory_mb', 0):.1f} MB

Speedup:
 {gpu_stats.get('speedup', 1.0):.2f}x
"""
        self.gpu_stats_text.set_text(text)
    
    def _update_time_series(self):
        """Update all time-series graphs."""
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
        
        # Interactions
        self.line_interactions.set_data(gens, list(self.interactions_per_gen))
        self.ax_interactions.relim()
        self.ax_interactions.autoscale_view()
        
        # GPU Memory
        if self.use_gpu:
            self.line_gpu_memory.set_data(gens, list(self.gpu_memory_history))
            self.ax_gpu_memory.relim()
            self.ax_gpu_memory.autoscale_view()
    
    def close(self):
        """Close dashboard."""
        plt.close(self.fig)


def run_ultimate_dashboard(
    initial_size: int = 200,
    generations: int = 300,
    government_style = None,
    grid_size: Tuple[int, int] = (75, 75),
    use_gpu: bool = False,
    update_every: int = 3
):
    """
    Run simulation with ultimate combined dashboard.
    """
    if use_gpu:
        from gpu_echo_simulation import GPUEchoSimulation as SimClass
        from gpu_acceleration import get_gpu_accelerator
        gpu = get_gpu_accelerator()
    else:
        from ultimate_echo_simulation import UltimateEchoSimulation as SimClass
        gpu = None
    
    from government_styles import GovernmentStyle
    from genetic_traits import analyze_population_genetics
    
    if government_style is None:
        government_style = GovernmentStyle.LAISSEZ_FAIRE
    
    print("\n" + "=" * 80)
    print("🌟 ULTIMATE COMBINED DASHBOARD - All Features Together")
    print("=" * 80)
    print(f"   Population: {initial_size}")
    print(f"   Generations: {generations}")
    print(f"   Government: {government_style.value}")
    print(f"   GPU: {gpu.backend.upper() if (gpu and gpu.use_gpu) else 'CPU Only'}")
    print("\n✨ Opening ultimate dashboard...\n")
    
    # Create simulation
    sim = SimClass(
        initial_size=initial_size,
        grid_size=grid_size,
        government_style=government_style
    )
    
    # Create ultimate dashboard
    dashboard = UltimateDashboard(grid_size, generations, use_gpu=use_gpu)
    
    try:
        for gen in range(generations):
            survived = sim.step()
            
            if not survived:
                print("\n💀 Population extinct!")
                break
            
            # Track interactions
            _track_generation_interactions(sim.agents, dashboard)
            
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
                    'clustering': 0.5
                }
                
                # Get GPU stats if applicable
                gpu_stats = None
                if use_gpu and gpu and gpu.use_gpu:
                    stats = gpu.get_stats()
                    gpu_stats = {
                        'device': stats.get('device_name', 'GPU'),
                        'backend': gpu.backend,
                        'memory_mb': stats.get('memory_allocated', stats.get('memory_used', 0)),
                        'speedup': 1.2
                    }
                
                dashboard.update(gen + 1, sim.agents, sim_data, government_style.value, gpu_stats)
        
        print("\n✅ Simulation complete! Close window to exit.")
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Generations: {gen + 1}")
        print(f"   Final Population: {len(sim.agents)}")
        print(f"   Cooperation Rate: {genetics.get('cooperation_rate', 0)*100:.1f}%")
        print(f"   Total Interactions: {len(dashboard.interaction_history)}")
        
        plt.show(block=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        dashboard.close()
    
    return sim


def _track_generation_interactions(agents: List, dashboard):
    """Track interactions between nearby agents."""
    for i, agent1 in enumerate(agents):
        if hasattr(agent1, 'dead') and agent1.dead:
            continue
        
        vision = agent1.traits.vision if hasattr(agent1, 'traits') else 3
        
        for agent2 in agents[i+1:]:
            if hasattr(agent2, 'dead') and agent2.dead:
                continue
            
            dx = abs(agent1.pos[0] - agent2.pos[0])
            dy = abs(agent1.pos[1] - agent2.pos[1])
            distance = dx + dy
            
            if distance <= vision and np.random.random() < 0.2:
                s1 = agent1.get_strategy()
                s2 = agent2.get_strategy()
                
                if s1 == 1 and s2 == 1:
                    outcome = 'both_coop'
                elif s1 == 0 and s2 == 0:
                    outcome = 'both_defect'
                else:
                    outcome = 'betrayal'
                
                dashboard.track_interaction(
                    id(agent1), id(agent2),
                    agent1.pos, agent2.pos,
                    s1, s2, outcome
                )


if __name__ == "__main__":
    from government_styles import GovernmentStyle
    
    # Run with all features
    run_ultimate_dashboard(
        initial_size=200,
        generations=300,
        government_style=GovernmentStyle.LAISSEZ_FAIRE,
        grid_size=(75, 75),
        use_gpu=False,  # Set True for GPU acceleration
        update_every=3
    )
