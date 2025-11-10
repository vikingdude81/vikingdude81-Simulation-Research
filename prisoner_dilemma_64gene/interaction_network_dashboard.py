"""
🕸️ INTERACTION NETWORK DASHBOARD - Visualize Agent Relationships
=================================================================

Shows agent interactions as an animated network:
- Nodes = Agents (color by strategy: green=cooperate, red=defect)
- Edges = Recent interactions (thickness = frequency, color = outcome)
- Activity highlights = Flash when interaction happens
- Clusters = Groups of frequently interacting agents

Features:
- Real-time network updates
- Spatial clustering visualization
- Interaction history tracking
- Activity heatmap
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import deque, defaultdict
import time

# Enable interactive mode
plt.ion()


class InteractionNetworkDashboard:
    """
    Real-time network visualization of agent interactions.
    """
    
    def __init__(self, grid_size: Tuple[int, int], total_generations: int):
        self.grid_size = grid_size
        self.total_generations = total_generations
        
        # Interaction tracking
        self.interaction_history = deque(maxlen=100)  # Last 100 interactions
        self.interaction_counts = defaultdict(int)  # (agent_id1, agent_id2) -> count
        self.recent_interactions = deque(maxlen=20)  # Very recent for flash effect
        
        # Data storage
        self.history_length = 200
        self.generations = deque(maxlen=self.history_length)
        self.population_history = deque(maxlen=self.history_length)
        self.cooperation_history = deque(maxlen=self.history_length)
        self.interactions_per_gen = deque(maxlen=self.history_length)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('🕸️ AGENT INTERACTION NETWORK - Real-Time Social Dynamics', 
                         fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(2, 3, hspace=0.25, wspace=0.3)
        
        # Left: Large interaction network
        self.ax_network = self.fig.add_subplot(gs[:, 0:2])
        
        # Right top: Activity heatmap
        self.ax_activity = self.fig.add_subplot(gs[0, 2])
        
        # Right bottom: Statistics
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        
        # Initialize plots
        self._init_network()
        self._init_activity()
        self._init_stats()
        
        # Show window
        plt.show(block=False)
        plt.pause(0.1)
        
        self.start_time = time.time()
        self.generation_interactions = []
    
    def _init_network(self):
        """Initialize interaction network visualization."""
        self.ax_network.set_title('🕸️ Agent Interaction Network', fontweight='bold', fontsize=12)
        self.ax_network.set_xlim(0, self.grid_size[0])
        self.ax_network.set_ylim(0, self.grid_size[1])
        self.ax_network.set_xlabel('X Position')
        self.ax_network.set_ylabel('Y Position')
        self.ax_network.set_facecolor('#1a1a1a')  # Dark background
        
        # Legend
        coop_patch = mpatches.Patch(color='lime', label='Cooperators')
        defect_patch = mpatches.Patch(color='red', label='Defectors')
        interact_line = mpatches.Patch(color='cyan', label='Interactions')
        self.ax_network.legend(handles=[coop_patch, defect_patch, interact_line], 
                              loc='upper right', fontsize=8)
    
    def _init_activity(self):
        """Initialize activity heatmap."""
        self.ax_activity.set_title('🔥 Activity Heatmap', fontweight='bold', fontsize=10)
        
        # Create empty heatmap
        heatmap_size = (20, 20)
        self.activity_data = np.zeros(heatmap_size)
        self.activity_im = self.ax_activity.imshow(
            self.activity_data,
            cmap='hot',
            interpolation='bilinear',
            vmin=0,
            vmax=10,
            aspect='auto'
        )
        
        cbar = plt.colorbar(self.activity_im, ax=self.ax_activity, fraction=0.046)
        cbar.set_label('Interaction Frequency', rotation=270, labelpad=15, fontsize=8)
        self.ax_activity.set_xticks([])
        self.ax_activity.set_yticks([])
    
    def _init_stats(self):
        """Initialize statistics panel."""
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(
            0.05, 0.5, '', 
            fontsize=9, 
            family='monospace',
            verticalalignment='center'
        )
    
    def track_interaction(self, agent1_id: int, agent2_id: int, 
                         agent1_pos: Tuple, agent2_pos: Tuple,
                         agent1_strategy: int, agent2_strategy: int,
                         outcome: str):
        """
        Track an interaction between two agents.
        
        Args:
            agent1_id, agent2_id: Agent identifiers
            agent1_pos, agent2_pos: Agent positions
            agent1_strategy, agent2_strategy: 0=defect, 1=cooperate
            outcome: 'both_coop', 'both_defect', 'betrayal'
        """
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
        
        # Count this interaction
        pair = tuple(sorted([agent1_id, agent2_id]))
        self.interaction_counts[pair] += 1
    
    def update(self, generation: int, agents: List, sim_data: Dict, 
               government_style: str = "laissez_faire"):
        """Update all dashboard elements."""
        # Add to history
        self.generations.append(generation)
        self.population_history.append(sim_data.get('population', 0))
        self.cooperation_history.append(sim_data.get('cooperation', 0) * 100)
        self.interactions_per_gen.append(len(self.generation_interactions))
        
        # Update visualizations
        self._update_network(agents)
        self._update_activity(agents)
        self._update_stats(generation, sim_data, government_style)
        
        # Clear generation interactions for next round
        self.generation_interactions = []
        
        # Decay recent interactions
        current_time = time.time()
        while self.recent_interactions and current_time - self.recent_interactions[0]['timestamp'] > 0.5:
            self.recent_interactions.popleft()
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _update_network(self, agents: List):
        """Update network visualization with nodes and edges."""
        self.ax_network.clear()
        self.ax_network.set_title('🕸️ Agent Interaction Network', fontweight='bold', fontsize=12)
        self.ax_network.set_xlim(0, self.grid_size[0])
        self.ax_network.set_ylim(0, self.grid_size[1])
        self.ax_network.set_xlabel('X Position')
        self.ax_network.set_ylabel('Y Position')
        self.ax_network.set_facecolor('#1a1a1a')
        
        if not agents:
            return
        
        # Draw interaction edges (connections)
        edges = []
        edge_colors = []
        edge_widths = []
        
        # Sample interactions to draw (don't overwhelm visualization)
        sampled_interactions = list(self.interaction_history)[-50:]  # Last 50
        
        for interaction in sampled_interactions:
            pos1 = interaction['pos1']
            pos2 = interaction['pos2']
            
            # Create edge
            edges.append([pos1, pos2])
            
            # Color by outcome
            outcome = interaction['outcome']
            if outcome == 'both_coop':
                edge_colors.append('lime')
                edge_widths.append(1.5)
            elif outcome == 'both_defect':
                edge_colors.append('red')
                edge_widths.append(1.0)
            else:  # betrayal
                edge_colors.append('orange')
                edge_widths.append(1.2)
        
        # Draw edges
        if edges:
            lc = LineCollection(edges, colors=edge_colors, linewidths=edge_widths, 
                               alpha=0.3, zorder=1)
            self.ax_network.add_collection(lc)
        
        # Draw recent interactions with flash effect
        for interaction in self.recent_interactions:
            age = time.time() - interaction['timestamp']
            if age < 0.5:  # Flash for 0.5 seconds
                alpha = 1.0 - (age / 0.5)  # Fade out
                pos1 = interaction['pos1']
                pos2 = interaction['pos2']
                self.ax_network.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                    color='yellow', linewidth=3, alpha=alpha, zorder=3)
        
        # Draw agent nodes
        positions = np.array([agent.pos for agent in agents if not (hasattr(agent, 'dead') and agent.dead)])
        strategies = np.array([agent.get_strategy() for agent in agents if not (hasattr(agent, 'dead') and agent.dead)])
        
        if len(positions) > 0:
            # Cooperators
            coop_mask = strategies == 1
            if np.any(coop_mask):
                self.ax_network.scatter(positions[coop_mask, 0], positions[coop_mask, 1],
                                       c='lime', s=20, alpha=0.8, edgecolors='white',
                                       linewidths=0.5, zorder=2)
            
            # Defectors
            defect_mask = strategies == 0
            if np.any(defect_mask):
                self.ax_network.scatter(positions[defect_mask, 0], positions[defect_mask, 1],
                                       c='red', s=20, alpha=0.8, edgecolors='white',
                                       linewidths=0.5, zorder=2)
        
        # Legend
        coop_patch = mpatches.Patch(color='lime', label='Cooperators')
        defect_patch = mpatches.Patch(color='red', label='Defectors')
        interact_line = mpatches.Patch(color='cyan', alpha=0.3, label='Past Interactions')
        flash_line = mpatches.Patch(color='yellow', label='Active Interactions')
        self.ax_network.legend(handles=[coop_patch, defect_patch, interact_line, flash_line], 
                              loc='upper right', fontsize=8)
    
    def _update_activity(self, agents: List):
        """Update activity heatmap showing interaction density."""
        heatmap_size = (20, 20)
        self.activity_data = np.zeros(heatmap_size)
        
        # Scale factors
        scale_x = self.grid_size[0] / heatmap_size[1]
        scale_y = self.grid_size[1] / heatmap_size[0]
        
        # Add interaction positions to heatmap
        for interaction in self.interaction_history:
            pos1 = interaction['pos1']
            pos2 = interaction['pos2']
            
            # Add both positions
            for pos in [pos1, pos2]:
                x = min(int(pos[0] / scale_x), heatmap_size[1] - 1)
                y = min(int(pos[1] / scale_y), heatmap_size[0] - 1)
                self.activity_data[y, x] += 1
        
        # Update image
        self.activity_im.set_data(self.activity_data)
        self.activity_im.set_clim(vmin=0, vmax=max(10, self.activity_data.max()))
    
    def _update_stats(self, generation: int, sim_data: Dict, government_style: str):
        """Update statistics text."""
        elapsed = time.time() - self.start_time
        speed = generation / elapsed if elapsed > 0 else 0
        progress = (generation / self.total_generations) * 100
        
        # Calculate network statistics
        total_interactions = len(self.interaction_history)
        unique_pairs = len(self.interaction_counts)
        avg_interactions_per_pair = (sum(self.interaction_counts.values()) / unique_pairs) if unique_pairs > 0 else 0
        
        # Recent interaction rate
        recent_rate = len(self.generation_interactions)
        
        stats = f"""📊 NETWORK STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━
Generation: {generation}/{self.total_generations}
Progress: {progress:.1f}%

👥 Population: {sim_data.get('population', 0):,}
🤝 Cooperation: {sim_data.get('cooperation', 0)*100:.1f}%

🕸️ INTERACTIONS:
Total (recent): {total_interactions}
This generation: {recent_rate}
Unique pairs: {unique_pairs}
Avg per pair: {avg_interactions_per_pair:.1f}

🔥 ACTIVITY:
High activity zones: {np.sum(self.activity_data > 5)}
Isolated agents: {sim_data.get('population', 0) - len([a for i in self.interaction_history for a in [i['agent1_id'], i['agent2_id']]])}

💰 Economy:
Avg wealth: {sim_data.get('avg_wealth', 0):.1f}
Total wealth: {sim_data.get('total_wealth', 0):,.0f}

⚡ Performance:
Speed: {speed:.2f} gen/s
Time: {elapsed:.1f}s
"""
        
        self.stats_text.set_text(stats)
    
    def close(self):
        """Close the dashboard."""
        plt.close(self.fig)


def run_with_interaction_network(
    initial_size: int = 100,
    generations: int = 300,
    government_style = None,
    grid_size: Tuple[int, int] = (50, 50),
    update_every: int = 3
):
    """
    Run simulation with interaction network visualization.
    
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
    
    print("\n🕸️ Starting simulation with interaction network visualization...")
    print(f"   Population: {initial_size}")
    print(f"   Generations: {generations}")
    print(f"   Government: {government_style.value}")
    print("\n✨ Network window opening...\n")
    
    # Create simulation
    sim = UltimateEchoSimulation(
        initial_size=initial_size,
        grid_size=grid_size,
        government_style=government_style
    )
    
    # Create dashboard
    dashboard = InteractionNetworkDashboard(grid_size, generations)
    
    try:
        for gen in range(generations):
            # Step simulation WITH interaction tracking
            survived = sim.step()
            
            if not survived:
                print("\n💀 Population went extinct!")
                break
            
            # Track interactions from this generation
            # We need to modify the simulation to expose interactions
            # For now, reconstruct likely interactions based on proximity
            _track_generation_interactions(sim.agents, dashboard)
            
            # Update dashboard
            if gen % update_every == 0 or gen == generations - 1:
                genetics = analyze_population_genetics(sim.agents)
                
                sim_data = {
                    'population': len(sim.agents),
                    'cooperation': genetics.get('cooperation_rate', 0),
                    'avg_wealth': np.mean([a.wealth for a in sim.agents]) if sim.agents else 0,
                    'total_wealth': sum(a.wealth for a in sim.agents),
                    'avg_age': genetics.get('avg_age', 0)
                }
                
                dashboard.update(gen + 1, sim.agents, sim_data, government_style.value)
        
        print("\n✅ Simulation complete! Close the window to exit.")
        plt.show(block=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
    
    finally:
        dashboard.close()
    
    return sim


def _track_generation_interactions(agents: List, dashboard: InteractionNetworkDashboard):
    """
    Reconstruct interactions by finding agents within vision range.
    
    Note: This is an approximation. Ideally, the simulation would emit interaction events.
    """
    for i, agent1 in enumerate(agents):
        if hasattr(agent1, 'dead') and agent1.dead:
            continue
        
        # Find nearby agents (within vision range)
        vision = agent1.traits.vision if hasattr(agent1, 'traits') else 3
        
        for agent2 in agents[i+1:]:
            if hasattr(agent2, 'dead') and agent2.dead:
                continue
            
            # Calculate distance
            dx = abs(agent1.pos[0] - agent2.pos[0])
            dy = abs(agent1.pos[1] - agent2.pos[1])
            distance = dx + dy
            
            # If within vision, likely interacted
            if distance <= vision and np.random.random() < 0.3:  # Sample 30% to avoid clutter
                # Determine outcome
                s1 = agent1.get_strategy()
                s2 = agent2.get_strategy()
                
                if s1 == 1 and s2 == 1:
                    outcome = 'both_coop'
                elif s1 == 0 and s2 == 0:
                    outcome = 'both_defect'
                else:
                    outcome = 'betrayal'
                
                dashboard.track_interaction(
                    agent1_id=id(agent1),
                    agent2_id=id(agent2),
                    agent1_pos=agent1.pos,
                    agent2_pos=agent2.pos,
                    agent1_strategy=s1,
                    agent2_strategy=s2,
                    outcome=outcome
                )


if __name__ == "__main__":
    from government_styles import GovernmentStyle
    
    print("=" * 70)
    print("🕸️ INTERACTION NETWORK DASHBOARD - Social Dynamics Visualization")
    print("=" * 70)
    
    run_with_interaction_network(
        initial_size=100,
        generations=300,
        government_style=GovernmentStyle.LAISSEZ_FAIRE,
        grid_size=(50, 50),
        update_every=2  # Update frequently to see interactions
    )
