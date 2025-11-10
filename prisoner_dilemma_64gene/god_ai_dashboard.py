"""
📊🧠 REAL-TIME GOD-AI DASHBOARD

Interactive matplotlib dashboard showing:
- Population dynamics over time
- Cooperation rate evolution
- Wealth distribution
- God intervention timeline (color-coded by type)
- Intervention effectiveness metrics
- External shock events
- Live spatial grid visualization

Updates in real-time during simulation.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import json
from collections import deque
from colorama import Fore, Style

# Import our God simulation
from prisoner_echo_god import (
    GodEchoPopulation, InterventionType, run_god_echo_simulation
)

class LiveGodDashboard:
    """Real-time dashboard for God-AI interventions."""
    
    def __init__(self, population: GodEchoPopulation, max_history: int = 500):
        self.population = population
        self.max_history = max_history
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('🧠👁️ God-AI Controller Dashboard - Real-Time Monitoring', 
                         fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Main metrics
        self.ax_population = self.fig.add_subplot(gs[0, 0])
        self.ax_cooperation = self.fig.add_subplot(gs[0, 1])
        self.ax_wealth = self.fig.add_subplot(gs[0, 2])
        
        # Row 2: Interventions and shocks
        self.ax_interventions = self.fig.add_subplot(gs[1, :])
        
        # Row 3: Detailed metrics
        self.ax_clustering = self.fig.add_subplot(gs[2, 0])
        self.ax_diversity = self.fig.add_subplot(gs[2, 1])
        self.ax_inequality = self.fig.add_subplot(gs[2, 2])
        
        # Row 4: Spatial grid and intervention stats
        self.ax_grid = self.fig.add_subplot(gs[3, 0:2])
        self.ax_intervention_stats = self.fig.add_subplot(gs[3, 2])
        
        # Style
        self.setup_axes()
        
        # Intervention colors
        self.intervention_colors = {
            InterventionType.STIMULUS: '#FFD700',          # Gold
            InterventionType.WELFARE: '#00CED1',           # Dark Turquoise
            InterventionType.SPAWN_TRIBE: '#FF69B4',       # Hot Pink
            InterventionType.EMERGENCY_REVIVAL: '#FF4500', # Orange Red
            InterventionType.FORCED_COOPERATION: '#9370DB', # Medium Purple
            'shock': '#DC143C'                             # Crimson (for external shocks)
        }
        
        # Show legend
        plt.ion()  # Interactive mode
        plt.show()
    
    def setup_axes(self):
        """Configure all axes."""
        
        # Population
        self.ax_population.set_title('Population Over Time', fontweight='bold')
        self.ax_population.set_xlabel('Generation')
        self.ax_population.set_ylabel('Agent Count')
        self.ax_population.grid(True, alpha=0.3)
        
        # Cooperation
        self.ax_cooperation.set_title('Cooperation Rate', fontweight='bold')
        self.ax_cooperation.set_xlabel('Generation')
        self.ax_cooperation.set_ylabel('Cooperation %')
        self.ax_cooperation.set_ylim([0, 1])
        self.ax_cooperation.grid(True, alpha=0.3)
        
        # Wealth
        self.ax_wealth.set_title('Average Wealth', fontweight='bold')
        self.ax_wealth.set_xlabel('Generation')
        self.ax_wealth.set_ylabel('Resources')
        self.ax_wealth.grid(True, alpha=0.3)
        
        # Interventions timeline
        self.ax_interventions.set_title('God Interventions & External Shocks Timeline', fontweight='bold')
        self.ax_interventions.set_xlabel('Generation')
        self.ax_interventions.set_ylabel('Event Type')
        self.ax_interventions.set_ylim([-0.5, 6.5])
        
        # Clustering
        self.ax_clustering.set_title('Tribe Clustering', fontweight='bold')
        self.ax_clustering.set_xlabel('Generation')
        self.ax_clustering.set_ylabel('Clustering %')
        self.ax_clustering.set_ylim([0, 1])
        self.ax_clustering.grid(True, alpha=0.3)
        
        # Diversity
        self.ax_diversity.set_title('Genetic Diversity', fontweight='bold')
        self.ax_diversity.set_xlabel('Generation')
        self.ax_diversity.set_ylabel('Unique Tags %')
        self.ax_diversity.set_ylim([0, 1])
        self.ax_diversity.grid(True, alpha=0.3)
        
        # Inequality
        self.ax_inequality.set_title('Wealth Inequality', fontweight='bold')
        self.ax_inequality.set_xlabel('Generation')
        self.ax_inequality.set_ylabel('Max/Min Ratio')
        self.ax_inequality.grid(True, alpha=0.3)
        
        # Spatial grid
        self.ax_grid.set_title('Spatial Distribution (Wealth)', fontweight='bold')
        self.ax_grid.set_aspect('equal')
        self.ax_grid.axis('off')
        
        # Intervention stats
        self.ax_intervention_stats.set_title('Intervention Summary', fontweight='bold')
        self.ax_intervention_stats.axis('off')
    
    def update(self, generation: int):
        """Update dashboard with current generation data."""
        
        # Clear axes (except grid and stats which we redraw)
        for ax in [self.ax_population, self.ax_cooperation, self.ax_wealth,
                   self.ax_clustering, self.ax_diversity, self.ax_inequality]:
            ax.clear()
        
        # Re-setup after clearing
        self.setup_axes()
        
        # Get history
        history = self.population.history
        gens = list(range(len(history['population'])))
        
        if not gens:
            return
        
        # === ROW 1: MAIN METRICS ===
        
        # Population
        self.ax_population.plot(gens, history['population'], 
                               color='#2E86DE', linewidth=2, label='Population')
        self.ax_population.fill_between(gens, history['population'], alpha=0.3, color='#2E86DE')
        
        # Cooperation
        coop_pct = [c * 100 for c in history['cooperation']]
        self.ax_cooperation.plot(gens, coop_pct, 
                                color='#10AC84', linewidth=2, label='Cooperation')
        self.ax_cooperation.fill_between(gens, coop_pct, alpha=0.3, color='#10AC84')
        self.ax_cooperation.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        
        # Wealth
        avg_wealth = [r / max(p, 1) for r, p in zip(history['resources'], history['population'])]
        self.ax_wealth.plot(gens, avg_wealth, 
                           color='#F39C12', linewidth=2, label='Avg Wealth')
        self.ax_wealth.fill_between(gens, avg_wealth, alpha=0.3, color='#F39C12')
        
        # === ROW 2: INTERVENTION TIMELINE ===
        
        self.ax_interventions.clear()
        self.ax_interventions.set_title('God Interventions & External Shocks Timeline', fontweight='bold')
        self.ax_interventions.set_xlabel('Generation')
        self.ax_interventions.set_ylabel('Event Type')
        self.ax_interventions.set_ylim([-0.5, 6.5])
        
        # Y-axis labels for event types
        event_types = ['Shock', 'Stimulus', 'Welfare', 'Spawn Tribe', 'Emergency', 'Forced Coop']
        self.ax_interventions.set_yticks(range(len(event_types)))
        self.ax_interventions.set_yticklabels(event_types)
        
        # Plot external shocks
        for i, shock in enumerate(history['shocks']):
            if shock is not None:
                self.ax_interventions.scatter(i, 0, color=self.intervention_colors['shock'], 
                                            s=100, marker='x', linewidths=3, alpha=0.8)
        
        # Plot God interventions
        intervention_type_to_y = {
            InterventionType.STIMULUS: 1,
            InterventionType.WELFARE: 2,
            InterventionType.SPAWN_TRIBE: 3,
            InterventionType.EMERGENCY_REVIVAL: 4,
            InterventionType.FORCED_COOPERATION: 5
        }
        
        for intervention in self.population.god.intervention_history:
            gen = intervention.generation
            itype = intervention.intervention_type
            
            if itype in intervention_type_to_y:
                y_pos = intervention_type_to_y[itype]
                color = self.intervention_colors[itype]
                
                # Draw intervention marker
                self.ax_interventions.scatter(gen, y_pos, color=color, 
                                            s=200, marker='o', alpha=0.8, edgecolors='black')
                
                # Draw vertical line to show intervention moment
                self.ax_interventions.axvline(x=gen, color=color, alpha=0.2, linestyle='--')
        
        self.ax_interventions.grid(True, alpha=0.3, axis='x')
        
        # === ROW 3: DETAILED METRICS ===
        
        # Clustering
        clustering_pct = [c * 100 for c in history['clustering']]
        self.ax_clustering.plot(gens, clustering_pct, 
                               color='#9B59B6', linewidth=2)
        self.ax_clustering.fill_between(gens, clustering_pct, alpha=0.3, color='#9B59B6')
        
        # Diversity (calculate from current state)
        diversity_history = []
        for gen_idx in gens:
            # We don't store this in history, so calculate from intervention records
            # For now, use a proxy - just show trend
            diversity_history.append(len(self.population.agents) * 0.15)  # Placeholder
        
        # Better: Calculate actual diversity if we have agent snapshots
        if self.population.agents:
            unique_tags = len(set(a.tag for a in self.population.agents))
            current_diversity = unique_tags / len(self.population.agents)
            diversity_history = [current_diversity] * len(gens)  # Simplified
        
        self.ax_diversity.plot(gens, [d * 100 if isinstance(d, float) and d <= 1 else d for d in diversity_history], 
                              color='#E74C3C', linewidth=2)
        
        # Inequality (calculate from intervention records)
        inequality_history = []
        for intervention in self.population.god.intervention_history:
            if 'wealth_inequality' in intervention.before_state:
                inequality_history.append((intervention.generation, 
                                         intervention.before_state['wealth_inequality']))
        
        if inequality_history:
            i_gens, i_vals = zip(*inequality_history)
            self.ax_inequality.plot(i_gens, i_vals, 
                                   color='#E67E22', linewidth=2, marker='o')
        
        # === ROW 4: SPATIAL GRID & STATS ===
        
        # Spatial grid visualization
        self.ax_grid.clear()
        self.ax_grid.set_title('Spatial Distribution (Wealth)', fontweight='bold')
        self.ax_grid.set_aspect('equal')
        self.ax_grid.axis('off')
        
        # Create wealth heatmap
        grid_size = 40  # Show 40×40 sample
        wealth_grid = np.zeros((grid_size, grid_size))
        
        for agent in self.population.agents:
            x, y = agent.position
            if x < grid_size and y < grid_size:
                wealth_grid[y, x] = agent.resources
        
        # Plot heatmap
        im = self.ax_grid.imshow(wealth_grid, cmap='hot', interpolation='nearest', 
                                aspect='auto', vmin=0, vmax=np.max(wealth_grid) if np.max(wealth_grid) > 0 else 1)
        
        # Add colorbar (with error handling)
        try:
            # Remove existing colorbar if present
            if hasattr(self, '_colorbar') and self._colorbar is not None:
                self._colorbar.remove()
            
            self._colorbar = plt.colorbar(im, ax=self.ax_grid, fraction=0.046, pad=0.04)
            self._colorbar.set_label('Resources', rotation=270, labelpad=15)
        except Exception as e:
            # Skip colorbar if there's an error (doesn't affect core functionality)
            pass
        
        # Intervention statistics
        self.ax_intervention_stats.clear()
        self.ax_intervention_stats.set_title('Intervention Summary', fontweight='bold')
        self.ax_intervention_stats.axis('off')
        
        god_stats = self.population.god.get_summary_stats()
        
        stats_text = f"""
{'='*30}
CURRENT GENERATION: {generation}

POPULATION STATS:
  Size: {len(self.population.agents)}
  Avg Wealth: {avg_wealth[-1]:.1f}
  Cooperation: {coop_pct[-1]:.1f}%
  Clustering: {clustering_pct[-1]:.1f}%

GOD INTERVENTIONS:
  Total: {god_stats.get('total_interventions', 0)}
"""
        
        if 'interventions_by_type' in god_stats:
            stats_text += "\nBy Type:\n"
            for itype, count in god_stats['interventions_by_type'].items():
                stats_text += f"  {itype}: {count}\n"
        
        stats_text += f"""
EXTERNAL SHOCKS:
  Total: {sum(1 for s in history['shocks'] if s is not None)}

BIRTHS/DEATHS:
  Total Births: {sum(history['births'])}
  Total Deaths: {sum(history['deaths'])}
"""
        
        self.ax_intervention_stats.text(0.05, 0.95, stats_text, 
                                       transform=self.ax_intervention_stats.transAxes,
                                       fontsize=9, verticalalignment='top',
                                       fontfamily='monospace',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Force update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def save_dashboard(self, filename: Optional[str] = None):
        """Save current dashboard state."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'outputs/god_ai/dashboard_{timestamp}.png'
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n{Fore.GREEN}✓ Dashboard saved: {filename}{Style.RESET_ALL}")

def run_with_live_dashboard(
    generations: int = 500,
    initial_size: int = 100,
    god_mode: str = "RULE_BASED",
    update_frequency: int = 5
):
    """
    Run God-AI simulation with live matplotlib dashboard.
    
    Args:
        generations: Number of generations to run
        initial_size: Starting population
        god_mode: "RULE_BASED", "ML_BASED", "API_BASED", or "DISABLED"
        update_frequency: Update dashboard every N generations
    """
    
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.MAGENTA}🧠👁️  INITIALIZING GOD-AI SIMULATION WITH LIVE DASHBOARD 👁️🧠{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}God Mode: {Fore.YELLOW}{god_mode}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Generations: {Fore.YELLOW}{generations}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Dashboard Updates: Every {Fore.YELLOW}{update_frequency}{Style.RESET_ALL} generations")
    print(f"\n{Fore.GREEN}Opening dashboard...{Style.RESET_ALL}\n")
    
    # Create population
    from prisoner_echo_god import GodEchoPopulation
    population = GodEchoPopulation(initial_size=initial_size, god_mode=god_mode)
    
    # Create dashboard
    dashboard = LiveGodDashboard(population)
    
    print(f"{Fore.GREEN}✓ Dashboard initialized{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Running simulation...{Style.RESET_ALL}\n")
    
    # Run simulation with dashboard updates
    for gen in range(generations):
        deaths, births, shock_msg, god_msg = population.step()
        
        # Update dashboard
        if gen % update_frequency == 0 or shock_msg or god_msg:
            dashboard.update(gen)
        
        # Print events to console
        if shock_msg:
            print(f"{Fore.RED}Gen {gen}: {shock_msg}{Style.RESET_ALL}")
        if god_msg:
            print(f"{Fore.MAGENTA}Gen {gen}: {god_msg}{Style.RESET_ALL}")
        
        # Check extinction
        if not population.agents:
            print(f"\n{Fore.RED}💀 EXTINCTION at generation {gen}{Style.RESET_ALL}")
            dashboard.update(gen)
            break
    
    # Final update
    dashboard.update(population.generation)
    
    # Print summary
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.YELLOW}🏁 SIMULATION COMPLETE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    
    if population.agents:
        print(f"{Fore.GREEN}✅ Population survived!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Final Population: {Fore.CYAN}{len(population.agents)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Final Cooperation: {Fore.CYAN}{population.history['cooperation'][-1]*100:.1f}%{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}💀 Population went extinct{Style.RESET_ALL}")
    
    # God summary
    god_stats = population.god.get_summary_stats()
    print(f"\n{Fore.MAGENTA}🧠 God-AI Summary:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Total Interventions: {Fore.YELLOW}{god_stats.get('total_interventions', 0)}{Style.RESET_ALL}")
    
    # Save dashboard
    dashboard.save_dashboard()
    
    # Save results
    population.save_results()
    
    # Keep dashboard open
    print(f"\n{Fore.YELLOW}Dashboard is open. Close the window to exit.{Style.RESET_ALL}")
    plt.ioff()
    plt.show()
    
    return population, dashboard

if __name__ == "__main__":
    # Run with live dashboard
    population, dashboard = run_with_live_dashboard(
        generations=500,
        initial_size=100,
        god_mode="RULE_BASED",
        update_frequency=10  # Update every 10 generations
    )
