"""
🎨 VISUAL FEEDBACK SYSTEM FOR ULTIMATE ECHO
===========================================

Real-time visualization with:
- World map (agents positioned on grid)
- Population graph
- Cooperation tracker
- Event log
- Progress bars
- Live statistics

Uses rich library for terminal-based UI.
"""

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time

try:
    from rich import console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Install rich for visual feedback: pip install rich")


class VisualFeedback:
    """
    Real-time visual feedback system for Echo simulation.
    
    Features:
    - ASCII world map with agent density
    - Live population/cooperation graphs
    - Event log
    - Progress bars
    - Statistics panels
    """
    
    def __init__(self, grid_size: Tuple[int, int], total_generations: int):
        if not RICH_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        self.console = Console()
        self.grid_size = grid_size
        self.total_generations = total_generations
        
        # Layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        self.layout["left"].split_column(
            Layout(name="world", ratio=3),
            Layout(name="events", ratio=1)
        )
        
        self.layout["right"].split_column(
            Layout(name="stats", ratio=1),
            Layout(name="genetics", ratio=1)
        )
        
        # Event log
        self.events: List[str] = []
        self.max_events = 10
        
        # History for mini-graphs
        self.pop_history: List[int] = []
        self.coop_history: List[float] = []
        self.max_history = 50
        
        # Progress tracking
        self.generation = 0
        self.start_time = time.time()
    
    def _make_world_map(self, agents: List, map_size: Tuple[int, int] = (40, 20)) -> Panel:
        """
        Create ASCII world map showing agent positions.
        
        Color-coded by:
        - Green: Cooperators
        - Red: Defectors
        - Brightness: Wealth
        """
        # Create density grid
        grid = np.zeros(map_size)
        coop_grid = np.zeros(map_size)
        
        # Scale factor
        scale_x = self.grid_size[0] / map_size[1]
        scale_y = self.grid_size[1] / map_size[0]
        
        for agent in agents:
            if hasattr(agent, 'dead') and agent.dead:
                continue
            
            # Map to display grid
            x = int(agent.pos[0] / scale_x)
            y = int(agent.pos[1] / scale_y)
            
            if 0 <= y < map_size[0] and 0 <= x < map_size[1]:
                grid[y, x] += 1
                if agent.get_strategy() == 1:
                    coop_grid[y, x] += 1
        
        # Render as ASCII
        chars = " ░▒▓█"
        lines = []
        
        for y in range(map_size[0]):
            line = ""
            for x in range(map_size[1]):
                count = int(grid[y, x])
                if count == 0:
                    line += " "
                elif count > len(chars) - 1:
                    # Color by cooperation rate
                    coop_rate = coop_grid[y, x] / count if count > 0 else 0
                    if coop_rate > 0.7:
                        line += f"[green]{chars[-1]}[/green]"
                    elif coop_rate > 0.3:
                        line += f"[yellow]{chars[-1]}[/yellow]"
                    else:
                        line += f"[red]{chars[-1]}[/red]"
                else:
                    line += chars[count]
            lines.append(line)
        
        world_text = "\n".join(lines)
        
        return Panel(
            world_text,
            title=f"🌍 World Map ({self.grid_size[0]}×{self.grid_size[1]}) - Green=Coop, Red=Defect",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def _make_stats_panel(self, sim_data: Dict) -> Panel:
        """Create statistics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Population
        pop = sim_data.get('population', 0)
        max_pop = sim_data.get('max_population', pop)
        table.add_row("👥 Population", f"{pop:,} / {max_pop:,}")
        
        # Cooperation
        coop = sim_data.get('cooperation', 0) * 100
        coop_color = "green" if coop > 60 else "yellow" if coop > 30 else "red"
        table.add_row("🤝 Cooperation", f"[{coop_color}]{coop:.1f}%[/{coop_color}]")
        
        # Wealth
        avg_wealth = sim_data.get('avg_wealth', 0)
        total_wealth = sim_data.get('total_wealth', 0)
        table.add_row("💰 Avg Wealth", f"{avg_wealth:.1f}")
        table.add_row("💎 Total Wealth", f"{total_wealth:,.0f}")
        
        # Age
        avg_age = sim_data.get('avg_age', 0)
        oldest = sim_data.get('oldest_agent', 0)
        table.add_row("⏳ Avg Age", f"{avg_age:.1f} gen")
        table.add_row("👴 Oldest", f"{oldest} gen")
        
        # Speed
        elapsed = time.time() - self.start_time
        speed = self.generation / elapsed if elapsed > 0 else 0
        table.add_row("⚡ Speed", f"{speed:.2f} gen/s")
        
        return Panel(table, title="📊 Statistics", border_style="green", box=box.ROUNDED)
    
    def _make_genetics_panel(self, sim_data: Dict) -> Panel:
        """Create genetics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Trait", style="magenta")
        table.add_column("Value", style="white")
        
        # Metabolism
        metabolism = sim_data.get('avg_metabolism', 0)
        table.add_row("🔥 Metabolism", f"{metabolism:.2f}")
        
        # Vision
        vision = sim_data.get('avg_vision', 0)
        table.add_row("👁️ Vision", f"{vision:.2f}")
        
        # Lifespan
        lifespan = sim_data.get('avg_lifespan', 0)
        table.add_row("💗 Lifespan", f"{lifespan:.1f}")
        
        # Clustering
        clustering = sim_data.get('clustering', 0) * 100
        table.add_row("🏘️ Clustering", f"{clustering:.1f}%")
        
        return Panel(table, title="🧬 Genetics", border_style="magenta", box=box.ROUNDED)
    
    def _make_events_panel(self) -> Panel:
        """Create events log panel."""
        # Show last N events
        recent_events = self.events[-self.max_events:]
        
        if not recent_events:
            text = "[dim]No events yet...[/dim]"
        else:
            text = "\n".join(recent_events)
        
        return Panel(
            text,
            title="📜 Event Log",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def _make_header(self, government_style: str) -> Panel:
        """Create header panel."""
        progress = (self.generation / self.total_generations) * 100
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * self.generation / self.total_generations)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        header_text = Text()
        header_text.append("🌍 ULTIMATE ECHO SIMULATION ", style="bold cyan")
        header_text.append(f"| Gov: {government_style.upper()} ", style="bold yellow")
        header_text.append(f"| Gen: {self.generation}/{self.total_generations} ", style="bold white")
        header_text.append(f"| {bar} {progress:.1f}%", style="green")
        
        return Panel(header_text, style="bold", box=box.HEAVY)
    
    def _make_footer(self) -> Panel:
        """Create footer panel."""
        elapsed = time.time() - self.start_time
        remaining = (self.total_generations - self.generation) / (self.generation / elapsed) if self.generation > 0 else 0
        
        footer_text = Text()
        footer_text.append("⏱️ Elapsed: ", style="cyan")
        footer_text.append(f"{elapsed:.1f}s ", style="white")
        footer_text.append("| Est. Remaining: ", style="cyan")
        footer_text.append(f"{remaining:.1f}s ", style="white")
        footer_text.append("| Press Ctrl+C to stop", style="dim")
        
        return Panel(footer_text, style="dim", box=box.ROUNDED)
    
    def update(
        self,
        generation: int,
        agents: List,
        sim_data: Dict,
        government_style: str = "laissez_faire"
    ):
        """Update the visual display."""
        if not self.enabled:
            return
        
        self.generation = generation
        
        # Update history
        self.pop_history.append(sim_data.get('population', 0))
        self.coop_history.append(sim_data.get('cooperation', 0))
        
        if len(self.pop_history) > self.max_history:
            self.pop_history.pop(0)
            self.coop_history.pop(0)
        
        # Update layout
        self.layout["header"].update(self._make_header(government_style))
        self.layout["world"].update(self._make_world_map(agents))
        self.layout["events"].update(self._make_events_panel())
        self.layout["stats"].update(self._make_stats_panel(sim_data))
        self.layout["genetics"].update(self._make_genetics_panel(sim_data))
        self.layout["footer"].update(self._make_footer())
    
    def add_event(self, event_text: str, style: str = "white"):
        """Add event to log."""
        timestamp = f"[dim]Gen {self.generation}:[/dim]"
        self.events.append(f"{timestamp} [{style}]{event_text}[/{style}]")
    
    def get_layout(self):
        """Get the layout for rendering."""
        return self.layout


def create_visual_simulation(
    initial_size: int = 100,
    generations: int = 500,
    government_style = None,
    grid_size: Tuple[int, int] = (50, 50),
    update_every: int = 5
):
    """
    Create and run simulation with visual feedback.
    
    Args:
        initial_size: Starting population
        generations: Number of generations
        government_style: GovernmentStyle enum
        grid_size: World grid size
        update_every: Update display every N generations
    """
    if not RICH_AVAILABLE:
        print("⚠️ Visual feedback requires 'rich' library")
        print("Install with: pip install rich")
        return None
    
    from ultimate_echo_simulation import UltimateEchoSimulation
    from government_styles import GovernmentStyle
    
    if government_style is None:
        government_style = GovernmentStyle.LAISSEZ_FAIRE
    
    # Create simulation
    sim = UltimateEchoSimulation(
        initial_size=initial_size,
        grid_size=grid_size,
        government_style=government_style
    )
    
    # Create visual feedback
    visual = VisualFeedback(grid_size, generations)
    
    # Run with live display
    with Live(visual.get_layout(), refresh_per_second=4, screen=True) as live:
        for gen in range(generations):
            # Step simulation
            survived = sim.step()
            
            if not survived:
                visual.add_event("💀 EXTINCTION!", "red bold")
                break
            
            # Update visual every N generations
            if gen % update_every == 0:
                # Gather data
                from genetic_traits import analyze_population_genetics
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
                
                visual.update(gen + 1, sim.agents, sim_data, government_style.value)
                live.update(visual.get_layout())
            
            # Log events
            if sim.history['policy_actions'] and sim.history['policy_actions'][-1].generation == gen + 1:
                action = sim.history['policy_actions'][-1]
                visual.add_event(f"🏛️ {action.action_type}: {action.affected_agents} agents", "yellow")
            
            if sim.history['shocks'] and sim.history['shocks'][-1]['generation'] == gen + 1:
                shock = sim.history['shocks'][-1]
                visual.add_event(f"⚠️ {shock['type']}: {shock['affected']} affected", "red")
        
        # Final update
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
        
        visual.add_event("✅ SIMULATION COMPLETE!", "green bold")
        visual.update(generations, sim.agents, sim_data, government_style.value)
        live.update(visual.get_layout())
        
        # Pause to show final state
        import time
        time.sleep(3)
    
    return sim


if __name__ == "__main__":
    if not RICH_AVAILABLE:
        print("❌ Please install rich: pip install rich")
    else:
        print("🎨 Starting visual simulation...")
        from government_styles import GovernmentStyle
        
        create_visual_simulation(
            initial_size=100,
            generations=200,
            government_style=GovernmentStyle.LAISSEZ_FAIRE,
            update_every=2
        )
