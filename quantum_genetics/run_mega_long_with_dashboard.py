"""
Mega-Long Evolution with Live Terminal Dashboard
Beautiful real-time visualization of evolution progress

Features:
- Live fitness charts
- Population DNA visualization
- Elite genome display
- Innovation event tracking
- Performance metrics
- Progress bars and sparklines
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
from quantum_genetic_agents import QuantumAgent
import json
from datetime import datetime
import time
import sys

# Rich library for beautiful terminal UI
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.text import Text
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    print("⚠️  'rich' library not found. Installing for beautiful dashboard...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.text import Text
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True

console = Console()

class EvolutionDashboard:
    """Real-time terminal dashboard for evolution monitoring"""
    
    def __init__(self, total_generations):
        self.total_generations = total_generations
        self.fitness_history = []
        self.innovation_events = []
        self.start_time = time.time()
        self.generation_times = []
        self.best_genome_history = []
        self.diversity_history = []
        
    def create_sparkline(self, data, width=40):
        """Create ASCII sparkline from data"""
        if len(data) < 2:
            return "▁" * width
        
        # Sample data to fit width
        if len(data) > width:
            indices = np.linspace(0, len(data)-1, width, dtype=int)
            data = [data[i] for i in indices]
        
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return "▄" * len(data)
        
        # Map to spark characters
        spark_chars = "▁▂▃▄▅▆▇█"
        normalized = [(x - min_val) / (max_val - min_val) for x in data]
        sparkline = "".join(spark_chars[min(int(x * len(spark_chars)), len(spark_chars)-1)] for x in normalized)
        
        return sparkline
    
    def create_dna_visualization(self, genome):
        """Create visual representation of genome"""
        mu, omega, d, phi = genome
        
        # Create color-coded bars
        mu_bar = "█" * min(int(mu * 10), 30)
        omega_bar = "█" * min(int(omega * 10), 30)
        d_bar = "█" * min(int(d * 1000), 30)
        phi_bar = "█" * min(int(phi * 30), 30)
        
        dna_text = Text()
        dna_text.append("μ ", style="bold cyan")
        dna_text.append(f"{mu:.4f} ", style="cyan")
        dna_text.append(mu_bar + "\n", style="cyan")
        
        dna_text.append("ω ", style="bold magenta")
        dna_text.append(f"{omega:.4f} ", style="magenta")
        dna_text.append(omega_bar + "\n", style="magenta")
        
        dna_text.append("d ", style="bold green")
        dna_text.append(f"{d:.6f} ", style="green")
        dna_text.append(d_bar + "\n", style="green")
        
        dna_text.append("φ ", style="bold yellow")
        dna_text.append(f"{phi:.4f} ", style="yellow")
        dna_text.append(phi_bar, style="yellow")
        
        return dna_text
    
    def create_population_heatmap(self, population, param_index=0):
        """Create ASCII heatmap of population parameter distribution"""
        genomes = np.array([agent[1] for agent in population])
        param_vals = genomes[:, param_index]
        
        # Create histogram
        hist, bins = np.histogram(param_vals, bins=20)
        max_count = max(hist) if max(hist) > 0 else 1
        
        # Create vertical bar chart
        chart_height = 8
        chart = []
        for i in range(chart_height, 0, -1):
            row = ""
            for count in hist:
                bar_height = (count / max_count) * chart_height
                if bar_height >= i:
                    row += "█"
                else:
                    row += " "
            chart.append(row)
        
        return "\n".join(chart)
    
    def create_layout(self, evolution, generation, analyzer):
        """Create the main dashboard layout"""
        layout = Layout()
        
        # Main structure: header, body (3 columns), footer
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split body into 3 columns
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Further split left column
        layout["left"].split(
            Layout(name="champion", ratio=1),
            Layout(name="population", ratio=1)
        )
        
        # Split center column
        layout["center"].split(
            Layout(name="fitness", ratio=1),
            Layout(name="diversity", ratio=1)
        )
        
        # Split right column
        layout["right"].split(
            Layout(name="performance", ratio=1),
            Layout(name="innovations", ratio=1)
        )
        
        # === HEADER ===
        progress = (generation / self.total_generations) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / (generation + 1)) * (self.total_generations - generation) if generation > 0 else 0
        
        header_text = Text()
        header_text.append("🧬 MEGA-LONG EVOLUTION DASHBOARD ", style="bold white on blue")
        header_text.append(f"Gen {generation}/{self.total_generations} ", style="bold yellow")
        header_text.append(f"({progress:.1f}%) ", style="bold green")
        header_text.append(f"⏱️  {elapsed/60:.1f}m elapsed, {eta/60:.1f}m remaining", style="cyan")
        
        layout["header"].update(Panel(Align.center(header_text), border_style="blue"))
        
        # === CHAMPION DNA ===
        best_agent = evolution.population[0]
        best_fitness = best_agent[0]
        best_genome = best_agent[1]
        
        dna_viz = self.create_dna_visualization(best_genome)
        
        champion_content = Text()
        champion_content.append(f"🏆 Fitness: {best_fitness:.6f}\n\n", style="bold gold1")
        champion_content.append(dna_viz)
        
        layout["champion"].update(Panel(
            champion_content,
            title="[bold cyan]Champion Genome[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
        
        # === POPULATION STATS ===
        fitness_array = np.array([agent[0] for agent in evolution.population])
        genomes = np.array([agent[1] for agent in evolution.population])
        
        pop_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        pop_table.add_column("Metric", style="cyan")
        pop_table.add_column("Value", style="white")
        
        pop_table.add_row("Population", f"{len(evolution.population)}")
        pop_table.add_row("Mean Fitness", f"{np.mean(fitness_array):.6f}")
        pop_table.add_row("Median Fitness", f"{np.median(fitness_array):.6f}")
        pop_table.add_row("Std Dev", f"{np.std(fitness_array):.6f}")
        pop_table.add_row("Top 10%", f"{np.percentile(fitness_array, 90):.6f}")
        pop_table.add_row("Bottom 10%", f"{np.percentile(fitness_array, 10):.6f}")
        
        # Add parameter means
        pop_table.add_row("─" * 15, "─" * 15)
        pop_table.add_row("Avg μ", f"{np.mean(genomes[:, 0]):.4f}")
        pop_table.add_row("Avg ω", f"{np.mean(genomes[:, 1]):.4f}")
        pop_table.add_row("Avg d", f"{np.mean(genomes[:, 2]):.6f}")
        
        layout["population"].update(Panel(
            pop_table,
            title="[bold magenta]Population Stats[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED
        ))
        
        # === FITNESS EVOLUTION ===
        if len(self.fitness_history) > 1:
            sparkline = self.create_sparkline(self.fitness_history, width=50)
            
            fitness_content = Text()
            fitness_content.append("Best Fitness Over Time\n\n", style="bold green")
            fitness_content.append(sparkline, style="green")
            fitness_content.append(f"\n\n📈 Current: {best_fitness:.6f}\n", style="bold white")
            
            if len(self.fitness_history) > 1:
                improvement = ((best_fitness - self.fitness_history[0]) / self.fitness_history[0] * 100) if self.fitness_history[0] > 0 else 0
                fitness_content.append(f"📊 Total Improvement: {improvement:+.1f}%\n", style="yellow")
                
                recent_improvement = ((best_fitness - self.fitness_history[-10]) / self.fitness_history[-10] * 100) if len(self.fitness_history) > 10 and self.fitness_history[-10] > 0 else 0
                fitness_content.append(f"🔥 Last 10 gens: {recent_improvement:+.1f}%", style="cyan")
        else:
            fitness_content = Text("Collecting data...", style="dim")
        
        layout["fitness"].update(Panel(
            fitness_content,
            title="[bold green]Fitness Evolution[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # === DIVERSITY ===
        if len(self.diversity_history) > 1:
            mu_diversity = [d['mu'] for d in self.diversity_history]
            d_diversity = [d['d'] for d in self.diversity_history]
            
            diversity_content = Text()
            diversity_content.append("μ Diversity (Mutation Rate)\n", style="bold cyan")
            diversity_content.append(self.create_sparkline(mu_diversity, width=50), style="cyan")
            diversity_content.append(f"  {mu_diversity[-1]:.4f}\n\n", style="white")
            
            diversity_content.append("d Diversity (Decoherence)\n", style="bold green")
            diversity_content.append(self.create_sparkline(d_diversity, width=50), style="green")
            diversity_content.append(f"  {d_diversity[-1]:.6f}\n\n", style="white")
            
            # Convergence status
            if mu_diversity[-1] < 0.05:
                diversity_content.append("🎯 Approaching convergence!", style="bold yellow")
            else:
                diversity_content.append("🌊 Population still diverse", style="dim")
        else:
            diversity_content = Text("Collecting data...", style="dim")
        
        layout["diversity"].update(Panel(
            diversity_content,
            title="[bold yellow]Population Diversity[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        ))
        
        # === PERFORMANCE METRICS ===
        if len(self.generation_times) > 0:
            avg_time = np.mean(self.generation_times[-10:])
            agents_per_sec = len(evolution.population) / avg_time if avg_time > 0 else 0
            
            perf_table = Table(show_header=False, box=box.SIMPLE)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="white")
            
            perf_table.add_row("⚡ Throughput", f"{agents_per_sec:.0f} agents/sec")
            perf_table.add_row("⏱️  Gen Time", f"{avg_time:.2f}s")
            perf_table.add_row("🎯 Evaluations", f"{(generation + 1) * len(evolution.population):,}")
            perf_table.add_row("💾 Total Memory", f"{len(evolution.population) * 4 * 32 / 1024:.1f} KB")
            
            # GPU info if available
            perf_table.add_row("─" * 20, "─" * 20)
            perf_table.add_row("🔥 Device", "CUDA (GPU)")
            perf_table.add_row("🧠 ML Predictor", "Active")
            
            if hasattr(evolution, 'current_mutation_rate'):
                perf_table.add_row("🎲 Mutation Rate", f"{evolution.current_mutation_rate:.3f}")
        else:
            perf_table = Text("Collecting data...", style="dim")
        
        layout["performance"].update(Panel(
            perf_table,
            title="[bold blue]Performance[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        ))
        
        # === INNOVATION EVENTS ===
        if len(self.innovation_events) > 0:
            innov_table = Table(show_header=True, header_style="bold red", box=box.SIMPLE)
            innov_table.add_column("Gen", style="yellow", width=6)
            innov_table.add_column("Jump", style="green", width=10)
            innov_table.add_column("New Fitness", style="cyan", width=12)
            
            # Show last 5 innovations
            for event in self.innovation_events[-5:]:
                innov_table.add_row(
                    str(event['generation']),
                    f"+{event['improvement_pct']:.1f}%",
                    f"{event['new_fitness']:.6f}"
                )
            
            # Create a Group to combine Text and Table
            from rich.console import Group
            innov_header = Text(f"🚀 Total Events: {len(self.innovation_events)}\n", style="bold red")
            innov_content = Group(innov_header, innov_table)
        else:
            innov_content = Text("No major breakthroughs yet...\nWaiting for >10% fitness jump", style="dim")
        
        layout["innovations"].update(Panel(
            innov_content,
            title="[bold red]Innovation Events[/bold red]",
            border_style="red",
            box=box.ROUNDED
        ))
        
        # === FOOTER ===
        footer_text = Text()
        footer_text.append("🧬 ", style="bold")
        
        # Add d=0.005 convergence indicator
        current_d = np.mean(genomes[:, 2])
        if abs(current_d - 0.005) < 0.001:
            footer_text.append("✅ d≈0.005 OPTIMAL ", style="bold green")
        else:
            footer_text.append(f"d={current_d:.6f} ", style="yellow")
        
        footer_text.append("│ ", style="dim")
        
        # ML training indicator
        if generation > 0 and generation % 50 == 0:
            footer_text.append("🧠 ML TRAINING IN PROGRESS... ", style="bold magenta blink")
        else:
            next_training = 50 - (generation % 50)
            footer_text.append(f"Next ML training in {next_training} gens ", style="dim")
        
        footer_text.append("│ ", style="dim")
        footer_text.append("Press Ctrl+C to stop gracefully", style="dim italic")
        
        layout["footer"].update(Panel(Align.center(footer_text), border_style="dim"))
        
        return layout

class MegaLongAnalyzer:
    """Analyzer for mega-long evolution runs"""
    
    def __init__(self):
        self.snapshots = []
        self.convergence_detected = False
        self.convergence_generation = None
        self.innovation_events = []
        self.ml_training_history = []
        
    def take_snapshot(self, evolution, generation):
        """Take detailed population snapshot"""
        fitness = np.array([agent[0] for agent in evolution.population])
        genomes = np.array([agent[1] for agent in evolution.population])
        
        fitness_stats = {
            'mean': float(np.mean(fitness)),
            'std': float(np.std(fitness)),
            'min': float(np.min(fitness)),
            'max': float(np.max(fitness)),
            'median': float(np.median(fitness)),
            'q25': float(np.percentile(fitness, 25)),
            'q75': float(np.percentile(fitness, 75)),
            'top1%': float(np.percentile(fitness, 99)),
            'top10%': float(np.percentile(fitness, 90)),
            'top5%': float(np.percentile(fitness, 95)),
            'bottom10%': float(np.percentile(fitness, 10))
        }
        
        genome_diversity = {
            'mu': float(np.std(genomes[:, 0])),
            'omega': float(np.std(genomes[:, 1])),
            'd': float(np.std(genomes[:, 2])),
            'phi': float(np.std(genomes[:, 3])),
            'mu_mean': float(np.mean(genomes[:, 0])),
            'omega_mean': float(np.mean(genomes[:, 1])),
            'd_mean': float(np.mean(genomes[:, 2])),
            'phi_mean': float(np.mean(genomes[:, 3]))
        }
        
        elite_threshold = np.percentile(fitness, 90)
        elite_mask = fitness >= elite_threshold
        
        elite_comparison = {
            'elite_mu_mean': float(np.mean(genomes[elite_mask, 0])),
            'non_elite_mu_mean': float(np.mean(genomes[~elite_mask, 0])),
            'elite_omega_mean': float(np.mean(genomes[elite_mask, 1])),
            'non_elite_omega_mean': float(np.mean(genomes[~elite_mask, 1])),
            'elite_d_mean': float(np.mean(genomes[elite_mask, 2])),
            'non_elite_d_mean': float(np.mean(genomes[~elite_mask, 2])),
        }
        
        snapshot = {
            'generation': generation,
            'fitness_stats': fitness_stats,
            'genome_diversity': genome_diversity,
            'elite_comparison': elite_comparison,
            'mutation_rate': float(evolution.current_mutation_rate) if hasattr(evolution, 'current_mutation_rate') else 0.3
        }
        
        self.snapshots.append(snapshot)
        
        # Check convergence
        if len(self.snapshots) >= 20:
            recent_diversity = [s['genome_diversity']['mu'] for s in self.snapshots[-20:]]
            if max(recent_diversity) < 0.03 and not self.convergence_detected:
                self.convergence_detected = True
                self.convergence_generation = generation
        
        return snapshot
    
    def detect_innovation_events(self, dashboard):
        """Detect major fitness jumps"""
        if len(self.snapshots) < 2:
            return []
        
        for i in range(1, len(self.snapshots)):
            prev_best = self.snapshots[i-1]['fitness_stats']['max']
            curr_best = self.snapshots[i]['fitness_stats']['max']
            
            improvement = (curr_best - prev_best) / prev_best if prev_best > 0 else 0
            
            if improvement > 0.10:
                event = {
                    'generation': self.snapshots[i]['generation'],
                    'prev_fitness': float(prev_best),
                    'new_fitness': float(curr_best),
                    'improvement_pct': float(improvement * 100)
                }
                
                # Only add if not already recorded
                if event not in self.innovation_events:
                    self.innovation_events.append(event)
                    dashboard.innovation_events.append(event)
        
        return self.innovation_events

def run_mega_long_with_dashboard(population_size=1000, generations=1000):
    """Run mega-long evolution with live dashboard"""
    
    console.print("\n🚀 [bold cyan]Initializing Mega-Long Evolution Dashboard...[/bold cyan]\n")
    time.sleep(1)
    
    # Initialize evolution
    evolution = AdaptiveMutationEvolution(
        population_size=population_size,
        strategy='ml_adaptive'
    )
    
    # Initialize dashboard and analyzer
    dashboard = EvolutionDashboard(generations)
    analyzer = MegaLongAnalyzer()
    
    start_time = time.time()
    
    try:
        with Live(console=console, refresh_per_second=2) as live:  # Update every 0.5 seconds
            for gen in range(generations):
                gen_start = time.time()
                
                # Adapt mutation rate
                evolution.adapt_mutation_rate(gen)
                
                # Evaluate population
                evolution.evaluate_population(environment='standard')
                
                # Evolve to next generation
                evolution.evolve_generation()
                
                # Record timing
                gen_time = time.time() - gen_start
                dashboard.generation_times.append(gen_time)
                
                # Update dashboard every generation (but only refreshes at 2 Hz)
                best_fitness = evolution.population[0][0]
                dashboard.fitness_history.append(best_fitness)
                dashboard.best_genome_history.append(evolution.population[0][1].copy())
                
                # Take snapshot every 10 generations
                if gen % 10 == 0:
                    snapshot = analyzer.take_snapshot(evolution, gen)
                    dashboard.diversity_history.append(snapshot['genome_diversity'])
                    
                    # Detect innovations
                    analyzer.detect_innovation_events(dashboard)
                
                # ML training marker
                if gen > 0 and gen % 50 == 0:
                    analyzer.ml_training_history.append(gen)
                
                # Update live display
                live.update(dashboard.create_layout(evolution, gen, analyzer))
                
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]⚠️  Evolution interrupted by user[/bold yellow]")
        console.print(f"[cyan]Completed {gen}/{generations} generations[/cyan]\n")
    
    # Final snapshot
    final_snapshot = analyzer.take_snapshot(evolution, gen)
    
    total_time = time.time() - start_time
    
    console.print("\n" + "=" * 80)
    console.print("[bold green]✅ MEGA-LONG EVOLUTION COMPLETE![/bold green]")
    console.print("=" * 80)
    console.print(f"[cyan]Total runtime: {total_time/60:.1f} minutes ({total_time:.0f} seconds)[/cyan]")
    console.print(f"[cyan]Generations completed: {gen + 1}/{generations}[/cyan]")
    console.print(f"[cyan]Total evaluations: {(gen + 1) * population_size:,}[/cyan]")
    console.print(f"[cyan]Average throughput: {((gen + 1) * population_size) / total_time:.1f} agents/second[/cyan]")
    
    return evolution, analyzer, dashboard

def save_results(evolution, analyzer, dashboard, timestamp):
    """Save results to JSON and generate final visualization"""
    
    console.print("\n[cyan]💾 Saving results...[/cyan]")
    
    # Export JSON
    export_data = {
        'experiment': 'mega_long_evolution_dashboard',
        'population_size': len(evolution.population),
        'generations': len(dashboard.fitness_history),
        'total_snapshots': len(analyzer.snapshots),
        'snapshots': analyzer.snapshots,
        'innovation_events': analyzer.innovation_events,
        'ml_training_history': analyzer.ml_training_history,
        'convergence_detected': analyzer.convergence_detected,
        'convergence_generation': analyzer.convergence_generation,
        'champion_genome': {
            'mu': float(evolution.population[0][1][0]),
            'omega': float(evolution.population[0][1][1]),
            'd': float(evolution.population[0][1][2]),
            'phi': float(evolution.population[0][1][3]),
            'fitness': float(evolution.population[0][0])
        }
    }
    
    json_filename = f"mega_long_analysis_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    console.print(f"[green]✅ Data exported: {json_filename}[/green]")
    
    # Quick summary visualization
    console.print("\n[cyan]📊 Generating summary visualization...[/cyan]")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Fitness evolution
    axes[0, 0].plot(dashboard.fitness_history, 'g-', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Best Fitness Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Innovation events
    if analyzer.innovation_events:
        gens = [e['generation'] for e in analyzer.innovation_events]
        improvements = [e['improvement_pct'] for e in analyzer.innovation_events]
        axes[0, 1].bar(range(len(gens)), improvements, color='red', alpha=0.7)
        axes[0, 1].set_title(f'Innovation Events (n={len(gens)})', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Event #')
        axes[0, 1].set_ylabel('Improvement %')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Diversity
    if len(dashboard.diversity_history) > 0:
        mu_div = [d['mu'] for d in dashboard.diversity_history]
        d_div = [d['d'] for d in dashboard.diversity_history]
        gen_points = list(range(0, len(mu_div) * 10, 10))
        
        axes[1, 0].plot(gen_points, mu_div, 'r-', linewidth=2, label='μ diversity', alpha=0.8)
        axes[1, 0].plot(gen_points, d_div, 'g-', linewidth=2, label='d diversity', alpha=0.8)
        axes[1, 0].set_title('Population Diversity', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Std Dev')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Final population scatter
    genomes = np.array([agent[1] for agent in evolution.population])
    fitness = np.array([agent[0] for agent in evolution.population])
    
    scatter = axes[1, 1].scatter(genomes[:, 0], genomes[:, 2], c=fitness, 
                                 cmap='viridis', s=30, alpha=0.6, edgecolors='black')
    axes[1, 1].scatter([genomes[0, 0]], [genomes[0, 2]], s=500, c='gold', 
                      marker='*', edgecolors='red', linewidths=2, zorder=10)
    axes[1, 1].axhline(0.005, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 1].set_title('Final Population: μ vs d', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('μ (mutation)')
    axes[1, 1].set_ylabel('d (decoherence)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Fitness')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_filename = f"mega_long_evolution_{timestamp}.png"
    plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
    
    console.print(f"[green]✅ Visualization saved: {viz_filename}[/green]")

def main():
    """Main execution with dashboard"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run evolution with dashboard
    evolution, analyzer, dashboard = run_mega_long_with_dashboard(
        population_size=1000,
        generations=1000
    )
    
    # Save results
    save_results(evolution, analyzer, dashboard, timestamp)
    
    console.print("\n[bold green]🎉 ALL DONE![/bold green]\n")

if __name__ == "__main__":
    main()
