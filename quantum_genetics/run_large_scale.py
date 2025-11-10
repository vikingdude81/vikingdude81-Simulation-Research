"""
🚀⚛️ LARGE-SCALE GPU EVOLUTION WITH LIVE VISUALIZATION
======================================================

Real-time terminal dashboard showing:
- Color-coded DNA evolution
- Population fitness heatmap
- Top agents leaderboard
- Progress bar with ETA
- Mutation rate dynamics
- Generation statistics
"""

import numpy as np
import time
from datetime import datetime, timedelta
import sys
import os

# Try to import colorama for Windows color support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("⚠️  Install colorama for colors: pip install colorama")
    
    # Fallback color class
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution


class LiveDashboard:
    """Real-time terminal dashboard for evolution visualization"""
    
    def __init__(self, population_size, total_generations):
        self.population_size = population_size
        self.total_generations = total_generations
        self.start_time = None
        self.generation_times = []
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def genome_to_color_blocks(self, genome):
        """Convert genome [μ, ω, d, φ] to color-coded blocks"""
        mu, omega, d, phi = genome
        
        # Map parameters to colors
        colors = []
        
        # μ (mutation_rate): 0.01-3.0 → RED gradient
        mu_intensity = min(int((mu / 3.0) * 100), 100)
        if mu_intensity < 33:
            colors.append(f'{Fore.RED}░{Style.RESET_ALL}')
        elif mu_intensity < 66:
            colors.append(f'{Fore.RED}{Style.BRIGHT}▒{Style.RESET_ALL}')
        else:
            colors.append(f'{Fore.RED}{Style.BRIGHT}█{Style.RESET_ALL}')
        
        # ω (oscillation_freq): 0.1-2.0 → GREEN gradient
        omega_intensity = min(int((omega / 2.0) * 100), 100)
        if omega_intensity < 33:
            colors.append(f'{Fore.GREEN}░{Style.RESET_ALL}')
        elif omega_intensity < 66:
            colors.append(f'{Fore.GREEN}{Style.BRIGHT}▒{Style.RESET_ALL}')
        else:
            colors.append(f'{Fore.GREEN}{Style.BRIGHT}█{Style.RESET_ALL}')
        
        # d (decoherence): 0.005-0.1 → BLUE gradient
        d_intensity = min(int((d / 0.1) * 100), 100)
        if d_intensity < 33:
            colors.append(f'{Fore.BLUE}░{Style.RESET_ALL}')
        elif d_intensity < 66:
            colors.append(f'{Fore.BLUE}{Style.BRIGHT}▒{Style.RESET_ALL}')
        else:
            colors.append(f'{Fore.BLUE}{Style.BRIGHT}█{Style.RESET_ALL}')
        
        # φ (phase): 0-2π → MAGENTA gradient
        phi_intensity = min(int((phi / (2 * np.pi)) * 100), 100)
        if phi_intensity < 33:
            colors.append(f'{Fore.MAGENTA}░{Style.RESET_ALL}')
        elif phi_intensity < 66:
            colors.append(f'{Fore.MAGENTA}{Style.BRIGHT}▒{Style.RESET_ALL}')
        else:
            colors.append(f'{Fore.MAGENTA}{Style.BRIGHT}█{Style.RESET_ALL}')
        
        return ''.join(colors)
    
    def fitness_to_bar(self, fitness, max_fitness, width=20):
        """Convert fitness to colored bar"""
        if max_fitness == 0:
            ratio = 0
        else:
            ratio = min(fitness / max_fitness, 1.0)
        
        filled = int(ratio * width)
        bar = '█' * filled + '░' * (width - filled)
        
        # Color based on fitness level
        if ratio > 0.8:
            return f'{Fore.GREEN}{Style.BRIGHT}{bar}{Style.RESET_ALL}'
        elif ratio > 0.5:
            return f'{Fore.YELLOW}{bar}{Style.RESET_ALL}'
        elif ratio > 0.2:
            return f'{Fore.CYAN}{bar}{Style.RESET_ALL}'
        else:
            return f'{Fore.RED}{bar}{Style.RESET_ALL}'
    
    def render_progress_bar(self, current, total, width=50):
        """Render progress bar with ETA"""
        ratio = current / total
        filled = int(ratio * width)
        bar = '█' * filled + '░' * (width - filled)
        
        # Calculate ETA
        if self.generation_times and current > 0:
            avg_time = np.mean(self.generation_times[-10:])  # Last 10 gens
            remaining = total - current
            eta_seconds = remaining * avg_time
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta)
        else:
            eta_str = "calculating..."
        
        # Elapsed time
        if self.start_time:
            elapsed = timedelta(seconds=int(time.time() - self.start_time))
            elapsed_str = str(elapsed)
        else:
            elapsed_str = "00:00:00"
        
        percentage = ratio * 100
        
        return (f'{Fore.CYAN}{Style.BRIGHT}{bar}{Style.RESET_ALL} '
                f'{percentage:5.1f}% | '
                f'Elapsed: {Fore.GREEN}{elapsed_str}{Style.RESET_ALL} | '
                f'ETA: {Fore.YELLOW}{eta_str}{Style.RESET_ALL}')
    
    def render_population_heatmap(self, population, top_n=10):
        """Render top N agents as fitness heatmap"""
        sorted_pop = sorted(population, reverse=True, key=lambda x: x[0])[:top_n]
        max_fitness = sorted_pop[0][0] if sorted_pop else 1.0
        
        lines = []
        lines.append(f"\n{Fore.YELLOW}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}🏆 TOP {top_n} AGENTS{Style.RESET_ALL}")
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")
        
        for i, (fitness, genome, agent_id) in enumerate(sorted_pop, 1):
            # Rank color
            if i == 1:
                rank_str = f'{Fore.YELLOW}{Style.BRIGHT}#{i:2d}{Style.RESET_ALL}'
            elif i <= 3:
                rank_str = f'{Fore.CYAN}{Style.BRIGHT}#{i:2d}{Style.RESET_ALL}'
            else:
                rank_str = f'{Fore.WHITE}#{i:2d}{Style.RESET_ALL}'
            
            # DNA visualization
            dna = self.genome_to_color_blocks(genome)
            
            # Fitness bar
            fit_bar = self.fitness_to_bar(fitness, max_fitness, width=30)
            
            # Genome values with colors
            mu, omega, d, phi = genome
            genome_str = (f'{Fore.RED}μ={mu:.3f}{Style.RESET_ALL} '
                         f'{Fore.GREEN}ω={omega:.3f}{Style.RESET_ALL} '
                         f'{Fore.BLUE}d={d:.4f}{Style.RESET_ALL} '
                         f'{Fore.MAGENTA}φ={phi:.3f}{Style.RESET_ALL}')
            
            lines.append(f'{rank_str} {dna} {fit_bar} {fitness:10.6f} | {genome_str}')
        
        return '\n'.join(lines)
    
    def render_statistics(self, population, mutation_rate, diversity, generation):
        """Render generation statistics"""
        fitnesses = [agent[0] for agent in population]
        
        best = max(fitnesses)
        avg = np.mean(fitnesses)
        std = np.std(fitnesses)
        worst = min(fitnesses)
        
        lines = []
        lines.append(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        lines.append(f"{Fore.CYAN}{Style.BRIGHT}📊 GENERATION {generation} STATISTICS{Style.RESET_ALL}")
        lines.append(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")
        
        # Fitness stats
        lines.append(f'{Fore.GREEN}{Style.BRIGHT}Best Fitness:{Style.RESET_ALL}     {best:12.6f}  '
                    f'{self.fitness_to_bar(best, best, width=30)}')
        lines.append(f'{Fore.YELLOW}Average Fitness:{Style.RESET_ALL}  {avg:12.6f}  '
                    f'{self.fitness_to_bar(avg, best, width=30)}')
        lines.append(f'{Fore.RED}Worst Fitness:{Style.RESET_ALL}    {worst:12.6f}  '
                    f'{self.fitness_to_bar(worst, best, width=30)}')
        lines.append(f'{Fore.WHITE}Std Deviation:{Style.RESET_ALL}   {std:12.6f}')
        
        # Population metrics
        lines.append(f'\n{Fore.MAGENTA}{Style.BRIGHT}Mutation Rate:{Style.RESET_ALL}    {mutation_rate:12.6f}  '
                    f'{self.fitness_to_bar(mutation_rate, 3.0, width=30)}')
        lines.append(f'{Fore.CYAN}Diversity:{Style.RESET_ALL}         {diversity:12.6f}  '
                    f'{self.fitness_to_bar(diversity, 1.0, width=30)}')
        
        return '\n'.join(lines)
    
    def render_full_dashboard(self, evo, generation, gen_time):
        """Render complete dashboard - only clear every 5 generations to reduce flicker"""
        # Only clear screen every 5 generations to reduce flicker
        if generation % 5 == 1:
            self.clear_screen()
        
        # Use carriage return to update progress line
        sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear line
        
        # Header (only show once at start or every 5 gens)
        if generation == 1 or generation % 5 == 1:
            print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
            print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️  LARGE-SCALE QUANTUM EVOLUTION - LIVE DASHBOARD{Style.RESET_ALL}')
            print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
            print(f'{Fore.WHITE}Hardware: {Fore.CYAN}{Style.BRIGHT}NVIDIA RTX 4070 Ti (12GB){Style.RESET_ALL} | '
                  f'Population: {Fore.YELLOW}{Style.BRIGHT}{self.population_size}{Style.RESET_ALL} agents | '
                  f'Strategy: {Fore.MAGENTA}{Style.BRIGHT}ML Adaptive{Style.RESET_ALL}')
            print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
        
        # Progress bar (update every generation with carriage return)
        sys.stdout.write(f'\r{Fore.YELLOW}{Style.BRIGHT}GEN {generation:3d}/{self.total_generations}{Style.RESET_ALL} ')
        sys.stdout.write(self.render_progress_bar(generation, self.total_generations))
        sys.stdout.flush()
        
        # Statistics and leaderboard (only every 5 generations)
        if generation % 5 == 0:
            mutation_rate = evo.current_mutation_rate
            diversity = evo._calculate_diversity()
            print(self.render_statistics(evo.population, mutation_rate, diversity, generation))
            print(self.render_population_heatmap(evo.population, top_n=10))
            
            # Generation time
            print(f'\n{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
            print(f'{Fore.WHITE}Generation Time: {Fore.GREEN}{gen_time:.3f}s{Style.RESET_ALL} | '
                  f'Avg: {Fore.YELLOW}{np.mean(self.generation_times[-10:]) if self.generation_times else 0:.3f}s{Style.RESET_ALL}')
            print(f'{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')


def run_large_scale_evolution():
    """Run large-scale evolution with live dashboard"""
    
    # Configuration
    POPULATION_SIZE = 200  # Large population for GPU
    GENERATIONS = 100      # Extended run
    ENVIRONMENT = 'standard'
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀 INITIALIZING LARGE-SCALE EVOLUTION{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.CYAN}Population Size:{Style.RESET_ALL} {POPULATION_SIZE}')
    print(f'{Fore.CYAN}Generations:{Style.RESET_ALL} {GENERATIONS}')
    print(f'{Fore.CYAN}Environment:{Style.RESET_ALL} {ENVIRONMENT}')
    print(f'{Fore.CYAN}Strategy:{Style.RESET_ALL} ML Adaptive with GPU')
    print(f'\n{Fore.YELLOW}Loading...{Style.RESET_ALL}')
    
    # Create evolution engine
    evo = AdaptiveMutationEvolution(
        population_size=POPULATION_SIZE,
        strategy='ml_adaptive',
        use_gpu=True
    )
    
    # Create dashboard
    dashboard = LiveDashboard(POPULATION_SIZE, GENERATIONS)
    dashboard.start_time = time.time()
    
    print(f'{Fore.GREEN}✅ Initialized!{Style.RESET_ALL}')
    time.sleep(1)
    
    # Custom evolution loop with live updates
    for gen in range(GENERATIONS):
        gen_start = time.time()
        
        # Adapt mutation rate
        evo.adapt_mutation_rate(gen)
        
        # Evaluate population
        evo.evaluate_population(ENVIRONMENT)
        
        # Track metrics
        best_fitness = evo.population[0][0]
        avg_fitness = np.mean([agent[0] for agent in evo.population])
        diversity = evo._calculate_diversity()
        
        evo.best_fitness_history.append(best_fitness)
        evo.avg_fitness_history.append(avg_fitness)
        evo.diversity_history.append(diversity)
        
        # Collect training data for ML predictor
        if evo.strategy == 'ml_adaptive' and gen > 0:
            features = evo.predictor.extract_features(
                evo.population, gen, evo.best_fitness_history
            )
            evo.predictor.add_training_data(
                features, 
                evo.current_mutation_rate,
                best_fitness
            )
        
        # Evolve to next generation
        if gen < GENERATIONS - 1:  # Don't evolve after last generation
            evo.evolve_generation()
        
        gen_time = time.time() - gen_start
        dashboard.generation_times.append(gen_time)
        
        # Render dashboard every generation
        dashboard.render_full_dashboard(evo, gen + 1, gen_time)
        
        # Train predictor periodically
        if evo.strategy == 'ml_adaptive' and (gen + 1) % 50 == 0:
            print(f'\n{Fore.MAGENTA}{Style.BRIGHT}🎓 Training ML predictor...{Style.RESET_ALL}')
            evo.predictor.train(epochs=30)
            time.sleep(1)
    
    # Final summary
    total_time = time.time() - dashboard.start_time
    
    dashboard.clear_screen()
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}✅ EVOLUTION COMPLETE!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    print(f'{Fore.CYAN}{Style.BRIGHT}FINAL RESULTS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Total Time:{Style.RESET_ALL}           {Fore.GREEN}{total_time:.2f}s{Style.RESET_ALL} '
          f'({timedelta(seconds=int(total_time))})')
    print(f'{Fore.WHITE}Avg Time per Gen:{Style.RESET_ALL}    {Fore.YELLOW}{total_time/GENERATIONS:.3f}s{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Total Generations:{Style.RESET_ALL}   {Fore.CYAN}{GENERATIONS}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Population Size:{Style.RESET_ALL}     {Fore.CYAN}{POPULATION_SIZE}{Style.RESET_ALL}')
    
    # Best agent
    best_fitness, best_genome, _ = evo.population[0]
    print(f'\n{Fore.YELLOW}{Style.BRIGHT}🏆 CHAMPION AGENT:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Fitness:{Style.RESET_ALL}              {Fore.GREEN}{Style.BRIGHT}{best_fitness:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}DNA:{Style.RESET_ALL}                  {dashboard.genome_to_color_blocks(best_genome)}')
    mu, omega, d, phi = best_genome
    print(f'{Fore.WHITE}Genome:{Style.RESET_ALL}')
    print(f'  {Fore.RED}μ (mutation):{Style.RESET_ALL}      {mu:.6f}')
    print(f'  {Fore.GREEN}ω (oscillation):{Style.RESET_ALL}   {omega:.6f}')
    print(f'  {Fore.BLUE}d (decoherence):{Style.RESET_ALL}   {d:.6f}')
    print(f'  {Fore.MAGENTA}φ (phase):{Style.RESET_ALL}         {phi:.6f}')
    
    # Fitness improvement
    initial_fitness = evo.best_fitness_history[0]
    final_fitness = evo.best_fitness_history[-1]
    improvement = ((final_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}FITNESS IMPROVEMENT:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Initial:{Style.RESET_ALL}              {Fore.RED}{initial_fitness:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Final:{Style.RESET_ALL}                {Fore.GREEN}{Style.BRIGHT}{final_fitness:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Improvement:{Style.RESET_ALL}          {Fore.YELLOW}{Style.BRIGHT}{improvement:+.2f}%{Style.RESET_ALL}')
    
    # Mutation rate evolution
    print(f'\n{Fore.MAGENTA}{Style.BRIGHT}MUTATION RATE ADAPTATION:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Initial:{Style.RESET_ALL}              {Fore.RED}{evo.mutation_rate_history[0]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Final:{Style.RESET_ALL}                {Fore.GREEN}{evo.mutation_rate_history[-1]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Min:{Style.RESET_ALL}                  {Fore.CYAN}{min(evo.mutation_rate_history):.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Max:{Style.RESET_ALL}                  {Fore.YELLOW}{max(evo.mutation_rate_history):.6f}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Visualize final results
    print(f'{Fore.CYAN}Generating final visualization...{Style.RESET_ALL}')
    evo.visualize_results()
    
    return evo


if __name__ == "__main__":
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️  LARGE-SCALE QUANTUM EVOLUTION{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.YELLOW}This will run a large-scale evolution with live visualization.{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}Press Ctrl+C at any time to stop.{Style.RESET_ALL}\n')
    
    try:
        evo = run_large_scale_evolution()
        
        print(f'\n{Fore.GREEN}{Style.BRIGHT}✅ All done! Results saved and visualizations generated.{Style.RESET_ALL}')
        print(f'{Fore.CYAN}Check the generated PNG files for detailed analysis.{Style.RESET_ALL}\n')
        
    except KeyboardInterrupt:
        print(f'\n\n{Fore.YELLOW}⚠️  Evolution interrupted by user.{Style.RESET_ALL}')
        print(f'{Fore.CYAN}Partial results may be available.{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'\n\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
