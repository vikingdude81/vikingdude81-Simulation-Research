"""
🚀⚛️ LARGE-SCALE GPU EVOLUTION - SMOOTH VERSION (NO FLICKER)
============================================================

Compact real-time dashboard with minimal screen updates:
- Single-line progress updates
- Detailed stats every 10 generations
- No flickering or blinking
"""

import numpy as np
import time
from datetime import datetime, timedelta
import sys
import os

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution


class SmoothDashboard:
    """Smooth, flicker-free terminal dashboard"""
    
    def __init__(self, population_size, total_generations):
        self.population_size = population_size
        self.total_generations = total_generations
        self.start_time = None
        self.generation_times = []
        self.last_full_update = 0
    
    def genome_to_emoji(self, genome):
        """Convert genome to emoji representation"""
        mu, omega, d, phi = genome
        
        # Map to emojis based on intensity
        emojis = []
        
        # μ (mutation): 🔥=high, 🌡️=med, ❄️=low
        if mu > 2.0:
            emojis.append('🔥')
        elif mu > 1.0:
            emojis.append('🌡️')
        else:
            emojis.append('❄️')
        
        # ω (oscillation): ⚡=fast, ⚙️=med, 🐌=slow
        if omega > 1.5:
            emojis.append('⚡')
        elif omega > 0.8:
            emojis.append('⚙️')
        else:
            emojis.append('🐌')
        
        # d (decoherence): 💎=stable, 🌊=med, 💨=volatile
        if d < 0.02:
            emojis.append('💎')
        elif d < 0.05:
            emojis.append('🌊')
        else:
            emojis.append('💨')
        
        return ''.join(emojis)
    
    def compact_progress(self, generation, evo):
        """Single-line compact progress update"""
        # Calculate metrics
        ratio = generation / self.total_generations
        percentage = ratio * 100
        
        # ETA calculation
        if self.generation_times and generation > 0:
            avg_time = np.mean(self.generation_times[-10:])
            remaining = self.total_generations - generation
            eta_seconds = remaining * avg_time
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "calculating"
        
        # Elapsed time
        if self.start_time:
            elapsed = timedelta(seconds=int(time.time() - self.start_time))
            elapsed_str = str(elapsed).split('.')[0]
        else:
            elapsed_str = "00:00:00"
        
        # Best fitness and mutation rate
        best_fitness = evo.population[0][0]
        mutation_rate = evo.current_mutation_rate
        diversity = evo._calculate_diversity()
        
        # Champion DNA
        champion_dna = self.genome_to_emoji(evo.population[0][1])
        
        # Build compact progress bar
        bar_width = 25
        filled = int(ratio * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Color fitness based on magnitude
        if best_fitness > 1.0:
            fit_color = Fore.RED
        elif best_fitness > 0.01:
            fit_color = Fore.YELLOW
        elif best_fitness > 0.0001:
            fit_color = Fore.GREEN
        else:
            fit_color = Fore.CYAN
        
        # Compact one-line status
        line = (f'\r{Fore.CYAN}{Style.BRIGHT}GEN {generation:3d}/{self.total_generations}{Style.RESET_ALL} '
                f'{Fore.GREEN}{bar}{Style.RESET_ALL} {percentage:5.1f}% | '
                f'⏱️{elapsed_str} ETA:{eta_str} | '
                f'🏆{champion_dna} {fit_color}{best_fitness:.6f}{Style.RESET_ALL} | '
                f'{Fore.MAGENTA}μ={mutation_rate:.3f}{Style.RESET_ALL} '
                f'{Fore.YELLOW}D={diversity:.3f}{Style.RESET_ALL}')
        
        sys.stdout.write(line + ' ' * 10)  # Extra spaces to clear any leftover text
        sys.stdout.flush()
    
    def full_update(self, generation, evo):
        """Detailed update every N generations"""
        print('\n')  # New line after progress bar
        print(f'{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
        print(f'{Fore.YELLOW}{Style.BRIGHT}📊 GENERATION {generation} DETAILED STATS{Style.RESET_ALL}')
        print(f'{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
        
        # Top 5 agents
        sorted_pop = sorted(evo.population, reverse=True, key=lambda x: x[0])[:5]
        print(f'{Fore.CYAN}{Style.BRIGHT}🏆 TOP 5 AGENTS:{Style.RESET_ALL}\n')
        
        for i, (fitness, genome, agent_id) in enumerate(sorted_pop, 1):
            mu, omega, d, phi = genome
            dna = self.genome_to_emoji(genome)
            
            if i == 1:
                rank_color = Fore.YELLOW + Style.BRIGHT
            elif i == 2:
                rank_color = Fore.WHITE + Style.BRIGHT
            else:
                rank_color = Fore.CYAN
            
            print(f'{rank_color}#{i}{Style.RESET_ALL} {dna} '
                  f'Fitness: {Fore.GREEN}{fitness:10.6f}{Style.RESET_ALL} | '
                  f'{Fore.RED}μ={mu:.3f}{Style.RESET_ALL} '
                  f'{Fore.GREEN}ω={omega:.3f}{Style.RESET_ALL} '
                  f'{Fore.BLUE}d={d:.4f}{Style.RESET_ALL} '
                  f'{Fore.MAGENTA}φ={phi:.3f}{Style.RESET_ALL}')
        
        # Population statistics
        fitnesses = [agent[0] for agent in evo.population]
        print(f'\n{Fore.CYAN}{Style.BRIGHT}📈 POPULATION STATS:{Style.RESET_ALL}\n')
        print(f'Best:    {Fore.GREEN}{max(fitnesses):10.6f}{Style.RESET_ALL}')
        print(f'Average: {Fore.YELLOW}{np.mean(fitnesses):10.6f}{Style.RESET_ALL}')
        print(f'Worst:   {Fore.RED}{min(fitnesses):10.6f}{Style.RESET_ALL}')
        print(f'Std Dev: {Fore.WHITE}{np.std(fitnesses):10.6f}{Style.RESET_ALL}')
        
        # Timing
        if self.generation_times:
            avg_time = np.mean(self.generation_times[-10:])
            print(f'\n{Fore.CYAN}{Style.BRIGHT}⏱️  TIMING:{Style.RESET_ALL}\n')
            print(f'Last Gen:     {Fore.GREEN}{self.generation_times[-1]:.3f}s{Style.RESET_ALL}')
            print(f'Avg (last 10): {Fore.YELLOW}{avg_time:.3f}s{Style.RESET_ALL}')
        
        print(f'\n{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')


def run_smooth_evolution():
    """Run large-scale evolution with smooth, flicker-free updates"""
    
    # Configuration
    POPULATION_SIZE = 200
    GENERATIONS = 100
    ENVIRONMENT = 'standard'
    UPDATE_INTERVAL = 10  # Full stats every N generations
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️  LARGE-SCALE QUANTUM EVOLUTION (SMOOTH MODE){Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.CYAN}Population:{Style.RESET_ALL} {POPULATION_SIZE} agents')
    print(f'{Fore.CYAN}Generations:{Style.RESET_ALL} {GENERATIONS}')
    print(f'{Fore.CYAN}Environment:{Style.RESET_ALL} {ENVIRONMENT}')
    print(f'{Fore.CYAN}Strategy:{Style.RESET_ALL} ML Adaptive + GPU')
    print(f'{Fore.CYAN}Update Interval:{Style.RESET_ALL} Every {UPDATE_INTERVAL} generations')
    print(f'\n{Fore.YELLOW}🔧 Initializing...{Style.RESET_ALL}\n')
    
    # Create evolution engine
    evo = AdaptiveMutationEvolution(
        population_size=POPULATION_SIZE,
        strategy='ml_adaptive',
        use_gpu=True
    )
    
    # Create dashboard
    dashboard = SmoothDashboard(POPULATION_SIZE, GENERATIONS)
    dashboard.start_time = time.time()
    
    print(f'{Fore.GREEN}✅ Ready! Starting evolution...{Style.RESET_ALL}\n')
    time.sleep(1)
    
    # Custom evolution loop
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
        
        # Collect training data
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
        if gen < GENERATIONS - 1:
            evo.evolve_generation()
        
        gen_time = time.time() - gen_start
        dashboard.generation_times.append(gen_time)
        
        # Compact progress update (every generation)
        dashboard.compact_progress(gen + 1, evo)
        
        # Full update (every N generations)
        if (gen + 1) % UPDATE_INTERVAL == 0 or gen == GENERATIONS - 1:
            dashboard.full_update(gen + 1, evo)
        
        # Train predictor periodically
        if evo.strategy == 'ml_adaptive' and (gen + 1) % 50 == 0:
            print(f'\n{Fore.MAGENTA}{Style.BRIGHT}🎓 Training ML predictor...{Style.RESET_ALL}')
            evo.predictor.train(epochs=30)
            print(f'{Fore.GREEN}✅ Training complete!{Style.RESET_ALL}\n')
    
    # Final summary
    print('\n')
    total_time = time.time() - dashboard.start_time
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}✅ EVOLUTION COMPLETE!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Performance metrics
    print(f'{Fore.CYAN}{Style.BRIGHT}⏱️  PERFORMANCE:{Style.RESET_ALL}\n')
    print(f'Total Time:        {Fore.GREEN}{total_time:.2f}s{Style.RESET_ALL} ({timedelta(seconds=int(total_time))})')
    print(f'Avg Time per Gen:  {Fore.YELLOW}{total_time/GENERATIONS:.3f}s{Style.RESET_ALL}')
    print(f'Generations:       {Fore.CYAN}{GENERATIONS}{Style.RESET_ALL}')
    print(f'Population Size:   {Fore.CYAN}{POPULATION_SIZE}{Style.RESET_ALL}')
    print(f'Total Simulations: {Fore.MAGENTA}{GENERATIONS * POPULATION_SIZE:,}{Style.RESET_ALL}')
    
    # Champion
    best_fitness, best_genome, _ = evo.population[0]
    dna = dashboard.genome_to_emoji(best_genome)
    mu, omega, d, phi = best_genome
    
    print(f'\n{Fore.YELLOW}{Style.BRIGHT}🏆 CHAMPION AGENT:{Style.RESET_ALL}\n')
    print(f'DNA:      {dna}')
    print(f'Fitness:  {Fore.GREEN}{Style.BRIGHT}{best_fitness:.6f}{Style.RESET_ALL}')
    print(f'Genome:   {Fore.RED}μ={mu:.6f}{Style.RESET_ALL} '
          f'{Fore.GREEN}ω={omega:.6f}{Style.RESET_ALL} '
          f'{Fore.BLUE}d={d:.6f}{Style.RESET_ALL} '
          f'{Fore.MAGENTA}φ={phi:.6f}{Style.RESET_ALL}')
    
    # Improvement
    initial_fitness = evo.best_fitness_history[0]
    final_fitness = evo.best_fitness_history[-1]
    improvement = ((final_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}📈 FITNESS IMPROVEMENT:{Style.RESET_ALL}\n')
    print(f'Initial:      {Fore.RED}{initial_fitness:.6f}{Style.RESET_ALL}')
    print(f'Final:        {Fore.GREEN}{Style.BRIGHT}{final_fitness:.6f}{Style.RESET_ALL}')
    print(f'Improvement:  {Fore.YELLOW}{Style.BRIGHT}{improvement:+.2f}%{Style.RESET_ALL}')
    
    # Mutation adaptation
    print(f'\n{Fore.MAGENTA}{Style.BRIGHT}🧬 MUTATION RATE DYNAMICS:{Style.RESET_ALL}\n')
    print(f'Initial:  {Fore.RED}{evo.mutation_rate_history[0]:.6f}{Style.RESET_ALL}')
    print(f'Final:    {Fore.GREEN}{evo.mutation_rate_history[-1]:.6f}{Style.RESET_ALL}')
    print(f'Min:      {Fore.CYAN}{min(evo.mutation_rate_history):.6f}{Style.RESET_ALL}')
    print(f'Max:      {Fore.YELLOW}{max(evo.mutation_rate_history):.6f}{Style.RESET_ALL}')
    print(f'Range:    {Fore.WHITE}{max(evo.mutation_rate_history) - min(evo.mutation_rate_history):.6f}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Visualize
    print(f'{Fore.CYAN}📊 Generating visualizations...{Style.RESET_ALL}')
    evo.visualize_results()
    print(f'{Fore.GREEN}✅ Done! Check the PNG files.{Style.RESET_ALL}\n')
    
    return evo


if __name__ == "__main__":
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️  SMOOTH LARGE-SCALE EVOLUTION{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.YELLOW}Single-line progress updates with detailed stats every 10 generations.{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}No flickering or screen clearing!{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}Press Ctrl+C to stop.{Style.RESET_ALL}\n')
    
    try:
        evo = run_smooth_evolution()
        print(f'\n{Fore.GREEN}{Style.BRIGHT}✅ Success! All results saved.{Style.RESET_ALL}\n')
        
    except KeyboardInterrupt:
        print(f'\n\n{Fore.YELLOW}⚠️  Stopped by user.{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'\n\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
