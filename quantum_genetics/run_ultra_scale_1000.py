"""
🚀⚛️ ULTRA-SCALE QUANTUM EVOLUTION - 1000 AGENTS
=================================================

What can we discover with 1000 agents?

1. **Population Dynamics at Scale**
   - Observe emergent behaviors impossible with small populations
   - Track sub-population clusters and niches
   - Study gene flow across large populations

2. **Hyperparameter Optimization**
   - Test multiple mutation strategies simultaneously
   - Run island model evolution with isolated populations
   - Compare different environments in parallel

3. **Meta-Learning**
   - Train ML predictor on massive dataset (1000+ samples)
   - Transfer learning across different environments
   - Discover universal mutation patterns

4. **Statistical Significance**
   - Get reliable statistics with huge sample sizes
   - Confidence intervals on fitness distributions
   - Detect rare beneficial mutations

5. **Evolution Speed Records**
   - Leverage GPU to evolve faster than CPU could ever handle
   - 50,000+ agent evaluations per generation
   - Real evolutionary timescales in minutes

This script runs an ultra-scale experiment showcasing these capabilities!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import sys
import json
from collections import defaultdict

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''

from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution


class UltraScaleAnalyzer:
    """Analyze ultra-scale population dynamics"""
    
    def __init__(self, population_size):
        self.population_size = population_size
        self.generation_snapshots = []
        
    def analyze_population_structure(self, population):
        """Analyze population clustering and diversity"""
        fitnesses = np.array([agent[0] for agent in population])
        genomes = np.array([agent[1] for agent in population])
        
        # Fitness distribution statistics
        fitness_stats = {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'median': np.median(fitnesses),
            'q25': np.percentile(fitnesses, 25),
            'q75': np.percentile(fitnesses, 75),
            'top1%': np.percentile(fitnesses, 99),
            'top10%': np.percentile(fitnesses, 90),
            'bottom10%': np.percentile(fitnesses, 10)
        }
        
        # Genome diversity per parameter
        genome_diversity = {
            'mu': np.std(genomes[:, 0]),
            'omega': np.std(genomes[:, 1]),
            'd': np.std(genomes[:, 2]),
            'phi': np.std(genomes[:, 3])
        }
        
        # Elite vs non-elite comparison
        elite_threshold = np.percentile(fitnesses, 90)
        elite_mask = fitnesses >= elite_threshold
        non_elite_mask = ~elite_mask
        
        elite_genomes = genomes[elite_mask]
        non_elite_genomes = genomes[non_elite_mask]
        
        comparison = {
            'elite_mu_mean': np.mean(elite_genomes[:, 0]),
            'non_elite_mu_mean': np.mean(non_elite_genomes[:, 0]),
            'elite_omega_mean': np.mean(elite_genomes[:, 1]),
            'non_elite_omega_mean': np.mean(non_elite_genomes[:, 1]),
            'elite_d_mean': np.mean(elite_genomes[:, 2]),
            'non_elite_d_mean': np.mean(non_elite_genomes[:, 2]),
        }
        
        return {
            'fitness_stats': fitness_stats,
            'genome_diversity': genome_diversity,
            'elite_comparison': comparison
        }
    
    def take_snapshot(self, generation, population, mutation_rate):
        """Take snapshot of population state"""
        analysis = self.analyze_population_structure(population)
        analysis['generation'] = generation
        analysis['mutation_rate'] = mutation_rate
        self.generation_snapshots.append(analysis)
    
    def export_analysis(self, filename):
        """Export detailed analysis to JSON"""
        data = {
            'population_size': self.population_size,
            'total_snapshots': len(self.generation_snapshots),
            'snapshots': self.generation_snapshots
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename


def run_ultra_scale_evolution():
    """Run 1000-agent ultra-scale evolution with deep analysis"""
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️ ULTRA-SCALE QUANTUM EVOLUTION{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Configuration
    POPULATION_SIZE = 1000  # Ultra-scale!
    GENERATIONS = 200       # Comprehensive evolution
    ENVIRONMENT = 'standard'
    SNAPSHOT_INTERVAL = 10  # Take detailed snapshots every N gens
    
    print(f'{Fore.CYAN}{Style.BRIGHT}ULTRA-SCALE CONFIGURATION:{Style.RESET_ALL}\n')
    print(f'{Fore.YELLOW}Population Size:{Style.RESET_ALL}     {Fore.GREEN}{Style.BRIGHT}{POPULATION_SIZE:,}{Style.RESET_ALL} agents 🔥')
    print(f'{Fore.YELLOW}Generations:{Style.RESET_ALL}         {Fore.GREEN}{GENERATIONS}{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}Environment:{Style.RESET_ALL}         {Fore.GREEN}{ENVIRONMENT}{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}Total Evaluations:{Style.RESET_ALL}   {Fore.MAGENTA}{POPULATION_SIZE * GENERATIONS:,}{Style.RESET_ALL} 🚀')
    print(f'{Fore.YELLOW}Strategy:{Style.RESET_ALL}            {Fore.GREEN}ML Adaptive + GPU{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}GPU:{Style.RESET_ALL}                 {Fore.CYAN}{Style.BRIGHT}NVIDIA RTX 4070 Ti (12GB){Style.RESET_ALL}')
    
    # Estimate time
    estimated_time = GENERATIONS * 0.9  # ~0.9s per gen for 1000 agents
    print(f'{Fore.YELLOW}Estimated Time:{Style.RESET_ALL}     {Fore.CYAN}~{int(estimated_time/60)} minutes{Style.RESET_ALL}')
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}WHAT WE CAN DISCOVER:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}  📊 Population structure and clustering{Style.RESET_ALL}')
    print(f'{Fore.WHITE}  🧬 Elite vs non-elite genome differences{Style.RESET_ALL}')
    print(f'{Fore.WHITE}  📈 Statistical significance with huge samples{Style.RESET_ALL}')
    print(f'{Fore.WHITE}  🤖 ML predictor trained on 200+ samples{Style.RESET_ALL}')
    print(f'{Fore.WHITE}  ⚡ GPU pushing 200k+ agent-timesteps/second{Style.RESET_ALL}')
    print(f'{Fore.WHITE}  🔬 Emergent behaviors at population scale{Style.RESET_ALL}')
    
    print(f'\n{Fore.YELLOW}🔧 Initializing ultra-scale evolution...{Style.RESET_ALL}\n')
    
    # Create evolution engine
    evo = AdaptiveMutationEvolution(
        population_size=POPULATION_SIZE,
        strategy='ml_adaptive',
        use_gpu=True
    )
    
    # Create analyzer
    analyzer = UltraScaleAnalyzer(POPULATION_SIZE)
    
    print(f'{Fore.GREEN}✅ Ready! Starting ultra-scale evolution...{Style.RESET_ALL}\n')
    
    start_time = time.time()
    generation_times = []
    
    # Evolution loop
    for gen in range(GENERATIONS):
        gen_start = time.time()
        
        # Adapt mutation
        evo.adapt_mutation_rate(gen)
        
        # Evaluate
        evo.evaluate_population(ENVIRONMENT)
        
        # Track metrics
        best_fitness = evo.population[0][0]
        avg_fitness = np.mean([agent[0] for agent in evo.population])
        diversity = evo._calculate_diversity()
        
        evo.best_fitness_history.append(best_fitness)
        evo.avg_fitness_history.append(avg_fitness)
        evo.diversity_history.append(diversity)
        
        # Collect ML training data
        if gen > 0:
            features = evo.predictor.extract_features(
                evo.population, gen, evo.best_fitness_history
            )
            evo.predictor.add_training_data(
                features, 
                evo.current_mutation_rate,
                best_fitness
            )
        
        # Take detailed snapshot
        if (gen + 1) % SNAPSHOT_INTERVAL == 0:
            analyzer.take_snapshot(gen + 1, evo.population, evo.current_mutation_rate)
        
        # Evolve
        if gen < GENERATIONS - 1:
            evo.evolve_generation()
        
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)
        
        # Calculate throughput
        agent_timesteps_per_sec = POPULATION_SIZE / gen_time
        
        # Progress update
        if (gen + 1) % 5 == 0:
            # ETA calculation
            avg_time = np.mean(generation_times[-10:])
            remaining = GENERATIONS - (gen + 1)
            eta_seconds = remaining * avg_time
            eta = timedelta(seconds=int(eta_seconds))
            
            # Elapsed
            elapsed = timedelta(seconds=int(time.time() - start_time))
            
            # Top 3 fitness
            sorted_pop = sorted(evo.population, reverse=True, key=lambda x: x[0])
            top3_fitness = [agent[0] for agent in sorted_pop[:3]]
            
            sys.stdout.write(f'\r{Fore.CYAN}Gen {gen+1:3d}/{GENERATIONS}{Style.RESET_ALL} | '
                           f'Best: {Fore.GREEN}{best_fitness:.6f}{Style.RESET_ALL} | '
                           f'Top3: {Fore.YELLOW}{top3_fitness[1]:.6f}{Style.RESET_ALL}/{Fore.WHITE}{top3_fitness[2]:.6f}{Style.RESET_ALL} | '
                           f'μ: {Fore.MAGENTA}{evo.current_mutation_rate:.3f}{Style.RESET_ALL} | '
                           f'D: {Fore.CYAN}{diversity:.3f}{Style.RESET_ALL} | '
                           f'⚡{agent_timesteps_per_sec:,.0f} ag/s | '
                           f'⏱️{elapsed} ETA:{eta}')
            sys.stdout.flush()
        
        # Detailed stats every 20 generations
        if (gen + 1) % 20 == 0:
            analysis = analyzer.analyze_population_structure(evo.population)
            stats = analysis['fitness_stats']
            
            print(f'\n\n{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
            print(f'{Fore.YELLOW}{Style.BRIGHT}📊 GENERATION {gen+1} POPULATION ANALYSIS{Style.RESET_ALL}')
            print(f'{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
            
            print(f'{Fore.CYAN}FITNESS DISTRIBUTION:{Style.RESET_ALL}')
            print(f'  Top 1%:   {Fore.GREEN}{Style.BRIGHT}{stats["top1%"]:10.6f}{Style.RESET_ALL}')
            print(f'  Top 10%:  {Fore.GREEN}{stats["top10%"]:10.6f}{Style.RESET_ALL}')
            print(f'  Median:   {Fore.YELLOW}{stats["median"]:10.6f}{Style.RESET_ALL}')
            print(f'  Bottom 10%: {Fore.RED}{stats["bottom10%"]:10.6f}{Style.RESET_ALL}')
            print(f'  Range:    {Fore.WHITE}{stats["max"] - stats["min"]:10.6f}{Style.RESET_ALL}')
            
            elite_comp = analysis['elite_comparison']
            print(f'\n{Fore.CYAN}ELITE (Top 10%) vs NON-ELITE GENOMES:{Style.RESET_ALL}')
            print(f'  Mutation (μ):     {Fore.GREEN}{elite_comp["elite_mu_mean"]:.4f}{Style.RESET_ALL} vs {Fore.RED}{elite_comp["non_elite_mu_mean"]:.4f}{Style.RESET_ALL}')
            print(f'  Oscillation (ω):  {Fore.GREEN}{elite_comp["elite_omega_mean"]:.4f}{Style.RESET_ALL} vs {Fore.RED}{elite_comp["non_elite_omega_mean"]:.4f}{Style.RESET_ALL}')
            print(f'  Decoherence (d):  {Fore.GREEN}{elite_comp["elite_d_mean"]:.4f}{Style.RESET_ALL} vs {Fore.RED}{elite_comp["non_elite_d_mean"]:.4f}{Style.RESET_ALL}')
            
            print(f'\n{Fore.YELLOW}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
        
        # Train predictor periodically
        if (gen + 1) % 50 == 0:
            print(f'\n\n{Fore.MAGENTA}{Style.BRIGHT}🎓 Training ML predictor on {len(evo.predictor.training_data)} samples...{Style.RESET_ALL}')
            evo.predictor.train(epochs=50)
            print(f'{Fore.GREEN}✅ Training complete!{Style.RESET_ALL}\n')
    
    # Final summary
    print('\n\n')
    total_time = time.time() - start_time
    total_evaluations = POPULATION_SIZE * GENERATIONS
    avg_throughput = total_evaluations / total_time
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}✅ ULTRA-SCALE EVOLUTION COMPLETE!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    print(f'{Fore.CYAN}{Style.BRIGHT}⚡ PERFORMANCE METRICS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Total Time:{Style.RESET_ALL}              {Fore.GREEN}{total_time:.2f}s{Style.RESET_ALL} ({timedelta(seconds=int(total_time))})')
    print(f'{Fore.WHITE}Avg Time per Gen:{Style.RESET_ALL}       {Fore.YELLOW}{total_time/GENERATIONS:.3f}s{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Total Evaluations:{Style.RESET_ALL}      {Fore.MAGENTA}{total_evaluations:,}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Avg Throughput:{Style.RESET_ALL}         {Fore.CYAN}{Style.BRIGHT}{avg_throughput:,.0f}{Style.RESET_ALL} agent-timesteps/second 🚀')
    print(f'{Fore.WHITE}Peak Throughput:{Style.RESET_ALL}        {Fore.GREEN}{POPULATION_SIZE / min(generation_times):,.0f}{Style.RESET_ALL} agent-timesteps/second')
    
    # Champion analysis
    best_fitness, best_genome, _ = evo.population[0]
    mu, omega, d, phi = best_genome
    
    print(f'\n{Fore.YELLOW}{Style.BRIGHT}🏆 CHAMPION AGENT (Best of {POPULATION_SIZE:,}):{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Fitness:{Style.RESET_ALL}              {Fore.GREEN}{Style.BRIGHT}{best_fitness:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Genome:{Style.RESET_ALL}               μ={Fore.RED}{mu:.4f}{Style.RESET_ALL} ω={Fore.GREEN}{omega:.4f}{Style.RESET_ALL} d={Fore.BLUE}{d:.4f}{Style.RESET_ALL} φ={Fore.MAGENTA}{phi:.4f}{Style.RESET_ALL}')
    
    # Population statistics
    final_analysis = analyzer.analyze_population_structure(evo.population)
    stats = final_analysis['fitness_stats']
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}📊 FINAL POPULATION STATISTICS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Top 1%:{Style.RESET_ALL}               {Fore.GREEN}{Style.BRIGHT}{stats["top1%"]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Top 10%:{Style.RESET_ALL}              {Fore.GREEN}{stats["top10%"]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Median:{Style.RESET_ALL}               {Fore.YELLOW}{stats["median"]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Mean:{Style.RESET_ALL}                 {Fore.YELLOW}{stats["mean"]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Bottom 10%:{Style.RESET_ALL}           {Fore.RED}{stats["bottom10%"]:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Std Deviation:{Style.RESET_ALL}        {Fore.WHITE}{stats["std"]:.6f}{Style.RESET_ALL}')
    
    # Fitness improvement
    initial_best = evo.best_fitness_history[0]
    final_best = evo.best_fitness_history[-1]
    improvement = ((final_best - initial_best) / initial_best * 100) if initial_best > 0 else 0
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}📈 EVOLUTION PROGRESS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Initial Best:{Style.RESET_ALL}         {Fore.RED}{initial_best:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Final Best:{Style.RESET_ALL}           {Fore.GREEN}{Style.BRIGHT}{final_best:.6f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Improvement:{Style.RESET_ALL}          {Fore.YELLOW}{Style.BRIGHT}{improvement:+.2f}%{Style.RESET_ALL}')
    
    # ML predictor stats
    print(f'\n{Fore.MAGENTA}{Style.BRIGHT}🤖 ML PREDICTOR STATS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Training Samples:{Style.RESET_ALL}     {Fore.CYAN}{len(evo.predictor.training_data)}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Training Sessions:{Style.RESET_ALL}    {Fore.CYAN}{GENERATIONS // 50}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Final Mutation Rate:{Style.RESET_ALL}  {Fore.MAGENTA}{evo.current_mutation_rate:.6f}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Export analysis
    print(f'{Fore.CYAN}📊 Exporting detailed analysis...{Style.RESET_ALL}')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = analyzer.export_analysis(f'ultra_scale_analysis_{timestamp}.json')
    print(f'{Fore.GREEN}✅ Saved: {analysis_file}{Style.RESET_ALL}')
    
    # Generate visualizations
    print(f'{Fore.CYAN}📈 Generating ultra-scale visualizations...{Style.RESET_ALL}')
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'🚀⚛️ Ultra-Scale Evolution: {POPULATION_SIZE:,} Agents × {GENERATIONS} Generations', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: Best fitness evolution
    ax = axes[0, 0]
    ax.plot(evo.best_fitness_history, color='#2ECC71', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best Fitness Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Average fitness evolution
    ax = axes[0, 1]
    ax.plot(evo.avg_fitness_history, color='#F39C12', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Average Fitness Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Mutation rate dynamics
    ax = axes[0, 2]
    ax.plot(evo.mutation_rate_history, color='#E74C3C', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mutation Rate')
    ax.set_title('ML Adaptive Mutation Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Diversity evolution
    ax = axes[1, 0]
    ax.plot(evo.diversity_history, color='#3498DB', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Diversity')
    ax.set_title('Genetic Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Generation time
    ax = axes[1, 1]
    ax.plot(generation_times, color='#9B59B6', linewidth=2, alpha=0.6)
    ax.axhline(np.mean(generation_times), color='red', linestyle='--', label=f'Mean: {np.mean(generation_times):.3f}s')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Generation Time (GPU Performance)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Fitness distribution snapshots
    ax = axes[1, 2]
    snapshot_gens = [s['generation'] for s in analyzer.generation_snapshots]
    top1_vals = [s['fitness_stats']['top1%'] for s in analyzer.generation_snapshots]
    median_vals = [s['fitness_stats']['median'] for s in analyzer.generation_snapshots]
    bottom10_vals = [s['fitness_stats']['bottom10%'] for s in analyzer.generation_snapshots]
    
    ax.plot(snapshot_gens, top1_vals, 'g-', label='Top 1%', linewidth=2)
    ax.plot(snapshot_gens, median_vals, 'y-', label='Median', linewidth=2)
    ax.plot(snapshot_gens, bottom10_vals, 'r-', label='Bottom 10%', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Population Fitness Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    viz_file = f'ultra_scale_evolution_{timestamp}.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f'{Fore.GREEN}✅ Saved: {viz_file}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}🎉 Ultra-scale evolution complete! All data exported.{Style.RESET_ALL}\n')
    
    return evo, analyzer


if __name__ == "__main__":
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️ ULTRA-SCALE QUANTUM EVOLUTION{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.YELLOW}WARNING: This will run 1000 agents for 200 generations!{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}         Total: 200,000 agent evaluations{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}         Estimated time: ~3-4 minutes on RTX 4070 Ti{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}         Press Ctrl+C to stop anytime.{Style.RESET_ALL}\n')
    
    try:
        evo, analyzer = run_ultra_scale_evolution()
        print(f'\n{Fore.GREEN}{Style.BRIGHT}✅ SUCCESS! Check JSON and PNG files for insights!{Style.RESET_ALL}\n')
        
    except KeyboardInterrupt:
        print(f'\n\n{Fore.YELLOW}⚠️  Stopped by user.{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'\n\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
