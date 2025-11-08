"""
Generate Training Data for ML Fitness Surrogate
==============================================
Simulate random genomes to build a training dataset.
This is better than using old data because:
1. Uses the FIXED fitness function (numerically stable)
2. Controlled genome distribution sampling
3. Fresh, high-quality data
"""

import numpy as np
import json
from pathlib import Path
from quantum_genetic_agents import QuantumAgent
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

def sample_genome_parameter(param_name):
    """Sample a genome parameter from appropriate range"""
    if param_name == 'mu':  # mutation_rate
        return np.random.uniform(1.0, 5.0)
    elif param_name == 'omega':  # oscillation_freq
        return np.random.uniform(0.1, 2.5)
    elif param_name == 'd':  # decoherence_rate
        # Focus on stable range around 0.005
        return np.random.choice([
            np.random.uniform(0.001, 0.01),   # 70% in stable range
            np.random.uniform(0.01, 0.03),    # 20% higher
            np.random.uniform(0.0001, 0.001)  # 10% very low
        ], p=[0.7, 0.2, 0.1])
    elif param_name == 'phi':  # phase_offset
        return np.random.uniform(0.0, 2 * np.pi)
    else:
        raise ValueError(f"Unknown parameter: {param_name}")

def generate_random_genome():
    """Generate a random genome with realistic parameter ranges"""
    return np.array([
        sample_genome_parameter('mu'),
        sample_genome_parameter('omega'),
        sample_genome_parameter('d'),
        sample_genome_parameter('phi')
    ])

def evaluate_genome(genome, environment='standard', timesteps=100):
    """Simulate a genome and return its fitness"""
    agent = QuantumAgent(0, genome, environment)
    
    for t in range(timesteps):
        agent.evolve(t)
    
    return agent.get_final_fitness()

def generate_training_dataset(num_samples=10000, environments=['standard'], timesteps=100):
    """
    Generate training dataset by simulating random genomes
    
    Args:
        num_samples: Number of genome-fitness pairs to generate
        environments: List of environments to test in
        timesteps: Number of timesteps per simulation
    
    Returns:
        genomes: Array of [μ, ω, d, φ] genomes
        fitnesses: Corresponding fitness values
        metadata: Additional info
    """
    print(f"\n{'='*70}")
    print("GENERATING TRAINING DATA FOR ML SURROGATE")
    print(f"{'='*70}")
    print(f"  Samples: {num_samples:,}")
    print(f"  Environments: {environments}")
    print(f"  Timesteps per simulation: {timesteps}")
    print(f"{'='*70}\n")
    
    genomes = []
    fitnesses = []
    env_labels = []
    
    samples_per_env = num_samples // len(environments)
    
    for env in environments:
        print(f"\n🌍 Generating {samples_per_env:,} samples for '{env}' environment...")
        
        for i in tqdm(range(samples_per_env), desc=f"  {env}"):
            # Generate random genome
            genome = generate_random_genome()
            
            # Evaluate fitness
            fitness = evaluate_genome(genome, env, timesteps)
            
            # Store results
            genomes.append(genome.tolist())
            fitnesses.append(float(fitness))
            env_labels.append(env)
    
    # Convert to arrays
    genomes = np.array(genomes)
    fitnesses = np.array(fitnesses)
    
    # Calculate statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"  Total samples: {len(genomes):,}")
    print(f"\n  Fitness statistics:")
    print(f"    Mean: {np.mean(fitnesses):.6f}")
    print(f"    Std: {np.std(fitnesses):.6f}")
    print(f"    Min: {np.min(fitnesses):.6f}")
    print(f"    Max: {np.max(fitnesses):.6f}")
    print(f"    Median: {np.median(fitnesses):.6f}")
    
    print(f"\n  Genome parameter ranges:")
    for i, param in enumerate(['μ (mutation)', 'ω (oscillation)', 'd (decoherence)', 'φ (phase)']):
        print(f"    {param:25s}: [{genomes[:, i].min():.4f}, {genomes[:, i].max():.4f}]")
    
    # Check for invalid values
    invalid_count = np.sum(~np.isfinite(fitnesses))
    if invalid_count > 0:
        print(f"\n  ⚠ Warning: {invalid_count} invalid fitness values detected")
        # Remove invalid values
        valid_mask = np.isfinite(fitnesses)
        genomes = genomes[valid_mask]
        fitnesses = fitnesses[valid_mask]
        env_labels = [e for e, v in zip(env_labels, valid_mask) if v]
        print(f"  ✓ Removed invalid values, {len(genomes):,} samples remaining")
    
    print(f"{'='*70}\n")
    
    metadata = {
        'num_samples': len(genomes),
        'environments': environments,
        'timesteps': timesteps,
        'fitness_stats': {
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses)),
            'median': float(np.median(fitnesses))
        },
        'genome_ranges': {
            'mu': [float(genomes[:, 0].min()), float(genomes[:, 0].max())],
            'omega': [float(genomes[:, 1].min()), float(genomes[:, 1].max())],
            'd': [float(genomes[:, 2].min()), float(genomes[:, 2].max())],
            'phi': [float(genomes[:, 3].min()), float(genomes[:, 3].max())]
        }
    }
    
    return genomes, fitnesses, env_labels, metadata

def plot_dataset_distribution(genomes, fitnesses, save_path):
    """Visualize the generated dataset"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    param_names = ['μ (mutation)', 'ω (oscillation)', 'd (decoherence)', 'φ (phase)']
    
    # Plot each genome parameter
    for i, (ax, param) in enumerate(zip(axes[0, :], param_names)):
        ax.hist(genomes[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of {param}', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    # Plot fitness distribution
    axes[1, 0].hist(fitnesses, bins=100, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Fitness', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Fitness Distribution', fontsize=13)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot fitness vs decoherence (key parameter)
    axes[1, 1].scatter(genomes[:, 2], fitnesses, alpha=0.3, s=10)
    axes[1, 1].set_xlabel('d (decoherence)', fontsize=12)
    axes[1, 1].set_ylabel('Fitness', fontsize=12)
    axes[1, 1].set_title('Fitness vs Decoherence Rate', fontsize=13)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot fitness vs mutation rate
    axes[1, 2].scatter(genomes[:, 0], fitnesses, alpha=0.3, s=10, color='purple')
    axes[1, 2].set_xlabel('μ (mutation)', fontsize=12)
    axes[1, 2].set_ylabel('Fitness', fontsize=12)
    axes[1, 2].set_title('Fitness vs Mutation Rate', fontsize=13)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Dataset visualization saved to: {save_path}")
    plt.close()

def save_training_data(genomes, fitnesses, env_labels, metadata, filename):
    """Save training data to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata,
        'data': [
            {
                'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome,
                'fitness': float(fitness),
                'environment': env
            }
            for genome, fitness, env in zip(genomes, fitnesses, env_labels)
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size = Path(filename).stat().st_size / 1024  # KB
    print(f"✓ Training data saved to: {filename}")
    print(f"  File size: {file_size:.1f} KB")

def main():
    # Configuration
    NUM_SAMPLES = 10000
    ENVIRONMENTS = ['standard']  # Can add more: ['standard', 'gentle', 'harsh']
    TIMESTEPS = 100
    
    script_dir = Path(__file__).parent
    
    # Generate dataset
    genomes, fitnesses, env_labels, metadata = generate_training_dataset(
        num_samples=NUM_SAMPLES,
        environments=ENVIRONMENTS,
        timesteps=TIMESTEPS
    )
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = script_dir / f'ml_training_data_{timestamp}.json'
    save_training_data(genomes, fitnesses, env_labels, metadata, data_filename)
    
    # Visualize
    plot_filename = script_dir / f'training_data_distribution_{timestamp}.png'
    plot_dataset_distribution(genomes, fitnesses, plot_filename)
    
    print(f"\n{'='*70}")
    print("✓ TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Data file: {data_filename.name}")
    print(f"  Plot file: {plot_filename.name}")
    print(f"  Total samples: {len(genomes):,}")
    print(f"  Ready for ML training!")
    print(f"{'='*70}\n")
    
    return genomes, fitnesses, env_labels, metadata

if __name__ == '__main__':
    genomes, fitnesses, env_labels, metadata = main()
