"""
🚀⚛️ ADAPTIVE MUTATION WITH GPU ACCELERATION & ML PREDICTION
=============================================================

Implements intelligent mutation strategies that adapt based on:
1. Population diversity (self-adjusting rates)
2. Fitness progress (increase when stagnating, decrease when improving)
3. ML prediction (neural network predicts optimal mutation rate)
4. GPU acceleration (parallel fitness evaluation & evolution)

Key Features:
- Adaptive mutation rates (0.01 to 3.0 range)
- PyTorch GPU acceleration for batch processing
- Neural network mutation predictor
- Real-time performance monitoring
- Comparative analysis (fixed vs adaptive)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import json
from datetime import datetime
import time

# GPU/ML Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch available! Using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available. GPU acceleration disabled.")
    print("   Install with: pip install torch torchvision")

# Sklearn for fallback ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================================
# NEURAL NETWORK MUTATION PREDICTOR (GPU-Accelerated)
# ============================================================================

class MutationPredictorNN(nn.Module):
    """
    Neural network that predicts optimal mutation rate based on:
    - Current fitness statistics
    - Population diversity
    - Generation number
    - Recent fitness trend
    """
    def __init__(self, input_size=10, hidden_sizes=[64, 32, 16]):
        super(MutationPredictorNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output: mutation rate (0.01 to 3.0)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output 0-1, scale to 0.01-3.0
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x) * 2.99 + 0.01  # Scale to [0.01, 3.0]


class MutationRatePredictor:
    """
    Manages mutation rate prediction using either:
    1. Neural network (GPU-accelerated if available)
    2. GradientBoosting (CPU fallback)
    """
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        if self.use_gpu:
            self.model = MutationPredictorNN().to(DEVICE)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            self.scaler = None
            print("✅ Using GPU-accelerated neural network predictor")
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
            print("✅ Using CPU GradientBoosting predictor")
        
        self.training_data = []
        self.is_trained = False
        
    def extract_features(self, population, generation, fitness_history):
        """
        Extract features from current evolutionary state
        
        Returns:
        - 10D feature vector for prediction
        """
        fitnesses = [agent[0] for agent in population]  # Extract fitness values
        
        features = [
            np.mean(fitnesses),                          # Current avg fitness
            np.std(fitnesses),                           # Fitness variance
            np.max(fitnesses),                           # Best fitness
            np.min(fitnesses),                           # Worst fitness
            np.max(fitnesses) - np.min(fitnesses),      # Fitness range
            generation / 1000.0,                         # Normalized generation
            len(fitness_history),                        # History length
            np.mean(fitness_history[-10:]) if len(fitness_history) >= 10 else 0,  # Recent avg
            np.std(fitness_history[-10:]) if len(fitness_history) >= 10 else 0,   # Recent std
            self._calculate_diversity(population)        # Population diversity
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_diversity(self, population):
        """Calculate genome diversity in population"""
        genomes = [agent[1] for agent in population]
        if len(genomes) < 2:
            return 0.0
        
        diversity = 0.0
        for i in range(len(genomes)):
            for j in range(i+1, len(genomes)):
                diversity += np.linalg.norm(np.array(genomes[i]) - np.array(genomes[j]))
        
        return diversity / (len(genomes) * (len(genomes) - 1) / 2)
    
    def predict_mutation_rate(self, population, generation, fitness_history):
        """Predict optimal mutation rate"""
        features = self.extract_features(population, generation, fitness_history)
        
        if not self.is_trained:
            # Default adaptive strategy before training
            return self._adaptive_default(population, fitness_history)
        
        if self.use_gpu:
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
                prediction = self.model(features_tensor).cpu().item()
            return np.clip(prediction, 0.01, 3.0)
        else:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            return np.clip(prediction, 0.01, 3.0)
    
    def _adaptive_default(self, population, fitness_history):
        """Default adaptive strategy (no ML)"""
        if len(fitness_history) < 20:
            return 0.3  # High exploration initially
        
        recent_trend = np.mean(fitness_history[-10:]) - np.mean(fitness_history[-20:-10])
        diversity = self._calculate_diversity(population)
        
        # Increase mutation if stagnating or low diversity
        if recent_trend < 0.001 or diversity < 0.1:
            return min(2.0, 0.5 + diversity * 5)
        else:
            return max(0.1, 0.3 - recent_trend * 10)
    
    def add_training_data(self, features, optimal_mutation_rate, resulting_fitness):
        """Store training data (features -> mutation rate -> fitness improvement)"""
        self.training_data.append({
            'features': features,
            'mutation_rate': optimal_mutation_rate,
            'fitness': resulting_fitness
        })
    
    def train(self, epochs=50):
        """Train the predictor on collected data"""
        if len(self.training_data) < 50:
            print(f"⚠️  Not enough training data ({len(self.training_data)} samples). Need at least 50.")
            return
        
        print(f"\n🎓 Training mutation predictor on {len(self.training_data)} samples...")
        
        # Prepare data
        X = np.array([d['features'] for d in self.training_data])
        y = np.array([d['mutation_rate'] for d in self.training_data])
        
        if self.use_gpu:
            self._train_neural_network(X, y, epochs)
        else:
            self._train_gradient_boosting(X, y)
        
        self.is_trained = True
        print("✅ Training complete!")
    
    def _train_neural_network(self, X, y, epochs):
        """Train neural network (GPU)"""
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    def _train_gradient_boosting(self, X, y):
        """Train gradient boosting (CPU)"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        score = self.model.score(X_scaled, y)
        print(f"   R² Score: {score:.4f}")


# ============================================================================
# GPU-ACCELERATED QUANTUM AGENT (Batch Processing)
# ============================================================================

class QuantumAgentGPU:
    """
    GPU-accelerated version of QuantumAgent
    Processes entire populations in parallel
    """
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.device = DEVICE if self.use_gpu else None
        
    def simulate_population_batch(self, genomes, environment='standard', timesteps=80):
        """
        Simulate entire population in parallel on GPU
        
        Args:
            genomes: List of [μ, ω, d, φ] genome arrays
            environment: Environment type
            timesteps: Number of simulation steps
        
        Returns:
            List of fitness values
        """
        if not self.use_gpu:
            return self._simulate_cpu(genomes, environment, timesteps)
        
        # Convert to tensors
        genomes_tensor = torch.FloatTensor(genomes).to(self.device)  # [N, 4]
        batch_size = len(genomes)
        
        # Initialize states [N, 4] - [energy, coherence, phase, fitness]
        states = torch.zeros((batch_size, 4), device=self.device)
        states[:, 0] = 1.0  # Initial energy
        states[:, 1] = 1.0  # Initial coherence
        
        # Environment modifiers
        env_factors = {
            'standard': (1.0, 1.0),
            'harsh': (1.5, 0.8),
            'gentle': (0.7, 1.2),
            'chaotic': (1.0, 1.0),  # Will add noise
            'oscillating': (1.0, 1.0)  # Will modulate
        }
        env_factor, fitness_mod = env_factors.get(environment, (1.0, 1.0))
        
        # Extract genome parameters
        mu = genomes_tensor[:, 0]      # mutation_rate
        omega = genomes_tensor[:, 1]   # oscillation_freq
        d = genomes_tensor[:, 2]       # decoherence_rate
        phi = genomes_tensor[:, 3]     # phase_offset
        
        fitness_history = []
        
        for t in range(1, timesteps):
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            # Environment modulation
            if environment == 'chaotic':
                env_factor = 1.0 + torch.rand(1, device=self.device).item() * 0.4 - 0.2
            elif environment == 'oscillating':
                env_factor = 1.0 + 0.3 * torch.sin(t_tensor * 0.2).item()
            
            # Energy evolution: E = E * cos(ω*t*env) + μ*randn()
            noise = torch.randn(batch_size, device=self.device)
            states[:, 0] = states[:, 0] * torch.cos(omega * t_tensor * env_factor) + mu * noise * 0.1
            
            # Coherence evolution: C = C * exp(-d*t*env) + μ*randn()
            noise = torch.randn(batch_size, device=self.device)
            states[:, 1] = states[:, 1] * torch.exp(-d * t_tensor * env_factor) + mu * noise * 0.01
            states[:, 1] = torch.clamp(states[:, 1], 0.0, 1.0)
            
            # Phase evolution: θ = (θ + φ*t) mod 2π
            states[:, 2] = (states[:, 2] + phi * t_tensor) % (2 * np.pi)
            
            # Fitness: F = |E| * C * modifier
            states[:, 3] = torch.abs(states[:, 0]) * states[:, 1] * fitness_mod
            
            fitness_history.append(states[:, 3].clone())
        
        # Calculate final fitness (skip first 20 steps)
        fitness_tensor = torch.stack(fitness_history[20:])  # [timesteps-20, batch_size]
        
        avg_fitness = fitness_tensor.mean(dim=0)
        std_fitness = fitness_tensor.std(dim=0)
        stability = 1.0 / (1.0 + std_fitness)
        
        final_coherence = states[:, 1]
        coherence_decay = 1.0 - final_coherence
        longevity_penalty = torch.exp(-coherence_decay * 2.0)
        
        final_fitness = avg_fitness * stability * longevity_penalty
        
        return final_fitness.cpu().numpy().tolist()
    
    def _simulate_cpu(self, genomes, environment, timesteps):
        """CPU fallback (slower)"""
        from quantum_genetic_agents import QuantumAgent
        
        fitnesses = []
        for genome in genomes:
            agent = QuantumAgent(0, genome, environment=environment)
            for t in range(1, timesteps):
                agent.evolve(t)
            fitnesses.append(agent.get_final_fitness())
        
        return fitnesses


# ============================================================================
# ADAPTIVE MUTATION EVOLUTION ENGINE
# ============================================================================

class AdaptiveMutationEvolution:
    """
    Evolution engine with adaptive mutation rates
    
    Strategies:
    1. Fixed: Constant mutation rate (baseline)
    2. Simple Adaptive: Rule-based adaptation
    3. ML Adaptive: Neural network predicts optimal rate
    4. GPU Accelerated: Parallel population processing
    """
    def __init__(self, 
                 population_size=30,
                 strategy='ml_adaptive',  # 'fixed', 'simple_adaptive', 'ml_adaptive'
                 use_gpu=True):
        self.population_size = population_size
        self.strategy = strategy
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        # Initialize population
        self.population = []
        self.initialize_population()
        
        # ML Predictor
        if strategy == 'ml_adaptive':
            self.predictor = MutationRatePredictor(use_gpu=use_gpu)
        else:
            self.predictor = None
        
        # GPU Agent
        if self.use_gpu:
            self.gpu_agent = QuantumAgentGPU(use_gpu=True)
        else:
            self.gpu_agent = None
        
        # Tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.mutation_rate_history = []
        self.diversity_history = []
        
        self.current_mutation_rate = 0.3  # Default
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for i in range(self.population_size):
            genome = self.create_genome()
            self.population.append([0.0, genome, i])  # [fitness, genome, id]
    
    def create_genome(self):
        """Create random genome [μ, ω, d, φ]"""
        return [
            np.random.uniform(0.01, 0.3),   # mutation_rate
            np.random.uniform(0.5, 2.0),    # oscillation_freq
            np.random.uniform(0.01, 0.1),   # decoherence_rate
            np.random.uniform(0.1, 0.5)     # phase_offset
        ]
    
    def evaluate_population(self, environment='standard'):
        """Evaluate fitness for entire population"""
        if self.use_gpu and self.gpu_agent:
            # GPU batch processing
            genomes = [agent[1] for agent in self.population]
            fitnesses = self.gpu_agent.simulate_population_batch(genomes, environment)
            
            for i, fitness in enumerate(fitnesses):
                self.population[i][0] = fitness
        else:
            # CPU sequential processing
            from quantum_genetic_agents import QuantumAgent
            
            for i, agent in enumerate(self.population):
                genome = agent[1]
                q_agent = QuantumAgent(i, genome, environment=environment)
                for t in range(1, 80):
                    q_agent.evolve(t)
                self.population[i][0] = q_agent.get_final_fitness()
        
        # Sort by fitness
        self.population.sort(reverse=True, key=lambda x: x[0])
    
    def adapt_mutation_rate(self, generation):
        """Determine mutation rate based on strategy"""
        if self.strategy == 'fixed':
            self.current_mutation_rate = 0.3
            
        elif self.strategy == 'simple_adaptive':
            # Rule-based adaptation
            if len(self.best_fitness_history) < 20:
                self.current_mutation_rate = 0.5  # High initial exploration
            else:
                recent_improvement = (self.best_fitness_history[-1] - 
                                     self.best_fitness_history[-20])
                diversity = self._calculate_diversity()
                
                if recent_improvement < 0.001 or diversity < 0.1:
                    # Stagnating: increase mutation
                    self.current_mutation_rate = min(2.0, self.current_mutation_rate * 1.2)
                else:
                    # Improving: decrease mutation
                    self.current_mutation_rate = max(0.05, self.current_mutation_rate * 0.9)
        
        elif self.strategy == 'ml_adaptive':
            # ML prediction
            self.current_mutation_rate = self.predictor.predict_mutation_rate(
                self.population, generation, self.best_fitness_history
            )
        
        self.mutation_rate_history.append(self.current_mutation_rate)
    
    def _calculate_diversity(self):
        """Calculate population diversity"""
        genomes = [agent[1] for agent in self.population]
        if len(genomes) < 2:
            return 0.0
        
        diversity = 0.0
        for i in range(len(genomes)):
            for j in range(i+1, len(genomes)):
                diversity += np.linalg.norm(np.array(genomes[i]) - np.array(genomes[j]))
        
        return diversity / (len(genomes) * (len(genomes) - 1) / 2)
    
    def evolve_generation(self):
        """Single generation evolution"""
        new_population = []
        
        # Elitism: Keep top 3
        new_population.extend(self.population[:3])
        
        # Breed new agents
        while len(new_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            child_genome = self._crossover(parent1[1], parent2[1])
            
            # Mutate with adaptive rate
            child_genome = self._mutate(child_genome, self.current_mutation_rate)
            
            new_population.append([0.0, child_genome, len(new_population)])
        
        self.population = new_population
    
    def _tournament_select(self, tournament_size=3):
        """Tournament selection"""
        tournament = np.random.choice(len(self.population), tournament_size, replace=False)
        return max([self.population[i] for i in tournament], key=lambda x: x[0])
    
    def _crossover(self, genome1, genome2):
        """Single-point crossover"""
        point = np.random.randint(1, len(genome1))
        return genome1[:point] + genome2[point:]
    
    def _mutate(self, genome, mutation_rate):
        """Gaussian mutation with adaptive rate"""
        mutated = genome.copy()
        for i in range(len(mutated)):
            if np.random.random() < 0.1:  # 10% probability per gene
                # Mutation strength proportional to mutation_rate
                mutated[i] += np.random.randn() * mutation_rate * 0.1
                
                # Clip to valid ranges
                ranges = [(0.01, 3.0), (0.1, 2.0), (0.005, 0.1), (0.0, 2*np.pi)]
                mutated[i] = np.clip(mutated[i], ranges[i][0], ranges[i][1])
        
        return mutated
    
    def run(self, generations=100, environment='standard', train_predictor=True):
        """Run evolution with adaptive mutation"""
        print(f"\n{'='*70}")
        print(f"🚀 ADAPTIVE MUTATION EVOLUTION - {self.strategy.upper()}")
        print(f"{'='*70}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {generations}")
        print(f"Environment: {environment}")
        print(f"GPU: {self.use_gpu}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for gen in range(generations):
            gen_start = time.time()
            
            # Adapt mutation rate
            self.adapt_mutation_rate(gen)
            
            # Evaluate population
            self.evaluate_population(environment)
            
            # Track metrics
            best_fitness = self.population[0][0]
            avg_fitness = np.mean([agent[0] for agent in self.population])
            diversity = self._calculate_diversity()
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Collect training data for ML predictor
            if self.strategy == 'ml_adaptive' and gen > 0:
                features = self.predictor.extract_features(
                    self.population, gen, self.best_fitness_history
                )
                self.predictor.add_training_data(
                    features, 
                    self.current_mutation_rate,
                    best_fitness
                )
            
            # Evolve to next generation
            self.evolve_generation()
            
            gen_time = time.time() - gen_start
            
            if (gen + 1) % 10 == 0:
                print(f"Gen {gen+1:3d} | Best: {best_fitness:.6f} | "
                      f"Avg: {avg_fitness:.6f} | Mutation: {self.current_mutation_rate:.4f} | "
                      f"Diversity: {diversity:.4f} | Time: {gen_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ Evolution complete!")
        print(f"Total time: {total_time:.2f}s ({total_time/generations:.3f}s per generation)")
        print(f"Final best fitness: {self.best_fitness_history[-1]:.6f}")
        print(f"{'='*70}\n")
        
        # Train predictor if using ML
        if self.strategy == 'ml_adaptive' and train_predictor:
            self.predictor.train(epochs=50)
        
        return {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'mutation_rates': self.mutation_rate_history,
            'diversity': self.diversity_history,
            'time': total_time,
            'final_best': self.population[0]
        }
    
    def visualize_results(self):
        """Create visualization of adaptive mutation performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = range(len(self.best_fitness_history))
        
        # Fitness evolution
        ax = axes[0, 0]
        ax.plot(generations, self.best_fitness_history, 'g-', label='Best', linewidth=2)
        ax.plot(generations, self.avg_fitness_history, 'b--', label='Average', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mutation rate adaptation
        ax = axes[0, 1]
        ax.plot(generations, self.mutation_rate_history, 'r-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mutation Rate')
        ax.set_title(f'Adaptive Mutation Rate ({self.strategy})')
        ax.grid(True, alpha=0.3)
        
        # Population diversity
        ax = axes[1, 0]
        ax.plot(generations, self.diversity_history, 'm-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity')
        ax.set_title('Population Diversity')
        ax.grid(True, alpha=0.3)
        
        # Mutation vs Fitness correlation
        ax = axes[1, 1]
        scatter = ax.scatter(self.mutation_rate_history, self.best_fitness_history, 
                           c=generations, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Mutation Rate')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Mutation-Fitness Relationship')
        plt.colorbar(scatter, ax=ax, label='Generation')
        
        plt.tight_layout()
        plt.savefig(f'adaptive_mutation_{self.strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# COMPARATIVE BENCHMARK
# ============================================================================

def compare_strategies(population_size=30, generations=100, environment='standard'):
    """
    Compare all mutation strategies:
    1. Fixed mutation (baseline)
    2. Simple adaptive
    3. ML adaptive (CPU)
    4. ML adaptive (GPU) - if available
    """
    print("\n" + "="*80)
    print("🔬 COMPARATIVE MUTATION STRATEGY BENCHMARK")
    print("="*80 + "\n")
    
    strategies = ['fixed', 'simple_adaptive', 'ml_adaptive']
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy.upper()}")
        print(f"{'='*80}\n")
        
        # Test with GPU if available and ML strategy
        use_gpu = (strategy == 'ml_adaptive' and TORCH_AVAILABLE)
        
        evo = AdaptiveMutationEvolution(
            population_size=population_size,
            strategy=strategy,
            use_gpu=use_gpu
        )
        
        result = evo.run(generations=generations, environment=environment)
        results[strategy] = result
        
        evo.visualize_results()
    
    # Comparative visualization
    _plot_comparative_results(results)
    
    return results


def _plot_comparative_results(results):
    """Create comparative visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'fixed': 'blue', 'simple_adaptive': 'green', 'ml_adaptive': 'red'}
    
    # Best fitness comparison
    ax = axes[0, 0]
    for strategy, result in results.items():
        ax.plot(result['best_fitness'], label=strategy, 
               color=colors.get(strategy, 'black'), linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best Fitness Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average fitness comparison
    ax = axes[0, 1]
    for strategy, result in results.items():
        ax.plot(result['avg_fitness'], label=strategy,
               color=colors.get(strategy, 'black'), linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Average Fitness Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mutation rate strategies
    ax = axes[1, 0]
    for strategy, result in results.items():
        if 'mutation_rates' in result:
            ax.plot(result['mutation_rates'], label=strategy,
                   color=colors.get(strategy, 'black'), linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mutation Rate')
    ax.set_title('Mutation Rate Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final performance bar chart
    ax = axes[1, 1]
    strategies_list = list(results.keys())
    final_fitness = [results[s]['final_best'][0] for s in strategies_list]
    times = [results[s]['time'] for s in strategies_list]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_fitness, width, label='Final Fitness', color='steelblue')
    bars2 = ax2.bar(x + width/2, times, width, label='Time (s)', color='coral')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Final Best Fitness', color='steelblue')
    ax2.set_ylabel('Time (seconds)', color='coral')
    ax.set_title('Performance Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_list, rotation=45, ha='right')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'mutation_strategy_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
               dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀⚛️ ADAPTIVE MUTATION WITH GPU & ML - QUANTUM EVOLUTION")
    print("="*80)
    
    # Quick test
    print("\n[1] Quick Test - Single Strategy")
    print("[2] Full Comparison - All Strategies")
    print("[3] GPU Stress Test")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        strategy = input("Strategy (fixed/simple_adaptive/ml_adaptive): ").strip()
        evo = AdaptiveMutationEvolution(
            population_size=30,
            strategy=strategy,
            use_gpu=True
        )
        results = evo.run(generations=100, environment='standard')
        evo.visualize_results()
        
    elif choice == "2":
        results = compare_strategies(
            population_size=30,
            generations=100,
            environment='standard'
        )
        
    elif choice == "3":
        if TORCH_AVAILABLE:
            print("\n🔥 GPU Stress Test - Large Population")
            evo = AdaptiveMutationEvolution(
                population_size=100,  # Large population
                strategy='ml_adaptive',
                use_gpu=True
            )
            results = evo.run(generations=200, environment='harsh')
            evo.visualize_results()
        else:
            print("❌ PyTorch not available for GPU test")
    
    else:
        print("Invalid option")
