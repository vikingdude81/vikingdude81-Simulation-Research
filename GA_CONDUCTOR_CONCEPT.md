# Self-Modifying Genetic Algorithm: The GA Conductor 🎼

**Date**: November 5, 2025  
**Concept**: ML model that controls the ENTIRE evolutionary process, not just mutation

---

## 🎯 The Evolution of Control

### Level 1: Basic GA (Where we started)
```python
# Fixed parameters, no adaptation
mutation_rate = 0.1        # Never changes
crossover_rate = 0.7       # Never changes
population_size = 100      # Never changes
selection = 'tournament'   # Never changes
```

### Level 2: Reactive ML (Current system)
```python
# ML predicts mutation rate based on results
mutation_rate = ml_model.predict(current_state)  # Adapts!
# But everything else is still fixed
```

### Level 3: GA Conductor (The breakthrough!)
```python
# ML controls EVERYTHING
mutation_rate, crossover_rate, selection_pressure, population_delta = ml_model.predict(current_state)

# The model can:
- Adjust mutation and crossover dynamically
- Change selection pressure (elitism dial)
- Add/remove agents (immigration/culling)
- Even change the fitness function!
```

---

## 📊 Enhanced Inputs: Deeper Agent Metrics

### Current Inputs (Population-Wide Stats)
```python
inputs = [
    avg_fitness,      # Average across all agents
    best_fitness,     # Single best agent
    worst_fitness,    # Single worst agent
    diversity,        # Overall genetic diversity
    gini_coefficient  # Wealth inequality (0-1)
]
```

**Problem**: These are **lagging indicators**. By the time `avg_fitness` drops, it's already too late!

### Enhanced Inputs (Early Warning System)

#### 1. Wealth Distribution Percentiles
```python
# Instead of just Gini, see the SHAPE of distribution
wealth_percentiles = {
    'bottom_10_pct': 50,      # Bottom 10% avg wealth
    'bottom_25_pct': 120,     # Bottom quartile
    'median': 500,            # Middle agent
    'top_25_pct': 2000,       # Top quartile
    'top_10_pct': 5000,       # Top 10%
    'top_1_pct': 15000        # The elite
}
```

**What the model can learn**:
```python
if bottom_10_pct < 50 and median > 500:
    # Poverty crisis brewing! Bottom collapsing while middle is fine
    # avg_wealth won't show this yet (still high)
    action = "WELFARE_INTERVENTION"
    # Inject resources to bottom 10% NOW

if top_1_pct > 10 * median:
    # Extreme inequality, elite hoarding
    action = "REDISTRIBUTE"
    # Tax the rich, fund mutation in poor agents
```

#### 2. Fitness Distribution Shape
```python
fitness_distribution = {
    'bottom_10_pct_fitness': 0.1,
    'median_fitness': 0.5,
    'top_10_pct_fitness': 0.9,
    'fitness_std_dev': 0.2,      # How spread out?
    'fitness_skewness': -0.5     # Is it skewed?
}
```

**What the model can learn**:
```python
if fitness_std_dev < 0.1:
    # Everyone has similar fitness - CONVERGENCE!
    # Need massive exploration
    mutation_rate = 2.0
    add_random_agents = 50

if fitness_skewness > 1.0:
    # Most agents are bad, few are good (right-skewed)
    # Need aggressive selection
    selection_pressure = HIGH
    tournament_size = 20
```

#### 3. Agent Age Distribution
```python
age_metrics = {
    'avg_agent_age': 25,           # How many generations survived
    'oldest_agent': 150,           # The ancient one
    'youngest_agents_pct': 0.05,   # Only 5% are young!
    'age_diversity': 0.3           # Age spread
}
```

**What the model can learn**:
```python
if avg_agent_age > 50 and youngest_agents_pct < 0.1:
    # STAGNATION! Old guard is entrenched
    # No new ideas breaking through
    action = "EXTINCTION_EVENT"
    # Kill oldest 30%, inject 50 random youth
    
if oldest_agent > 200:
    # One agent has survived 200 generations!
    # It's dominating, but might be local optimum
    action = "TARGETED_MUTATION"
    # Force mutate the ancient one's descendants
```

#### 4. Strategy Diversity
```python
strategy_metrics = {
    'unique_strategies': 15,        # Out of 100 agents
    'dominant_strategy_pct': 0.6,   # 60% use same strategy
    'strategy_entropy': 0.3,        # Low entropy = low diversity
    'novelty_score': 0.1            # New strategies emerging?
}
```

**What the model can learn**:
```python
if dominant_strategy_pct > 0.8:
    # 80% of agents are clones!
    # Monoculture - dangerous
    action = "DIVERSITY_INJECTION"
    mutation_rate = 3.0
    crossover_rate = 0.2  # Stop mixing clones
    
if novelty_score < 0.05:
    # No new ideas in 10 generations
    action = "RANDOM_SEEDING"
    # Add 20 completely random agents
```

---

## 🎮 Enhanced Outputs: Full GA Control

### Current Output (Single Value)
```python
output = mutation_rate  # 0-2
```

### Enhanced Output (Multi-Dimensional Control)

#### Version 1: Basic Multi-Parameter
```python
output = {
    'mutation_rate': 0.5,      # 0-2
    'crossover_rate': 0.7,     # 0-1
    'selection_pressure': 0.8, # 0-1 (weak to strong)
}
```

#### Version 2: Full GA Conductor
```python
output = {
    # Evolution parameters
    'mutation_rate': 0.5,
    'crossover_rate': 0.7,
    'selection_pressure': 0.8,
    
    # Population dynamics
    'population_delta': +20,           # Add 20 agents (or -20 to remove)
    'immigration_type': 'random',      # 'random', 'elite_copy', 'mutated_best'
    'culling_strategy': 'bottom_20',   # Which agents to remove if negative delta
    
    # Selection method
    'selection_method': 'tournament',  # Switch between methods
    'tournament_size': 10,             # If tournament selected
    
    # Special actions
    'extinction_event': False,         # Nuclear option: kill 50%
    'elite_preservation': True,        # Protect top 5 agents
    'diversity_injection': False,      # Add random agents
    
    # Intervention types (god-mode)
    'welfare_amount': 0,               # Resources to bottom 10%
    'tax_rate': 0.0,                   # Tax top 10% (0-0.5)
}
```

---

## 🧬 Implementation: The Conductor Model

### Architecture

```python
import torch
import torch.nn as nn

class GAConductor(nn.Module):
    """
    The GA Conductor: Controls entire evolutionary process
    """
    
    def __init__(self, input_size=25):
        super().__init__()
        
        # Shared feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Evolution parameters head
        self.evolution_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # mutation, crossover, selection_pressure
        )
        
        # Population dynamics head
        self.population_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # population_delta, immigration_strength
        )
        
        # Special actions head (binary decisions)
        self.actions_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # extinction, diversity_inject, welfare, etc.
        )
        
        # Selection method head (categorical)
        self.selection_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # tournament, roulette, elite
        )
    
    def forward(self, x):
        # Encode input state
        features = self.encoder(x)
        
        # Evolution parameters
        evolution = self.evolution_head(features)
        mutation_rate = torch.sigmoid(evolution[:, 0]) * 2.0      # 0-2
        crossover_rate = torch.sigmoid(evolution[:, 1])           # 0-1
        selection_pressure = torch.sigmoid(evolution[:, 2])       # 0-1
        
        # Population dynamics
        population = self.population_head(features)
        pop_delta = torch.tanh(population[:, 0]) * 50            # -50 to +50
        immigration_strength = torch.sigmoid(population[:, 1])    # 0-1
        
        # Special actions (binary)
        actions = torch.sigmoid(self.actions_head(features))
        extinction_event = actions[:, 0] > 0.5
        diversity_inject = actions[:, 1] > 0.5
        welfare = actions[:, 2] * 1000  # 0-1000 resources
        tax_rate = actions[:, 3] * 0.5  # 0-0.5 tax
        elite_preserve = actions[:, 4] > 0.5
        
        # Selection method (categorical)
        selection_logits = self.selection_head(features)
        selection_method = torch.argmax(selection_logits, dim=1)
        # 0=tournament, 1=roulette, 2=elite
        
        return {
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'selection_pressure': selection_pressure,
            'population_delta': pop_delta,
            'immigration_strength': immigration_strength,
            'extinction_event': extinction_event,
            'diversity_inject': diversity_inject,
            'welfare': welfare,
            'tax_rate': tax_rate,
            'elite_preserve': elite_preserve,
            'selection_method': selection_method
        }
```

---

## 🎓 Training Strategy

### Input Features (25 total)

```python
input_vector = [
    # Population-wide stats (10)
    avg_fitness, best_fitness, worst_fitness,
    diversity, generation, time_since_improvement,
    fitness_trend, diversity_trend, stagnation_indicator,
    convergence_speed,
    
    # Wealth distribution (6)
    wealth_bottom_10, wealth_bottom_25, wealth_median,
    wealth_top_25, wealth_top_10, gini_coefficient,
    
    # Fitness distribution (4)
    fitness_bottom_10, fitness_median, fitness_top_10,
    fitness_std_dev,
    
    # Age metrics (3)
    avg_agent_age, oldest_agent_age, young_agents_pct,
    
    # Strategy diversity (2)
    unique_strategies_count, dominant_strategy_pct
]
```

### Training Approach: Reinforcement Learning

Instead of supervised learning (predict what worked), use **RL** (learn what works):

```python
class GAConductorTrainer:
    """Train the conductor using RL"""
    
    def __init__(self):
        self.model = GAConductor()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []  # Experience replay buffer
    
    def train_episode(self):
        """Run one GA simulation with the conductor controlling"""
        
        # Initialize GA
        population = initialize_population(100)
        total_reward = 0
        episode_memory = []
        
        for generation in range(500):
            # Get current state
            state = self.compute_state(population, generation)
            
            # Model predicts actions
            with torch.no_grad():
                actions = self.model(state)
            
            # Execute actions in GA
            population = self.apply_actions(population, actions)
            
            # Compute reward
            reward = self.compute_reward(population, generation)
            total_reward += reward
            
            # Store experience
            episode_memory.append({
                'state': state,
                'actions': actions,
                'reward': reward
            })
            
            # Run one generation of GA
            population = evolve_one_generation(
                population,
                mutation_rate=actions['mutation_rate'],
                crossover_rate=actions['crossover_rate'],
                selection_pressure=actions['selection_pressure']
            )
        
        # Add episode to memory
        self.memory.extend(episode_memory)
        
        # Train on batch from memory
        if len(self.memory) > 1000:
            self.train_on_batch()
        
        return total_reward
    
    def compute_reward(self, population, generation):
        """Reward function for RL"""
        
        # Component 1: Fitness improvement
        fitness_reward = (best_fitness - prev_best_fitness) * 100
        
        # Component 2: Diversity maintenance
        diversity_reward = current_diversity * 10
        
        # Component 3: Efficiency (penalize unnecessary actions)
        efficiency_penalty = -abs(population_delta) * 0.1
        
        # Component 4: Convergence speed bonus
        if reached_target_fitness:
            convergence_bonus = 1000 / generation  # Faster = better
        else:
            convergence_bonus = 0
        
        # Component 5: Stability (avoid thrashing)
        if action_changed_a_lot:
            stability_penalty = -50
        else:
            stability_penalty = 0
        
        total_reward = (
            fitness_reward +
            diversity_reward +
            efficiency_penalty +
            convergence_bonus +
            stability_penalty
        )
        
        return total_reward
```

---

## 🚀 Specific Learned Behaviors

### Behavior 1: The Immigration Wave
```python
Situation: Low diversity, fitness stuck at 0.7 for 20 generations
Model learns:
  → mutation_rate = 0.3 (moderate)
  → population_delta = +50 (ADD 50 random agents!)
  → diversity_inject = True
  
Result: Diversity spikes, new ideas compete, fitness jumps to 0.85
Lesson: Sometimes you need fresh blood, not just mutation
```

### Behavior 2: The Elite Refinement Phase
```python
Situation: High fitness (0.92), high diversity still
Model learns:
  → selection_pressure = 0.95 (VERY high)
  → mutation_rate = 0.02 (tiny)
  → crossover_rate = 0.9 (high mixing)
  → elite_preserve = True (protect top 10)
  
Result: Rapid convergence to 0.95, no wasted exploration
Lesson: Late game needs polish, not more ideas
```

### Behavior 3: The Extinction Event
```python
Situation: avg_agent_age > 100, dominant_strategy_pct = 0.9
Model learns:
  → extinction_event = True (KILL 50% of population!)
  → population_delta = +30 (replace with random)
  → mutation_rate = 1.5 (high)
  
Result: Population "reset", escapes local optimum
Lesson: Sometimes you need creative destruction
```

### Behavior 4: The Welfare Intervention
```python
Situation: bottom_10_wealth < 50, top_10_wealth > 5000
Model learns:
  → welfare = 500 resources to bottom 10%
  → tax_rate = 0.2 (tax top 10%)
  → mutation_rate for poor = 0.8 (give them a chance)
  
Result: Bottom agents can compete, diversity maintained
Lesson: Inequality kills innovation
```

---

## 📈 Expected Results

### Standard GA (Fixed Parameters)
```
Convergence: 300 generations
Best fitness: 0.85
Final diversity: 0.1 (converged)
CPU time: 100% (ran all 300 gens)
```

### ML-Guided GA (Mutation Only)
```
Convergence: 150 generations (50% faster)
Best fitness: 0.90 (better solution)
Final diversity: 0.2 (maintained better)
CPU time: 50% (early convergence)
```

### GA Conductor (Full Control)
```
Convergence: 80 generations (73% faster!)
Best fitness: 0.95 (excellent solution)
Final diversity: 0.3 (still exploring)
CPU time: 27% (very efficient)
Actions taken:
  - 3 immigration waves (+120 agents total)
  - 2 extinction events (-50% population)
  - 15 welfare interventions
  - 8 selection method switches
```

---

## 🎯 Implementation Priority

### Phase 1: Enhanced Inputs (This Week)
1. ✅ Add wealth percentiles
2. ✅ Add fitness distribution
3. ✅ Add age metrics
4. ✅ Add strategy diversity

### Phase 2: Basic Multi-Output (Next Week)
1. ✅ Train mutation + crossover predictor
2. ✅ Add selection pressure output
3. ✅ Test on trading specialists

### Phase 3: Population Dynamics (Week 3)
1. Add immigration/culling actions
2. Train with RL instead of supervised learning
3. Test extinction events

### Phase 4: Full Conductor (Week 4)
1. All outputs enabled
2. Self-modifying GA fully functional
3. Compare to all baselines

---

## 💡 Why This Is Revolutionary

1. **Adaptive Intelligence**: The GA adapts to its own state in real-time
2. **Meta-Optimization**: The optimizer optimizes itself
3. **Strategic Reasoning**: Learns when to explore vs exploit
4. **Efficient**: Reaches better solutions faster
5. **Robust**: Can escape local optima through creative destruction

This is essentially **AlphaGo for genetic algorithms** - a neural network that learns to play the "game" of evolution itself!

---

**Ready to build this?** We can start with Phase 1 (enhanced inputs) today! 🚀
