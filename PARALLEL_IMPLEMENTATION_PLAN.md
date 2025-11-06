# Integrated Plan: Phase 2 + GA Conductor 🚀

**Date**: November 5, 2025  
**Goal**: Train trading specialists WITH GA Conductor controlling the evolution

---

## 🎯 The Integration

### What We're Building

```
Phase 2: Trading Specialists
    ↓
    Uses
    ↓
GA Conductor (Context-Aware Trainer)
    ↓
    Produces
    ↓
Optimally Trained Specialists
```

---

## 📋 Parallel Implementation Plan

### Track 1: Phase 2 Trading Specialists (Core Work)
**Goal**: Get specialists trained and working  
**Time**: 4-6 hours  
**Priority**: HIGH (This is our deliverable)

### Track 2: GA Conductor (Enhancement)
**Goal**: Make training smarter and faster  
**Time**: 2-3 hours (parallel work)  
**Priority**: MEDIUM (This is our innovation)

---

## 🏗️ Track 1: Phase 2 Implementation

### Step 1: Label Historical Data (30 min)
```python
# label_historical_regimes.py
detector = RegimeDetector()
df = pd.read_csv('DATA/yf_btc_1d.csv')

# Add regime labels
regimes = []
for i in range(len(df)):
    window = df.iloc[max(0, i-100):i+1]
    regime = detector.detect_regime(window)
    regimes.append(regime)

df['regime'] = regimes
df.to_csv('DATA/btc_with_regimes.csv')

# Split by regime
volatile_periods = df[df['regime'] == 'volatile']
trending_periods = df[df['regime'] == 'trending']
ranging_periods = df[df['regime'] == 'ranging']
crisis_periods = df[df['regime'] == 'crisis']

print(f"""
Regime Distribution:
  Volatile: {len(volatile_periods)} days
  Trending: {len(trending_periods)} days
  Ranging:  {len(ranging_periods)} days
  Crisis:   {len(crisis_periods)} days
""")
```

### Step 2: Define Trading Specialist (1 hour)
```python
# trading_specialist.py

class TradingSpecialist:
    """
    A trading agent optimized for specific market regime
    
    Genome (8 genes):
    [
        stop_loss_pct,        # 0.01-0.05 (1-5%)
        take_profit_pct,      # 0.02-0.20 (2-20%)
        position_size_pct,    # 0.01-0.10 (1-10%)
        entry_threshold,      # 0.0-1.0 (signal strength)
        exit_threshold,       # 0.0-1.0 (signal strength)
        max_hold_time,        # 1-14 days
        volatility_scaling,   # 0.5-2.0 (ATR multiplier)
        momentum_weight,      # 0.0-1.0 (trend vs mean-reversion)
    ]
    """
    
    def __init__(self, genome, regime_type):
        self.genome = np.array(genome)
        self.regime_type = regime_type
        self.trades = []
        self.equity_curve = []
    
    def generate_signal(self, market_data, predictions, current_position=0):
        """
        Generate trading signal based on genome
        
        Args:
            market_data: DataFrame with OHLCV + indicators
            predictions: ML model predictions (price direction)
            current_position: Current position (-1, 0, 1)
        
        Returns:
            signal: -1 (sell), 0 (hold), 1 (buy)
            position_size: 0.0-1.0 (fraction of capital)
        """
        # Unpack genome
        stop_loss = self.genome[0]
        take_profit = self.genome[1]
        position_size = self.genome[2]
        entry_threshold = self.genome[3]
        exit_threshold = self.genome[4]
        max_hold_time = int(self.genome[5])
        volatility_scaling = self.genome[6]
        momentum_weight = self.genome[7]
        
        # Get current market state
        current_price = market_data['close'].iloc[-1]
        atr = market_data['atr'].iloc[-1] if 'atr' in market_data else current_price * 0.02
        prediction_strength = abs(predictions[-1])
        
        # Calculate signal
        if current_position == 0:
            # No position - look for entry
            if prediction_strength > entry_threshold:
                signal = 1 if predictions[-1] > 0 else -1
                
                # Scale position by volatility
                scaled_position = position_size * (1.0 / volatility_scaling)
                scaled_position = np.clip(scaled_position, 0.01, 0.10)
                
                return signal, scaled_position
            else:
                return 0, 0.0
        
        else:
            # Have position - check exit conditions
            if prediction_strength < exit_threshold:
                # Signal weakened - exit
                return 0, 0.0
            
            # Check hold time limit
            if self.current_hold_time >= max_hold_time:
                return 0, 0.0
            
            # Hold position
            return current_position, position_size
    
    def evaluate_fitness(self, historical_data, predictions):
        """
        Backtest specialist on its regime data
        
        Returns:
            fitness: Combined score (Sharpe + Win Rate - Drawdown)
        """
        equity = 10000  # Starting capital
        position = 0
        entry_price = 0
        self.trades = []
        self.equity_curve = [equity]
        
        for i in range(len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            pred = predictions[i]
            
            # Generate signal
            signal, size = self.generate_signal(
                historical_data.iloc[:i+1],
                predictions[:i+1],
                position
            )
            
            # Execute trade
            if signal != position:
                # Close current position
                if position != 0:
                    pnl = (current_price - entry_price) * position * equity * abs(position)
                    equity += pnl
                    self.trades.append({
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': pnl / (equity - pnl)
                    })
                
                # Open new position
                if signal != 0:
                    entry_price = current_price
                    position = signal
            
            self.equity_curve.append(equity)
        
        # Calculate fitness metrics
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Win rate
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Total return
        total_return = (equity - 10000) / 10000
        
        # Combined fitness (higher is better)
        fitness = (
            sharpe * 10 +           # Sharpe ratio weight
            win_rate * 5 +          # Win rate weight
            total_return * 20 -     # Total return weight
            max_drawdown * 10       # Drawdown penalty
        )
        
        return fitness
```

### Step 3: Standard GA Trainer (1 hour)
```python
# specialist_trainer.py

class SpecialistTrainer:
    """Standard GA trainer for trading specialists"""
    
    def __init__(self, regime_type, training_data, predictions):
        self.regime_type = regime_type
        self.data = training_data
        self.predictions = predictions
        self.population_size = 100
        self.generations = 1000
        
        # Genome bounds for each regime type
        self.bounds = self.get_bounds_for_regime(regime_type)
    
    def get_bounds_for_regime(self, regime_type):
        """Different starting bounds for different regimes"""
        
        if regime_type == 'volatile':
            return {
                'stop_loss': (0.01, 0.03),      # Tight stops
                'take_profit': (0.03, 0.10),    # Quick profits
                'position_size': (0.02, 0.05),  # Small positions
                'entry_threshold': (0.5, 0.8),  # Moderate signals
                'exit_threshold': (0.3, 0.6),   # Quick exits
                'max_hold_time': (1, 5),        # Short holds
                'volatility_scaling': (0.8, 1.5),
                'momentum_weight': (0.6, 0.9)   # Favor momentum
            }
        
        elif regime_type == 'trending':
            return {
                'stop_loss': (0.02, 0.05),      # Wider stops
                'take_profit': (0.10, 0.25),    # Let winners run
                'position_size': (0.05, 0.10),  # Larger positions
                'entry_threshold': (0.6, 0.9),  # Strong signals
                'exit_threshold': (0.4, 0.7),   # Hold longer
                'max_hold_time': (5, 14),       # Longer holds
                'volatility_scaling': (1.0, 2.0),
                'momentum_weight': (0.7, 1.0)   # Strong momentum
            }
        
        elif regime_type == 'ranging':
            return {
                'stop_loss': (0.02, 0.04),      # Medium stops
                'take_profit': (0.03, 0.08),    # Target middle
                'position_size': (0.03, 0.07),  # Medium positions
                'entry_threshold': (0.4, 0.7),  # Moderate signals
                'exit_threshold': (0.3, 0.6),   # Quick exits
                'max_hold_time': (2, 7),        # Medium holds
                'volatility_scaling': (0.5, 1.2),
                'momentum_weight': (0.2, 0.5)   # Favor mean reversion
            }
        
        else:  # crisis
            return {
                'stop_loss': (0.005, 0.015),    # Very tight
                'take_profit': (0.01, 0.03),    # Quick scalps
                'position_size': (0.005, 0.02), # Minimal
                'entry_threshold': (0.8, 0.95), # Very strong signals
                'exit_threshold': (0.5, 0.8),   # Fast exits
                'max_hold_time': (1, 3),        # Very short
                'volatility_scaling': (0.3, 0.8),
                'momentum_weight': (0.3, 0.6)
            }
    
    def initialize_population(self):
        """Create initial random population within bounds"""
        population = []
        for _ in range(self.population_size):
            genome = []
            for param, (low, high) in self.bounds.items():
                genome.append(np.random.uniform(low, high))
            population.append(genome)
        return np.array(population)
    
    def train(self):
        """Train specialist using standard GA"""
        
        population = self.initialize_population()
        best_genome = None
        best_fitness = -np.inf
        fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness for all agents
            fitness_scores = []
            for genome in population:
                specialist = TradingSpecialist(genome, self.regime_type)
                fitness = specialist.evaluate_fitness(self.data, self.predictions)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_genome = population[gen_best_idx].copy()
            
            fitness_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'diversity': np.std(population)
            })
            
            if gen % 100 == 0:
                print(f"Gen {gen}: Best={best_fitness:.2f}, Avg={np.mean(fitness_scores):.2f}")
            
            # Selection
            elite_size = 10
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = population[elite_indices]
            
            # Create next generation
            next_population = elite.copy()
            
            while len(next_population) < self.population_size:
                # Tournament selection
                tournament_size = 5
                tournament = np.random.choice(len(population), tournament_size)
                parent1_idx = tournament[np.argmax(fitness_scores[tournament])]
                parent1 = population[parent1_idx]
                
                tournament = np.random.choice(len(population), tournament_size)
                parent2_idx = tournament[np.argmax(fitness_scores[tournament])]
                parent2 = population[parent2_idx]
                
                # Crossover
                if np.random.random() < 0.7:
                    crossover_point = np.random.randint(1, len(parent1))
                    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                else:
                    child = parent1.copy()
                
                # Mutation
                mutation_rate = 0.1
                for i in range(len(child)):
                    if np.random.random() < mutation_rate:
                        param_name = list(self.bounds.keys())[i]
                        low, high = self.bounds[param_name]
                        child[i] += np.random.normal(0, (high - low) * 0.1)
                        child[i] = np.clip(child[i], low, high)
                
                next_population = np.vstack([next_population, child])
            
            population = next_population[:self.population_size]
        
        return best_genome, best_fitness, fitness_history
```

---

## 🧠 Track 2: GA Conductor Integration

### Step 1: Enhanced Trainer with Context (1 hour)
```python
# enhanced_trainer.py

class EnhancedTrainer(SpecialistTrainer):
    """Context-aware trainer with adaptive parameters"""
    
    def __init__(self, regime_type, training_data, predictions):
        super().__init__(regime_type, training_data, predictions)
        
        # Context features
        self.context = {
            'regime_type': regime_type,
            'data_size': len(training_data),
            'volatility': training_data['close'].pct_change().std(),
            'trend_strength': self.calculate_trend_strength(training_data)
        }
        
        # Adaptive parameters (will be learned by conductor)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.selection_pressure = 0.8
    
    def train_with_adaptation(self):
        """Train with adaptive parameters"""
        
        population = self.initialize_population()
        best_genome = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_population(population)
            
            # Update best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_genome = population[gen_best_idx].copy()
            
            # Compute state for adaptive control
            state = self.compute_state(population, fitness_scores, gen)
            
            # Adapt parameters based on state
            self.adapt_parameters(state)
            
            # Create next generation with adapted parameters
            population = self.evolve_population(
                population,
                fitness_scores,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate
            )
        
        return best_genome, best_fitness
    
    def adapt_parameters(self, state):
        """Adapt GA parameters based on current state"""
        
        # Simple heuristics (will be replaced by ML model later)
        diversity = state['diversity']
        fitness_improvement = state['fitness_improvement']
        
        # Low diversity → increase mutation
        if diversity < 0.1:
            self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
        
        # Stagnant fitness → try something different
        if fitness_improvement < 0.01:
            self.mutation_rate = min(1.0, self.mutation_rate * 1.2)
            self.crossover_rate = max(0.3, self.crossover_rate * 0.9)
        
        # Good progress → keep refining
        if fitness_improvement > 0.05:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
```

### Step 2: Collect Training Data for Conductor (30 min)
```python
# collect_conductor_training_data.py

def collect_training_data_for_conductor():
    """
    Run multiple training sessions with different parameters
    to create training data for GA Conductor
    """
    
    training_samples = []
    
    # Try different parameter combinations
    mutation_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    crossover_rates = [0.3, 0.5, 0.7, 0.9]
    population_sizes = [50, 100, 200]
    
    for mutation in mutation_rates:
        for crossover in crossover_rates:
            for pop_size in population_sizes:
                # Run short training session
                trainer = SpecialistTrainer('volatile', volatile_data, predictions)
                trainer.mutation_rate = mutation
                trainer.crossover_rate = crossover
                trainer.population_size = pop_size
                
                # Train for 200 generations
                trainer.generations = 200
                genome, fitness, history = trainer.train()
                
                # Record what worked
                for gen_data in history:
                    sample = {
                        # State
                        'avg_fitness': gen_data['avg_fitness'],
                        'diversity': gen_data['diversity'],
                        'generation': gen_data['generation'],
                        
                        # Context
                        'population_size': pop_size / 200,  # Normalized
                        'crossover_rate': crossover,
                        'mutation_rate': mutation,
                        
                        # Outcome (target)
                        'fitness_improvement': gen_data['best_fitness'] - gen_data['avg_fitness'],
                        'success': fitness > threshold
                    }
                    training_samples.append(sample)
    
    return training_samples
```

### Step 3: Train GA Conductor Model (1 hour)
```python
# train_ga_conductor.py

import torch
import torch.nn as nn

class GAConductor(nn.Module):
    """ML model that predicts optimal GA parameters"""
    
    def __init__(self):
        super().__init__()
        
        # Input: 13 features (state + context)
        self.network = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Output: mutation_rate, crossover_rate, selection_pressure
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        output = self.network(x)
        
        # Constrain outputs to valid ranges
        mutation_rate = torch.sigmoid(output[:, 0]) * 2.0      # 0-2
        crossover_rate = torch.sigmoid(output[:, 1])           # 0-1
        selection_pressure = torch.sigmoid(output[:, 2])       # 0-1
        
        return mutation_rate, crossover_rate, selection_pressure

# Train it
training_data = collect_training_data_for_conductor()
model = GAConductor()
# ... training loop ...
```

---

## 🎯 Combined Workflow

```python
# main_phase2.py - The complete system

def train_all_specialists_with_conductor():
    """Train all 4 specialists using GA Conductor"""
    
    # Step 1: Label data
    print("📊 Labeling historical data...")
    labeled_data = label_historical_regimes()
    
    # Step 2: Load/train GA Conductor
    print("🧠 Loading GA Conductor...")
    conductor = GAConductor()
    # conductor.load_state_dict(torch.load('ga_conductor.pth'))  # If pre-trained
    
    # Step 3: Train each specialist
    specialists = {}
    
    for regime in ['volatile', 'trending', 'ranging', 'crisis']:
        print(f"\n🎯 Training {regime} specialist...")
        
        # Get regime-specific data
        regime_data = labeled_data[labeled_data['regime'] == regime]
        
        if len(regime_data) < 100:
            print(f"⚠️  Only {len(regime_data)} days, using conservative defaults")
            continue
        
        # Create trainer with conductor
        trainer = EnhancedTrainer(
            regime_type=regime,
            training_data=regime_data,
            predictions=get_predictions(regime_data)
        )
        
        # Option 1: Standard GA (baseline)
        # genome, fitness, history = trainer.train()
        
        # Option 2: Enhanced with conductor
        genome, fitness = trainer.train_with_adaptation()
        
        # Save specialist
        specialists[regime] = {
            'genome': genome.tolist(),
            'fitness': fitness,
            'regime_type': regime,
            'training_days': len(regime_data)
        }
        
        print(f"✅ {regime}: fitness={fitness:.2f}")
    
    # Save all specialists
    with open('trading_specialists.json', 'w') as f:
        json.dump(specialists, f, indent=2)
    
    print("\n🎉 All specialists trained!")
    return specialists
```

---

## 📈 Expected Timeline

**Today (4-6 hours total)**:

```
Hour 1: Label data + Define specialist class
Hour 2: Build standard trainer
Hour 3: Train Volatile specialist (baseline)
Hour 4: Train Trending specialist (baseline)
Hour 5: Train Ranging specialist (baseline)
Hour 6: Validate and document results

Parallel (if time):
Hour 2-3: Build enhanced trainer with simple adaptation
Hour 4-5: Collect conductor training data
```

**Tomorrow (2-3 hours)**:
```
Hour 1: Train GA Conductor model
Hour 2: Retrain specialists with conductor
Hour 3: Compare baseline vs conductor results
```

---

## 🎯 Success Metrics

### Phase 2 Success
- ✅ 4 specialists trained
- ✅ Each has positive fitness on test data
- ✅ Specialists outperform buy-and-hold
- ✅ Ready for Phase 3 (meta-controller)

### GA Conductor Success
- ✅ Converges 30-50% faster than standard GA
- ✅ Finds better solutions (higher fitness)
- ✅ Adapts to different regime contexts
- ✅ Can be used for future training

---

**Ready to start?** I recommend:

1. **Start with Track 1** (Phase 2 standard implementation)
2. **Build Track 2 in parallel** as time allows
3. **Compare results tomorrow** (baseline vs conductor)

This gives us:
- ✅ Guaranteed deliverable (Phase 2 specialists)
- ✅ Innovation bonus (GA Conductor)
- ✅ Publishable research (if conductor works well)

Let's do this! 🚀

