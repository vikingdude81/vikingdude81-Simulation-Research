# Phase 2 Preparation: Train Trading Specialists 🚀

**Date**: November 4, 2025  
**Status**: Phase 1 Complete ✅ | Phase 2 Ready to Start  
**Branch**: ml-quantum-integration (pushed to GitHub)

---

## ✅ What Was Committed Today

### 5 Commits Pushed to GitHub:

1. **docs: Add Phase 1 trading system documentation** (5 files)
   - Complete regime detection documentation
   - Before/after calibration comparison
   - 6-phase implementation plan
   - Progress tracking and commit summary

2. **feat: Add crypto-calibrated regime detection system** (3 files)
   - regime_detector.py (RegimeDetector class)
   - calibrate_crypto_thresholds.py (statistical calibration)
   - validate_regime_detector.py (historical validation)

3. **data: Add regime calibration and validation results** (2 files)
   - Calibration statistics and thresholds
   - 13 event validations with accuracy metrics

4. **docs: Add multi-quantum ensemble documentation** (3 files)
   - Complete multi-quantum reference
   - 300-gen breakthrough discovery
   - Specialist genomes for reuse

5. **feat: Add 300-gen validation and multi-quantum controller** (2 files)
   - 300-gen test script
   - Multi-quantum controller implementation

**Total**: 15 files committed and pushed ✅

---

## 🎯 Tomorrow's Objectives: Phase 2

### Goal: Train 4 Trading Specialists

Each specialist will be trained using genetic algorithms (like prisoner's dilemma specialists) but optimized for specific market regimes.

### 4 Specialists to Build:

#### 1. Volatile_Market_Specialist 🌪️
**Target Regime**: VIX 62-99, ADX < 51  
**Example Periods**: Bull run peak (Jan-Apr 2021), Summer 2021 consolidation

**Strategy Parameters** (genome):
```python
[
    stop_loss_pct,        # Tight stops: 1-2%
    take_profit_pct,      # Quick profits: 3-5%
    position_size_pct,    # Smaller positions: 2-5%
    entry_threshold,      # How strong signal before entry
    exit_threshold,       # How weak signal before exit
    max_hold_time,        # Short holds: 1-3 days
    volatility_scaling,   # Adjust size based on ATR
    momentum_weight,      # How much to favor momentum
]
```

**Training Approach**:
- Label historical data with regime detector
- Extract all "volatile" periods
- Generate price/prediction data for those periods
- Train GA for 1000 generations
- Fitness = Sharpe ratio + win rate - drawdown
- Save best genome

#### 2. Trending_Market_Specialist 📈
**Target Regime**: ADX > 51, moderate VIX  
**Example Periods**: Bull run start (Oct-Dec 2020), Bear market start (Dec 2021-Mar 2022)

**Strategy Parameters** (genome):
```python
[
    stop_loss_pct,        # ATR-based stops: 2-4 * ATR
    take_profit_pct,      # Let winners run: 10-20%
    position_size_pct,    # Larger positions: 5-10%
    trend_confirmation,   # How many bars to confirm trend
    pullback_entry,       # Enter on pullbacks vs breakouts
    max_hold_time,        # Longer holds: 5-14 days
    add_on_profit,        # Pyramid into winners
    trailing_stop_pct,    # Trail stop to lock profits
]
```

**Training Approach**:
- Extract all "trending" periods
- Focus on trend-following metrics
- Train GA for 1000 generations
- Fitness = Total profit + trend capture ratio
- Penalize counter-trend trades

#### 3. Ranging_Market_Specialist ↔️
**Target Regime**: ADX < 27, VIX < 62  
**Example Periods**: 2023 recovery, Recent period (2024-2025)

**Strategy Parameters** (genome):
```python
[
    stop_loss_pct,        # Beyond range: 3-5%
    take_profit_pct,      # Target middle: 2-4%
    position_size_pct,    # Medium positions: 3-7%
    support_threshold,    # How close to support to buy
    resistance_threshold, # How close to resistance to sell
    mean_reversion_speed, # How fast to fade extremes
    range_confirmation,   # How many bars to confirm range
    oscillator_weight,    # RSI/Stochastic importance
]
```

**Training Approach**:
- Extract all "ranging" periods
- Focus on mean-reversion metrics
- Train GA for 1000 generations
- Fitness = Win rate + profit factor
- Penalize trend-following in ranging markets

#### 4. Crisis_Manager 🛡️
**Target Regime**: VIX > 99  
**Example Periods**: COVID crash (Feb-Mar 2020)

**Strategy Parameters** (genome):
```python
[
    stop_loss_pct,        # Very tight: 0.5-1%
    take_profit_pct,      # Quick scalps: 1-2%
    position_size_pct,    # Minimal: 0.5-2%
    risk_threshold,       # When to shut down completely
    max_trades_per_day,   # Limit exposure: 1-2
    defensive_mode,       # Cash % to hold: 80-95%
    volatility_cutoff,    # Stop trading if VIX too high
    recovery_detection,   # When to resume normal trading
]
```

**Training Approach**:
- Extract all "crisis" periods (rare!)
- Focus on capital preservation
- Train GA for 1000 generations
- Fitness = Max drawdown minimization + survival
- Penalize aggressive trading in crisis

---

## 🛠️ Implementation Plan for Tomorrow

### Step 1: Label Historical Data (30 min)
```python
# label_historical_regimes.py
detector = RegimeDetector()
df = pd.read_csv('DATA/yf_btc_1d.csv')

# Add regime column
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

print(f"Volatile: {len(volatile_periods)} days")
print(f"Trending: {len(trending_periods)} days")
print(f"Ranging: {len(ranging_periods)} days")
print(f"Crisis: {len(crisis_periods)} days")
```

### Step 2: Create Training Framework (1 hour)
```python
# trading_specialist_trainer.py
class TradingSpecialist:
    def __init__(self, genome, regime_type):
        self.genome = genome
        self.regime_type = regime_type
    
    def generate_signal(self, market_data, predictions):
        # Unpack genome
        stop_loss = self.genome[0]
        take_profit = self.genome[1]
        position_size = self.genome[2]
        # ... use other parameters
        
        # Generate buy/sell/hold signal
        return signal, position_size
    
    def evaluate_fitness(self, historical_data):
        # Backtest specialist on its regime
        trades = []
        equity_curve = []
        
        for i in range(len(historical_data)):
            signal, size = self.generate_signal(...)
            # Execute trade
            # Track P&L
        
        # Calculate fitness
        sharpe = calculate_sharpe(equity_curve)
        win_rate = winning_trades / total_trades
        drawdown = max_drawdown(equity_curve)
        
        fitness = sharpe * 10 + win_rate * 5 - drawdown * 2
        return fitness

class SpecialistTrainer:
    def __init__(self, regime_type, training_data):
        self.regime_type = regime_type
        self.data = training_data
        self.population_size = 100
        self.generations = 1000
    
    def train(self):
        # Initialize population
        population = [random_genome() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for genome in population:
                specialist = TradingSpecialist(genome, self.regime_type)
                fitness = specialist.evaluate_fitness(self.data)
                fitness_scores.append(fitness)
            
            # Selection, crossover, mutation
            population = genetic_algorithm_step(population, fitness_scores)
            
            if gen % 100 == 0:
                print(f"Gen {gen}: Best fitness = {max(fitness_scores):.2f}")
        
        # Return best genome
        best_idx = fitness_scores.index(max(fitness_scores))
        return population[best_idx]
```

### Step 3: Train Each Specialist (2-3 hours total)
```python
# train_all_specialists.py

# Load labeled data
df = pd.read_csv('DATA/btc_with_regimes.csv')

# Train Volatile specialist
print("Training Volatile_Market_Specialist...")
volatile_data = df[df['regime'] == 'volatile']
volatile_trainer = SpecialistTrainer('volatile', volatile_data)
volatile_genome = volatile_trainer.train()
print(f"Best genome: {volatile_genome}")

# Train Trending specialist
print("\nTraining Trending_Market_Specialist...")
trending_data = df[df['regime'] == 'trending']
trending_trainer = SpecialistTrainer('trending', trending_data)
trending_genome = trending_trainer.train()

# Train Ranging specialist
print("\nTraining Ranging_Market_Specialist...")
ranging_data = df[df['regime'] == 'ranging']
ranging_trainer = SpecialistTrainer('ranging', ranging_data)
ranging_genome = ranging_trainer.train()

# Train Crisis specialist
print("\nTraining Crisis_Manager...")
crisis_data = df[df['regime'] == 'crisis']
if len(crisis_data) < 100:
    print("Warning: Limited crisis data. Using conservative defaults.")
    crisis_genome = [0.005, 0.01, 0.01, ...]  # Conservative defaults
else:
    crisis_trainer = SpecialistTrainer('crisis', crisis_data)
    crisis_genome = crisis_trainer.train()

# Save all specialists
specialists = {
    'volatile': {
        'genome': volatile_genome.tolist(),
        'regime_type': 'volatile',
        'training_days': len(volatile_data),
        'best_fitness': ...
    },
    'trending': { ... },
    'ranging': { ... },
    'crisis': { ... }
}

with open('trading_specialists.json', 'w') as f:
    json.dump(specialists, f, indent=2)

print("\n✅ All specialists trained and saved!")
```

### Step 4: Validate Specialists (1 hour)
```python
# validate_specialists.py

# Load specialists
with open('trading_specialists.json', 'r') as f:
    specialists = json.load(f)

# Test on held-out data (last year)
test_data = df[df.index >= '2024-01-01']

for regime_type, config in specialists.items():
    print(f"\nTesting {regime_type} specialist:")
    
    # Filter test data for this regime
    regime_test = test_data[test_data['regime'] == regime_type]
    
    if len(regime_test) < 10:
        print(f"  Skipped: Only {len(regime_test)} test days")
        continue
    
    # Backtest specialist
    specialist = TradingSpecialist(config['genome'], regime_type)
    results = backtest(specialist, regime_test)
    
    print(f"  Total P&L: ${results['pnl']:,.2f}")
    print(f"  Win rate: {results['win_rate']:.1f}%")
    print(f"  Sharpe ratio: {results['sharpe']:.2f}")
    print(f"  Max drawdown: {results['drawdown']:.1f}%")
    print(f"  Trades: {results['num_trades']}")
```

---

## 📋 Phase 2 Checklist

### Data Preparation
- [ ] Label all historical BTC data with regimes
- [ ] Split data by regime type
- [ ] Verify sufficient data for each regime (except crisis)
- [ ] Create train/test split (2020-2023 train, 2024-2025 test)

### Specialist Training
- [ ] Build TradingSpecialist class
- [ ] Build SpecialistTrainer class
- [ ] Define genome structure for each specialist type
- [ ] Train Volatile_Market_Specialist (1000 gen)
- [ ] Train Trending_Market_Specialist (1000 gen)
- [ ] Train Ranging_Market_Specialist (1000 gen)
- [ ] Train/Define Crisis_Manager (conservative defaults if data sparse)

### Validation
- [ ] Backtest each specialist on its regime
- [ ] Calculate performance metrics (Sharpe, win rate, drawdown)
- [ ] Compare to buy-and-hold baseline
- [ ] Verify specialists don't overtrade
- [ ] Check specialists are regime-specific (don't work on wrong regime)

### Documentation
- [ ] Save all specialist genomes to JSON
- [ ] Document each specialist's strategy
- [ ] Record training metrics and performance
- [ ] Create before/after comparison
- [ ] Update progress summary

---

## 🎓 Key Learnings to Apply

From prisoner's dilemma specialists:
1. **Start simple**: Don't overcomplicate genome at first
2. **Track everything**: Log all trades, decisions, fitness scores
3. **Validate properly**: Use separate test data
4. **Save checkpoints**: Save best genome every 100 generations
5. **Visualize progress**: Plot fitness over generations

From regime detector:
1. **Use real data**: Train on actual BTC price action
2. **Validate historically**: Test on known market events
3. **Accept limitations**: 70% accuracy is excellent
4. **Document thresholds**: Make everything reproducible

---

## 🚀 Expected Outcomes

After Phase 2 completion:
- ✅ 4 trained specialists with validated genomes
- ✅ Performance metrics for each specialist
- ✅ Saved to `trading_specialists.json`
- ✅ Ready for Phase 3 (meta-controller)

Estimated time: 4-6 hours of work
- Data prep: 30 min
- Framework: 1 hour
- Training: 2-3 hours (parallel if possible)
- Validation: 1 hour
- Documentation: 30 min

---

## 📁 Files to Create Tomorrow

```
label_historical_regimes.py          # Step 1: Add regime labels to data
trading_specialist.py                # Specialist class definition
specialist_trainer.py                # GA training framework
train_all_specialists.py             # Train all 4 specialists
validate_specialists.py              # Backtest and validate
trading_specialists.json             # Saved specialist genomes
PHASE_2_RESULTS.md                   # Documentation of results
```

---

## 💡 Quick Start for Tomorrow

1. **Run regime labeling**:
```bash
python label_historical_regimes.py
```

2. **Train specialists**:
```bash
python train_all_specialists.py
```

3. **Validate**:
```bash
python validate_specialists.py
```

4. **Review and commit**:
```bash
git add .
git commit -m "feat: Add trained trading specialists"
git push
```

---

**Status**: ✅ Phase 1 Complete and Committed  
**Ready**: Phase 2 prep complete  
**Tomorrow**: Train specialists 🚀  

Get some rest! Big day tomorrow! 💪😴
