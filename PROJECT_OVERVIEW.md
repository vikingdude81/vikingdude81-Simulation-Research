# 🧬 Three Evolutionary Systems - Project Overview

**Modular GA & ABM Implementations for Trading & Game Theory**

Created: October 30, 2025

---

## 📁 Project Structure

You now have **THREE independent, modular evolutionary systems**:

```
PRICE-DETECTION-TEST-1/
├── ga_trading_agents/          # Trading strategy evolution
│   ├── simple_ga.py
│   ├── simple_abm_standalone.py
│   ├── trading_agent.py
│   ├── strategy_evolution.py
│   ├── watch_evolution.py
│   ├── visualize_evolution.py
│   ├── evolve_on_real_data.py
│   └── README.md
│
├── prisoner_dilemma_simple/    # 3-gene game theory
│   ├── prisoner_evolution.py
│   ├── visualize_prisoner.py
│   └── README.md
│
└── prisoner_dilemma_64gene/    # 64-gene lookup table
    ├── prisoner_64gene.py
    ├── visualize_64gene.py
    └── README.md
```

---

## 🎯 System 1: GA Trading Agents

**Purpose**: Evolve profitable trading strategies for crypto markets

### Key Features
- **5-gene chromosome**: Maps market conditions → trading actions
- **Market conditions**: TRENDING_UP, TRENDING_DOWN, VOLATILE, STABLE, BREAKOUT
- **Actions**: BUY, SELL, HOLD
- **Fitness**: Portfolio value after trading
- **Real data**: Tests on actual BTC/ETH/SOL price history

### Example Strategy
```
Chromosome: "B-S-H-B-H"
├─ TRENDING_UP → BUY
├─ TRENDING_DOWN → SELL
├─ VOLATILE → HOLD
├─ STABLE → BUY
└─ BREAKOUT → HOLD

Result: +36.4% portfolio growth
```

### Run It
```bash
cd ga_trading_agents
python evolve_on_real_data.py  # Test on crypto data
python watch_evolution.py       # Watch live
python visualize_evolution.py   # Create dashboard
```

### Latest Results
- ✅ Ran 30 generations successfully
- ✅ Strategy "B-S-H-B-H" emerged as dominant
- ✅ Final portfolio: $13,640 from $10,000 initial
- ✅ All code committed to GitHub (commit f5ef7ee)

---

## 🎯 System 2: Prisoner's Dilemma (Simple)

**Purpose**: Evolve cooperation strategies using 3-gene memory

### Key Features
- **3-gene chromosome**: [First move, vs Cooperate, vs Defect]
- **Game**: Prisoner's Dilemma (Cooperate/Defect)
- **Fitness**: Total score in round-robin tournament
- **Famous strategies**:
  - `CDC` = Tit-for-Tat (copy opponent)
  - `DDD` = Always Defect
  - `CCC` = Always Cooperate
  - `CCD` = Grudger (never forgive)

### Example Evolution
```
Generation 1:  Mixed population (CDC, DDD, CCC, etc.)
Generation 10: DDD dominates (84%)
Generation 50: DDD wins with fitness 3434

Interpretation: In this run, defectors took over!
```

### Run It
```bash
cd prisoner_dilemma_simple
python prisoner_evolution.py      # Run evolution
python visualize_prisoner.py      # Create dashboard
```

### Theory
Based on:
- John Holland's "Hidden Order" (Pages 82-84)
- Robert Axelrod's tournaments (1980s)
- Evolution of Cooperation research

**Key Insight**: Cooperation CAN evolve, but depends on initial conditions!

---

## 🎯 System 3: Prisoner's Dilemma (64-Gene)

**Purpose**: Full 3-move memory lookup table (Holland's page 82)

### Key Features
- **64-gene chromosome**: Complete strategy for all 3-move histories
- **History encoding**: 4^3 = 64 possible 3-move sequences
- **Index formula**: `(oldest * 16) + (middle * 4) + newest`
- **Tit-for-Tat**: `"CDCDCDCD..."` (alternating pattern)
- **Search space**: 2^64 ≈ 18 quintillion strategies!

### Example Chromosome
```
Index 0:  Gene for history [(C,C), (C,C), (C,C)]
Index 1:  Gene for history [(C,C), (C,C), (C,D)]
...
Index 63: Gene for history [(D,D), (D,D), (D,D)]
```

### Visualization
Creates 8×8 heatmap showing:
- Green cells = Cooperate
- Red cells = Defect
- Checkerboard pattern = Tit-for-Tat

### Run It
```bash
cd prisoner_dilemma_64gene
python prisoner_64gene.py        # Run evolution (100 gen)
python visualize_64gene.py       # Create dashboard
```

### Expected Convergence
- **Generation 1**: Random (50% C, 50% D)
- **Generation 50**: TFT-like (70-90% similarity)
- **Generation 100**: Near-perfect TFT (>90%)

---

## 🔗 Connections Between Systems

### Shared Concepts
All three use Holland's framework:

| Concept | Trading | Simple PD | 64-Gene PD |
|---------|---------|-----------|------------|
| **Chromosome** | 5 market→action genes | 3 memory genes | 64 history→action genes |
| **Fitness** | Portfolio value | Game score | Tournament score |
| **Selection** | Tournament | Tournament | Tournament |
| **Crossover** | Single-point | Single-point | Single-point |
| **Mutation** | Random flip | Random C↔D | Random C↔D |
| **Emergence** | Profitable strategies | Cooperation | Tit-for-Tat pattern |

### Key Differences

**Complexity**:
- Trading: 5 genes → 3^5 = 243 possible strategies
- Simple PD: 3 genes → 2^3 = 8 possible strategies
- 64-Gene PD: 64 genes → 2^64 = 18 quintillion strategies

**Evolution Speed**:
- Trading: 15-30 generations
- Simple PD: 10-30 generations
- 64-Gene PD: 50-100 generations

### Why Keep Them Separate?

1. **Learning**: Simple PD teaches GA/ABM basics
2. **Theory**: 64-gene demonstrates Holland's exact system
3. **Application**: Trading shows real-world usage
4. **Modularity**: Can experiment with each independently
5. **Cross-pollination**: Can borrow features between systems

---

## 🔬 Future Integration Ideas

### Idea 1: Memory-Based Trading
Bring 64-gene lookup table concept to trading:
- **Current**: 5 market conditions
- **Enhanced**: 3-period history of market conditions
- **Genes**: 5^3 = 125 genes (one for each history sequence)
- **Benefit**: Agents remember recent market behavior

### Idea 2: Spatial Trading
Bring ABM grid concept to trading:
- Agents on a grid (like `simple_abm_standalone.py`)
- Each agent trades only with neighbors
- Successful strategies spread geographically
- **Emergence**: Clusters of similar strategies form

### Idea 3: Tagged Strategies
Add tags from Holland's system:
- Each strategy has a "tag" (e.g., "aggressive", "conservative")
- Agents choose which strategies to compete against
- Co-evolution of tags and strategies
- **Benefit**: Strategy diversity maintained

### Idea 4: Multi-Agent Trading
Combine all three:
- Population of traders (GA Trading Agents)
- Traders play Prisoner's Dilemma for partnerships
- Partners cooperate on trades
- 64-gene memory for partner history
- **Result**: Social trading networks evolve

### Idea 5: Hybrid Ensemble
Combine evolved GA strategies with ML models:
- XGBoost predicts price direction
- LSTM predicts volatility
- GA strategy decides BUY/SELL/HOLD
- **Ensemble**: Combine all signals with learned weights

---

## 📊 Current Status

### ✅ Completed
- [x] GA Trading Agents system (30 gen, working)
- [x] Prisoner's Dilemma Simple (50 gen, working)
- [x] Prisoner's Dilemma 64-gene (ready to test)
- [x] Real-time visualizations for trading
- [x] Static dashboards for all three
- [x] Comprehensive documentation
- [x] GitHub integration (ml-pipeline-full branch)

### 🔄 Ready to Test
- [ ] Run 64-gene evolution (100 generations)
- [ ] Create visualizations for 64-gene
- [ ] Test on different initial conditions
- [ ] Compare convergence rates

### 🚀 Next Steps
1. **Test 64-gene system** fully
2. **Commit new Prisoner's Dilemma systems** to GitHub
3. **Experiment** with parameters in all three
4. **Compare results** across systems
5. **Begin integration** (pick one idea from above)

---

## 🧠 Theoretical Foundation

All three systems implement concepts from:

**John Holland (1995) - "Hidden Order"**
- Chapter 3: Agent-Based Models
- Chapter 4: Genetic Algorithms
- Page 82: 64-position classifier systems
- Page 84: Axelrod's tournaments

**Key Principles**:
1. **Building Blocks** (Schemas): Good patterns are discovered and propagated
2. **Emergence**: Complex behaviors arise from simple rules
3. **Adaptation**: No centralized control needed
4. **Competition**: Drives improvement without explicit objectives

**Why It Works**:
- **Exploration**: Mutation finds new strategies
- **Exploitation**: Selection keeps good strategies
- **Balance**: Crossover mixes good parts from different strategies
- **Robustness**: Population diversity prevents premature convergence

---

## 📖 References

- Holland, J. H. (1995). *Hidden Order: How Adaptation Builds Complexity*
- Axelrod, R. (1984). *The Evolution of Cooperation*
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
- Mitchell, M. (1996). *An Introduction to Genetic Algorithms*

---

## 🎯 Quick Command Reference

```bash
# GA Trading Agents
cd ga_trading_agents
python evolve_on_real_data.py       # Test on BTC/ETH/SOL
python watch_evolution.py           # Live visualization
python visualize_evolution.py       # Static dashboard

# Simple Prisoner's Dilemma
cd prisoner_dilemma_simple
python prisoner_evolution.py        # 50 generations
python visualize_prisoner.py        # Create dashboard

# 64-Gene Prisoner's Dilemma
cd prisoner_dilemma_64gene
python prisoner_64gene.py           # 100 generations
python visualize_64gene.py          # Create dashboard

# Check all systems
cd ..
dir ga_trading_agents
dir prisoner_dilemma_simple
dir prisoner_dilemma_64gene
```

---

**Happy Evolving! 🧬🎨📈**
