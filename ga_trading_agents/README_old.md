# 🧬 Genetic Algorithm Trading Agents
**Combining Holland's Hidden Order Concepts with Crypto Trading**

## 🎯 Project Overview

This project implements a hybrid system combining:
1. **Genetic Algorithm (GA)** - Evolves trading strategies
2. **Agent-Based Model (ABM)** - Simulates market with competing agents
3. **Real Crypto Data** - Uses actual price data from your ML pipeline

## 📚 Based on "Hidden Order" by John Holland

### Key Concepts Implemented:
- **Chromosomes** = Trading strategies encoded as strings (e.g., "B-S-H-B-S")
- **Fitness** = Agent's profit/wealth after trading period
- **Selection** = Best traders pass strategies to next generation
- **Crossover** = Combine successful strategies from two parent traders
- **Mutation** = Random strategy changes for exploration
- **Emergence** = Market behavior emerges from agent interactions

### Connection to Book Examples:
- **Echo Model (Fig 4.2)** → Our trading market simulation
- **IPD Tournament (Chapter 4)** → Our strategy competition
- **Resource Exchange (Fig 3.3)** → Our crypto trading
- **Adaptation (Chapter 5)** → Our GA optimization

## 🏗️ Project Structure

```
ga_trading_agents/
├── README.md                      # This file
├── simple_ga.py                   # Basic GA demo (evolve string)
├── simple_abm.py                  # Basic ABM demo (Mesa agents)
├── trading_agent.py               # Trading agent with GA chromosome
├── market_model.py                # Mesa market simulation
├── strategy_evolution.py          # Combined GA+ABM system
├── run_simulation.py              # Main entry point
└── analysis_tools.py              # Visualization and analysis
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install mesa matplotlib pandas numpy
```

### 2. Run Basic Examples
```bash
# Try the basic GA (evolve "HIDDENORDER")
python simple_ga.py

# Try the basic ABM (wealth distribution)
python simple_abm.py
```

### 3. Run Trading Simulation
```bash
# Run full GA + trading simulation
python run_simulation.py
```

## 🧬 How It Works

### Phase 1: Initialization
- Create 100 trading agents with random strategy chromosomes
- Each chromosome encodes: Buy/Sell/Hold decisions based on market conditions
- Example: `"B-S-H-B-H"` = "Buy on upturn, Sell on downturn, Hold on flat..."

### Phase 2: Market Simulation (ABM)
- Agents trade on real crypto price data
- Each agent follows its chromosome strategy
- Track profit/loss for each agent (fitness score)

### Phase 3: Evolution (GA)
- Select top 20% of profitable agents
- Breed them (crossover) to create new strategies
- Mutate slightly for exploration
- Replace worst performers with new generation

### Phase 4: Repeat
- Run for 50-100 generations
- Watch strategies evolve and improve
- Discover emergent trading patterns

## 📊 Expected Outcomes

### Generation 1:
- Random strategies
- ~50% win rate
- Wild profit variance

### Generation 50:
- Evolved strategies
- ~65-75% win rate
- Discovered market patterns
- Emergent cooperation/competition

### Emergent Phenomena:
- **Momentum traders** may emerge (buy rising, sell falling)
- **Contrarian traders** may emerge (buy dips, sell peaks)
- **Market efficiency** improves over time
- **Strategy diversity** maintained via mutation

## 🎯 Strategy Chromosome Format

Each agent has a chromosome string encoding its strategy:

```
Format: "G1-G2-G3-G4-G5-G6-G7-G8"

Where each gene (G) can be:
- B = Buy signal
- S = Sell signal  
- H = Hold signal

Applied to conditions:
G1: Strong uptrend (price +2%+)
G2: Weak uptrend (price +0.5% to +2%)
G3: Flat (price -0.5% to +0.5%)
G4: Weak downtrend (price -0.5% to -2%)
G5: Strong downtrend (price -2%-) 
G6: High volume
G7: Low volume
G8: Random exploration
```

Example chromosomes:
- `"B-B-H-S-S-H-H-B"` = Momentum trader
- `"S-H-H-H-B-B-H-S"` = Contrarian trader
- `"H-H-H-H-H-H-H-H"` = Passive holder

## 📈 Metrics Tracked

### Agent Level:
- Total wealth/profit
- Win rate (% profitable trades)
- Trade count
- Strategy chromosome
- Generation born

### Population Level:
- Average fitness per generation
- Best fitness per generation
- Strategy diversity (unique chromosomes)
- Convergence rate

### Market Level:
- Price impact from agent trading
- Liquidity (trade volume)
- Volatility
- Efficiency (spread)

## 🔬 Research Questions

1. What strategies naturally evolve?
2. Do different market conditions favor different strategies?
3. How does diversity affect market stability?
4. Can GA discover better strategies than hand-coded rules?
5. Do cooperative behaviors emerge?

## 🎨 Visualizations

The system generates:
1. **Evolution Chart** - Fitness over generations
2. **Strategy Distribution** - Histogram of evolved strategies
3. **Wealth Distribution** - Agent wealth inequality (Gini curve)
4. **Trading Patterns** - Buy/sell signals over time
5. **Market Impact** - Price movements from agent actions

## 🏆 Success Criteria

A successful evolution should show:
- ✅ Increasing average fitness over generations
- ✅ Emergence of dominant strategy patterns
- ✅ Maintained diversity (not all identical)
- ✅ Better than random trading (>50% win rate)
- ✅ Adaptation to market regime changes

## 🚀 Next Steps

1. Run basic examples to understand components
2. Run full simulation with crypto data
3. Analyze evolved strategies
4. Test on out-of-sample data
5. Compare to your ML models
6. Hybrid approach: GA + ML ensemble?

---

**Let's evolve some winning trading strategies! 🧬📈**
