# 🧬 Prisoner's Dilemma Evolution (Simple 3-Gene)

**Evolutionary Game Theory Simulation**

Based on John Holland's *"Hidden Order"* (Pages 82-84) and Axelrod's Tournaments

---

## 🎯 What Is This?

This simulates the evolution of strategies in the **Prisoner's Dilemma** game using a Genetic Algorithm. Agents with different strategies compete in a round-robin tournament, and the most successful strategies survive and reproduce.

### The Prisoner's Dilemma

Two players choose to either **Cooperate (C)** or **Defect (D)**:

| Your Move | Opponent Move | Your Payoff | Opponent Payoff |
|-----------|---------------|-------------|-----------------|
| C         | C             | 3 (Reward)  | 3 (Reward)      |
| D         | C             | 5 (Temptation) | 0 (Sucker)   |
| C         | D             | 0 (Sucker)  | 5 (Temptation)  |
| D         | D             | 1 (Punishment) | 1 (Punishment) |

The dilemma: **Mutual defection is worse than mutual cooperation, but defecting always gives a higher individual payoff.**

---

## 🧬 Chromosome Structure

Each agent has a **3-gene chromosome** representing their strategy:

```
Gene 0: Action on FIRST move (C or D)
Gene 1: Action if opponent's LAST move was 'C'
Gene 2: Action if opponent's LAST move was 'D'
```

### Famous Strategies

| Chromosome | Name | Description |
|------------|------|-------------|
| **CDC** | Tit-for-Tat | Start nice, then copy opponent's last move |
| **DDD** | Always Defect | Never cooperate, always betray |
| **CCC** | Always Cooperate | Always trust, never defect |
| **DDC** | Suspicious TFT | Start mean, but forgive cooperation |
| **CCD** | Grudger | Start nice, but never forgive defection |

---

## 🚀 Quick Start

### Run Basic Evolution
```bash
cd prisoner_dilemma_simple
python prisoner_evolution.py
```

This will:
- Create 50 agents with random strategies
- Run 50 generations of evolution
- Show the dominant strategies that emerge

### Create Visualization Dashboard
```bash
python visualize_prisoner.py
```

Creates a 4-panel dashboard showing:
1. Fitness evolution over time
2. Strategy diversity and convergence
3. Final population distribution
4. Strategy interpretations

---

## 📊 Expected Results

**Tit-for-Tat (CDC) usually wins!**

This matches Axelrod's famous tournaments from the 1980s. The winning strategy is typically:
- **Nice**: Never defect first
- **Retaliatory**: Punish defection immediately
- **Forgiving**: Return to cooperation after punishment

You'll see the population start with random strategies (`DCD`, `CDD`, `DDC`, etc.) and converge to `CDC` (Tit-for-Tat) or similar cooperative strategies.

---

## 🔬 How It Works

### 1. Initialization
```python
sim = PrisonerEvolution(
    population_size=50,     # Number of agents
    elite_size=5,           # Top agents that survive
    mutation_rate=0.15,     # Chance of gene mutation
    crossover_rate=0.7,     # Chance of crossover
    rounds_per_matchup=50   # Games per pair
)
```

### 2. Fitness Evaluation
Each agent plays against **every other agent** in a round-robin tournament. Fitness = total score across all games.

### 3. Selection
- Top 5 agents (elite) survive automatically
- Others selected probabilistically based on fitness

### 4. Reproduction
- **Crossover**: Two parents combine genes
- **Mutation**: Random gene flips (C↔D)

### 5. Repeat
New generation replaces old, and the cycle continues.

---

## 🎮 Experiments to Try

### Experiment 1: Always Defect World
Start with a population of all `DDD`:
```python
# Modify prisoner_evolution.py __init__
for i in range(population_size):
    chromosome = "DDD"  # All defectors
    self.population.append(PrisonerAgent(i, chromosome))
```

**Question**: Can cooperation emerge from pure defection?

### Experiment 2: Higher Mutation
```python
sim = PrisonerEvolution(mutation_rate=0.5)
```

**Question**: Does high mutation prevent convergence?

### Experiment 3: Longer Games
```python
sim = PrisonerEvolution(rounds_per_matchup=200)
```

**Question**: Do longer interactions favor cooperation more?

---

## 📚 Theory: Why Tit-for-Tat Wins

From Axelrod's research and Holland's analysis:

1. **Nice strategies** do well because mutual cooperation (3,3) beats mutual defection (1,1)
2. **Retaliatory strategies** prevent exploitation by always-defect
3. **Forgiving strategies** allow return to mutual cooperation
4. **Simple strategies** are more robust (less can go wrong)

Tit-for-Tat has all four properties!

---

## 🔗 Connection to Holland's "Hidden Order"

This implements concepts from Chapter 3 & 4:

- **Chromosomes**: The 3-gene strategy strings
- **Fitness**: Total score in tournament
- **Selection**: Tournament-style (like Figure 3.4)
- **Adaptation**: Strategies evolve without external guidance
- **Emergence**: Cooperation emerges from selfish competition!

The key insight: **The winning strategy wasn't programmed—it evolved naturally.**

---

## 📈 Next Steps

Ready to go deeper? Check out:

1. **`prisoner_dilemma_64gene/`** - Full 64-gene lookup table (3-move memory)
2. **`ga_trading_agents/`** - Apply these concepts to trading strategies
3. Add **tagging** - Let agents choose who to play against
4. Add **mutation** - Strategies can evolve during their lifetime
5. **Spatial structure** - Agents only play neighbors on a grid

---

## 🧠 Key Insight

> "The most effective strategy is not the most ruthless, but the most **cooperative** and **forgiving**—yet willing to defend itself."
> 
> — Robert Axelrod, *The Evolution of Cooperation*

This simulation proves that **cooperation can evolve through pure self-interest** when individuals interact repeatedly.

---

## 📖 References

- Holland, J. H. (1995). *Hidden Order: How Adaptation Builds Complexity* (Pages 82-84)
- Axelrod, R. (1984). *The Evolution of Cooperation*
- Axelrod, R., & Hamilton, W. D. (1981). "The Evolution of Cooperation", *Science*, 211(4489)

---

**Happy Evolving! 🧬**
