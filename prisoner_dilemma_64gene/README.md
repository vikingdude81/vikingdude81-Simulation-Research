# 🧬 Prisoner's Dilemma Evolution (64-Gene Lookup Table)

**Advanced Evolutionary Game Theory Simulation**

Based on John Holland's *"Hidden Order"* (Page 82) - The Full Implementation

---

## 🎯 What Is This?

This implements the **complete 64-gene chromosome** described on page 82 of Holland's book. Unlike the simple 3-gene version, this uses a full **lookup table** that remembers the last **3 joint moves** (not just 1).

### Why 64 Genes?

Each move creates 4 possible joint outcomes: `(C,C)`, `(C,D)`, `(D,C)`, `(D,D)`

With 3-move memory: $4 \times 4 \times 4 = 64$ unique histories

The chromosome is a **lookup table** mapping every possible history to an action.

**Total possible strategies**: $2^{64} \approx 18,446,744,073,709,551,616$ (18 quintillion!)

---

## 🧬 Chromosome Structure

### The Lookup Table

```
Index 0:  Action if history = [(C,C), (C,C), (C,C)]
Index 1:  Action if history = [(C,C), (C,C), (C,D)]
Index 2:  Action if history = [(C,C), (C,C), (D,C)]
...
Index 63: Action if history = [(D,D), (D,D), (D,D)]
```

### Index Calculation

```python
# Each joint move maps to a number:
(C,C) = 0, (C,D) = 1, (D,C) = 2, (D,D) = 3

# History: [oldest, middle, newest]
index = (oldest * 16) + (middle * 4) + newest
```

### Example

History: `[(D,C), (C,C), (C,D)]`

```
oldest = (D,C) = 2
middle = (C,C) = 0
newest = (C,D) = 1

index = (2 * 16) + (0 * 4) + 1 = 33

Action = chromosome[33]
```

---

## 🎯 Tit-for-Tat in 64 Genes

The famous **Tit-for-Tat** strategy only looks at the opponent's **last move** (ignoring earlier history):

```
If opponent's last move = C → Play C
If opponent's last move = D → Play D
```

In the 64-gene encoding, this becomes:

```
"CDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCD"
```

**Pattern**: Alternating C/D based on whether the index is even (opponent cooperated last) or odd (opponent defected last).

---

## 🚀 Quick Start

### Run Evolution
```bash
cd prisoner_dilemma_64gene
python prisoner_64gene.py
```

### Create Visualization Dashboard
```bash
python visualize_64gene.py
```

Creates a comprehensive 5-panel dashboard:
1. **Fitness Evolution** - Best and average over generations
2. **TFT Similarity** - Convergence to Tit-for-Tat pattern
3. **Chromosome Heatmap** - Visual lookup table (8x8 grid)
4. **Gene Distribution** - C vs D ratio over time
5. **Strategy Analysis** - Text interpretation

---

## 📊 Expected Results

**Tit-for-Tat (or close variants) will dominate!**

You'll typically see:
- **Initial**: Random 64-gene strings (50% C, 50% D)
- **Mid Evolution**: Strategies with high C-percentage emerge (60-70% C)
- **Final**: Near-perfect TFT (>90% similarity) or TFT-variants

### Why TFT Works

Even in the vast 64-gene space:
1. **Nice**: High percentage of C genes prevents exploitation
2. **Retaliatory**: D genes activate when opponent defects
3. **Forgiving**: Returns to C after punishment
4. **Simple Pattern**: The CD-CD-CD pattern is robust to noise

---

## 🔬 How It Works

### Evolution Loop

```python
sim = AdvancedPrisonerEvolution(
    population_size=50,
    elite_size=5,
    mutation_rate=0.01,     # Lower than 3-gene (more genes to mutate)
    crossover_rate=0.7,
    rounds_per_matchup=100  # Longer games for 3-move memory
)

sim.run(generations=100)
```

### Fitness Evaluation

Each agent plays **every other agent** in a round-robin tournament:
- 100 rounds per matchup
- Both agents maintain 3-move history
- Fitness = total accumulated score

### Genetic Operators

**Crossover**: 
```python
parent1: "CDCDCDCD...CDCD"
parent2: "DDCCDDCC...DDCC"
         ↓ cut at position 32
child:   "CDCDCDCD...DDCC"
```

**Mutation**: Each gene has 1% chance to flip (C↔D)

---

## 🎮 Experiments to Try

### Experiment 1: Seed with Tit-for-Tat
Start one agent with perfect TFT:

```python
# In prisoner_64gene.py __init__
self.population[0] = AdvancedPrisonerAgent(0, create_tit_for_tat())
```

**Question**: Does TFT spread faster or slower than evolving naturally?

### Experiment 2: Different Mutation Rates

```python
sim = AdvancedPrisonerEvolution(mutation_rate=0.05)  # 5x higher
```

**Question**: Does higher mutation prevent convergence to TFT?

### Experiment 3: Longer Memory

Extend to **4-move history** (256 genes):
- Change `create_random_chromosome()` to 256 genes
- Update history to 4 moves
- Adjust index calculation: `(h0*64) + (h1*16) + (h2*4) + h3`

**Question**: Does longer memory help or hurt?

### Experiment 4: Noise

Add 5% chance agents make "mistakes":

```python
# In play_prisoner_dilemma()
move1 = agent1.get_move()
if random.random() < 0.05:  # 5% error rate
    move1 = 'D' if move1 == 'C' else 'C'
```

**Question**: Can TFT survive in a noisy environment?

---

## 📈 Visualization Details

### Chromosome Heatmap

The 64-gene string is displayed as an **8×8 grid**:

```
Row 0: Genes 0-7   (History starts (C,C), (C,C), ...)
Row 1: Genes 8-15  (History starts (C,C), (C,D), ...)
...
Row 7: Genes 56-63 (History starts (D,D), (D,D), ...)
```

- **Green cells** = Cooperate (C)
- **Red cells** = Defect (D)

**Perfect TFT** shows a checkerboard pattern (alternating green/red).

### TFT Similarity Chart

Tracks how many of the 64 genes match the TFT chromosome:

```
Similarity = (matching genes / 64) * 100%
```

You'll see this climb from ~50% (random) to >90% (near-TFT).

---

## 🧠 Key Insights from Holland

From pages 82-84 of *Hidden Order*:

1. **"The chromosome is a policy"**: Each 64-gene string is a complete strategy for all situations

2. **"Adaptation discovers robust strategies"**: Evolution finds TFT without being told what to look for

3. **"Schemas emerge"**: The CD-CD-CD pattern is a **building block** (schema) that GA discovers

4. **"Competition drives cooperation"**: Selfish agents evolve to cooperate for mutual benefit

---

## 🔗 Comparison to Simple 3-Gene Version

| Feature | 3-Gene | 64-Gene |
|---------|--------|---------|
| **Memory** | 1 move | 3 moves |
| **Genes** | 3 | 64 |
| **Possible Strategies** | 8 | 18 quintillion |
| **Complexity** | Simple | Complex |
| **Convergence Speed** | Fast (10-30 gen) | Slower (50-100 gen) |
| **Best Strategy** | CDC (TFT) | CD-pattern (TFT) |

**Key Difference**: The 64-gene version can encode **conditional** strategies:
- "If we've both cooperated for 3 moves, keep cooperating"
- "If opponent defected twice in a row, defect"
- "If I defected but opponent cooperated, try to repair"

These nuanced strategies are impossible in the 3-gene version.

---

## 📚 Theory: Schema and Building Blocks

Holland's **Schema Theorem** explains why TFT emerges:

The CD-CD-CD pattern is a **schema** (template):
```
**CD**CD**CD**CD**CD**  (where * = don't care)
```

This schema says: "In even positions, use C; in odd positions, use D"

**Why it's good**:
- It appears in many different chromosomes
- Crossover preserves it
- It has high fitness

The GA **implicitly** discovers and **propagates** this building block!

---

## 🎯 Connection to Trading Strategies

This same principle applies to `ga_trading_agents/`:

| Prisoner's Dilemma | Trading |
|-------------------|---------|
| 64 genes = 64 market histories | 5 genes = 5 market conditions |
| Action = C or D | Action = BUY/SELL/HOLD |
| Fitness = Game score | Fitness = Portfolio value |
| TFT emerges | Profitable strategies emerge |

**Next step**: Combine the memory-based approach from 64-gene with trading signals!

---

## 📖 References

- Holland, J. H. (1995). *Hidden Order* (Page 82: "64-Position Classifier System")
- Axelrod, R. (1984). *The Evolution of Cooperation*
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*

---

**Ready to evolve 18 quintillion strategies! 🧬**
