# 📊 Quantum Genome Data Analysis

**Generated**: November 2, 2025  
**Purpose**: Complete analysis of all evolved genomes, JSON structures, and fitness patterns

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [JSON Data Structure](#json-data-structure)
3. [Genome Types](#genome-types)
4. [Genome Comparison](#genome-comparison)
5. [Evolution Strategies](#evolution-strategies)
6. [Fitness Analysis](#fitness-analysis)
7. [Environment Performance](#environment-performance)
8. [Key Insights](#key-insights)

---

## 1. Overview

### **Available Genomes** (14 Production Files)

Located in: `quantum_genetics/data/genomes/production/`

| Filename | Type | Generation | Fitness | Strategy |
|----------|------|------------|---------|----------|
| **co_evolved_best_gen_5878.json** | Co-evolved | 5878 | -0.00491 | Long co-evolution |
| **co_evolved_best_gen_2117.json** | Co-evolved | 2117 | 1.6788 | Mid co-evolution |
| **co_evolved_best_gen_770.json** | Co-evolved | 770 | 1.5350 | Early co-evolution |
| **best_individual_genome.json** | Best individual | 300 | 3.60×10¹¹ | Standard evolution |
| **best_individual_hybrid_genome.json** | Best individual | 400 | 1.62×10¹⁵ | Hybrid ensemble |
| **best_individual_long_evolution_genome.json** | Best individual | 500+ | High | Extended evolution |
| **best_individual_more_populations_genome.json** | Best individual | 300+ | High | Multi-population |
| **averaged_ensemble_genome.json** | Ensemble average | 300 | 1.4354 | 15 populations |
| **averaged_hybrid_genome.json** | Ensemble average | 400 | High | Hybrid averaging |
| **averaged_long_evolution_genome.json** | Ensemble average | 500+ | High | Extended averaging |
| **averaged_more_populations_genome.json** | Ensemble average | 300+ | High | Multi-population avg |
| **phase_focused_best.json** | Phase-optimized | ? | 2.73×10¹² | Phase evolution |
| **custom_1761985337_genome.json** | Custom | 0 | 0.0 | Manual design |
| **custom_1761985388_genome.json** | Custom | 0 | 0.0 | Manual design |

---

## 2. JSON Data Structure

### **Standard Genome JSON Format**

```json
{
  "genome": {
    "mutation_rate": float,      // μ: 0.01-6.5
    "oscillation_freq": float,   // ω: 0.2-2.0
    "decoherence_rate": float,   // d: 0.01-0.1
    "phase_offset": float        // φ: 0.0-6.28 (0-2π)
  },
  "export_timestamp": "ISO-8601 timestamp",
  "metadata": {
    "type": "evolution_strategy",
    "generation": int,
    "fitness": float,
    "...": "additional fields based on type"
  }
}
```

### **Metadata Fields by Type**

#### **Type: best_individual**
```json
{
  "fitness": float,
  "generation": int,
  "population_id": int,
  "type": "best_individual",
  "environment_performance": {
    "standard": {"fitness": float, "std": float},
    "harsh": {"fitness": float, "std": float},
    "gentle": {"fitness": float, "std": float},
    "chaotic": {"fitness": float, "std": float},
    "oscillating": {"fitness": float, "std": float}
  },
  "robustness_score": float  // Average across environments
}
```

#### **Type: co_evolved**
```json
{
  "type": "co_evolved",
  "generation": int,
  "fitness": float,
  "interactions": int,          // Total agent-agent interactions
  "successful_learns": int      // Successful learning events
}
```

#### **Type: averaged_ensemble**
```json
{
  "fitness": float,
  "generation": int,
  "type": "averaged_ensemble",
  "n_populations": int,         // Number of populations averaged
  "environment_performance": {...},
  "robustness_score": float
}
```

#### **Type: phase_focused**
```json
{
  "fitness": float,
  "phase": float,               // Target phase value
  "mutation": float,            // Mutation rate used
  "decoherence": float,
  "type": "phase_focused",
  "strategy": "strong_elitism"  // Evolution strategy
}
```

#### **Type: custom**
```json
{
  "type": "custom",
  "created": "ISO-8601 timestamp",
  "fitness": 0.0,               // Not yet tested
  "parameters": {               // Mirror of genome
    "mutation_rate": float,
    "oscillation_freq": float,
    "decoherence_rate": float,
    "phase_offset": float
  }
}
```

---

## 3. Genome Types

### **3.1 Co-Evolved Genomes** 
Evolution through agent-agent interactions and learning.

#### **Gen 770 (Early Co-evolution)**
```json
{
  "genome": {
    "mutation_rate": 0.7277,
    "oscillation_freq": 1.6946,
    "decoherence_rate": 0.0183,
    "phase_offset": 0.5118
  },
  "metadata": {
    "generation": 770,
    "fitness": 1.5350,
    "interactions": 259,
    "successful_learns": 91
  }
}
```
**Analysis**: 
- High oscillation freq (1.69) - highly dynamic agent
- Moderate mutation (0.73) - balanced exploration
- Learning success rate: 91/259 = 35%

#### **Gen 2117 (Mid Co-evolution)**
```json
{
  "genome": {
    "mutation_rate": 0.6885,
    "oscillation_freq": 0.2482,
    "decoherence_rate": 0.0183,
    "phase_offset": 0.5262
  },
  "metadata": {
    "generation": 2117,
    "fitness": 1.6788,
    "interactions": 734,
    "successful_learns": 262
  }
}
```
**Analysis**:
- Lower oscillation (0.25) - more stable agent
- Increased interactions (734 vs 259) - more experience
- Learning success rate: 262/734 = 36% (slight improvement)
- **9.4% fitness increase** over Gen 770

#### **Gen 5878 (Advanced Co-evolution)**
```json
{
  "genome": {
    "mutation_rate": 0.3171,
    "oscillation_freq": 0.2570,
    "decoherence_rate": 0.0111,
    "phase_offset": 0.1839
  },
  "metadata": {
    "generation": 5878,
    "fitness": -0.0049,
    "interactions": 1932,
    "successful_learns": 687
  }
}
```
**Analysis**:
- Much lower mutation (0.32) - refined, less exploration needed
- Very low decoherence (0.011) - excellent coherence preservation
- Massive interactions (1932) - extensive learning
- Learning success rate: 687/1932 = 36% (consistent)
- **NEGATIVE fitness**: Possible overfitting or different fitness function

**Co-Evolution Trend**:
```
Gen 770  → Gen 2117 → Gen 5878
μ: 0.73  →  0.69     →  0.32    (Decreasing mutation rate)
ω: 1.69  →  0.25     →  0.26    (Stabilizing oscillation)
d: 0.018 →  0.018    →  0.011   (Improving coherence)
I: 259   →  734      →  1932    (Increasing interactions)
S: 91    →  262      →  687     (More learning events)
```

---

### **3.2 Best Individual Genomes**
Top-performing agents from standard genetic algorithm.

#### **Standard Evolution (Gen 300)**
```json
{
  "genome": {
    "mutation_rate": 1.5410,
    "oscillation_freq": 0.9630,
    "decoherence_rate": 0.0110,
    "phase_offset": 0.5535
  },
  "metadata": {
    "fitness": 3.60×10¹¹,
    "generation": 300,
    "environment_performance": {
      "gentle": {"fitness": 7.34×10¹³, "std": 1.47×10¹⁴},
      "standard": {"fitness": 1.23×10⁵, "std": 2.47×10⁵},
      "oscillating": {"fitness": 326.77, "std": 654.90},
      "chaotic": {"fitness": 72.71, "std": 97.97},
      "harsh": {"fitness": 0.00053, "std": 0.0013}
    },
    "robustness_score": 2.94×10¹³
  }
}
```
**Analysis**:
- **EXTREME fitness in gentle environment** (10¹³ scale)
- Very high mutation rate (1.54) - unusual for GA
- Excellent decoherence (0.011) - matches co-evolution best
- **Environment-dependent**: 10¹³× difference between gentle/harsh

#### **Hybrid Evolution (Gen 400)**
```json
{
  "genome": {
    "mutation_rate": 3.0329,
    "oscillation_freq": 0.8639,
    "decoherence_rate": 0.0159,
    "phase_offset": 0.2626
  },
  "metadata": {
    "fitness": 1.62×10¹⁵,
    "generation": 400,
    "experiment": "Hybrid",
    "config": {
      "n_ensemble": 20,
      "population_size": 30,
      "generations": 400
    },
    "environment_performance": {
      "harsh": {"fitness": 2.03×10¹⁵, "std": 4.06×10¹⁵},
      "chaotic": {"fitness": 1.11×10¹⁵, "std": 2.22×10¹⁵},
      "oscillating": {"fitness": 1.89×10⁷, "std": 3.77×10⁷},
      "standard": {"fitness": -53682, "std": 107365},
      "gentle": {"fitness": -98252, "std": 215367}
    },
    "robustness_score": 1.26×10¹⁵
  }
}
```
**Analysis**:
- **HIGHEST mutation rate** (3.03) - extreme exploration
- **Inverse environment preference**: Excels in harsh/chaotic
- Negative fitness in standard/gentle - specialized genome
- **Ultra-high fitness**: 10¹⁵ scale in harsh environments

---

### **3.3 Phase-Focused Genome**
Optimized for phase stability and control.

```json
{
  "genome": {
    "mutation_rate": 2.9728,
    "oscillation_freq": 0.7442,
    "decoherence_rate": 0.011,
    "phase_offset": 3.1864
  },
  "metadata": {
    "fitness": 2.73×10¹²,
    "phase": 3.1864,
    "strategy": "strong_elitism"
  }
}
```
**Analysis**:
- **Extreme phase offset** (3.19 ≈ π) - opposite phase
- Very high mutation (2.97) - similar to hybrid
- Perfect decoherence (0.011) - optimal value
- **Phase-specific fitness**: 10¹² scale

---

### **3.4 Averaged Ensemble Genome**
Statistical average across 15 populations.

```json
{
  "genome": {
    "mutation_rate": 0.6349,
    "oscillation_freq": 0.6968,
    "decoherence_rate": 0.0120,
    "phase_offset": 0.3339
  },
  "metadata": {
    "fitness": 1.4354,
    "n_populations": 15,
    "environment_performance": {
      "gentle": {"fitness": 2661.81, "std": 5322.50},
      "oscillating": {"fitness": 179.88, "std": 360.04},
      "harsh": {"fitness": 1.50, "std": 2.71},
      "standard": {"fitness": 1.44, "std": 2.96},
      "chaotic": {"fitness": 1.15, "std": 2.29}
    },
    "robustness_score": 1138.10
  }
}
```
**Analysis**:
- **Moderate all parameters** - balanced consensus
- Much lower fitness than best individuals (averaging effect)
- **Stable across environments** - low variance
- **Generalizes well** - robustness score 1138

---

### **3.5 Custom Genomes**
Manually designed genomes for testing hypotheses.

```json
{
  "genome": {
    "mutation_rate": 0.06,
    "oscillation_freq": 1.98,
    "decoherence_rate": 0.022,
    "phase_offset": 0.06
  },
  "metadata": {
    "type": "custom",
    "fitness": 0.0,
    "parameters": {...}
  }
}
```
**Analysis**:
- Very low mutation (0.06) - minimal exploration
- High oscillation (1.98) - very dynamic
- Higher decoherence (0.022) - faster coherence loss
- **Not yet tested** - fitness = 0.0

---

## 4. Genome Comparison

### **Parameter Ranges Observed**

| Parameter | Min | Max | Mean | Std Dev | Optimal Range |
|-----------|-----|-----|------|---------|---------------|
| **mutation_rate** | 0.06 | 3.03 | 1.18 | 0.95 | 0.011 (low) or 2.5-3.0 (high) |
| **oscillation_freq** | 0.25 | 1.98 | 0.81 | 0.56 | 0.7-1.0 (balanced) |
| **decoherence_rate** | 0.011 | 0.022 | 0.014 | 0.004 | 0.011 (optimal) |
| **phase_offset** | 0.06 | 3.19 | 0.61 | 0.87 | 0.25-0.55 (typical) |

### **Bimodal Mutation Rate Distribution**

```
Low Mutation Group (Co-evolution):
  μ ∈ [0.3, 0.7]  →  Fitness: 1.5-1.7
  
High Mutation Group (Best individuals, phase-focused):
  μ ∈ [1.5, 3.0]  →  Fitness: 10¹¹-10¹⁵
  
Custom/Conservative:
  μ = 0.06        →  Untested
```

**Hypothesis**: High mutation rates act as "energy injection" rather than exploration, leading to extreme fitness values in certain environments.

---

## 5. Evolution Strategies

### **5.1 Standard Genetic Algorithm**
- Files: `best_individual_*.json`, `averaged_*.json`
- Population: 30 agents
- Generations: 300-500
- Selection: Elitism (top 3) + fitness-proportional
- Crossover: Single-point
- Mutation: 10% probability, Gaussian noise

### **5.2 Co-Evolution**
- Files: `co_evolved_best_gen_*.json`
- Mechanism: Agent-agent interactions with learning
- Metrics: interactions, successful_learns
- Trend: Decreasing mutation, increasing stability over time

### **5.3 Hybrid Ensemble**
- Files: `best_individual_hybrid_genome.json`
- Config: 20 ensembles, 30 agents each, 400 generations
- Result: **Highest fitness** (10¹⁵ scale)
- Specialization: Thrives in harsh environments

### **5.4 Phase-Focused Evolution**
- Files: `phase_focused_best.json`
- Strategy: Strong elitism for phase stability
- Target: π phase (3.14) achieved (3.19)
- Result: 10¹² scale fitness

### **5.5 Multi-Population**
- Files: `*_more_populations_genome.json`
- Mechanism: Multiple independent populations
- Benefit: Diversity preservation
- Output: Both best individual and averaged ensemble

### **5.6 Long Evolution**
- Files: `*_long_evolution_genome.json`
- Duration: 500+ generations
- Benefit: Convergence to stable optima
- Output: Both best individual and averaged ensemble

---

## 6. Fitness Analysis

### **Fitness Ranges by Strategy**

| Strategy | Min Fitness | Max Fitness | Typical Range |
|----------|-------------|-------------|---------------|
| **Co-evolution** | -0.0049 | 1.68 | 1.5-1.7 |
| **Standard GA** | 1.44 | 3.60×10¹¹ | 10⁵-10¹¹ |
| **Hybrid** | 1.62×10¹⁵ | 1.62×10¹⁵ | 10¹⁵ |
| **Phase-focused** | 2.73×10¹² | 2.73×10¹² | 10¹² |
| **Ensemble avg** | 1.15 | 2661.81 | 1-3000 |

### **Fitness Calculation**

```python
def get_final_fitness(self):
    fitness_values = self.fitness_history[20:]  # Skip initialization
    
    avg_fitness = np.mean(fitness_values)
    stability = 1 / (1 + np.std(fitness_values))
    
    coherence_decay = 1 - self.traits[1]  # 1 - final coherence
    longevity_penalty = np.exp(-coherence_decay * 2)
    
    return avg_fitness * stability * longevity_penalty
```

**Components**:
1. **avg_fitness**: Mean fitness over lifetime
2. **stability**: Inverse variance (rewards consistency)
3. **longevity_penalty**: Penalizes coherence loss

---

## 7. Environment Performance

### **Best Individual Genome - Environment Response**

| Environment | Fitness | Std Dev | Rank |
|-------------|---------|---------|------|
| **Gentle** | 7.34×10¹³ | 1.47×10¹⁴ | 🥇 1st |
| **Standard** | 1.23×10⁵ | 2.47×10⁵ | 🥈 2nd |
| **Oscillating** | 326.77 | 654.90 | 🥉 3rd |
| **Chaotic** | 72.71 | 97.97 | 4th |
| **Harsh** | 0.00053 | 0.0013 | 5th |

**Ratio**: Gentle/Harsh = 1.38×10¹⁷ (17 orders of magnitude!)

### **Hybrid Genome - Environment Response**

| Environment | Fitness | Std Dev | Rank |
|-------------|---------|---------|------|
| **Harsh** | 2.03×10¹⁵ | 4.06×10¹⁵ | 🥇 1st |
| **Chaotic** | 1.11×10¹⁵ | 2.22×10¹⁵ | 🥈 2nd |
| **Oscillating** | 1.89×10⁷ | 3.77×10⁷ | 🥉 3rd |
| **Standard** | -53,682 | 107,365 | 4th |
| **Gentle** | -98,252 | 215,367 | 5th |

**Inverse specialization**: Thrives where others fail!

### **Environment Difficulty Ranking**

Based on averaged ensemble performance:

1. **Gentle** (easiest): 2661.81 fitness
2. **Oscillating**: 179.88 fitness
3. **Harsh**: 1.50 fitness
4. **Standard**: 1.44 fitness
5. **Chaotic** (hardest): 1.15 fitness

---

## 8. Key Insights

### **🔑 Finding 1: Bimodal Mutation Strategy**
- **Low mutation (0.3-0.7)**: Co-evolution, stable convergence
- **High mutation (1.5-3.0)**: Explosive fitness in specific environments
- No successful genomes in middle range (0.8-1.4)

### **🔑 Finding 2: Decoherence Optimal**
- **d = 0.011** appears across multiple top genomes
- Co-evolved Gen 5878: 0.011
- Best individual: 0.011
- Phase-focused: 0.011
- **Hypothesis**: Critical value for coherence-fitness balance

### **🔑 Finding 3: Environment Specialization**
- Standard GA genomes: Excel in gentle environments
- Hybrid genomes: Excel in harsh/chaotic environments
- **No single genome dominates all environments**
- Trade-off between specialization and generalization

### **🔑 Finding 4: Co-Evolution Learning Plateau**
- Learning success rate stable at ~36% across generations
- Fitness improves 770→2117, then drops at 5878
- **Possible overfitting** or fitness function change

### **🔑 Finding 5: Ensemble Averaging Effect**
- Averaged genomes: Much lower fitness than best individuals
- But: **More robust** across environments (lower std dev)
- **Robustness vs Performance trade-off**

### **🔑 Finding 6: Extreme Fitness Magnitudes**
- Standard GA: 10¹¹-10¹³ scale
- Hybrid: 10¹⁵ scale
- Co-evolution: 1-2 scale
- **Different fitness landscapes** or calculation methods?

### **🔑 Finding 7: Phase Offset Patterns**
- Typical range: 0.2-0.6
- Phase-focused: 3.19 (≈ π, opposite phase)
- **π phase**: Special attractor in phase space

### **🔑 Finding 8: Oscillation Frequency Stabilization**
- Co-evolution: 1.69 → 0.25 → 0.26 (stabilizes)
- Best individuals: Stay in 0.7-1.0 range
- **Low oscillation** correlates with maturity

---

## 🎯 Recommendations

### **For Improving Fitness:**
1. **Explore high mutation regime** (μ = 2.5-3.5)
2. **Fix decoherence at 0.011** (optimal value)
3. **Target gentle environment** for maximum fitness
4. **Use hybrid ensemble** for harsh environment specialization
5. **Test π/2 and 3π/2 phase offsets** (quadrature phases)

### **For Robust Agents:**
1. **Use averaged ensemble approach** (lower fitness, higher stability)
2. **Target oscillation freq 0.7-1.0** (balanced)
3. **Test in chaotic environment** (worst-case scenario)
4. **Co-evolution for generalization** (consistent learning)

### **For Research:**
1. **Investigate negative fitness** in Gen 5878
2. **Understand fitness scale differences** (10⁰ vs 10¹⁵)
3. **Test middle mutation range** (0.8-1.4) - unexplored
4. **Analyze d=0.011 optimality** - why this specific value?
5. **Build multi-environment genome** that doesn't specialize

---

## 📂 File Locations

```
quantum_genetics/data/genomes/production/
├── co_evolved_best_gen_770.json
├── co_evolved_best_gen_2117.json
├── co_evolved_best_gen_5878.json
├── best_individual_genome.json
├── best_individual_hybrid_genome.json
├── best_individual_long_evolution_genome.json
├── best_individual_more_populations_genome.json
├── averaged_ensemble_genome.json
├── averaged_hybrid_genome.json
├── averaged_long_evolution_genome.json
├── averaged_more_populations_genome.json
├── phase_focused_best.json
├── custom_1761985337_genome.json
└── custom_1761985388_genome.json
```

---

## 🚀 Quick Commands

### Load and Test a Genome
```python
import json
from quantum_genetic_agents import QuantumAgent

# Load best hybrid genome
with open('data/genomes/production/best_individual_hybrid_genome.json') as f:
    data = json.load(f)
    genome = [
        data['genome']['mutation_rate'],
        data['genome']['oscillation_freq'],
        data['genome']['decoherence_rate'],
        data['genome']['phase_offset']
    ]

# Test in different environments
for env in ['standard', 'harsh', 'gentle', 'chaotic', 'oscillating']:
    agent = QuantumAgent(0, genome, environment=env)
    for t in range(1, 80):
        agent.evolve(t)
    print(f"{env:12s}: {agent.get_final_fitness():.6e}")
```

### Compare All Genomes
```python
import json
import os
from glob import glob

results = []
for file in glob('data/genomes/production/*.json'):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'file': os.path.basename(file),
            'type': data['metadata']['type'],
            'generation': data['metadata'].get('generation', 0),
            'fitness': data['metadata'].get('fitness', 0),
            'mutation': data['genome']['mutation_rate'],
            'decoherence': data['genome']['decoherence_rate']
        })

# Sort by fitness
results.sort(key=lambda x: x['fitness'], reverse=True)
for r in results[:5]:  # Top 5
    print(f"{r['file']:40s} | Fitness: {r['fitness']:.2e} | μ: {r['mutation']:.4f}")
```

---

**End of Genome Data Analysis** 🧬⚛️
