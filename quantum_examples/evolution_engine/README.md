# Evolution Engine

This directory contains genetic algorithm implementations for evolving quantum agents.

## Available Engines

### Main Evolution Systems
- **quantum_genetic_agents.py** - Primary evolution engine with parallel populations
  - 15 parallel populations
  - Multi-environment testing
  - Exports best individual and averaged ensemble genomes

### Specialized Evolution
- **multi_objective_evolution.py** - Optimize multiple objectives simultaneously
  - Pareto frontier optimization
  - Trade-off analysis between competing objectives

- **phase_focused_evolution.py** - Focus on phase parameter optimization
  - Specialized for quantum phase dynamics
  - Fine-tuned phase control

- **quantum_evolution_agents.py** - Quantum-specific evolutionary strategies
  - Quantum-inspired operators
  - Superposition-based selection

### Machine Learning
- **quantum_ml.py** - Machine learning genome prediction
  - Predict genome fitness
  - Learn optimal parameter combinations
  - Neural network-based evolution guidance

## Usage

Use the **Evolution Engine** workflow to run the main evolution engine, or run any evolution experiment directly:
```bash
python evolution_engine/quantum_genetic_agents.py
python evolution_engine/multi_objective_evolution.py
python evolution_engine/phase_focused_evolution.py
# ... or any other evolution experiment
```

## Output

Evolution engines produce:
- `best_individual_genome.json` - The best performing genome
- `averaged_ensemble_genome.json` - Averaged parameters from top performers
- Various analysis plots in the root directory

## Performance

Evolution experiments typically take several minutes to complete:
- Quick test: ~2-5 minutes
- Standard run: ~5-15 minutes
- Long evolution: ~30+ minutes

## Configuration

Edit `config.py` in the root directory to adjust:
- Number of populations
- Population size
- Number of generations
- Mutation and crossover rates
