# Multiscale Dynamics Integration

## Overview

This branch (`copilot/integrate-multiscale-dynamics`) extends the government simulation framework with multiscale modeling capabilities, allowing simultaneous analysis at individual (micro), community (meso), and national (macro) levels.

## What is Multiscale Modeling?

Multiscale modeling recognizes that complex systems operate at multiple levels of organization simultaneously:

- **Micro Scale**: Individual agents (citizens) with unique characteristics and behaviors
- **Meso Scale**: Communities and local groups with emergent properties
- **Macro Scale**: System-wide patterns and aggregate statistics

### Key Concepts

1. **Scale Coupling**: Information flows both up (aggregation) and down (disaggregation) between scales
2. **Emergence**: Macro patterns emerge from micro interactions
3. **Downward Causation**: Macro conditions constrain micro behavior
4. **Scale Separation**: Understanding what variance exists at each level

## Architecture

### Core Components

```
src/simulations/
├── government_simulation.py      # Base government simulation
├── multiscale_dynamics.py        # Multiscale framework
└── integrated_multiscale.py      # Integration layer
```

### Class Hierarchy

```
MultiscaleModel (Base)
├── GovernmentMultiscaleModel
└── IntegratedGovernmentModel
```

## Usage Examples

### Basic Multiscale Simulation

```python
from src.simulations.multiscale_dynamics import GovernmentMultiscaleModel

# Create multiscale model
model = GovernmentMultiscaleModel(
    n_agents=1000,
    n_communities=10
)

# Run simulation
history = model.run(n_steps=50)

# Analyze results at each scale
final_state = history[-1]
print(f"Micro: {final_state['micro']}")
print(f"Meso: {final_state['meso']}")
print(f"Macro: {final_state['macro']}")
```

### Integrated Government + Multiscale

```python
from src.simulations.integrated_multiscale import IntegratedGovernmentModel
from src.simulations.government_simulation import create_example_policies

# Create integrated model
model = IntegratedGovernmentModel(
    population_size=5000,
    n_communities=20,
    time_steps=100
)

# Get policies
policies = create_example_policies()

# Run with policies
history = model.run(policies)

# Analyze cross-scale dynamics
analysis = model.analyze_cross_scale_dynamics()
print(analysis['correlations'])
print(analysis['volatility'])
```

### Scale Transitions

```python
# Identify when dynamics shift between scales
transitions = model.identify_scale_transitions(threshold=2.0)

for t in transitions:
    print(f"Step {t['step']}: {t['type']} - {t['description']}")
```

## Key Features

### 1. Automatic Scale Coupling

The framework automatically maintains consistency across scales:

```python
# Upscaling: Aggregate micro → meso → macro
meso_data = model.upscale(ScaleLevel.MICRO, ScaleLevel.MESO)
macro_data = model.upscale(ScaleLevel.MESO, ScaleLevel.MACRO)

# Downscaling: Disaggregate macro → meso → micro
meso_data = model.downscale(ScaleLevel.MACRO, ScaleLevel.MESO)
micro_data = model.downscale(ScaleLevel.MESO, ScaleLevel.MICRO)
```

### 2. Cross-Scale Analysis

Analyze relationships between scales:

```python
analysis = model.analyze_cross_scale_dynamics()

# Correlations between scales
print(analysis['correlations']['micro_meso'])
print(analysis['correlations']['meso_macro'])

# Volatility at each scale
print(analysis['volatility']['micro'])
print(analysis['volatility']['macro'])
```

### 3. Policy Impact Across Scales

Track policy effects at all levels:

```python
result = model.implement_policy_multiscale(policy)

print(result['micro_impact'])     # Individual-level effects
print(result['meso_impact'])      # Community-level effects
print(result['macro_impact'])     # National-level effects
```

### 4. Scale Separation Analysis

Understand variance at each level:

```python
from src.simulations.multiscale_dynamics import analyze_scale_separation

separation = analyze_scale_separation(model)

print(f"Total variance: {separation['total_variance']}")
print(f"Within-group variance: {separation['within_group_variance']}")
print(f"Between-group variance: {separation['between_group_variance']}")
print(f"Variance ratio: {separation['variance_ratio']}")
```

## Research Applications

### 1. Understanding Emergence

Study how macro patterns emerge from micro interactions:

```python
# Track satisfaction at all levels
micro_sat = [h['micro_metrics']['avg_satisfaction'] for h in history]
macro_eff = [h['macro_metrics']['policy_effectiveness'] for h in history]

# See how individual satisfaction relates to policy effectiveness
correlation = np.corrcoef(micro_sat, macro_eff)[0, 1]
```

### 2. Community Dynamics

Analyze meso-level patterns:

```python
# Community cohesion over time
cohesions = [h['meso_metrics']['avg_cohesion'] for h in history]

# Resource inequality between communities
inequality = [h['meso_metrics']['resource_inequality'] for h in history]
```

### 3. Policy Design

Design policies considering all scales:

```python
# Test policy at multiple scales
single_scale_result = gov_sim.implement_policy(policy)
multi_scale_result = integrated_model.implement_policy_multiscale(policy)

# Compare insights
print("Single-scale only sees:", single_scale_result.keys())
print("Multi-scale also sees:", multi_scale_result.keys())
```

### 4. Scale Transitions

Identify critical transitions:

```python
# Find when micro changes cascade to macro
transitions = model.identify_scale_transitions()

cross_scale_events = [t for t in transitions if t['type'] == 'cross_scale']
print(f"Found {len(cross_scale_events)} cross-scale transitions")
```

## Methodological Advantages

### When to Use Multiscale Modeling

Use multiscale approaches when:

1. **Community Structure Matters**: Local interactions and group dynamics are important
2. **Emergence is Key**: Need to see how macro patterns arise from micro behavior
3. **Multiple Levels of Intervention**: Policies can target individuals, communities, or nation
4. **Scale-Dependent Phenomena**: Effects vary by scale of observation

### Comparison with Single-Scale

```python
from src.simulations.integrated_multiscale import compare_single_vs_multiscale

comparison = compare_single_vs_multiscale(
    policies=policies,
    population_size=1000,
    time_steps=50
)

print(comparison['insights']['use_multiscale_when'])
```

## Technical Details

### Scale States

Each scale maintains its own state:

```python
class ScaleState:
    level: ScaleLevel          # MICRO, MESO, or MACRO
    variables: Dict[str, Any]  # Scale-specific variables
    timestamp: int             # Current time
```

### Upscaling Operations

- **Micro → Meso**: Group agents spatially, calculate local averages
- **Meso → Macro**: Aggregate community statistics to national level

### Downscaling Operations

- **Macro → Meso**: Distribute national statistics to communities
- **Meso → Micro**: Disaggregate community properties to individuals

### Synchronization

The integrated model maintains bidirectional synchronization:

```python
def _synchronize_states(self):
    """Keep government sim and multiscale model aligned"""
    # Copy agent states
    for gov_agent, multi_agent in zip(gov_agents, multi_agents):
        multi_agent['state'] = gov_agent.satisfaction
        multi_agent['wealth'] = gov_agent.wealth
```

## Performance Considerations

### Computational Complexity

- **Single-Scale**: O(N) per time step
- **Multiscale**: O(N + C + 1) where C = number of communities
- **Coupling Overhead**: Additional O(N) for synchronization

### Optimization Tips

1. Use fewer communities for faster computation
2. Reduce coupling frequency if not needed every step
3. Use parallel processing for independent scales

### Memory Usage

Multiscale models store states at all levels:
- Micro: O(N) agents
- Meso: O(C) communities  
- Macro: O(1) global state

## Testing

Run tests for multiscale functionality:

```bash
# Test multiscale dynamics
pytest tests/test_multiscale_dynamics.py -v

# Test integration
pytest tests/test_integrated_multiscale.py -v

# Run all tests
pytest tests/ -v
```

## Advanced Topics

### Custom Scale Dynamics

Override update methods for custom behavior:

```python
class CustomMultiscaleModel(MultiscaleModel):
    def _update_micro(self):
        """Custom micro-level dynamics"""
        agents = self.micro_state.get_variable('agents', [])
        # Your custom logic here
        
    def _update_meso(self):
        """Custom meso-level dynamics"""
        # Your custom logic here
        
    def _update_macro(self):
        """Custom macro-level dynamics"""
        # Your custom logic here
```

### Alternative Aggregation Methods

Customize how information flows between scales:

```python
def _micro_to_meso(self):
    """Custom aggregation logic"""
    agents = self.micro_state.get_variable('agents', [])
    
    # Use network-based grouping instead of spatial
    # Use weighted averages instead of simple means
    # Consider temporal dynamics in aggregation
    
    return custom_groups
```

### Multi-Timescale Modeling

Run scales at different time steps:

```python
# Micro updates every step
# Meso updates every 5 steps
# Macro updates every 10 steps

if self.current_step % 5 == 0:
    self._update_meso()
if self.current_step % 10 == 0:
    self._update_macro()
```

## References

### Multiscale Modeling Theory

1. Weinan, E. (2011). Principles of Multiscale Modeling
2. Kevrekidis et al. (2003). Equation-free multiscale computation
3. Hoekstra et al. (2014). Multiscale modeling and simulation: A position paper

### Applications

1. Epstein, J. M. (2006). Generative Social Science (emergence)
2. Axelrod, R. (1997). The Complexity of Cooperation (scale transitions)
3. Arthur, W. B. (2015). Complexity and the Economy (economic multiscale)

## Contributing

Contributions to multiscale capabilities are welcome:

- New scale coupling methods
- Alternative aggregation/disaggregation schemes
- Performance optimizations
- Visualization tools for multiscale data
- Documentation improvements

## Citation

```bibtex
@software{multiscale_dynamics,
  title = {Multiscale Dynamics Integration for Government Simulation},
  author = {Simulation Research Team},
  year = {2024},
  url = {https://github.com/vikingdude81/vikingdude81-Simulation-Research}
}
```
