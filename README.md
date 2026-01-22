# Government & Complex Systems Simulation Research

This repository contains advanced simulation frameworks for studying government policy impacts, complex systems dynamics, and agent-based modeling.

## Overview

The simulation research focuses on understanding emergent behaviors in complex social, economic, and political systems through computational modeling.

## Branch: copilot/integrate-multiscale-dynamics

This branch integrates multiscale modeling capabilities with government policy simulation, enabling analysis at micro (individual), meso (community), and macro (national) levels simultaneously.

### Key Features

- **Agent-Based Modeling**: Sophisticated agent models representing citizens with diverse characteristics
- **Policy Simulation**: Framework for testing different policy interventions and measuring outcomes
- **Multiscale Dynamics**: Simultaneous modeling at micro, meso, and macro scales
- **Scale Coupling**: Automatic aggregation (upscaling) and disaggregation (downscaling) between levels
- **Cross-Scale Analysis**: Tools to analyze correlations and dynamics across scales
- **Economic Modeling**: Economic agents that make rational decisions based on market conditions
- **Social Network Dynamics**: Social agents that form opinions through network interactions
- **Adaptive Behavior**: Agents that learn and adapt strategies based on experience
- **Integrated Framework**: Seamless integration of single-scale and multiscale approaches

## Project Structure

```
src/
├── simulations/        # Core simulation engines
│   ├── government_simulation.py      # Base government policy simulation
│   ├── multiscale_dynamics.py        # Multiscale modeling framework
│   └── integrated_multiscale.py      # Integration layer
├── models/            # Agent-based models
│   └── agent_based_models.py
├── analysis/          # Analysis and visualization tools
│   └── simulation_analyzer.py
└── utils/             # Utility functions
    └── simulation_utils.py
docs/
└── MULTISCALE_INTEGRATION.md         # Detailed multiscale documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vikingdude81/vikingdude81-Simulation-Research.git
cd vikingdude81-Simulation-Research

# Checkout the multiscale dynamics branch
git checkout copilot/integrate-multiscale-dynamics

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running an Integrated Multiscale Simulation

```python
from src.simulations.integrated_multiscale import IntegratedGovernmentModel
from src.simulations.government_simulation import create_example_policies

# Initialize integrated model
model = IntegratedGovernmentModel(
    population_size=5000,
    n_communities=20,
    time_steps=100
)

# Create and implement policies
policies = create_example_policies()
history = model.run(policies)

# Analyze cross-scale dynamics
analysis = model.analyze_cross_scale_dynamics()
print(analysis['correlations'])  # See how scales relate
print(analysis['volatility'])     # Compare scale volatility
```

### Running a Basic Government Simulation

```python
from src.simulations.government_simulation import GovernmentSimulation, create_example_policies

# Initialize simulation
sim = GovernmentSimulation(
    population_size=10000,
    initial_budget=1000000,
    time_steps=100
)

# Create and implement policies
policies = create_example_policies()
history = sim.run_simulation(policies)

# Get results
summary = sim.get_summary_statistics()
print(summary)
```

### Running a Pure Multiscale Model

```python
from src.simulations.multiscale_dynamics import GovernmentMultiscaleModel

# Initialize multiscale model
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

### Analyzing Results

```python
from src.analysis.simulation_analyzer import SimulationAnalyzer

# Analyze simulation results
analyzer = SimulationAnalyzer(history)
print(analyzer.generate_report())

# Export data for further analysis
analyzer.export_data('simulation_results.json')
```

### Creating Custom Agents

```python
from src.models.agent_based_models import EconomicAgent, SocialAgent

# Create economic agents
economic_agents = [
    EconomicAgent(i, initial_wealth=1000) 
    for i in range(100)
]

# Create social network
social_agents = [SocialAgent(i) for i in range(100)]

# Connect agents in network
for i in range(len(social_agents) - 1):
    social_agents[i].add_neighbor(social_agents[i + 1])
```

## Simulation Methodologies

### Agent-Based Modeling (ABM)

Our framework uses agent-based modeling to simulate complex systems from the bottom up:

1. **Heterogeneous Agents**: Agents have diverse characteristics (wealth, education, preferences)
2. **Local Interactions**: Agents interact with neighbors in their network
3. **Emergent Behavior**: System-level patterns emerge from individual actions
4. **Adaptive Learning**: Agents learn and adapt strategies based on outcomes

### Policy Impact Assessment

The government simulation evaluates policies across multiple dimensions:

- **Economic Impact**: Changes in wealth distribution, inequality
- **Social Impact**: Citizen satisfaction and well-being
- **Budget Constraints**: Realistic fiscal limitations
- **Temporal Dynamics**: Short-term vs. long-term effects

### Network Topologies

Multiple network structures are supported:

- **Random Networks**: Erdős-Rényi graphs
- **Small World Networks**: Watts-Strogatz model
- **Scale-Free Networks**: Barabási-Albert model
- **Ring Networks**: Structured local connectivity

## Key Metrics

### Population-Level Metrics
- Average satisfaction
- Wealth distribution (mean, median, Gini coefficient)
- Social cohesion (network clustering)
- Opinion polarization

### Policy Effectiveness Metrics
- Cost-benefit ratio
- Implementation success rate
- Long-term sustainability
- Equity impact

## Advanced Features

### Multi-Scale Dynamics

The framework supports analyzing systems at multiple scales:
- **Micro**: Individual agent decisions
- **Meso**: Local network dynamics
- **Macro**: System-wide patterns

### Scenario Comparison

Compare different policy scenarios:

```python
# Run baseline scenario
sim1 = GovernmentSimulation(population_size=1000)
history1 = sim1.run_simulation([])

# Run intervention scenario
sim2 = GovernmentSimulation(population_size=1000)
history2 = sim2.run_simulation(policies)

# Compare results
analyzer1 = SimulationAnalyzer(history1)
analyzer2 = SimulationAnalyzer(history2)
comparison = analyzer1.compare_scenarios(analyzer2)
```

### Sensitivity Analysis

Test robustness of findings:

```python
# Run multiple simulations with different parameters
results = []
for population_size in [1000, 5000, 10000]:
    sim = GovernmentSimulation(population_size=population_size)
    history = sim.run_simulation(policies)
    results.append(history)
```

## Research Applications

### Policy Design
- Test policy interventions before implementation
- Optimize policy parameters for desired outcomes
- Identify unintended consequences

### Economic Modeling
- Study wealth inequality dynamics
- Analyze market behaviors
- Model economic interventions

### Social Dynamics
- Opinion formation and polarization
- Information diffusion in networks
- Social influence mechanisms

### Complex Systems Analysis
- Emergent phenomena identification
- Phase transitions and critical points
- Resilience and stability analysis

## Migrated from crypto-ml-trading-system

This simulation research framework was migrated from the original `crypto-ml-trading-system` repository to focus specifically on complex systems and government policy simulation research. The migration preserved:

- Complete commit history
- All model implementations
- Analysis tools and utilities
- Documentation and examples

## Contributing

Contributions are welcome! Areas of interest:

- New agent types and behaviors
- Additional policy models
- Enhanced analysis tools
- Visualization capabilities
- Performance optimizations

## Testing

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_government_simulation.py
```

## Documentation

Additional documentation available in the `docs/` directory:

- Architecture overview
- API reference
- Tutorial notebooks
- Best practices

## License

This research code is provided for academic and research purposes.

## Contact

For questions about this simulation research:
- Open an issue on GitHub
- See repository documentation

## References

Key methodological references:

1. Agent-Based Modeling: Epstein, J. M. (2006). Generative Social Science
2. Complex Systems: Miller & Page (2007). Complex Adaptive Systems
3. Network Science: Barabási (2016). Network Science
4. Policy Simulation: Gilbert & Troitzsch (2005). Simulation for the Social Scientist

## Citation

If you use this simulation framework in your research, please cite:

```
@software{simulation_research,
  title = {Government and Complex Systems Simulation Framework},
  author = {Simulation Research Team},
  year = {2024},
  url = {https://github.com/vikingdude81/vikingdude81-Simulation-Research}
}
```