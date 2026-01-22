# Simulation Research
## Government & Complex Systems Simulation

This repository contains complex systems simulation research:
- Government simulation models
- Multiscale dynamics integration
- Agent-based modeling for complex systems

## Branch Migration Status

This repository has been set up with simulation research code organized into dedicated branches.

### Migrated Branches

The following branches have been created with simulation research frameworks:

1. **`government-simulation-research`**
   - Agent-based government simulation framework
   - Legislative process modeling
   - Citizen satisfaction dynamics
   - Policy impact analysis
   - Contains: `government_simulation/` module with agents, simulation engine, and examples

2. **`copilot/integrate-multiscale-dynamics`**
   - Multiscale dynamics integration framework
   - Scale management (micro, meso, macro, meta levels)
   - Cross-scale coupling mechanisms
   - Temporal and spatial scale transformations
   - Contains: `multiscale_dynamics/` module with scale definitions, simulation engine, and examples

### Migration Notes

- Each branch contains a complete, self-contained simulation framework
- All code includes comprehensive documentation and example scripts
- Directory structures preserve modular organization
- No modifications to core simulation logic during setup

## Setup

### Accessing Branch Content

To work with a specific simulation framework, checkout the corresponding branch:

```bash
# For government simulation research
git checkout government-simulation-research

# For multiscale dynamics
git checkout copilot/integrate-multiscale-dynamics
```

### Running Examples

Each branch includes example scripts in the `examples/` directory:

```bash
# On government-simulation-research branch
python examples/basic_simulation.py

# On copilot/integrate-multiscale-dynamics branch
python examples/multiscale_example.py
```

### Requirements

Both frameworks use standard Python libraries:
- Python 3.7+
- No external dependencies for core functionality
- Optional: numpy, matplotlib for advanced analysis and visualization

More documentation available in each branch's module README.