# Market Consciousness Experiments

Experiments exploring the intersection of consciousness theory and market dynamics, integrating SNN models with interface theory.

## Overview

This directory extends the existing interface theory experiments to model markets as conscious agent networks and test Hoffman's interface theory in trading contexts.

## Experiments

### 1. Market as Conscious Agent (`market_as_conscious_agent.py`)
**Goal**: Model market as hierarchical conscious agent network

Framework:
- Traders = micro-agents with local perceptions
- Market = macro-agent emerging from trader interactions
- Apply consciousness metrics (H_mode, PR, coherence) to market

**Key Questions**:
- Does market exhibit emergent conscious-like behavior?
- Can consciousness metrics predict regime changes?
- How do trader networks create market coherence?

### 2. SNN Interface Theory (`snn_interface_theory.py`)
**Goal**: SNNs as interface between trader perception and market reality

Tests Hoffman's "fitness beats truth" in trading:
- Evolved SNN strategies (fitness-optimized)
- Accurate market models (truth-seeking)
- Which performs better in real trading?

**Key Questions**:
- Do fitness-optimized SNNs outperform accurate models?
- What perceptual interfaces emerge through evolution?
- How does interface compression affect trading performance?

## Connection to Existing Work

Builds on:
- `interface_theory_experiments/conscious_agent_network.py`
- `interface_theory_experiments/hierarchical_agents.py`
- `interface_theory_experiments/hoffman_fitness_vs_truth.py`

## Usage

```bash
python interface_theory_experiments/market_consciousness/market_as_conscious_agent.py
python interface_theory_experiments/market_consciousness/snn_interface_theory.py
```

## Dependencies

- PyTorch >= 2.0.0
- NumPy
- Existing interface theory utilities
- SNN models from `models/snn_trading_agent.py`

## Results

Results are saved to:
- `outputs/market_consciousness/` - Experiment outputs
- Console logs with consciousness metrics
